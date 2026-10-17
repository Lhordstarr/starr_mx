/**
 * Local model management through the Ollama daemon (pull / list / delete).
 *
 * Separate from providers/ollama.ts, which only speaks the chat API: these
 * are daemon-admin calls, and they belong to the settings page, not to a
 * conversation. The base URL mirrors the chat provider's default so both
 * follow OLLAMA_BASE_URL; it is read per call (not at module load) so tests
 * can point it at a stub.
 */
import { parseNDJSON } from './sse.js';

const defaultBaseUrl = () => process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434';

export interface LocalModel {
  name: string;
  size: number;
  modifiedAt: string;
  parameterSize?: string;
  quantization?: string;
  family?: string;
}

export interface PullProgress {
  status: string;
  digest?: string;
  total?: number;
  completed?: number;
}

/** Fetch that reports a dead daemon plainly instead of a bare TypeError. */
async function daemonFetch(path: string, init: RequestInit, signal?: AbortSignal): Promise<Response> {
  const timeout = AbortSignal.timeout(30_000);
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;
  try {
    return await fetch(`${defaultBaseUrl()}${path}`, { ...init, signal: combined });
  } catch (err) {
    // The caller's own abort (a cancelled pull, a caller's short budget) is
    // theirs to interpret. A *timeout* is ours — and it is not the same story
    // as a dead daemon: an Ollama busy loading a model looks exactly like
    // this, and "not reachable" would send the user to restart a daemon that
    // is working perfectly.
    if (signal?.aborted) throw err;
    if (err instanceof Error && err.name === 'TimeoutError') {
      throw new Error('Ollama did not answer within 30s — it is busy loading or replying.');
    }
    throw new Error(`Ollama is not reachable at ${defaultBaseUrl()}. Is the daemon running?`);
  }
}

function daemonError(res: Response, body: string): Error {
  // Ollama's own error text ({ "error": "..." }) is the useful half.
  try {
    const parsed = JSON.parse(body) as { error?: string };
    if (parsed?.error) return new Error(`Ollama: ${parsed.error}`);
  } catch {
    /* not JSON — fall through to the status */
  }
  return new Error(`Ollama error ${res.status}: ${body.slice(0, 200) || res.statusText}`);
}

/**
 * Model names as Ollama accepts them: registry paths, tags, versions.
 * Deliberately narrow — this travels to the daemon untouched.
 */
export function validateModelName(name: unknown): string | null {
  if (typeof name !== 'string' || !name.trim()) return 'Model name is required';
  if (name.length > 200) return 'Model name is too long';
  if (!/^[A-Za-z0-9._:/-]+$/.test(name)) return 'Invalid model name';
  return null;
}

/** Installed models, most recently modified first (mirrors the chat picker). */
export async function listLocalModels(signal?: AbortSignal): Promise<LocalModel[]> {
  const res = await daemonFetch('/api/tags', {}, signal);
  const body = await res.text();
  if (!res.ok) throw daemonError(res, body);
  const data = JSON.parse(body) as {
    models?: {
      name?: string;
      size?: number;
      modified_at?: string;
      details?: { parameter_size?: string; quantization_level?: string; family?: string };
    }[];
  };
  return (data.models ?? [])
    .filter((m) => typeof m.name === 'string')
    .map((m) => ({
      name: m.name as string,
      size: typeof m.size === 'number' ? m.size : 0,
      modifiedAt: m.modified_at ?? '',
      parameterSize: m.details?.parameter_size,
      quantization: m.details?.quantization_level,
      family: m.details?.family,
    }))
    .sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt));
}

/**
 * Pulls a model, yielding progress lines as the daemon reports them
 * (pulling manifest, downloading with completed/total, verifying, writing,
 * success). The caller stops at `status: 'success'`; an `error` line throws.
 */
export async function* pullModel(
  name: string,
  signal?: AbortSignal,
): AsyncGenerator<PullProgress, void, unknown> {
  const res = await daemonFetch(
    '/api/pull',
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ name, stream: true }),
    },
    signal,
  );
  if (!res.ok) throw daemonError(res, await res.text());
  for await (const event of parseNDJSON(res.body!)) {
    if (typeof event?.error === 'string' && event.error) {
      throw new Error(`Ollama: ${event.error}`);
    }
    if (typeof event?.status === 'string') {
      yield {
        status: event.status,
        digest: typeof event.digest === 'string' ? event.digest : undefined,
        total: typeof event.total === 'number' ? event.total : undefined,
        completed: typeof event.completed === 'number' ? event.completed : undefined,
      };
      if (event.status === 'success') return;
    }
  }
}

/** Removes an installed model. */
export async function deleteLocalModel(name: string, signal?: AbortSignal): Promise<void> {
  const res = await daemonFetch(
    '/api/delete',
    {
      method: 'DELETE',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ name }),
    },
    signal,
  );
  if (!res.ok) throw daemonError(res, await res.text());
}

/* ---- Running models ---- */

/** A model the daemon currently holds in memory, from `/api/ps`. */
export interface RunningModel {
  name: string;
  /** Total bytes the model occupies. */
  sizeBytes: number;
  /** How much of that is resident in video memory. */
  vramBytes: number;
  /** Ollama's own expiry, e.g. "5 minutes from now", or "never". */
  expires: string;
  device: 'gpu' | 'cpu' | 'mixed';
}

const mb = (n: number) => `${(n / 1048576).toFixed(0)} MB`;

/**
 * Models the daemon is holding right now.
 *
 * This is the list behind the resource manager: a model stays resident after a
 * reply finishes, holding its VRAM and RAM until Ollama's keep-alive expires or
 * it is told to let go. That is why a chat can look like it is still using the
 * GPU long after the last token arrived.
 */
export async function listRunningModels(signal?: AbortSignal): Promise<RunningModel[]> {
  const res = await daemonFetch('/api/ps', {}, signal);
  if (!res.ok) throw daemonError(res, await res.text());
  const data = JSON.parse(await res.text()) as {
    models?: {
      name?: string;
      size?: number;
      size_vram?: number;
      expires_at?: string;
    }[];
  };
  return (data.models ?? [])
    .filter((m) => typeof m.name === 'string' && m.name)
    .map((m) => {
      const size = m.size ?? 0;
      const vram = m.size_vram ?? 0;
      // The same split backends.ts reports: equal is entirely on the GPU,
      // zero is entirely on the CPU, and anything between is a model larger
      // than the card running on both at once.
      const frac = size > 0 ? vram / size : 0;
      const device: RunningModel['device'] =
        frac > 0.001 && frac < 0.999 ? 'mixed' : frac >= 0.999 ? 'gpu' : 'cpu';
      const expires = m.expires_at ? new Date(m.expires_at) : null;
      const expiresText =
        expires && !Number.isNaN(expires.getTime())
          ? Math.max(0, Math.round((expires.getTime() - Date.now()) / 1000))
          : null;
      return {
        name: m.name as string,
        sizeBytes: size,
        vramBytes: vram,
        expires:
          expiresText === null
            ? 'unknown'
            : expiresText > 90
              ? `${Math.round(expiresText / 60)} min`
              : `${Math.round(expiresText)}s`,
        device,
      };
    })
    .sort((a, b) => b.sizeBytes - a.sizeBytes);
}

/**
 * Releases a loaded model, freeing the memory it was holding.
 *
 * Ollama has no "unload" endpoint: `keep_alive: 0` on a generate request with
 * an empty prompt is the documented way to say "done with this one", and it is
 * what the daemon itself acts on when its own timer expires. Unloading the model
 * is the reversible, correct way to free a GPU — killing the daemon instead
 * would take every model down with it and break the app's own chat.
 */
export async function unloadModel(name: string, signal?: AbortSignal): Promise<void> {
  const problem = validateModelName(name);
  if (problem) throw new Error(problem);
  const res = await daemonFetch(
    '/api/generate',
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: name, keep_alive: 0, prompt: '' }),
    },
    signal,
  );
  if (!res.ok) throw daemonError(res, await res.text());
  // The daemon streams a JSON line; drain it so the socket is released.
  await res.text().catch(() => '');
}

export { mb as formatModelBytes };

/** One pull at a time: concurrent pulls thrash the disk and the progress UI. */
let pulling: string | null = null;

export function acquirePull(name: string): boolean {
  if (pulling) return false;
  pulling = name;
  return true;
}

export function releasePull(): void {
  pulling = null;
}
