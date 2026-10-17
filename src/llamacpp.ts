/**
 * The GGUF models on this machine, and the llama-server that runs them.
 *
 * llama.cpp was installable but unusable. The app fetched the binary, the
 * models page downloaded GGUF files, and nothing connected the two: the chat
 * screen's provider picker offered Ollama and whatever server the user had
 * started themselves, while a freshly downloaded Qwen3 GGUF sat on disk
 * invisible to every picker. This module is the missing half — it finds the
 * GGUF files (the app's own downloads and the shared Hugging Face cache other
 * tools wrote), decides which of them are chat models, and starts llama-server
 * with one loaded when a chat actually asks for it.
 *
 * Model ids are display names, not paths: `Qwen3-4B-Q4_K_M`. That is what the
 * composer chip and the conversation store, and what llama-server is told to
 * report through `--alias`, so the id survives a restart of the server and the
 * file moving between the downloads directory and the HF cache.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync } from 'node:fs';
import * as path from 'node:path';

import { listDownloads } from './huggingface.js';
import { CATALOG } from './model-catalog.js';
import { llamaServerPath } from './runtimes.js';

/**
 * Where the managed server listens. Deliberately not upstream's default of
 * 9931: anyone running their own llama-server gets that port, and a collision
 * would fail at bind time and read as "llama.cpp is broken" instead of "the
 * port is taken".
 */
const port = () => Number(process.env.LLAMACPP_PORT ?? 9932);

/**
 * Prompt context the server is started with.
 *
 * llama.cpp's default is the model's trained context, which for a modern GGUF
 * can be 128k — a KV cache larger than the model itself, allocated on a machine
 * that is usually CPU-only here. 16k is more conversation than the chat
 * transcript ever accumulates, and it keeps the memory footprint predictable.
 */
const ctxSize = () => Number(process.env.LLAMACPP_CTX ?? 16384);

/** How long the server may sit unused before it is shut down. */
const idleMs = () => Number(process.env.LLAMACPP_IDLE_MS ?? 10 * 60_000);

/** How long a model may take to load before the attempt is abandoned. */
const loadTimeoutMs = () => Number(process.env.LLAMACPP_LOAD_MS ?? 10 * 60_000);

/** How long a cached GGUF listing is trusted. */
const CACHE_MS = 5000;

/**
 * GGUF shapes that are not chat models.
 *
 * A GGUF is llama.cpp's format, but not all of llama.cpp is a chat model:
 * whisper GGUFs belong to whisper.cpp, a `mmproj` is a vision projector, and
 * the diffusion/vocoder files are other programs' weights. Offering them in a
 * chat picker would produce a model that fails to load with a loader error,
 * which is exactly the "downloaded it and nothing works" experience this
 * module exists to end.
 */
const NOT_CHAT =
  /(whisper|wav2vec|parakeet|vocoder|tts|mmproj|\bclip\b|\bvae\b|unet|controlnet|diffusion|stable-diffusion|sdxl|sd3|flux|embed|rerank)/i;

/**
 * Repos the catalog already knows are not chat models.
 *
 * The file name is a good signal and the catalog entry is a better one: the
 * catalog states the task outright, so a whisper GGUF is classified from its
 * task rather than from the word "whisper" happening to be in its name.
 */
const NON_TEXT_REPOS = new Map(
  CATALOG.flatMap((section) =>
    section.task === 'text'
      ? []
      : section.entries.map((e) => [e.repo, section.task] as const),
  ),
);

export interface GgufModel {
  /** Display name and the id the chat sends: the file name without `.gguf`. */
  id: string;
  repo: string;
  file: string;
  /** Absolute path to the weights. */
  path: string;
  sizeBytes: number;
  /** Where the bytes live: this app's downloads dir, or the shared HF cache. */
  location: 'models' | 'cache';
  /** True when llama-server can hold a conversation with this file. */
  chat: boolean;
  /** Why it cannot, when it cannot. */
  why?: string;
}

/** The file's name without its extension — the model id, before collisions. */
function baseId(file: string): string {
  const name = file.split('/').pop() ?? file;
  return name.replace(/\.gguf$/i, '');
}

let cached: { at: number; models: GgufModel[] } | null = null;

/**
 * Every GGUF file on this machine, with the chat models marked.
 *
 * Returned sorted by id, and cached briefly: the chat page asks for the
 * provider list on every load and every model switch, and walking the
 * downloads directory plus the whole Hugging Face cache each time would put a
 * filesystem walk in front of every repaint.
 */
export async function ggufModels(force = false): Promise<GgufModel[]> {
  if (!force && cached && Date.now() - cached.at < CACHE_MS) return cached.models;

  const files = await listDownloads().catch(() => []);
  const models: GgufModel[] = [];
  for (const f of files) {
    if (!/\.gguf$/i.test(f.file)) continue;
    const task = NON_TEXT_REPOS.get(f.repo);
    const chat = !task && !NOT_CHAT.test(f.file);
    models.push({
      id: baseId(f.file),
      repo: f.repo,
      file: f.file,
      // `root` is the repo directory the downloader wrote to; the file name may
      // contain subdirectories of its own, which are joined one segment at a
      // time so a slash in a name can never escape that root.
      path: path.join(f.root, ...f.file.split('/')),
      sizeBytes: f.size,
      location: f.location,
      chat,
      why: chat ? undefined : task ? `it is a ${task} model` : 'it is not a text model',
    });
  }

  // Two repos can ship the same file name. Qualifying only the collisions keeps
  // the common case short — `Qwen3-4B-Q4_K_M`, not `unsloth/Qwen3-4B-Q4_K_M`.
  const counts = new Map<string, number>();
  for (const m of models) counts.set(m.id, (counts.get(m.id) ?? 0) + 1);
  for (const m of models) {
    if ((counts.get(m.id) ?? 0) > 1) m.id = `${m.repo}/${m.id}`;
  }

  // Case-insensitive ordering, so `Qwen3-4B` sits with the q's rather than
  // above every lowercase name in the picker.
  models.sort((a, b) => a.id.localeCompare(b.id));
  cached = { at: Date.now(), models };
  return models;
}

/** The chat models only: what the provider offers. */
export async function ggufChatModels(force = false): Promise<GgufModel[]> {
  return (await ggufModels(force)).filter((m) => m.chat);
}

/** The last listing, without touching the filesystem. */
export function cachedGgufModels(): GgufModel[] {
  return cached?.models ?? [];
}

/**
 * Finds a model by the id the chat was told to use.
 *
 * Deliberately forgiving: a conversation can outlive a file being renamed or
 * moved, and a user can type an id by hand, so the file name and the lowercased
 * name are both accepted before giving up.
 */
export async function resolveGgufModel(id: string): Promise<GgufModel | null> {
  const want = String(id ?? '').trim();
  if (!want) return null;
  const all = await ggufModels();
  return (
    all.find((m) => m.id === want) ??
    all.find((m) => m.file === want) ??
    all.find((m) => baseId(m.file) === baseId(want)) ??
    all.find((m) => m.id.toLowerCase() === want.toLowerCase()) ??
    null
  );
}

interface Serving {
  child: ChildProcess;
  model: string;
}

/**
 * The supervisor llama-server actually runs under.
 *
 * A child process cannot tell that its parent was killed outright — SIGKILL
 * from a `tsx watch` reload, a crash, `kill -9` — so a plain spawn leaves
 * llama-server reparented to init, holding the model in RAM and the port this
 * app believes is free. The next chat then fails against a server it cannot
 * see. (That is not hypothetical: it happened here, and the orphan had to be
 * found by hand.)
 *
 * So the binary is started through a one-liner Node process whose stdin is a
 * pipe from the app. When the app goes, for any reason at all, the pipe closes,
 * and the supervisor stops the server and exits. It passes its own stdout and
 * stderr through, which is what keeps the app's error tail intact, and it
 * relays the server's exit code so a failure reads the same as before.
 */
const SUPERVISOR = `
const { spawn } = require('node:child_process');
const [bin, ...args] = process.argv.slice(1);
const child = spawn(bin, args, { stdio: ['ignore', 'inherit', 'inherit'] });
let stopping = false;
function stop() {
  if (stopping) return;
  stopping = true;
  try { child.kill('SIGTERM'); } catch {}
  const hard = setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, 4000);
  hard.unref();
}
child.on('exit', (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
process.on('SIGTERM', stop);
process.on('SIGINT', stop);
process.stdin.on('end', stop);
process.stdin.on('close', stop);
process.stdin.resume();
`;

let serving: Serving | null = null;
let starting: Promise<void> | null = null;
let idleTimer: NodeJS.Timeout | null = null;
/** The tail of the server's own output, kept for error messages. */
let tail: string[] = [];

/** The binary to run: the pinned build, or an override for tests and power users. */
function serverBinary(): string | null {
  const override = process.env.LLAMACPP_SERVER;
  if (override) return existsSync(override) ? override : null;
  const pinned = llamaServerPath();
  return pinned && existsSync(pinned) ? pinned : null;
}

export interface LlamaServerState {
  installed: boolean;
  running: boolean;
  model: string | null;
  pid: number | null;
}

export function llamaServerState(): LlamaServerState {
  return {
    installed: Boolean(serverBinary()),
    running: Boolean(serving),
    model: serving?.model ?? null,
    pid: serving?.child.pid ?? null,
  };
}

/** The last few lines the server printed, for an error that explains itself. */
function tailSuffix(): string {
  const last = tail.slice(-3).join(' · ');
  return last ? ` — llama-server said: ${last.slice(0, 400)}` : '';
}

function remember(chunk: Buffer) {
  for (const line of String(chunk).split(/\r?\n/)) {
    if (!line.trim()) continue;
    tail.push(line.trim());
    while (tail.length > 40) tail.shift();
  }
}

function clearIdle() {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = null;
}

/** Restarts the countdown: the server is shut down only once nothing is using it. */
function touchIdle() {
  clearIdle();
  const ms = idleMs();
  if (!Number.isFinite(ms) || ms <= 0) return;
  idleTimer = setTimeout(() => void stopLlamaServer(), ms);
  // Never a reason for the process to stay alive on its own.
  idleTimer.unref?.();
}

/**
 * Stops the managed server, politely first.
 *
 * SIGTERM, then SIGKILL after a grace period: the weights are memory-mapped, so
 * there is nothing to flush, but a server that is mid-generation deserves the
 * chance to finish writing its log before it goes.
 */
export async function stopLlamaServer(): Promise<void> {
  const current = serving;
  serving = null;
  clearIdle();
  if (!current) return;
  const { child } = current;
  if (child.exitCode !== null || child.signalCode) return;
  await new Promise<void>((resolve) => {
    const hard = setTimeout(() => {
      try {
        child.kill('SIGKILL');
      } catch {
        /* already gone */
      }
      resolve();
    }, 4000);
    child.once('exit', () => {
      clearTimeout(hard);
      resolve();
    });
    try {
      child.kill('SIGTERM');
    } catch {
      clearTimeout(hard);
      resolve();
    }
  });
}

/** Waits until the server answers its own health endpoint. */
async function waitForReady(child: ChildProcess, model: GgufModel, onProgress: (line: string) => void) {
  const started = Date.now();
  let lastNote = started;
  for (;;) {
    if (child.exitCode !== null || child.signalCode) {
      throw new Error(
        `llama-server stopped while loading ${model.id} (exit ${child.exitCode ?? child.signalCode})${tailSuffix()}`,
      );
    }
    try {
      // /health answers 503 until the weights are in, which is why this is not
      // a plain connect check: the port is open well before the model is ready.
      const res = await fetch(`http://127.0.0.1:${port()}/health`, {
        signal: AbortSignal.timeout(1500),
      });
      if (res.ok) return;
    } catch {
      /* not listening yet */
    }
    const waited = Date.now() - started;
    if (waited > loadTimeoutMs()) {
      throw new Error(
        `${model.id} did not finish loading within ${Math.round(loadTimeoutMs() / 1000)}s${tailSuffix()}`,
      );
    }
    if (Date.now() - lastNote > 3000) {
      lastNote = Date.now();
      onProgress(`Still loading ${model.id} — ${Math.round(waited / 1000)}s so far…`);
    }
    await new Promise((r) => setTimeout(r, 400));
  }
}

async function start(bin: string, model: GgufModel, onProgress: (line: string) => void) {
  // One model at a time: llama-server loads exactly the file it was given.
  await stopLlamaServer();

  onProgress(`Starting llama.cpp with ${model.id}…`);
  const args = [
    '--model',
    model.path,
    // The name the API reports, so /v1/models agrees with the picker and the
    // conversation that named this model still resolves after a restart.
    '--alias',
    model.id,
    '--host',
    '127.0.0.1',
    '--port',
    String(port()),
    '--ctx-size',
    String(ctxSize()),
    // One slot: the chat allows at most two generations at once, and a second
    // slot would double the KV cache for a queue that never needs it.
    '--parallel',
    '1',
    // Tool calls go through the model's own chat template. Enabled by default
    // in this build; passed explicitly so a default flipping upstream cannot
    // silently take tool support away.
    '--jinja',
    // The bundled web UI is for browsers, and this server only ever answers
    // this app's API calls.
    '--no-webui',
  ];

  tail = [];
  const child = spawn(process.execPath, ['-e', SUPERVISOR, bin, ...args], {
    // stdin is the supervisor's death watch, not a conversation partner: the
    // app never writes to it and the pipe closing is the whole message.
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  serving = { child, model: model.id };

  child.stdout?.on('data', remember);
  child.stderr?.on('data', remember);
  child.on('error', (err) => {
    if (serving?.child === child) serving = null;
    clearIdle();
    console.error('[llama.cpp] could not start llama-server:', err.message);
  });
  child.on('exit', (code, signal) => {
    const wasCurrent = serving?.child === child;
    if (wasCurrent) {
      serving = null;
      clearIdle();
    }
    if (wasCurrent && code) {
      console.error(`[llama.cpp] llama-server exited (${code ?? signal})${tailSuffix()}`);
    }
  });

  await waitForReady(child, model, onProgress);
  // A server that died between the health check and here would be caught above
  // on the next call; the idle timer only starts once it is genuinely serving.
  touchIdle();
}

/**
 * Brings up the server with `modelId` loaded, or does nothing when it is
 * already the model being served.
 *
 * Called before a chat request rather than on selection: loading several GB of
 * weights because someone opened a dropdown is exactly the kind of unasked-for
 * work that makes a local app feel hostile. Concurrent calls are serialised —
 * the second waits for the first and then restarts only if it needs a different
 * model, so two messages in a row never race to spawn two servers.
 */
export async function prepareLlama(
  modelId: string,
  onProgress: (line: string) => void,
): Promise<void> {
  const model = await resolveGgufModel(modelId);
  if (!model) {
    throw new Error(
      `llama.cpp: no GGUF model called "${String(modelId ?? '').slice(0, 80)}" on this machine. ` +
        'Pick one from the model list, or download it from the Models page.',
    );
  }
  if (!model.chat) {
    throw new Error(
      `llama.cpp: ${model.file} is not a chat model — ${model.why}. Pick a text model instead.`,
    );
  }
  const bin = serverBinary();
  if (!bin) {
    throw new Error(
      'llama.cpp is not installed on this machine. Install it under Settings → Runtimes, then send the message again.',
    );
  }

  if (serving?.model === model.id) {
    touchIdle();
    return;
  }
  if (starting) {
    // Someone else is already starting a server; wait for their attempt to
    // settle before deciding whether this request needs its own.
    await starting.catch(() => undefined);
    if (serving?.model === model.id) {
      touchIdle();
      return;
    }
  }

  const attempt = start(bin, model, onProgress);
  starting = attempt;
  try {
    await attempt;
  } finally {
    if (starting === attempt) starting = null;
  }
}

/** One line for the provider picker: what is here, and what is running. */
export function llamaHint(): string {
  if (!llamaServerState().installed) {
    return 'llama.cpp is not installed yet — install it under Settings → Runtimes';
  }
  const models = cachedGgufModels().filter((m) => m.chat);
  if (!models.length) {
    return 'No GGUF chat models on this machine yet — download one from Settings → Models';
  }
  const state = llamaServerState();
  if (state.running && state.model) return `Running with ${state.model} loaded`;
  return `${models.length} GGUF chat model${models.length === 1 ? '' : 's'} on this machine — loaded when you send a message`;
}
