import { chmod, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * User-added providers, saved next to the app in a gitignored 0600 file.
 *
 * Only OpenAI-compatible endpoints are supported: that one wire format covers
 * vLLM, LM Studio, llama.cpp, LocalAI, Together, Fireworks, DeepSeek and most
 * gateways, so it is the case worth supporting. A provider here is treated as
 * reachable by definition — the user added it deliberately — so the key is
 * optional and a 401 from the endpoint surfaces on its own if auth is needed.
 */

const file = path.resolve(process.env.CUSTOM_PROVIDERS_FILE ?? '.providers.json');

export interface CustomProvider {
  /** URL-safe, unique, and never collides with a built-in id. */
  id: string;
  label: string;
  /** Base URL including any /v1 suffix. */
  baseUrl: string;
  /** Empty string when the endpoint needs no auth. */
  apiKey: string;
  /** Whether the endpoint exposes /audio/transcriptions. */
  stt: boolean;
  createdAt: number;
}

/** Guards against a settings page creating an unbounded number of entries. */
const MAX_PROVIDERS = 50;
const MAX_LABEL = 60;
const MAX_URL = 300;

let cache: CustomProvider[] = [];

/** Loads saved providers at startup. A missing or corrupt file is not fatal. */
export async function loadCustomProviders(): Promise<void> {
  try {
    const raw = await readFile(file, 'utf8');
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) throw new Error('not an array');
    cache = parsed.flatMap((entry) => {
      const p = entry as Partial<CustomProvider>;
      if (typeof p.id !== 'string' || !p.id) return [];
      if (typeof p.label !== 'string' || typeof p.baseUrl !== 'string') return [];
      let url: URL;
      try {
        url = new URL(p.baseUrl);
      } catch {
        return []; // dropped rather than kept in a state that throws on use
      }
      return [
        {
          id: p.id,
          label: p.label.slice(0, MAX_LABEL) || p.id,
          baseUrl: url.toString().replace(/\/+$/, ''),
          apiKey: typeof p.apiKey === 'string' ? p.apiKey : '',
          stt: p.stt === true,
          createdAt: typeof p.createdAt === 'number' ? p.createdAt : Date.now(),
        },
      ];
    });
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
      console.warn(`Could not read ${file}: ${(err as Error).message}`);
    }
    cache = [];
  }
}

export function listCustomProviders(): CustomProvider[] {
  return cache;
}

export function customApiKey(id: string): string | undefined {
  return cache.find((p) => p.id === id)?.apiKey || undefined;
}

async function save(): Promise<void> {
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, `${JSON.stringify(cache, null, 2)}\n`, { mode: 0o600 });
  await chmod(file, 0o600);
}

/**
 * Only http(s) URLs are accepted. The app will POST to whatever is configured,
 * so anything else (file:, javascript:, data:) is rejected up front rather than
 * failing later inside a fetch.
 */
export function validateBaseUrl(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) throw new Error('Base URL is required');
  if (trimmed.length > MAX_URL) throw new Error('Base URL is too long');

  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new Error('Base URL must be a full URL, e.g. http://localhost:8000/v1');
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('Base URL must use http:// or https://');
  }
  if (url.username || url.password) {
    throw new Error('Put credentials in the API key field, not the URL');
  }
  // Trailing slashes would produce '//models' when the path is appended.
  return trimmed.replace(/\/+$/, '');
}

/** Slugifies a label into a usable id: "My LLM Server" -> "my-llm-server". */
function slugify(label: string): string {
  const base = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40);
  return base || 'provider';
}

export function validateLabel(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) throw new Error('Name is required');
  if (trimmed.length > MAX_LABEL) throw new Error(`Name must be ${MAX_LABEL} characters or fewer`);
  return trimmed;
}

function nextId(label: string, taken: string[]): string {
  const base = slugify(label);
  if (!taken.includes(base)) return base;
  for (let n = 2; n < 1000; n += 1) {
    const candidate = `${base}-${n}`;
    if (!taken.includes(candidate)) return candidate;
  }
  return `${base}-${Date.now().toString(36)}`;
}

export async function addCustomProvider(input: {
  label?: string;
  baseUrl?: string;
  apiKey?: string;
  stt?: boolean;
}): Promise<CustomProvider> {
  if (cache.length >= MAX_PROVIDERS) {
    throw new Error(`Limit of ${MAX_PROVIDERS} custom providers reached`);
  }
  const label = validateLabel(input?.label ?? '');
  const baseUrl = validateBaseUrl(input?.baseUrl ?? '');
  const taken = new Set([...cache.map((p) => p.id), ...reservedIds()]);

  const provider: CustomProvider = {
    id: nextId(label, [...taken]),
    label,
    baseUrl,
    apiKey: (input.apiKey ?? '').trim(),
    stt: input.stt === true,
    createdAt: Date.now(),
  };
  cache = [provider, ...cache];
  await save();
  return provider;
}

/** Ids that belong to the built-in registry and so can't be reused. */
function reservedIds(): string[] {
  // Imported lazily to avoid a cycle: index.ts pulls this module in.
  return ['ollama', 'openai', 'openrouter', 'anthropic', 'gemini', 'groq'];
}

export async function updateCustomProvider(
  id: string,
  patch: { label?: string; baseUrl?: string; apiKey?: string; stt?: boolean },
): Promise<CustomProvider> {
  const index = cache.findIndex((p) => p.id === id);
  if (index === -1) throw new Error(`No such provider: ${id}`);

  const current = cache[index];
  const label = patch?.label === undefined ? current.label : validateLabel(patch.label);
  const baseUrl = patch?.baseUrl === undefined ? current.baseUrl : validateBaseUrl(patch.baseUrl);

  const updated: CustomProvider = {
    ...current,
    label,
    baseUrl,
    // An omitted key keeps the stored one; an empty string clears it.
    apiKey: patch?.apiKey === undefined ? current.apiKey : (patch.apiKey ?? '').trim(),
    stt: patch?.stt === undefined ? current.stt : patch.stt === true,
  };
  cache = cache.map((p) => (p.id === id ? updated : p));
  await save();
  return updated;
}

export async function deleteCustomProvider(id: string): Promise<boolean> {
  const before = cache.length;
  cache = cache.filter((p) => p.id !== id);
  if (cache.length === before) return false;
  await save();
  return true;
}

export function customProvidersFilePath(): string {
  return file;
}