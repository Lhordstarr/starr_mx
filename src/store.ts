import { chmod, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * API keys saved from the settings page live in a gitignored file next to the
 * app, not in .env, so they never get committed and never sit in the browser.
 */
const keysFile = path.resolve(process.env.KEYS_FILE ?? '.keys.json');

/** Only these names may be written, so the file can't become an env-injection vector. */
export const ALLOWED_KEYS = [
  'OPENAI_API_KEY',
  'OPENROUTER_API_KEY',
  'ANTHROPIC_API_KEY',
  'GEMINI_API_KEY',
  'GROQ_API_KEY',
  'HF_API_TOKEN',
  'CLOUDFLARE_API_TOKEN',
  'CLOUDFLARE_ACCOUNT_ID',
  'STABLE_HORDE_API_KEY',
] as const;

export type AllowedKey = (typeof ALLOWED_KEYS)[number];

let cache: Record<string, string> = {};

/** Loads saved keys into memory at startup. A missing or corrupt file is not fatal. */
export async function loadKeys(): Promise<void> {
  try {
    const raw = await readFile(keysFile, 'utf8');
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    cache = {};
    for (const name of ALLOWED_KEYS) {
      const value = parsed[name];
      if (typeof value === 'string' && value.trim()) cache[name] = value.trim();
    }
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
      console.warn(`Could not read ${keysFile}: ${(err as Error).message}`);
    }
    cache = {};
  }
}

/**
 * Resolves a key at call time, preferring the settings page over .env.
 * Must not be cached at module load: users can change keys while the server runs.
 */
export function getKey(name: string): string | undefined {
  return cache[name] || process.env[name] || undefined;
}

/** Where the active value came from, so the UI can label it. */
export function keySource(name: string): 'settings' | 'env' | 'none' {
  if (cache[name]) return 'settings';
  if (process.env[name]) return 'env';
  return 'none';
}

/** e.g. sk-abcd…wxyz. Enough to recognise a key, never enough to use one. */
export function maskKey(value: string): string {
  if (value.length <= 8) return '••••';
  return `${value.slice(0, 4)}…${value.slice(-4)}`;
}

/** Persists the cache, keeping the file readable only by its owner. */
async function save(): Promise<void> {
  await mkdir(path.dirname(keysFile), { recursive: true });
  await writeFile(keysFile, `${JSON.stringify(cache, null, 2)}\n`, { mode: 0o600 });
  await chmod(keysFile, 0o600);
}

export async function setKey(name: AllowedKey, value: string): Promise<void> {
  const trimmed = value.trim();
  if (trimmed) cache[name] = trimmed;
  else delete cache[name];
  await save();
}

export async function clearKey(name: AllowedKey): Promise<void> {
  delete cache[name];
  await save();
}

export async function clearAllKeys(): Promise<void> {
  cache = {};
  await save();
}

export function keysFilePath(): string {
  return keysFile;
}