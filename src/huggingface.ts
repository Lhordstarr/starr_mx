/**
 * Hugging Face model browser and downloader for the Models settings tab.
 *
 * Public models are keyless; the saved HF_API_TOKEN (same field the image
 * providers use) is sent as a Bearer for gated or private repos. No key ever
 * leaves the server for browsing — it only rides the Authorization header.
 *
 * Downloads land under ./models/<repo-with---for-slashes>/ (HF_MODELS_DIR
 * overrides, .gitignored). Only validated repo/file names ever reach the
 * filesystem, and file writes go to a `.part` sibling renamed on completion,
 * so an interrupted download never shows up as a finished file.
 */
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, rename, rm, stat, unlink, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import * as path from 'node:path';
import { getKey } from './store.js';

const apiBase = () => process.env.HF_API_BASE ?? 'https://huggingface.co';

export const modelsDir = () => path.resolve(process.env.HF_MODELS_DIR ?? './models');

/**
 * The shared Hugging Face cache that hugs huggingface-cli, transformers and
 * every Python downloader on this machine. A model fetched there is a download
 * the user made even though this app did not write it — it used to be
 * invisible in every list, which made the lists look hardcoded. It can also
 * run from there: the snapshot files are ordinary files behind symlinks.
 * Env resolution matches huggingface_hub's own precedence.
 */
export const hubCacheDir = () => {
  if (process.env.HF_HUB_CACHE) return path.resolve(process.env.HF_HUB_CACHE);
  const home = process.env.HF_HOME;
  return path.resolve(
    home ? path.join(home, 'hub') : path.join(homedir(), '.cache', 'huggingface', 'hub'),
  );
};

/**
 * Avatar cache. An author's avatar is the mark that identifies the model's
 * maker, and it is what fills the browse card — the per-model social thumbnail
 * used to sit there, but that image is a 1200×648 card with a 32px logo
 * printed on it: cropped into a card strip the logo came out a few pixels
 * wide, unreadable. Avatars are 200×200, one per author, and a fraction of
 * the bytes. Fetched once, kept on disk, served from our own origin.
 */
export const avatarsDir = () => path.resolve(process.env.HF_AVATARS_DIR ?? './.hf-cache/avatars');

export interface HFSearchHit {
  id: string;
  likes: number;
  downloads: number;
  pipelineTag: string | null;
  libraryName: string | null;
  tags: string[];
  private: boolean;
}

export interface HFFile {
  name: string;
  size: number | null;
}

export interface HFModelDetail {
  id: string;
  likes: number;
  downloads: number;
  pipelineTag: string | null;
  libraryName: string | null;
  tags: string[];
  gated: boolean;
  private: boolean;
  usedStorage: number | null;
  files: HFFile[];
}

export interface HFDownload {
  repo: string;
  file: string;
  size: number;
  modifiedAt: string;
  /** Where the bytes live: this app's models dir, or the shared HF cache. */
  location: 'models' | 'cache';
  /** Absolute repo directory holding `file`; the only base callers may join. */
  root: string;
}

/**
 * `author/model` and nothing else. The id goes into URLs and the filesystem,
 * so anything outside the Hub's own charset is rejected here, not downstream.
 */
export function validateRepoId(id: unknown): string | null {
  if (typeof id !== 'string' || !id.trim()) return 'Model id is required';
  if (id.length > 200) return 'Model id is too long';
  if (!/^[\w.\-]+\/[\w.\-]+$/.test(id)) return 'Invalid model id (want author/model)';
  return null;
}

/**
 * A repo-relative file path (`model-Q4_K_M.gguf`, `onnx/model.onnx`).
 * Every segment is checked: no traversal, no absolute paths, no empties.
 */
export function validateFilePath(file: unknown): string | null {
  if (typeof file !== 'string' || !file) return 'File name is required';
  if (file.length > 500) return 'File name is too long';
  if (file.startsWith('/') || file.includes('\\')) return 'Invalid file name';
  for (const seg of file.split('/')) {
    if (!seg || seg === '.' || seg === '..') return 'Invalid file name';
    if (seg.length > 255) return 'Invalid file name';
    if (!/^[\w.\-+ ]+$/.test(seg)) return 'Invalid file name';
  }
  return null;
}

function authHeaders(): Record<string, string> {
  const token = getKey('HF_API_TOKEN');
  return token ? { authorization: `Bearer ${token}` } : {};
}

async function hfError(res: Response): Promise<Error> {
  let detail = '';
  try {
    const data = (await res.text()).slice(0, 300);
    try {
      const parsed = JSON.parse(data) as { error?: string };
      if (parsed?.error) detail = parsed.error;
    } catch {
      detail = data;
    }
  } catch {
    /* body unreadable */
  }
  if (res.status === 401 || res.status === 403) {
    return new Error(
      'Hugging Face refused this (gated or private model). Add an HF token in Settings.',
    );
  }
  if (res.status === 404) return new Error('No such model on Hugging Face.');
  return new Error(`Hugging Face error ${res.status}${detail ? `: ${detail}` : ''}`);
}

async function hfJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const timeout = AbortSignal.timeout(20_000);
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;
  const init = {
    headers: { ...authHeaders(), accept: 'application/json' },
    signal: combined,
  };
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (err) {
    // Same dead-keep-alive story as the tools fetcher: a pooled socket can
    // die between uses and surface as a bare network failure on a request
    // that never went out. One retry on a fresh socket; aborts never retry.
    if (signal?.aborted || (err instanceof Error && err.name === 'AbortError')) throw err;
    try {
      res = await fetch(url, init);
    } catch (err2) {
      if (signal?.aborted || (err2 instanceof Error && err2.name === 'AbortError')) throw err2;
      throw new Error(`Hugging Face is unreachable: ${(err2 as Error).message}`);
    }
  }
  if (!res.ok) throw await hfError(res);
  return (await res.json()) as T;
}

/** Search the Hub, or list trending when the query is empty. `sort` is
 *  downloads, likes, or lastModified. */
export async function searchModels(
  query: string,
  opts: { sort?: string; limit?: number; pipeline?: string; signal?: AbortSignal } = {},
): Promise<HFSearchHit[]> {
  const sort = opts.sort === 'likes' || opts.sort === 'lastModified' ? opts.sort : 'downloads';
  // The Hub caps a page at 100. The old ceiling of 25 was chosen for a small
  // card grid; with a denser layout and no ranking there is no reason to
  // starve the page.
  const limit = Math.min(Math.max(opts.limit ?? 8, 1), 100);
  const q = query.trim();
  // `pipeline` maps to the Hub's own pipeline_tag filter, which is how a
  // category chip asks for "the top image models" rather than trusting a text
  // search to happen to surface them. Whitelisted, because it reaches a URL.
  const pipelines = new Set([
    'text-to-image',
    'image-to-image',
    'image-to-text',
    'automatic-speech-recognition',
    'text-to-speech',
    'text-to-audio',
    'text-generation',
    'fill-mask',
  ]);
  const pipeline = pipelines.has(opts.pipeline ?? '') ? (opts.pipeline as string) : null;
  const url =
    `${apiBase()}/api/models?` +
    (q ? `search=${encodeURIComponent(q)}&` : '') +
    (pipeline ? `filter=${pipeline}&` : '') +
    `sort=${sort}&direction=-1&limit=${limit}`;
  const data = await hfJson<unknown[]>(url, opts.signal);
  if (!Array.isArray(data)) return [];
  return data
    .filter((m) => m && typeof m === 'object' && typeof (m as { id?: unknown }).id === 'string')
    .map((m) => {
      const r = m as Record<string, unknown>;
      return {
        id: r.id as string,
        likes: typeof r.likes === 'number' ? r.likes : 0,
        downloads: typeof r.downloads === 'number' ? r.downloads : 0,
        pipelineTag: typeof r.pipeline_tag === 'string' ? r.pipeline_tag : null,
        libraryName: typeof r.library_name === 'string' ? r.library_name : null,
        tags: Array.isArray(r.tags) ? r.tags.filter((t): t is string => typeof t === 'string') : [],
        private: r.private === true,
      };
    });
}

/** Full detail for one repo, with per-file byte sizes (`blobs=true`). */
/**
 * The files a model genuinely needs, fetched as a set.
 *
 * Most model repositories are not a single file: a CTranslate2 Whisper needs a
 * config and a tokenizer beside its weights, a diffusers checkpoint is a whole
 * directory tree. Downloading "the .bin" leaves something the loader rejects,
 * and the failure surfaces as a cryptic error deep inside a runtime rather
 * than as "you are missing two files".
 *
 * So the set is derived here, once, and the UI offers it as one action.
 */

/**
 * Files that are never needed to run a model.
 *
 * Extensionless repo furniture — LICENSE, README, CITATION, CHANGELOG — has no
 * extension to match on, so it is matched by name instead. Abert-base shipped
 * one in every download for nothing.
 */
const JUNK =
  /(^|\/)\.(git|github|huggingface)|\.(md|png|jpe?g|gif|webp|svg|gitattributes|gitignore)$/i;
const JUNK_NAME =
  /^(license|licence|readme|citation|changelog|contributing|training_args\.bin|trainer_state\.json|events\.out\.tfevents.*|\.gitattributes|\.gitignore|use[-_]?instructions)$/i;

/**
 * Demo clips and helper scripts. The kokoro clone tuner shipped 13 MB of wav
 * samples and a generator script — nothing here ever opens either, yet every
 * file landed on disk and showed up under "no local runtime".
 */
const JUNK_MEDIA_OR_SCRIPT =
  /\.(wav|mp3|flac|ogg|opus|m4a|aac|aiff?|mp4|mov|avi|mkv|webm|py|ipynb|sh|bat|ps1)$/i;

/**
 * Backends nothing here can execute.
 *
 * CoreML was the expensive omission: bert-base-uncased carries four CoreML
 * files weighing 532 MB — more than the actual weights — and the path contains
 * `.bin`, so nothing else filtered them out.
 */
const OTHER_BACKEND =
  /(flax|openvino|onnx_model|rust_model|tf_model|tflite|messagepack|coreml|mlmodel|mlpackage)/i;

export interface RepoFile {
  path: string;
  size: number | null;
}

export async function requiredFiles(
  repo: string,
  signal?: AbortSignal,
): Promise<RepoFile[]> {
  const data = await hfJson<Record<string, unknown>>(
    `${apiBase()}/api/models/${repo}?blobs=true`,
    signal,
  );
  const siblings = Array.isArray(data.siblings) ? data.siblings : [];
  const all = siblings
    .filter((s) => s && typeof s === 'object')
    .map((s) => {
      const sib = s as { rfilename?: unknown; size?: unknown };
      return {
        path: typeof sib.rfilename === 'string' ? sib.rfilename : '',
        size: typeof sib.size === 'number' ? sib.size : null,
      };
    })
    .filter((f) => f.path);

  const hasFp16 = new Set(
    all.filter((f) => f.path.endsWith('.fp16.safetensors'))
      .map((f) => f.path.replace(/\.fp16\.safetensors$/, '.safetensors')),
  );

  // safetensors and pytorch_model.bin are the same weights in two formats.
  // Prefer safetensors: it is the format every runtime here reads, it loads
  // without pickle, and taking both doubles the download for nothing.
  const preferSafetensors = all.some((f) => f.path.endsWith('.safetensors'));

  // A repo with model_index.json is a diffusers tree. Those also ship a
  // single-file checkpoint at the root for ComfyUI/A1111, and it is routinely
  // the largest thing in the repo — 5.2 GB of sd_turbo.safetensors that
  // from_pretrained never opens. Dropping it halves the download.
  const isDiffusers = all.some((f) => f.path === 'model_index.json');

  return all.filter((f) => {
    if (JUNK.test(f.path)) return false;
    if (JUNK_NAME.test(f.path.split('/').pop() ?? '')) return false;
    if (JUNK_MEDIA_OR_SCRIPT.test(f.path)) return false;
    if (OTHER_BACKEND.test(f.path)) return false;
    if (isDiffusers && !f.path.includes('/') && /\.(safetensors|ckpt|bin|pt|pth)$/i.test(f.path)) {
      return false;
    }
    // Same weights, two containers — keep the one every runtime here reads.
    if (preferSafetensors && /pytorch_model\.bin$/.test(f.path)) return false;
    // GGUF and ONNX are first-class here (llama.cpp and onnxruntime serve
    // them), so they are NOT "another backend" — but a repo that also ships
    // safetensors does not need the quantized twin as well, which is the same
    // either-way-too-much rule as pytorch_model.bin above.
    if (preferSafetensors && /\.(gguf|onnx)$/i.test(f.path)) return false;
    // Keep an fp32 weight only where there is no fp16 twin; both together is
    // twice the download for one model.
    if (f.path.endsWith('.safetensors') && hasFp16.has(f.path)) return false;
    return true;
  });
}

export async function modelDetail(repo: string, signal?: AbortSignal): Promise<HFModelDetail> {
  const data = await hfJson<Record<string, unknown>>(
    `${apiBase()}/api/models/${repo}?blobs=true`,
    signal,
  );
  const siblings = Array.isArray(data.siblings) ? data.siblings : [];
  const safe = (data.safetensors ?? {}) as { total?: unknown };
  return {
    id: typeof data.id === 'string' ? data.id : repo,
    likes: typeof data.likes === 'number' ? data.likes : 0,
    downloads: typeof data.downloads === 'number' ? data.downloads : 0,
    pipelineTag: typeof data.pipeline_tag === 'string' ? data.pipeline_tag : null,
    libraryName: typeof data.library_name === 'string' ? data.library_name : null,
    tags: Array.isArray(data.tags)
      ? data.tags.filter((t): t is string => typeof t === 'string')
      : [],
    gated: data.gated === true || typeof data.gated === 'object',
    private: data.private === true,
    usedStorage: typeof data.usedStorage === 'number' ? data.usedStorage : null,
    files: siblings
      .filter((s) => s && typeof s === 'object' && typeof (s as { rfilename?: unknown }).rfilename === 'string')
      .map((s) => {
        const sib = s as { rfilename: string; size?: unknown };
        return { name: sib.rfilename, size: typeof sib.size === 'number' ? sib.size : null };
      }),
  };
}

/* ---- Author avatars ---- */

/** How long a cached avatar stays fresh before we ask the Hub again. */
const AVATAR_TTL_MS = 7 * 24 * 60 * 60 * 1000;
/** Hard ceiling on one avatar; they are 200px images, so this is generous. */
const AVATAR_MAX_BYTES = 2 * 1024 * 1024;
/** Where the Hub serves avatars from; the allowlist keeps the fetch honest. */
const AVATAR_HOSTS = new Set(['cdn-avatars.huggingface.co', 'huggingface.co']);
/**
 * One author shown by twenty cards would otherwise be twenty API calls and
 * twenty image fetches racing on the same cache file. Keyed by author, so the
 * first caller fetches and the rest await that one request.
 */
const inflightAvatars = new Map<string, Promise<Buffer>>();

/** Image type from magic bytes — the CDN answers with PNG, JPEG or WebP. */
export function imageMime(body: Buffer): string | null {
  if (body.length > 8 && body[0] === 0x89 && body[1] === 0x50 && body[2] === 0x4e && body[3] === 0x47) {
    return 'image/png';
  }
  if (body.length > 3 && body[0] === 0xff && body[1] === 0xd8 && body[2] === 0xff) {
    return 'image/jpeg';
  }
  if (
    body.length > 12 &&
    body.toString('ascii', 0, 4) === 'RIFF' &&
    body.toString('ascii', 8, 12) === 'WEBP'
  ) {
    return 'image/webp';
  }
  return null;
}

/** Asks one of the Hub's avatar endpoints. Null when this author is not one. */
async function avatarUrlFor(
  kind: 'organizations' | 'users',
  author: string,
  init: RequestInit,
): Promise<string | null> {
  try {
    const res = await fetch(`${apiBase()}/api/${kind}/${encodeURIComponent(author)}/avatar`, {
      ...init,
      headers: { accept: 'application/json' },
    });
    if (!res.ok) return null;
    const data = (await res.json().catch(() => null)) as { avatarUrl?: unknown } | null;
    return typeof data?.avatarUrl === 'string' ? data.avatarUrl : null;
  } catch {
    // A network hiccup on one endpoint should not veto the other.
    return null;
  }
}

/**
 * The author's avatar — the logo that identifies who made the model. Orgs and
 * individual users are filed in different endpoints, so both are tried.
 * Returns the image bytes, serving from disk when the cached copy is still
 * inside the TTL. Throws when the author has no avatar, so the caller can
 * fall back to the generated tile instead of showing a broken image.
 */
export async function fetchAvatar(author: string, signal?: AbortSignal): Promise<Buffer> {
  // Author ids share the repo id alphabet, minus the slash.
  if (typeof author !== 'string' || !/^[\w.\-]+$/.test(author) || author.length > 96) {
    throw new Error('Invalid author');
  }

  const dir = avatarsDir();
  const dest = path.join(dir, `${author}.bin`);
  if (!path.resolve(dest).startsWith(path.resolve(dir) + path.sep)) {
    throw new Error('Invalid author');
  }

  try {
    const cached = await stat(dest);
    if (Date.now() - cached.mtimeMs < AVATAR_TTL_MS) return await readFile(dest);
    // Stale: drop it, but only after a fresh copy is in hand below.
    await rm(dest, { force: true });
  } catch {
    /* no usable cache entry — fall through to the network */
  }

  const existing = inflightAvatars.get(dest);
  if (existing) return await existing;

  const work = (async (): Promise<Buffer> => {
    const timeout = AbortSignal.timeout(20_000);
    const init = {
      redirect: 'follow' as const,
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    };

    const url =
      (await avatarUrlFor('organizations', author, init)) ??
      (await avatarUrlFor('users', author, init));
    if (!url) throw new Error(`No avatar for ${author}`);
    let parsed: URL;
    try {
      parsed = new URL(url);
    } catch {
      throw new Error('Avatar URL was not a URL');
    }
    if (parsed.protocol !== 'https:' || !AVATAR_HOSTS.has(parsed.hostname)) {
      throw new Error('Avatar URL is not on the avatars CDN');
    }

    let res: Response;
    try {
      res = await fetch(parsed, init);
    } catch (err) {
      if (signal?.aborted || (err instanceof Error && err.name === 'AbortError')) throw err;
      res = await fetch(parsed, init);
    }
    if (!res.ok) throw new Error(`No avatar for ${author} (HTTP ${res.status})`);
    const declared = Number(res.headers.get('content-length') ?? '');
    if (Number.isFinite(declared) && declared > AVATAR_MAX_BYTES) {
      throw new Error('Avatar too large');
    }
    const body = Buffer.from(await res.arrayBuffer());
    if (body.length > AVATAR_MAX_BYTES) throw new Error('Avatar too large');
    // Not an image means we cached an error page or a redirect body, which
    // would then serve as a permanently broken image for a week.
    if (!imageMime(body)) throw new Error('Avatar was not an image');

    await mkdir(dir, { recursive: true });
    // Same `.part` + rename discipline as model downloads: a reader never
    // sees a half-written file, even if two processes race here.
    const tmp = `${dest}.${process.pid}.part`;
    try {
      await writeFile(tmp, body);
      await rename(tmp, dest);
    } catch (err) {
      await rm(tmp, { force: true });
      throw err;
    }
    return body;
  })();

  inflightAvatars.set(dest, work);
  try {
    return await work;
  } finally {
    inflightAvatars.delete(dest);
  }
}

/** Weak validator for an avatar body, so repeat views can 304. */
export function avatarEtag(body: Buffer): string {
  return `W/"${createHash('sha1').update(body).digest('base64url').slice(0, 20)}"`;
}

export interface DownloadProgress {
  received: number;
  total: number | null;
}

/**
 * Streams one repo file into the downloads dir. Progress reports bytes; the
 * file appears under its final name only on completion (`.part` until then).
 * The download URL is built from validated parts under the API origin, so no
 * user-controlled host ever reaches fetch — SSRF is impossible by
 * construction, and CDN redirects stay within that trust.
 */
export async function downloadFile(
  repo: string,
  file: string,
  opts: { revision?: string; signal?: AbortSignal; onProgress?: (p: DownloadProgress) => void } = {},
): Promise<{ repo: string; file: string; bytes: number }> {
  const revision = opts.revision ?? 'main';
  if (!/^[\w.\-]+$/.test(revision)) throw new Error('Invalid revision');
  const repoProblem = validateRepoId(repo);
  if (repoProblem) throw new Error(repoProblem);
  const fileProblem = validateFilePath(file);
  if (fileProblem) throw new Error(fileProblem);
  const url = `${apiBase()}/${repo}/resolve/${revision}/${file
    .split('/')
    .map((s) => encodeURIComponent(s))
    .join('/')}`;
  const dir = path.join(modelsDir(), repo.replace(/\//g, '--'));
  const dest = path.join(dir, ...file.split('/'));
  const tmp = `${dest}.part`;
  // Resume support: if a partial file is on disk, ask for the rest.
  let existing = 0;
  try {
    existing = (await stat(tmp)).size;
  } catch {
    existing = 0;
  }
  const headers = { ...authHeaders() };
  if (existing > 0) headers['Range'] = `bytes=${existing}-`;
  let res: Response;
  const dlInit = {
    headers,
    signal: opts.signal,
    redirect: 'follow' as const,
  };
  try {
    res = await fetch(url, dlInit);
  } catch (err) {
    // One retry, same dead-socket reasoning as hfJson above. Only the start
    // retries — a stream dying mid-body is a real failure, not a stale pool.
    if (opts.signal?.aborted || (err instanceof Error && err.name === 'AbortError')) throw err;
    try {
      res = await fetch(url, dlInit);
    } catch (err2) {
      if (opts.signal?.aborted || (err2 instanceof Error && err2.name === 'AbortError')) throw err2;
      throw new Error(`Download failed: ${(err2 as Error).message}`);
    }
  }
  if (!res.ok || !res.body) {
    if (res.status === 401 || res.status === 403) {
      throw new Error('Hugging Face refused this (gated or private model). Add an HF token in Settings.');
    }
    throw new Error(`Download failed: HTTP ${res.status}`);
  }
  await mkdir(dir, { recursive: true });
  await mkdir(path.dirname(dest), { recursive: true });
  // Belt and suspenders with the segment validation above: never write
  // outside the downloads dir no matter what the caller passed.
  if (!path.resolve(dest).startsWith(path.resolve(dir) + path.sep)) {
    throw new Error('Invalid file name');
  }
  // The server either honoured the Range (206, append) or sent the whole file
  // again (200, truncate and restart). A 200 on a partial means the partial
  // is discarded, which is the safe fallback.
  let resumed = false;
  if (existing > 0 && res.status === 206) {
    resumed = true;
  } else if (existing > 0 && res.status === 200) {
    // Server ignored the range (or we never sent one): truncate and restart.
    existing = 0;
  }
  const totalHeader = res.headers.get('content-length');
  let total: number | null =
    totalHeader !== null && totalHeader !== '' ? Number(totalHeader) : null;
  if (resumed && total !== null) total = total + existing;
  // Content-Range is the authoritative total when resuming.
  const cr = res.headers.get('content-range');
  if (cr) {
    const m = cr.match(/\/(\d+)$/);
    if (m) total = Number(m[0].slice(1));
  }

  const { createWriteStream } = await import('node:fs');
  let received = existing;
  await new Promise<void>((resolve, reject) => {
    const out = createWriteStream(tmp, { flags: resumed ? 'a' : 'w' });
    out.on('error', reject);
    const reader = res.body!.getReader();
    const pump = async (): Promise<void> => {
      try {
        for (;;) {
          if (opts.signal?.aborted) throw new Error('Download stopped');
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            received += value.length;
            await new Promise<void>((res2, rej2) =>
              out.write(Buffer.from(value), (e) => (e ? rej2(e) : res2())),
            );
            opts.onProgress?.({ received, total });
          }
        }
        out.end(() => resolve());
      } catch (err) {
        out.destroy();
        reject(err);
      } finally {
        reader.releaseLock();
      }
    };
    pump();
  }).catch(async (err) => {
    // Keep the partial file on an abort so a later Resume continues from it;
    // only a real failure discards it.
    const aborted =
      opts.signal?.aborted || (err instanceof Error && err.name === 'AbortError') ||
      /Download stopped/i.test((err as Error)?.message ?? '');
    if (!aborted) await rm(tmp, { force: true });
    throw err;
  });
  const { rename: renameFile } = await import('node:fs/promises');
  await renameFile(tmp, dest);
  return { repo, file, bytes: received };
}

/** One download at a time: concurrent multi-GB writes thrash the disk. */
const downloading = new Map<string, number>();

/**
 * How long a lock may be held before it is treated as abandoned.
 *
 * The lock is a single global slot, so a route that returns early — an aborted
 * request, a killed process — leaves it held forever. Every later download of
 * that file then reports "already being downloading" and is skipped, which is
 * how a model ended up permanently two files short with no error anywhere.
 */
const LOCK_TTL_MS = 2 * 60 * 60_000;

export function acquireDownload(key: string): boolean {
  const now = Date.now();
  for (const [k, since] of downloading) {
    if (now - since > LOCK_TTL_MS) downloading.delete(k);
  }
  if (downloading.has(key)) return false;
  downloading.set(key, now);
  return true;
}

export function releaseDownload(key: string): void {
  downloading.delete(key);
}

/** How many downloads are in flight right now. */
export function inflightDownloads(): number {
  return downloading.size;
}

/** Everything previously downloaded, by scanning the dir (no manifest to drift). */
/**
 * Drops `.part` files left by an interrupted download, and any repo directory
 * left empty behind them.
 *
 * A cancelled transfer used to keep its partial bytes forever: nothing retries
 * it, nothing lists it, and it just occupies disk. Only parts older than an
 * hour are removed, so a download that is merely slow is never interrupted.
 */
export async function pruneStalePartials(maxAgeMs = 60 * 60_000): Promise<number> {
  const root = path.resolve(modelsDir());
  let removed = 0;
  const walk = async (dir: string, depth: number): Promise<boolean> => {
    let entries: string[];
    try {
      entries = await readdir(dir);
    } catch {
      return false;
    }
    for (const name of entries) {
      const full = path.join(dir, name);
      const info = await stat(full).catch(() => null);
      if (!info) continue;
      if (info.isDirectory()) {
        // `full` is always a child of `dir`, and the root is only ever walked,
        // never removed — so an empty child is always safe to drop. Two earlier
        // guards (`depth > 0`, then `dir !== root`) both silently skipped every
        // repo directory, since those are exactly the root's direct children.
        if (await walk(full, depth + 1)) {
          await rm(full, { recursive: true, force: true }).catch(() => {});
        }
        continue;
      }
      if (!name.endsWith('.part')) continue;
      if (Date.now() - info.mtimeMs < maxAgeMs) continue;
      await unlink(full).catch(() => {});
      removed++;
    }
    // Emptiness is decided after the removals, not before: a directory holding
    // only an abandoned .part is empty once that part is gone.
    try {
      return (await readdir(dir)).length === 0;
    } catch {
      return true;
    }
  };
  await walk(root, 0);
  return removed;
}

/** Where the server stages browser dictation clips. Not a model repository. */
const UPLOAD_NS = '_uploads';

export async function listDownloads(): Promise<HFDownload[]> {
  const out: HFDownload[] = [];

  // This app's own downloads: ./models/<author>--<name>/.
  const root = modelsDir();
  const appRepos = new Set<string>();
  const walkModels = async (dir: string, rel: string, repo: string): Promise<void> => {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      if (entry.name.endsWith('.part')) continue;
      // hf_hub_download bookkeeping inside a local_dir: CACHEDIR.TAG, lock
      // and .metadata files. Machinery, not model files.
      if (entry.name === '.cache' && entry.isDirectory()) continue;
      const full = path.join(dir, entry.name);
      const sub = rel ? `${rel}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        await walkModels(full, sub, repo);
      } else if (entry.isFile()) {
        const st = await stat(full);
        out.push({
          repo: repo.replace(/--/g, '/'),
          file: sub,
          size: st.size,
          modifiedAt: st.mtime.toISOString(),
          location: 'models',
          root: path.join(root, repo),
        });
      }
    }
  };
  let repos: string[] = [];
  try {
    repos = await readdir(root);
  } catch {
    /* no downloads yet */
  }
  // Staged dictation clips live under `_uploads/` and are not models. Listing
  // them made the hub claim the machine held three more downloads than it did,
  // each with a "no local runtime" note and a Delete button.
  repos = repos.filter((r) => r !== UPLOAD_NS);
  for (const repo of repos) {
    const full = path.join(root, repo);
    try {
      if (!(await stat(full)).isDirectory()) continue;
      const before = out.length;
      await walkModels(full, '', repo);
      // Only a repo that actually holds files shadows its cache copy; an
      // abandoned empty directory must not hide a finished download.
      if (out.length > before) appRepos.add(repo.replace(/--/g, '/'));
    } catch {
      /* raced deletion */
    }
  }

  // The shared Hugging Face cache. Repos another tool fetched land here, and a
  // repo that also exists in ./models is skipped: one model, one row.
  const cache = hubCacheDir();
  let cacheDirs: string[] = [];
  try {
    cacheDirs = await readdir(cache);
  } catch {
    /* no cache on this machine */
  }
  for (const dir of cacheDirs) {
    if (!dir.startsWith('models--')) continue;
    const repo = dir.slice('models--'.length).replace(/--/g, '/');
    if (appRepos.has(repo)) continue;
    const snapshots = path.join(cache, dir, 'snapshots');
    let commits: string[];
    try {
      commits = await readdir(snapshots);
    } catch {
      continue;
    }
    const newestFirst = await Promise.all(
      commits.map(async (c) => {
        const st = await stat(path.join(snapshots, c)).catch(() => null);
        return { dir: path.join(snapshots, c), mtime: st?.mtimeMs ?? 0 };
      }),
    );
    newestFirst.sort((a, b) => b.mtime - a.mtime);
    // Newest revision first; a repo can hold several, and the files fetched
    // last are the ones whose names the catalog checks for.
    const files = new Map<string, HFDownload>();
    const walkSnapshot = async (cur: string, rel: string, snapRoot: string): Promise<void> => {
      for (const entry of await readdir(cur, { withFileTypes: true })) {
        const full = path.join(cur, entry.name);
        const sub = rel ? `${rel}/${entry.name}` : entry.name;
        // Snapshot entries are symlinks into ../blobs, so what a Dirent says
        // about them is meaningless — stat follows to the real file, and a
        // dangling link (half a download) is skipped rather than listed.
        const st = await stat(full).catch(() => null);
        if (!st) continue;
        if (st.isDirectory()) {
          await walkSnapshot(full, sub, snapRoot);
        } else if (st.isFile() && !files.has(sub)) {
          files.set(sub, {
            repo,
            file: sub,
            size: st.size,
            modifiedAt: st.mtime.toISOString(),
            location: 'cache',
            root: snapRoot,
          });
        }
      }
    };
    for (const { dir: snap } of newestFirst) await walkSnapshot(snap, '', snap);
    for (const f of files.values()) out.push(f);
  }

  return out.sort((a, b) => a.repo.localeCompare(b.repo) || a.file.localeCompare(b.file));
}

/** Deletes one downloaded file and prunes dirs left empty (never above root). */
export async function deleteDownload(repo: string, file: string): Promise<void> {
  if (!repo || !repo.trim()) throw new Error('Model id is required');
  const fileProblem = validateFilePath(file);
  if (fileProblem) throw new Error(fileProblem);
  const root = path.resolve(modelsDir());
  const target = path.resolve(root, repo.replace(/\//g, '--'), ...file.split('/'));
  if (target === root || !target.startsWith(root + path.sep)) {
    throw new Error('Invalid file name');
  }
  await unlink(target);
  let dir = path.dirname(target);
  while (dir.startsWith(root + path.sep) || dir === root) {
    try {
      const left = await readdir(dir);
      if (left.length || dir === root) break;
      // `recursive` is required for a directory: plain `rm` throws EISDIR on
      // one, and the catch below swallowed that — so emptied subfolders
      // (voices/, snapshots/) stayed behind as litter after every delete.
      await rm(dir, { recursive: true });
    } catch {
      break;
    }
    dir = path.dirname(dir);
  }
}

/**
 * Removes one repository from the shared Hugging Face cache. Only that repo's
 * own folder is touched — the cache is read by other tools too, which is why
 * the UI spells the action out and takes the whole repo rather than single
 * files.
 */
export async function deleteCacheRepo(repo: string): Promise<void> {
  const repoProblem = validateRepoId(repo);
  if (repoProblem) throw new Error(repoProblem);
  const root = path.resolve(hubCacheDir());
  const slug = `models--${repo.replace(/\//g, '--')}`;
  const dir = path.resolve(root, slug);
  if (dir === root || !dir.startsWith(root + path.sep)) {
    throw new Error('Invalid repository name');
  }
  try {
    await stat(dir);
  } catch {
    throw new Error(`${repo} is not in the Hugging Face cache.`);
  }
  await rm(dir, { recursive: true, force: true });
  // huggingface_hub keeps a lock folder beside each repo; it is ours to clear
  // too, or an empty shell would linger for every removed repo.
  await rm(path.join(root, '.locks', slug), { recursive: true, force: true }).catch(() => {});
}
