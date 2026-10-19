/**
 * whisper.cpp: speech-to-text through ggml-org's own build, installed as a
 * pinned prebuilt binary into this app's runtime folder.
 *
 * This is the transcription sibling of `llamacpp.ts`, and it follows the same
 * shape deliberately — same pinned-build reasoning, same "no root, no Python,
 * no compiler" promise, same honest reporting. Two things differ, and both are
 * consequences of what whisper.cpp actually is rather than of this file:
 *
 *  - it is a one-shot CLI, not a server. llama-server holds a chat model in
 *    memory between messages, which is what pays for it; a dictation clip is
 *    a few seconds of audio and arrives minutes apart, so there is nothing to
 *    keep warm. `whisper-server` exists upstream and is deliberately not used.
 *
 *  - it is CPU-only on the prebuilt Linux binaries. See `whisperAsset()`.
 */

import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import * as path from 'node:path';

import { listDownloads } from './huggingface.js';
import { attachChild, finishProcess, registerProcess } from './processes.js';
import { runtimesDir } from './paths.js';

/**
 * The whisper.cpp build this app installs.
 *
 * A rolling build number, not a `vX.Y.Z` tag: the tagged releases carry no
 * binaries at all (they are source only), while every platform build is
 * published on the `bNNNNN` releases. Pinned rather than "latest" so an
 * install is reproducible and a bad upstream build cannot change what lands on
 * someone's disk unasked. Bump it deliberately, and re-check `whisperAsset()` —
 * which assets exist changes between builds, as the GPU ones have.
 */
export const WHISPER_BUILD = 'b5454';

/**
 * Where whisper.cpp's own models live.
 *
 * These are GGML `.bin` files, not GGUF. That is not a preference: whisper.cpp's
 * loader reads the GGML magic and rejects GGUF outright with "invalid model
 * data (bad magic)". There are GGUF Whisper conversions on the Hub, but they
 * are built for *transcribe.cpp*, a different project, and whisper.cpp cannot
 * open them. The two catalog entries that used to advertise GGUF Whisper as a
 * "whisper.cpp" runtime were pointing at exactly those files.
 */
export const WHISPER_REPO = 'ggerganov/whisper.cpp';

/** Where this installer's marker and archives live. */
export const whisperDir = (): string => path.join(runtimesDir(), 'whisper.cpp');

const cliName = (): string => (process.platform === 'win32' ? 'whisper-cli.exe' : 'whisper-cli');

/**
 * The CLI's absolute path, or null when it is not installed.
 *
 * Read from a marker written at install time rather than searched for, because
 * upstream's directory layout is not something this app should hard-code — and
 * because removal has to delete exactly the tree that was unpacked.
 */
export function whisperCliPath(): string | null {
  const marker = path.join(whisperDir(), 'build.txt');
  try {
    const dir = existsSync(marker) ? readFileSync(marker, 'utf8').trim() : '';
    if (dir) return path.join(runtimesDir(), dir, cliName());
  } catch {
    /* not installed, or the marker is gone */
  }
  return null;
}

export async function whisperCppInstalled(): Promise<boolean> {
  const p = whisperCliPath();
  if (!p) return false;
  try {
    const { stat } = await import('node:fs/promises');
    return (await stat(p)).isFile();
  } catch {
    return false;
  }
}

/**
 * The library search path the CLI should start with.
 *
 * The prebuilt archive keeps `libwhisper.so` and the per-microarchitecture
 * `libggml-cpu-*.so` beside the binary. Without this in the environment the
 * loader never finds them and the process dies with a bare
 * "libwhisper.so.1: cannot open shared object file" that names no path.
 *
 * Prepended rather than assigned, so a machine with its own libwhisper keeps it
 * ahead in the search order — the same reasoning as `libPathFor` in llamacpp.ts.
 */
function libPathFor(bin: string): string {
  const dir = path.dirname(bin);
  const parts = [dir, path.join(dir, 'lib'), path.resolve(dir, '..', 'lib')];
  if (process.env.LD_LIBRARY_PATH) parts.push(process.env.LD_LIBRARY_PATH);
  return parts.join(path.delimiter);
}

/**
 * The build to fetch for this machine, or null when upstream ships none for
 * this platform.
 *
 * GPU support is genuinely uneven here and the answer changes between builds,
 * so it is looked up rather than assumed:
 *
 *  - Linux publishes no CUDA or Vulkan archive at this pin. The Linux tarball
 *    contains only `libggml-cpu-*`, so an NVIDIA machine gets the CPU build.
 *    That is a real limitation, not an oversight — the installer says so in its
 *    blurb rather than pretending otherwise.
 *  - Windows does publish CUDA archives, named with the CUDA version in them.
 *
 * When a Linux CUDA archive does appear upstream the naming is
 * `whisper-<build>-bin-ubuntu-cuda-<version>-<arch>.tar.gz`, and the probe
 * below will pick it up without a code change beyond bumping WHISPER_BUILD.
 */
export type WhisperKind = 'cpu' | 'cuda';

export function whisperAsset(): { name: string; kind: WhisperKind } | null {
  const arch = process.arch === 'arm64' ? 'arm64' : 'x64';
  if (process.platform === 'linux') {
    return { name: `whisper-bin-ubuntu-${arch}.tar.gz`, kind: 'cpu' };
  }
  if (process.platform === 'win32') {
    // The x64 CPU build is published without a CPU marker (`Win32` is the
    // 32-bit one); an arm64 Windows box is not something this installer claims
    // to support, so it is told rather than given a binary that will not run.
    if (arch !== 'x64') return null;
    return { name: 'whisper-bin-Win32.zip', kind: 'cpu' };
  }
  // macOS is published as an xcframework, which is a different shape entirely.
  return null;
}

/* ------------------------------------------------------------------ models */

export interface WhisperModel {
  /** Display name, and the id the transcription request names. */
  id: string;
  repo: string;
  file: string;
  /** Absolute path to the weights. */
  path: string;
  sizeBytes: number;
  location: 'models' | 'cache';
  /** True when the file is a size whisper.cpp can actually load. */
  usable: boolean;
  /** Why not, when not. */
  why?: string;
}

/** `ggml-small.en-q8_0.bin` -> `small.en q8_0`, which reads far better in a picker. */
function labelFor(file: string): string {
  const name = file.split('/').pop() ?? file;
  const stem = name.replace(/^ggml-/, '').replace(/\.bin$/i, '');
  const quant = /\b(q8_0|q5_1|q5_0|f16|f32)\b/i.exec(stem);
  const base = quant ? stem.replace(quant[0], '') : stem;
  const label = base.replace(/[-_]+/g, ' ').trim();
  return quant ? `${label} (${quant[0]})` : label;
}

/**
 * Every downloaded whisper.cpp model, with the ones that cannot be loaded
 * flagged rather than hidden.
 *
 * The unusable case is not hypothetical: the Hub is full of GGUF Whisper
 * conversions that are perfectly good models for transcribe.cpp and mean
 * nothing here. Offering one and failing at load time is the worst outcome, so
 * they are listed with the reason attached.
 */
let cache: { at: number; models: WhisperModel[] } | null = null;
const CACHE_MS = 5000;

export async function whisperModels(force = false): Promise<WhisperModel[]> {
  if (!force && cache && Date.now() - cache.at < CACHE_MS) return cache.models;
  const files = await listDownloads().catch(() => []);
  const models: WhisperModel[] = [];
  for (const f of files) {
    if (f.repo !== WHISPER_REPO) continue;
    const name = f.file.split('/').pop() ?? f.file;
    if (!/\.bin$/i.test(name)) continue;
    // The repo also carries the non-quantised originals and a couple of test
    // fixtures. The names are a fixed convention, so this is the whole filter.
    if (!/^ggml-[a-z0-9.]+(-[a-z0-9_]+)*\.bin$/i.test(name)) continue;
    const usable = /^(ggml-(tiny|base|small|medium|large)(-v[0-9](-turbo)?)?(\.en)?(-q[0-9]_[0-9])?)\.bin$/i.test(name);
    models.push({
      id: name.replace(/^ggml-/, '').replace(/\.bin$/i, ''),
      repo: f.repo,
      file: f.file,
      path: path.join(f.root, ...f.file.split('/')),
      sizeBytes: f.size,
      location: f.location,
      usable,
      why: usable ? undefined : 'whisper.cpp cannot load this file — it needs one of its own ggml-*.bin models',
    });
  }
  models.sort((a, b) => a.id.localeCompare(b.id));
  cache = { at: Date.now(), models };
  return models;
}

export function cachedWhisperModels(): WhisperModel[] {
  return cache?.models ?? [];
}

export async function resolveWhisperModel(id: string): Promise<WhisperModel | null> {
  const want = String(id ?? '').trim();
  if (!want) return null;
  const all = await whisperModels();
  return (
    all.find((m) => m.id === want) ??
    all.find((m) => m.file === want) ??
    all.find((m) => m.id.toLowerCase() === want.toLowerCase()) ??
    null
  );
}

/* -------------------------------------------------------------- transcribe */

/** Audio containers whisper.cpp opens without help. */
const READABLE = new Set(['wav', 'flac', 'mp3', 'ogg']);

/**
 * A transcription result in the shape the browser already consumes.
 *
 * `whisper_ct2.py` returns `segments` and `transcribe.py` returns `chunks`, and
 * the Studio reads one while the Settings run panel reads the other. whisper.cpp
 * emits both here rather than making a third caller guess which one this runner
 * happens to use.
 */
export interface WhisperTranscript {
  text: string;
  language: string | null;
  duration: number | null;
  segments: Array<{ start: number; end: number; text: string }>;
  chunks: Array<{ start: number | null; end: number | null; text: string }>;
}

export class WhisperError extends Error {
  constructor(
    message: string,
    /** True when the failure is a missing runtime rather than bad input. */
    readonly missingRuntime = false,
  ) {
    super(message);
    this.name = 'WhisperError';
  }
}

/**
 * Transcribes one clip with `whisper-cli`.
 *
 * The JSON output file is the source of truth rather than stdout: whisper-cli
 * prints its own progress and model banner interleaved with the transcript, and
 * the file is structured, so parsing it avoids guessing where the text starts.
 */
export function transcribeWithWhisperCpp(opts: {
  model: WhisperModel;
  audioPath: string;
  /** BCP-47-ish code, or 'auto' to detect. */
  language?: string;
  timeoutMs?: number;
  onLog?: (line: string) => void;
}): Promise<{ text: string; pid: number; tail: string[] }> {
  const bin = whisperCliPath();
  if (!bin) {
    throw new WhisperError(
      'whisper.cpp is not installed on this machine. Install it under Settings → Runtimes, then transcribe again.',
      true,
    );
  }
  if (!opts.model.usable) {
    throw new WhisperError(
      `whisper.cpp cannot load ${opts.model.file} — ${opts.model.why}.`,
    );
  }

  const ext = path.extname(opts.audioPath).slice(1).toLowerCase();
  if (!READABLE.has(ext)) {
    // Reached only if a recording reaches the server without being decoded. The
    // browser converts before uploading for exactly this reason, so seeing it
    // means that step was skipped — say which, rather than letting the CLI fail
    // with something about a file header.
    throw new WhisperError(
      `whisper.cpp cannot read ${ext || 'that'} audio; it reads ${[...READABLE].join(', ')}. ` +
        'This recording has to be decoded to 16 kHz WAV first.',
    );
  }

  // Under the runtimes dir rather than a bare relative path, so a test that
  // points RUNTIMES_DIR somewhere throwaway does not leave transcripts in the
  // repository it was run from.
  const outBase = path.join(runtimesDir(), 'out', `whisper-${process.pid}-${Date.now()}`);
  const args = [
    '-m',
    opts.model.path,
    '-f',
    opts.audioPath,
    '-of',
    outBase,
    // Machine-readable output. Parsed below rather than scraped off stdout,
    // which also carries the model banner and progress.
    '-oj',
    '-np',
  ];
  // An English-only checkpoint is told nothing: handed `-l` it warns that the
  // option is being ignored, and `auto` is meaningless to it anyway. A
  // multilingual one is asked to detect rather than assumed to be English.
  // Matches `ggml-small.en.bin` and `ggml-small.en-q8_0.bin` alike. The
  // separator after `.en` is a dot in the plain name and a dash before a
  // quant suffix, so both have to be accepted.
  if (!/\.en([.-]|$)/i.test(opts.model.file)) {
    args.push('-l', opts.language || 'auto');
  }

  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, LD_LIBRARY_PATH: libPathFor(bin) },
    });
    const tail: string[] = [];
    const remember = (chunk: Buffer) => {
      for (const line of String(chunk).split(/\r?\n/)) {
        const t = line.trim();
        if (!t) continue;
        tail.push(t);
        while (tail.length > 40) tail.shift();
        opts.onLog?.(t);
      }
    };
    child.stdout?.on('data', remember);
    child.stderr?.on('data', remember);

    // Registered as soon as the pid exists, so a long transcription is visible
    // in the resource manager from its first moment and can be stopped from
    // there, exactly like the Python runners are.
    const job = registerProcess({
      kind: 'stt',
      script: 'whisper-cli',
      pid: child.pid ?? -1,
      device: whisperDevice(),
    });
    attachChild(job.id, child);

    const timer = setTimeout(
      () => {
        try {
          child.kill('SIGKILL');
        } catch {
          /* already gone */
        }
      },
      opts.timeoutMs ?? 20 * 60_000,
    );
    timer.unref?.();

    child.on('error', (err) => {
      clearTimeout(timer);
      finishProcess(job.id, null, null);
      reject(new WhisperError(`Could not start whisper-cli: ${err.message}`, true));
    });
    child.on('close', (code, signal) => {
      clearTimeout(timer);
      if (finishProcess(job.id, code ?? null, signal)?.stoppedByUser) {
        reject(new WhisperError('Stopped from the resource manager.'));
        return;
      }
      if (signal) {
        reject(new WhisperError(`whisper-cli was stopped (${signal}).`));
        return;
      }
      if (code !== 0) {
        reject(
          new WhisperError(
            `whisper-cli exited ${code}${tail.length ? ` — ${tail.slice(-3).join(' · ')}` : ''}`,
          ),
        );
        return;
      }
      resolve({ text: `${outBase}.json`, pid: child.pid ?? -1, tail });
    });
  });
}

/**
 * Turns whisper.cpp's JSON file into the shape the browser reads.
 *
 * Upstream's own schema: `{ result: { language }, transcription: [{ timestamps,
 * offsets, text }] }`, with both timestamps as `HH:MM:SS,mmm` strings and the
 * offsets already in milliseconds. The offsets are used because they need no
 * parsing and cannot drift from the segment boundaries.
 */
/** Upstream's own schema, as written to the JSON output file. */
interface WhisperJsonDoc {
  result?: { language?: string };
  transcription?: Array<{ offsets?: { from?: number; to?: number }; text?: string }>;
}

export function parseWhisperJson(raw: string): WhisperTranscript {
  let doc: WhisperJsonDoc;
  try {
    doc = JSON.parse(raw) as WhisperJsonDoc;
  } catch (err) {
    throw new WhisperError(
      `whisper.cpp wrote output this app cannot read: ${(err as Error).message}`,
    );
  }
  const rows = Array.isArray(doc.transcription) ? doc.transcription : [];
  const segments = rows.map((r) => ({
    start: Number(r.offsets?.from ?? 0) / 1000,
    end: Number(r.offsets?.to ?? 0) / 1000,
    text: String(r.text ?? '').trim(),
  }));
  const last = segments.length ? segments[segments.length - 1].end : 0;
  return {
    text: segments
      .map((s) => s.text)
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim(),
    language: doc.result?.language ?? null,
    duration: segments.length ? last : null,
    segments,
    chunks: segments.map((s) => ({ start: s.start, end: s.end, text: s.text })),
  };
}

/** The kind of compute the CLI is doing, for the process row. */
export function whisperDevice(): 'cpu' | 'gpu' {
  const bin = whisperCliPath();
  if (!bin) return 'cpu';
  try {
    return readdirSync(path.dirname(bin)).some((n) => /^libggml-(cuda|vulkan|hip)/i.test(n))
      ? 'gpu'
      : 'cpu';
  } catch {
    return 'cpu';
  }
}

/** Used by the installer and the selftests to prove a build is runnable. */
export function whisperVersion(): string | null {
  const bin = whisperCliPath();
  if (!bin) return null;
  const out = spawnSync(bin, ['--version'], {
    encoding: 'utf8',
    env: { ...process.env, LD_LIBRARY_PATH: libPathFor(bin) },
  });
  if (out.status !== 0) return null;
  return String(out.stdout || '').trim().split('\n')[0] || null;
}