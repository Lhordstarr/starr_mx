/**
 * Runs the optional Python inference scripts.
 *
 * Everything here is a fixed script chosen by the caller, an argument array
 * (never a shell string), and a timeout. The only free-form values are a
 * downloaded model directory and user text, and text goes to stdin rather than
 * argv so it can be arbitrarily long and can never be read as a flag.
 *
 * Model directories are resolved through listDownloads() so a caller can't
 * hand this module an arbitrary path: the name has to be a file the Hub
 * downloader actually wrote.
 */
import { spawn } from 'node:child_process';
import { access, readFile, stat } from 'node:fs/promises';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { hubCacheDir, listDownloads, modelsDir } from './huggingface.js';
import { attachChild, finishProcess, registerProcess, setProcessDevice } from './processes.js';
import { venvPython, type Device } from './runtimes.js';
import { WHISPER_REPO, parseWhisperJson, resolveWhisperModel, transcribeWithWhisperCpp } from './whispercpp.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const scriptsDir = path.resolve(here, '..', 'scripts', 'py');
const outputsDir = () => path.resolve(process.env.RUNTIME_OUT_DIR ?? './.runtimes/out');

/**
 * Which venv a task runs in. The shared venv covers speech-to-text and image
 * generation; Kokoro cannot import on the machine's interpreter, so its own
 * environment answers text-to-speech.
 */
const VENV_FOR_KIND = { stt: 'venv', image: 'venv', tts: 'venv-kokoro' } as const;

export interface RunResult {
  ok: boolean;
  /** Parsed JSON the script printed last. */
  data: Record<string, unknown> | null;
  /** Everything the script wrote to stderr, for the UI log. */
  log: string;
}

/** Where browser-uploaded clips are staged, relative to the models root. */
const UPLOAD_NS = '_uploads';

/**
 * Confirms the target is a real file this app knows about and returns its
 * absolute path. This is the only way a path reaches a script.
 *
 * Two sources are legitimate: a model on disk — either fetched by this app's
 * Hub downloader, or already sitting in the shared Hugging Face cache another
 * tool wrote — and a clip the user just uploaded. Both are checked against
 * what the app actually produced rather than by joining caller-supplied
 * strings, and both are re-checked against their roots afterwards as a last
 * gate before the path becomes a subprocess argument.
 */
async function resolveDownloaded(repo: string, file: string): Promise<string> {
  // An upload, addressed as `_uploads/clip-….webm` with a placeholder repo.
  if (file.startsWith(`${UPLOAD_NS}/`)) {
    const base = modelsDir();
    const full = path.resolve(base, ...file.split('/'));
    if (!full.startsWith(path.join(base, UPLOAD_NS) + path.sep)) {
      throw new Error('Upload path escaped the upload directory.');
    }
    try {
      await stat(full);
    } catch {
      throw new Error('That upload is no longer on disk.');
    }
    return full;
  }

  const files = await listDownloads();
  const hit = files.find((f) => f.repo === repo && f.file === file);
  if (!hit) throw new Error(`${file} is not downloaded yet — get it from the catalog first.`);
  // The root came from the scan of exactly the two places a download can live
  // — this app's models dir and the shared HF cache — and is checked against
  // them again here. A caller-supplied string never becomes a path.
  const roots = [modelsDir(), hubCacheDir()].map((root) => path.resolve(root));
  if (!roots.some((root) => hit.root.startsWith(root + path.sep))) {
    throw new Error('Resolved path escaped the model directories.');
  }
  return path.join(hit.root, ...file.split('/'));
}

async function run(
  script: string,
  args: string[],
  opts: {
    stdin?: string;
    env?: Record<string, string>;
    timeoutMs: number;
    onLog?: (s: string) => void;
    /** Shown in the resource manager, so a long job can be identified. */
    kind?: 'stt' | 'tts' | 'image';
    repo?: string;
    device?: string;
  },
): Promise<RunResult> {
  const py = venvPython(VENV_FOR_KIND[opts.kind ?? 'image']);
  return new Promise((resolve, reject) => {
    const child = spawn(py, [path.join(scriptsDir, script), ...args], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, ...opts.env },
    });

    // Registered as soon as the pid exists, so the job is visible in the
    // resource manager from its first moment rather than only once it is
    // producing output.
    const job = registerProcess({
      kind: opts.kind ?? 'image',
      script,
      pid: child.pid ?? -1,
      repo: opts.repo,
      device: opts.device ?? 'cpu',
    });
    attachChild(job.id, child);

    let out = '';
    let err = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error(`${script} exceeded ${Math.round(opts.timeoutMs / 1000)}s and was stopped.`));
    }, opts.timeoutMs);

    // The runners print `{"device": ...}` on stdout the moment pick_device
    // resolves — usually seconds before the model is even loaded. Reading
    // those lines as they arrive lets the resource manager say "CPU" for a job
    // that was asked for "auto" on a machine that has no CUDA, which is the
    // truth the request itself can never carry. Line-buffered: a chunk can end
    // mid-object.
    let lineBuf = '';
    const scanDevice = (chunk: string) => {
      lineBuf += chunk;
      let nl: number;
      while ((nl = lineBuf.indexOf('\n')) !== -1) {
        const line = lineBuf.slice(0, nl).trim();
        lineBuf = lineBuf.slice(nl + 1);
        if (!line.startsWith('{')) continue;
        try {
          const obj = JSON.parse(line) as { device?: unknown };
          if (typeof obj.device === 'string') setProcessDevice(job.id, obj.device);
        } catch {
          /* progress chatter, not a result */
        }
      }
    };

    child.stdout.on('data', (d) => {
      out += d;
      scanDevice(d.toString());
    });
    child.stderr.on('data', (d) => {
      err += d;
      opts.onLog?.(d.toString());
    });

    child.on('error', (e) => {
      clearTimeout(timer);
      finishProcess(job.id, null);
      const code = (e as NodeJS.ErrnoException).code;
      reject(
        code === 'ENOENT'
          ? new Error('No Python environment found. Install a runtime first.')
          : e,
      );
    });

    child.on('close', (code, signal) => {
      clearTimeout(timer);
      // Anything after the last newline still counts: the result line carries
      // the device too and may have arrived without a trailing one.
      scanDevice('\n');
      let data: Record<string, unknown> | null = null;
      // The scripts print one JSON object per line; the last is the answer.
      for (const line of out.split('\n')) {
        const t = line.trim();
        if (!t.startsWith('{')) continue;
        try {
          data = JSON.parse(t) as Record<string, unknown>;
        } catch {
          /* a partial line, not the result */
        }
      }
      // Before finishProcess moves the entry to `recent`: a job stopped from
      // the manager is dropped from `active`, and the label has to be right
      // for the moment it is still listed as running.
      if (typeof data?.device === 'string') setProcessDevice(job.id, data.device);
      // Read the finished entry back: stopProcess marks the job while it is
      // still live, so this is the only place the "the user asked for this" flag
      // can still be seen.
      const finished = finishProcess(job.id, code ?? null, signal);
      // A job stopped from the resource manager is a decision, not a fault.
      // Reporting it as a success leaves the caller with no output and no
      // explanation, which is the one result nobody can act on.
      if (finished?.stoppedByUser) {
        return resolve({
          ok: false,
          data: { error: 'Stopped from the resource manager.' },
          log: err,
        });
      }
      if (data?.error) return resolve({ ok: false, data, log: err });
      resolve({ ok: true, data, log: err });
    });

    if (opts.stdin !== undefined) child.stdin.end(opts.stdin);
    else child.stdin.end();
  });
}

/**
 * True when a downloaded repository is a CTranslate2 Whisper.
 *
 * The marker is `config.json` next to a `model.bin`: that pairing is what
 * CTranslate2 loads, and it is the difference between a 40 MB runner and a
 * 529 MB PyTorch one. Anything else falls through to the transformers runner.
 */
async function isCTranslate2(dir: string): Promise<boolean> {
  try {
    await access(path.join(dir, 'config.json'));
    await access(path.join(dir, 'model.bin'));
    return true;
  } catch {
    return false;
  }
}

export async function transcribe(opts: {
  repo: string;
  /** The model file, as the catalog names it (e.g. `model.bin`). */
  file: string;
  /**
   * The audio to transcribe: an `_uploads/…` path from the browser, or another
   * downloaded file.
   *
   * This is separate from `file` on purpose. They used to share one parameter,
   * so `resolveDownloaded` returned the *uploaded audio*, its directory became
   * the "model directory", and the runners were handed `./_uploads` as their
   * weights. Transcription could not have worked at any point.
   */
  audio: string;
  device: Device;
  onLog?: (s: string) => void;
}): Promise<RunResult> {
  const modelPath = await resolveDownloaded(opts.repo, opts.file);
  const audioPath = await resolveDownloaded(opts.repo, opts.audio);

  // whisper.cpp first, and by identity rather than by shape: it is the only
  // runner that wants the single weights file, where the two Python ones want
  // the directory holding them. Probing the extension alone would put a GGUF
  // Whisper — a transcribe.cpp model whisper.cpp cannot open — on this path,
  // which is the mistake that put two unloadable entries in the catalog.
  if (opts.repo === WHISPER_REPO) {
    return transcribeViaWhisperCpp(opts.repo, opts.file, modelPath, audioPath, opts.onLog);
  }

  // Both Python Whisper builds want the directory that holds the weights, not
  // the single file, because each loads its neighbours (config, tokenizer,
  // vocabulary) by relative name.
  const dir = path.dirname(modelPath);

  if (await isCTranslate2(dir)) {
    return run('whisper_ct2.py', [dir, audioPath, opts.device], {
      timeoutMs: 20 * 60_000,
      onLog: opts.onLog,
      kind: 'stt',
      repo: opts.repo,
      device: opts.device,
    });
  }
  return run('transcribe.py', [dir, audioPath, opts.device], {
    timeoutMs: 20 * 60_000,
    onLog: opts.onLog,
    kind: 'stt',
    repo: opts.repo,
    device: opts.device,
  });
}

/**
 * Runs one clip through whisper.cpp and folds its JSON back into the shape the
 * browser already reads.
 *
 * A separate path rather than a fourth argument to `run()`, because `run()`
 * speaks Python: it resolves the shared venv's interpreter and expects a script
 * that prints one JSON object on stdout. whisper-cli is a native binary that
 * writes its own file, and the failure of trying to force it through the Python
 * helper would be an interpreter that does not exist — on a machine that
 * installed whisper.cpp precisely because it has no usable Python.
 */
async function transcribeViaWhisperCpp(
  repo: string,
  file: string,
  modelPath: string,
  audioPath: string,
  onLog?: (s: string) => void,
): Promise<RunResult> {
  const model = await resolveWhisperModel(file) ?? {
    id: file.replace(/^ggml-/, '').replace(/\.bin$/i, ''),
    repo,
    file,
    path: modelPath,
    sizeBytes: 0,
    location: 'models' as const,
    usable: true,
  };
  try {
    const { text: jsonPath } = await transcribeWithWhisperCpp({ model, audioPath, onLog });
    const raw = await readFile(jsonPath, 'utf8');
    const parsed = parseWhisperJson(raw);
    return { ok: true, data: parsed as unknown as Record<string, unknown>, log: '' };
  } catch (err) {
    return {
      ok: false,
      data: { error: err instanceof Error ? err.message : String(err) },
      log: '',
    };
  }
}

export async function speak(opts: {
  repo: string;
  file: string;
  voice: string;
  speed: number;
  device: Device;
  text: string;
  onLog?: (s: string) => void;
}): Promise<{ result: RunResult; outPath: string }> {
  const modelDir = await resolveDownloaded(opts.repo, opts.file);
  const dir = path.dirname(modelDir);
  await mkdirSafe(outputsDir());
  const outPath = path.join(outputsDir(), `tts-${Date.now()}.wav`);
  const result = await run('speak.py', [dir, opts.voice, outPath, opts.device, String(opts.speed), opts.repo], {
    stdin: opts.text,
    timeoutMs: 10 * 60_000,
    onLog: opts.onLog,
    kind: 'tts',
    repo: opts.repo,
    device: opts.device,
  });
  return { result, outPath };
}

/** True when a directory is a diffusers checkpoint rather than a single file. */
async function looksLikeDiffusers(dir: string): Promise<boolean> {
  try {
    await access(path.join(dir, 'model_index.json'));
    return true;
  } catch {
    return false;
  }
}

export async function generateImageLocal(opts: {
  repo: string;
  file: string;
  prompt: string;
  negative?: string;
  steps: number;
  width: number;
  height: number;
  device: Device;
  onLog?: (s: string) => void;
}): Promise<{ result: RunResult; outPath: string }> {
  const modelDir = await resolveDownloaded(opts.repo, opts.file);
  // A diffusers model is a *directory tree* — model_index.json plus unet/,
  // vae/, text_encoder/, tokenizer/ — so the pipeline is loaded from the repo
  // root. Taking the parent of whatever file was downloaded handed it
  // `unet/` and diffusers raised "model_index.json not found", which reads
  // like a corrupt download rather than a path bug.
  const repoRoot = path.resolve(
    path.resolve(process.env.HF_MODELS_DIR ?? './models'),
    opts.repo.replace(/\//g, '--'),
  );
  const dir = await looksLikeDiffusers(repoRoot) ? repoRoot : path.dirname(modelDir);
  await mkdirSafe(outputsDir());
  const outPath = path.join(outputsDir(), `img-${Date.now()}.png`);
  const result = await run(
    'generate.py',
    [dir, outPath, opts.device, String(opts.steps), `${opts.width}x${opts.height}`],
    {
      stdin: opts.prompt,
      env: { NEGATIVE_PROMPT: opts.negative ?? '' },
      timeoutMs: 60 * 60_000,
      onLog: opts.onLog,
      kind: 'image',
      repo: opts.repo,
      device: opts.device,
    },
  );
  return { result, outPath };
}

async function mkdirSafe(dir: string): Promise<void> {
  const { mkdir } = await import('node:fs/promises');
  await mkdir(dir, { recursive: true });
}