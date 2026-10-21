/**
 * Speech engines that stay loaded.
 *
 * A voice used to be made by a process that lived for exactly as long as the
 * utterance: it loaded the weights, said one thing, and exited. Every reply
 * paid for the load again — several seconds for Kokoro, more for the ONNX
 * export, whose inference session is not cheap to build — and the resource
 * manager showed a job that was gone before it could be read.
 *
 * This is the other half of that story. The same runners, kept in one process
 * and asked for one utterance at a time, so the load is paid once and the
 * engine is still there for the next reply.
 *
 * The rules it follows:
 *
 *  - **It waits, it does not linger forever.** A loaded voice that nobody
 *    asks for is released after ten idle minutes, the same keep-alive the
 *    transformers chat server has used all along. Long enough that a person
 *    reading and answering is still served from memory; short enough that a
 *    machine nobody is talking to is not holding a model for hours.
 *  - **It can be let go of at any time.** The engine is an ordinary entry in
 *    the process registry, so the resource manager's button releases it, and
 *    the memory it holds goes with it.
 *  - **It never breaks the feature.** If an engine cannot start, the request
 *    falls back to the one-shot runner, which is slower and always worked.
 *
 * The engine speaks one JSON object per line in both directions; the protocol
 * and the reasoning behind it are in `scripts/py/tts_engine.py`.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  attachChild,
  finishProcess,
  registerProcess,
  setProcessDevice,
  setProcessWarm,
} from './processes.js';
import { venvPython, type Device } from './runtimes.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const scriptsDir = path.resolve(here, '..', 'scripts', 'py');

/** How long a loaded voice waits for its next job before it is released. */
const idleMs = () => Number(process.env.TTS_ENGINE_IDLE_MS ?? 10 * 60_000);

/** How long the model may take to load before the engine is given up on. */
const loadTimeoutMs = () => Number(process.env.TTS_ENGINE_LOAD_MS ?? 10 * 60_000);

/**
 * How long one utterance may take before the engine is written off.
 *
 * A request that never gets an answer would otherwise hang the browser's
 * fetch for ever, and the engine would sit there holding a model for a reply
 * nobody is waiting for. The engine is released and the request falls back to
 * the one-shot runner, which has a timeout of its own.
 */
const requestMs = () => Number(process.env.TTS_ENGINE_REQUEST_MS ?? 10 * 60_000);

/** How long an engine that failed to start is left alone before it is retried. */
const brokenTtlMs = () => Number(process.env.TTS_ENGINE_BROKEN_MS ?? 5 * 60_000);

export type TtsRunner = 'kokoro' | 'kokoro-onnx' | 'transformers';

export interface TtsEngineSpec {
  /** Which runner answers, and therefore which environment it runs in. */
  runner: TtsRunner;
  /** The directory holding the weights and everything they need. */
  dir: string;
  /** The device asked for; "auto" is resolved by the runner itself. */
  device: Device;
  /** The catalog repo, for the resource manager's label. */
  repo: string;
  /** The exact ONNX graph, for the ONNX runner. */
  modelFile?: string;
  /** Which Python environment holds the runtime this model needs. */
  venv: string;
  /** The runner script, so the resource manager names the right packages. */
  script: string;
}

export interface TtsUtterance {
  text: string;
  voice: string;
  speed: number;
  /** Where the wav goes. An absolute path this app chose. */
  outPath: string;
  onLog?: (s: string) => void;
}

export interface TtsEngineResult {
  ok: boolean;
  data: Record<string, unknown> | null;
  log: string;
}

/** A request waiting for its answer. */
interface Pending {
  resolve: (value: { ok: boolean; data: Record<string, unknown> }) => void;
  reject: (err: Error) => void;
  onLog?: (s: string) => void;
}

class Engine {
  readonly spec: TtsEngineSpec;
  readonly key: string;
  readonly child: ChildProcess;
  jobId = '';
  ready = false;
  dead = false;
  device = 'cpu';
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private buffer = '';
  private idleTimer: NodeJS.Timeout | null = null;
  private readonly logLines: string[] = [];
  /** Why the engine could not start, when that is what happened. */
  private startError: Error | null = null;
  private startWaiters: Array<{ resolve: () => void; reject: (err: Error) => void }> = [];

  constructor(spec: TtsEngineSpec, key: string, child: ChildProcess) {
    this.spec = spec;
    this.key = key;
    this.child = child;
  }

  /** Everything the engine has said on stderr, for the run log. */
  get log(): string {
    return this.logLines.join('');
  }

  /** The engine is loaded and waiting. */
  markReady(device: string): void {
    this.ready = true;
    this.device = device;
    if (this.jobId) setProcessDevice(this.jobId, device);
    // Whoever was waiting for the model to load is waiting for *this* line.
    // Without it the load looks like it never finishes, and the request that
    // asked for it waits for a model that is already there.
    this.startWaiters.forEach((w) => w.resolve());
    this.startWaiters = [];
  }

  /** The engine is working; it is not idle any more. */
  markWorking(): void {
    if (this.jobId) setProcessWarm(this.jobId, false);
  }

  /**
   * Starts the wait for the next job.
   *
   * Ten minutes by default, and the deadline is published rather than kept
   * private: the resource manager says when the engine will let go of the
   * memory it is holding, so "the voice is still there" and "the voice is
   * still there for another four minutes" are both answerable.
   */
  touchIdle(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    const ms = idleMs();
    if (!Number.isFinite(ms) || ms <= 0) {
      if (this.jobId) setProcessWarm(this.jobId, false);
      return;
    }
    const until = Date.now() + ms;
    if (this.jobId) setProcessWarm(this.jobId, true, until);
    this.idleTimer = setTimeout(() => {
      void this.release();
    }, ms);
    this.idleTimer.unref?.();
  }

  /** Lets go of the model. The exit handler does the rest. */
  async release(): Promise<void> {
    if (this.dead) return;
    try {
      this.child.kill('SIGTERM');
    } catch {
      /* already gone */
    }
  }

  /** Feeds a line of the engine's output into whatever is waiting for it. */
  private onLine(line: string): void {
    const text = line.trim();
    if (!text.startsWith('{')) return;
    let doc: Record<string, unknown>;
    try {
      doc = JSON.parse(text) as Record<string, unknown>;
    } catch {
      return; // a partial line, or chatter
    }

    // The device the runner resolved, reported the moment it knows.
    if (typeof doc.device === 'string' && doc.ready !== true) {
      this.device = doc.device;
      if (this.jobId) setProcessDevice(this.jobId, doc.device);
    }

    if (doc.ready === true) {
      this.markReady(typeof doc.device === 'string' ? doc.device : this.device);
      return;
    }
    if (doc.ready === false) {
      this.failStart(String(doc.error ?? 'the engine could not load the model'));
      return;
    }

    const id = typeof doc.id === 'number' ? doc.id : null;
    if (id === null) return;
    const waiter = this.pending.get(id);
    if (!waiter) return;
    if (doc.ok === true) {
      this.pending.delete(id);
      waiter.resolve({ ok: true, data: (doc.result as Record<string, unknown>) ?? {} });
      return;
    }
    if (doc.ok === false) {
      this.pending.delete(id);
      waiter.resolve({ ok: false, data: { error: String(doc.error ?? 'the engine failed') } });
    }
  }

  /** A start that will not happen: the engine said so, or it died trying. */
  private failStart(reason: string): void {
    this.dead = true;
    const err = new Error(reason);
    this.startError = err;
    for (const [, waiter] of this.pending) waiter.reject(err);
    this.pending.clear();
    this.startWaiters.forEach((w) => w.reject(err));
    this.startWaiters = [];
  }

  /** Resolves once the model is loaded. */
  waitForReady(timeoutMs: number): Promise<void> {
    if (this.ready) return Promise.resolve();
    if (this.startError) return Promise.reject(this.startError);
    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`the voice engine did not finish loading within ${Math.round(timeoutMs / 1000)}s`));
      }, timeoutMs);
      timer.unref?.();
      this.startWaiters.push({
        resolve: () => {
          clearTimeout(timer);
          resolve();
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        },
      });
    });
  }

  /** Reads the engine's output. */
  feed(chunk: string): void {
    this.buffer += chunk;
    let nl: number;
    while ((nl = this.buffer.indexOf('\n')) !== -1) {
      const line = this.buffer.slice(0, nl);
      this.buffer = this.buffer.slice(nl + 1);
      this.onLine(line);
    }
  }

  /** The engine's own notes, which the server streams into the run log. */
  note(text: string): void {
    this.logLines.push(text);
    while (this.logLines.length > 200) this.logLines.shift();
  }

  /** Passes the engine's notes to whoever is waiting for a voice right now. */
  broadcast(text: string): void {
    for (const [, waiter] of this.pending) waiter.onLog?.(text);
  }

  /** Sends one utterance and waits for the wav. */
  request(utterance: TtsUtterance): Promise<{ ok: boolean; data: Record<string, unknown> }> {
    if (this.dead) return Promise.reject(new Error('the voice engine is not running'));
    const id = this.nextId++;
    this.markWorking();
    return new Promise((resolve, reject) => {
      const limit = requestMs();
      const timer = Number.isFinite(limit) && limit > 0
        ? setTimeout(() => {
            this.pending.delete(id);
            // An engine that will not answer is not an engine anybody should
            // keep asking. It is let go of here, and the request falls back to
            // the one-shot runner rather than waiting a second time.
            this.dead = true;
            void this.release();
            reject(new Error(`the voice engine did not answer within ${Math.round(limit / 1000)}s`));
          }, limit)
        : null;
      timer?.unref?.();
      const settle = (fn: () => void): void => {
        if (timer) clearTimeout(timer);
        fn();
      };
      this.pending.set(id, {
        resolve: (value) => settle(() => resolve(value)),
        reject: (err) => settle(() => reject(err)),
        onLog: utterance.onLog,
      });
      const line =
        JSON.stringify({
          id,
          text: utterance.text,
          voice: utterance.voice,
          speed: utterance.speed,
          out: utterance.outPath,
        }) + '\n';
      const stdin = this.child.stdin;
      if (!stdin) {
        this.pending.delete(id);
        settle(() => reject(new Error('the voice engine has no way to accept work')));
        return;
      }
      stdin.write(line, (err) => {
        if (!err) return;
        this.pending.delete(id);
        settle(() =>
          reject(new Error(`the voice engine stopped accepting work: ${err.message}`)),
        );
      });
    });
  }

  /** The engine is gone. Everything waiting on it is told. */
  onExit(code: number | null, signal: NodeJS.Signals | null): void {
    this.dead = true;
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
    const err = new Error(
      signal ? `the voice engine stopped (${signal})` : `the voice engine stopped (exit ${code ?? 'unknown'})`,
    );
    for (const [, waiter] of this.pending) waiter.reject(err);
    this.pending.clear();
    this.startWaiters.forEach((w) => w.reject(err));
    this.startWaiters = [];
  }
}

const engines = new Map<string, Engine>();
/** Engines that could not start, so the next request does not try again at once. */
const broken = new Map<string, { at: number; reason: string }>();

/** One engine per model, per device. */
function engineKey(spec: TtsEngineSpec): string {
  return [spec.runner, spec.dir, spec.modelFile ?? '', String(spec.device)].join('|');
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Says one thing with a loaded voice, loading it if this is the first time.
 *
 * Returns null when the engine path could not be used at all, which tells the
 * caller to fall back to the one-shot runner rather than to fail.
 */
export async function speakViaEngine(
  spec: TtsEngineSpec,
  utterance: TtsUtterance,
): Promise<TtsEngineResult | null> {
  const key = engineKey(spec);
  const failure = broken.get(key);
  if (failure && Date.now() - failure.at < brokenTtlMs()) return null;

  let engine = engines.get(key);
  if (!engine || engine.dead) {
    try {
      engine = await startEngine(spec, key);
      engines.set(key, engine);
    } catch (err) {
      broken.set(key, { at: Date.now(), reason: message(err) });
      console.error('[tts-engine] could not start:', message(err));
      return null;
    }
  }

  try {
    const answer = await engine.request(utterance);
    engine.touchIdle();
    return { ok: answer.ok, data: answer.data, log: engine.log };
  } catch (err) {
    if (engine.dead) {
      // The engine died under the request. The one-shot runner is the older,
      // slower path that always worked, so the reply still gets made.
      broken.set(key, { at: Date.now(), reason: message(err) });
      engines.delete(key);
      console.error('[tts-engine] stopped mid-request:', message(err));
      return null;
    }
    // The engine is fine; this one request was not. It goes back to waiting,
    // with the same deadline as any other idle engine.
    engine.touchIdle();
    return { ok: false, data: { error: message(err) }, log: engine.log };
  }
}

/** Spawns the engine and waits for it to say the model is loaded. */
async function startEngine(spec: TtsEngineSpec, key: string): Promise<Engine> {
  const py = venvPython(spec.venv);
  const args = [
    path.join(scriptsDir, 'tts_engine.py'),
    '--runner',
    spec.runner,
    '--model-dir',
    spec.dir,
    '--device',
    String(spec.device),
  ];
  if (spec.repo) args.push('--repo', spec.repo);
  if (spec.modelFile) args.push('--model-file', spec.modelFile);

  const child = spawn(py, args, { stdio: ['pipe', 'pipe', 'pipe'] });
  const engine = new Engine(spec, key, child);

  const job = registerProcess({
    kind: 'tts',
    script: spec.script,
    pid: child.pid ?? -1,
    repo: spec.repo,
    device: spec.device,
  });
  engine.jobId = job.id;
  attachChild(job.id, child);

  child.stdout?.on('data', (d: Buffer) => engine.feed(String(d)));
  child.stderr?.on('data', (d: Buffer) => {
    const text = String(d);
    engine.note(text);
    // The notes go to the run log of whoever is waiting for a voice.
    engine.broadcast(text);
  });
  child.stdin?.on('error', () => {
    /* the engine exited before reading the request — its own output says why */
  });
  child.on('error', (err) => {
    engine.onExit(null, null);
    finishProcess(job.id, null);
    console.error('[tts-engine] could not run:', err.message);
  });
  child.on('exit', (code, signal) => {
    engine.onExit(code, signal);
    if (engines.get(key) === engine) engines.delete(key);
    finishProcess(job.id, code ?? null, signal);
  });

  try {
    await engine.waitForReady(loadTimeoutMs());
  } catch (err) {
    try {
      child.kill('SIGTERM');
    } catch {
      /* already gone */
    }
    throw err;
  }
  return engine;
}

/** Every engine this app is holding, for the resource manager and the tests. */
export function ttsEngines(): Array<{
  key: string;
  repo?: string;
  device: string;
  ready: boolean;
  dead: boolean;
  pid: number | undefined;
  jobId: string;
}> {
  return [...engines.values()].map((e) => ({
    key: e.key,
    repo: e.spec.repo,
    device: e.device,
    ready: e.ready,
    dead: e.dead,
    pid: e.child.pid,
    jobId: e.jobId,
  }));
}

/** Releases every engine. Used when the app is shutting down. */
export async function stopTtsEngines(): Promise<number> {
  const all = [...engines.values()];
  engines.clear();
  for (const engine of all) await engine.release();
  return all.length;
}

// A child that outlives its parent is a model sitting in memory with nothing
// left to ask it a question. The server's own shutdown stops the registered
// jobs, but an exit that is not a shutdown — a crash, a `process.exit` — has to
// take them along.
process.on('exit', () => {
  for (const engine of engines.values()) {
    try {
      engine.child.kill('SIGKILL');
    } catch {
      /* already gone */
    }
  }
});
