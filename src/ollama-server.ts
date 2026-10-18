/**
 * The Ollama daemon, started on demand and stopped when nobody is using it.
 *
 * Ollama ships as a daemon, not a per-request binary, so unlike llama.cpp there
 * is nothing to launch at request time — `ollama serve` is the thing that answers
 * anything, and on a machine where nothing has started it yet every provider
 * call fails with a bare "fetch failed" at `localhost:11434`. Telling the user
 * to open a terminal is the wrong answer for an app whose whole point is that it
 * starts what it needs.
 *
 * The lifecycle deliberately matches the managed llama-server: one process at a
 * time, visible in the same process manager, stopped from the same Stop button,
 * idle-shut down after `OLLAMA_IDLE_MS`, and killed when the app exits. Sharing
 * the contract is what makes "two chat backends" feel like one feature rather
 * than two half-built ones.
 */
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import {
  attachChild,
  finishProcess,
  registerProcess,
  registerStopHandler,
  unregisterStopHandler,
} from './processes.js';

const baseUrl = process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434';
/** Idle shutdown, same ten minutes llama.cpp uses unless told otherwise. */
const IDLE_MS = Number(process.env.OLLAMA_IDLE_MS ?? 600_000);

let serving: {
  child: ChildProcess;
  jobId: string;
  /** Set when the user stopped it, so the exit is not reported as a failure. */
  userStopped: boolean;
} | null = null;
let starting: Promise<void> | null = null;
let idleTimer: NodeJS.Timeout | null = null;
let lastUsed = 0;

/** True when something is already answering on the Ollama port. */
export async function ollamaReachable(timeoutMs = 2500): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(timeoutMs) });
    return res.ok;
  } catch {
    return false;
  }
}

function clearIdle(): void {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = null;
}

/**
 * Restarts the idle countdown.
 *
 * Deliberately not `unref()`ed. That was tried and is why an unref'd timer let
 * the process it was guarding exit early: nothing else held the loop open, and
 * the daemon died while a request was still in flight.
 */
function touchIdle(): void {
  lastUsed = Date.now();
  clearIdle();
  idleTimer = setTimeout(() => {
    void stopOllama('idle');
  }, IDLE_MS);
}

/** Where the `ollama` binary is, or null when it is not installed. */
export function ollamaBinary(): string | null {
  const fromPath = spawnSync('sh', ['-c', 'command -v ollama'], { encoding: 'utf8' });
  const found = String(fromPath.stdout ?? '').trim().split('\n')[0];
  if (fromPath.status === 0 && found) return found;
  for (const dir of ['/usr/local/bin/ollama', '/opt/homebrew/bin/ollama', '/usr/bin/ollama']) {
    try {
      if (spawnSync('test', ['-x', dir]).status === 0) return dir;
    } catch {
      /* keep looking */
    }
  }
  return null;
}

/** Starts the daemon if it is not already answering. Safe to call concurrently. */
export async function ensureOllamaRunning(
  onProgress: (line: string) => void = () => {},
): Promise<void> {
  touchIdle();
  if (serving && !serving.child.killed) return;
  // Something outside this app is already serving — a system service, or a
  // terminal the user started it in. Leave it alone; it is not ours to stop.
  if (await ollamaReachable()) return;
  if (starting) return starting;

  starting = (async () => {
    const bin = ollamaBinary();
    if (!bin) {
      throw new Error(
        'Ollama is installed but its `ollama` command is not on this machine, so the ' +
          'daemon cannot be started for you. Start it in a terminal with `ollama serve`.',
      );
    }
    onProgress('Starting the Ollama daemon…');
    const child = spawn(bin, ['serve'], { stdio: ['ignore', 'pipe', 'pipe'] });
    const job = registerProcess({
      kind: 'llm',
      script: 'ollama serve',
      pid: child.pid ?? -1,
      // Ollama picks its own device; there is nothing here that reports which.
      device: 'cpu',
    });
    serving = { child, jobId: job.id, userStopped: false };
    attachChild(job.id, child);
    registerStopHandler(job.id, () => stopOllama('user'));

    const tail: string[] = [];
    const note = (b: Buffer) => {
      const line = String(b).trim();
      if (line) tail.push(line);
    };
    child.stdout?.on('data', note);
    child.stderr?.on('data', note);

    child.on('error', () => {
      unregisterStopHandler(job.id);
      serving = null;
    });
    child.on('exit', (code, signal) => {
      const mine = serving?.child === child;
      const userStopped = serving?.userStopped ?? false;
      if (mine) serving = null;
      clearIdle();
      unregisterStopHandler(job.id);
      // A daemon that exited on its own is a failure worth recording; one the
      // user stopped from the process manager is the outcome they asked for.
      if (mine && !userStopped) {
        finishProcess(job.id, code ?? null, signal, false);
        console.error(
          `[ollama] the daemon exited${signal ? ` on ${signal}` : ` (${code})`}: ${tail.slice(-3).join(' | ')}`,
        );
      }
    });

    // /health answers only once the server is actually serving, which is a
    // different moment from the process having spawned.
    const deadline = Date.now() + 60_000;
    for (;;) {
      if (await ollamaReachable(2000)) break;
      if (Date.now() > deadline) {
        throw new Error(
          'The Ollama daemon did not start within a minute. Its own output was: ' +
            (tail.slice(-3).join(' | ') || 'nothing'),
        );
      }
      await new Promise((r) => setTimeout(r, 700));
    }
    onProgress('Ollama is up.');
  })();

  try {
    await starting;
  } finally {
    starting = null;
  }
}

/** Stops the daemon. `reason` is for the log; the user stop is the only one that
 *  suppresses the exit being reported as a failure. */
export async function stopOllama(reason: 'idle' | 'user' | 'shutdown' = 'user'): Promise<void> {
  const mine = serving;
  clearIdle();
  if (!mine) return;
  serving = null;
  mine.userStopped = true;
  if (reason === 'idle') console.log('[ollama] idle — stopping the daemon');
  try {
    mine.child.kill('SIGTERM');
  } catch {
    /* already gone */
  }
  // SIGTERM is a request. Give it a moment to honour it before escalating, so a
  // model still resident gets a clean unload rather than a kill.
  await new Promise((r) => setTimeout(r, 1500));
  try {
    if (mine.child.exitCode === null && !mine.child.killed) mine.child.kill('SIGKILL');
  } catch {
    /* already gone */
  }
}

/** Called when a request has just used the daemon. */
export function noteOllamaUsed(): void {
  lastUsed = Date.now();
  touchIdle();
}

let exitHooked = false;
/**
 * Stops the daemon when the app exits.
 *
 * Registered once, lazily, so a run that never touches Ollama never installs a
 * handler. Without it a "close the app" leaves `ollama serve` resident, holding
 * whatever model was last loaded — on a Colab-sized machine that is the
 * difference between the GPU being free and not.
 */
export function hookOllamaExit(): void {
  if (exitHooked) return;
  exitHooked = true;
  const stop = () => {
    void stopOllama('shutdown');
  };
  process.once('exit', stop);
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
}