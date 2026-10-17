/**
 * Registry of the processes this app spawned and is still running.
 *
 * The local runners in infer.ts are ordinary `spawn` children that used to be
 * created, awaited and thrown away — the handle never outlived the promise, so
 * there was no way to see one running or to stop it. That matters because these
 * are the long jobs: image generation is allowed to run for an hour, a CPU
 * transcription can hold a core for minutes, and neither is interruptible from
 * the UI. Stopping a runaway local job is the whole point of this file.
 *
 * Only children this process started are tracked. There is deliberately no
 * "kill by pid" entry point: the server has no authentication, so a route that
 * accepted an arbitrary pid from the browser would be an arbitrary-process
 * killer reachable by any page that can talk to the port. Jobs are named by an
 * id minted here, so a caller can only ever stop something this app started.
 */

import type { ChildProcess } from 'node:child_process';
import { readFile } from 'node:fs/promises';

export type ProcessKind = 'stt' | 'tts' | 'image';
export type ProcessState = 'running' | 'stopping' | 'exited';

export interface ManagedProcess {
  /** Server-minted. Never the pid, and never derived from request input. */
  id: string;
  pid: number;
  kind: ProcessKind;
  /** The runner script, e.g. `whisper_ct2.py`. */
  script: string;
  /**
   * Runtime group ids this job actually loads, taken from the script's imports.
   * See SCRIPT_RUNTIMES — PyTorch is a library inside a process, not a process
   * of its own, so it is reported as something a job *uses*.
   */
  runtimes: string[];
  /** Catalog repo the weights came from, when there is one. */
  repo?: string;
  /** Which compute device the job was told to use. */
  device: string;
  /**
   * What the runner resolved its device request to, reported by the script
   * itself (`cpu`, `cuda` or `rocm`) the moment it knows. The request above is
   * only what was *asked for* — "auto" on a machine whose torch is CPU-only is
   * not a GPU job — so the manager labels jobs with this when it is there.
   */
  resolvedDevice?: string;
  /**
   * Resident memory in bytes, read from /proc when the list is asked for.
   * A reading of the moment, not a fact about the job, so it is filled in by
   * the caller and never stored on the entry.
   */
  memoryBytes?: number | null;
  startedAt: number;
  endedAt?: number;
  state: ProcessState;
  /** Why it ended, for a job the user stopped rather than one that finished. */
  stoppedByUser?: boolean;
  exitCode?: number | null;
}

/**
 * What each runner actually imports.
 *
 * Read off the scripts rather than guessed: whisper_ct2.py uses
 * faster_whisper + ctranslate2 and deliberately never touches torch, which is
 * the whole reason it exists (40MB of runtime instead of 529MB). The other
 * three all build on torch. onnxruntime is absent on purpose — no bundled
 * runner uses it, and listing it would offer a "stop" for a process that can
 * never appear.
 */
const SCRIPT_RUNTIMES: Record<string, string[]> = {
  'whisper_ct2.py': ['faster-whisper'],
  'transcribe.py': ['torch', 'transformers'],
  'speak.py': ['torch', 'kokoro'],
  'generate.py': ['torch', 'diffusers'],
};

let seq = 0;
const live = new Map<string, { entry: ManagedProcess; child: ChildProcess }>();

/** Finished jobs stay listed briefly so the UI can show the transition. */
const recent: ManagedProcess[] = [];
const RECENT_TTL_MS = 60_000;
const RECENT_MAX = 20;

/** How long a politely-stopped process gets before it is killed outright. */
const SIGKILL_GRACE_MS = 4000;

export function registerProcess(opts: {
  kind: ProcessKind;
  script: string;
  pid: number;
  repo?: string;
  device: string;
}): ManagedProcess {
  seq += 1;
  const entry: ManagedProcess = {
    id: `job-${seq}`,
    pid: opts.pid,
    kind: opts.kind,
    script: opts.script,
    runtimes: SCRIPT_RUNTIMES[opts.script] ?? [],
    repo: opts.repo,
    device: opts.device,
    startedAt: Date.now(),
    state: 'running',
  };
  live.set(entry.id, { entry, child: undefined as unknown as ChildProcess });
  return entry;
}

/**
 * Binds the child handle so the job can actually be signalled.
 *
 * Registration happens at spawn time, when the pid is known but the handle is
 * not returned until the next tick; this closes that gap. Returns the id so the
 * caller can pass one object to both calls.
 */
export function attachChild(id: string, child: ChildProcess): void {
  const slot = live.get(id);
  if (slot) slot.child = child;
}

/**
 * Records what the runner resolved its device request to.
 *
 * The scripts print the resolved device on stdout as soon as `pick_device`
 * returns — before the model is even loaded — and infer.ts reads those lines
 * out of the running job's output. Values are the scripts' own: `cpu`, `cuda`
 * or `rocm`. An id that is already gone is ignored, because a job that died
 * between printing and this call has nothing left to relabel.
 */
export function setProcessDevice(id: string, device: string): void {
  const slot = live.get(id);
  if (slot && slot.entry.resolvedDevice !== device) slot.entry.resolvedDevice = device;
}

/**
 * Resident set size of a pid, in bytes, or null when it cannot be read.
 *
 * This is VmRSS out of /proc/<pid>/status, the figure `top` shows as RES: how
 * much memory the process actually holds right now, which for these jobs is
 * mostly model weights. Every failure — pid gone, /proc unreadable, a platform
 * without it — answers null, and the manager leaves the figure out rather than
 * showing a guess.
 */
export async function processMemoryBytes(pid: number): Promise<number | null> {
  if (process.platform !== 'linux' || !Number.isFinite(pid) || pid <= 0) return null;
  try {
    const status = await readFile(`/proc/${pid}/status`, 'utf8');
    const match = status.match(/^VmRSS:\s*(\d+)\s*kB/m);
    return match ? Number(match[1]) * 1024 : null;
  } catch {
    return null;
  }
}

/**
 * Moves a job out of the live set and into the recent list.
 *
 * Returns the finished entry, because the caller needs `stoppedByUser` *after*
 * the fact — stopProcess marks it while the job is still live, so by the time
 * the child closes there is nothing left to look it up on. Reading it from the
 * returned copy is the only way to tell "the user stopped this" from "this
 * finished on its own".
 */
export function finishProcess(
  id: string,
  code: number | null,
  signal?: NodeJS.Signals | null,
): ManagedProcess | null {
  const slot = live.get(id);
  if (!slot) return null;
  const { entry } = slot;
  entry.state = 'exited';
  entry.endedAt = Date.now();
  entry.exitCode = code;
  if (signal) entry.stoppedByUser = true;
  live.delete(id);
  recent.unshift({ ...entry });
  while (recent.length > RECENT_MAX) recent.pop();
  return { ...entry };
}

/**
 * Stops one job: SIGTERM, then SIGKILL if it is still there after the grace
 * period. Python is given the chance to clean up its own temporary output —
 * a TTS job killed mid-write leaves a half-written wav behind.
 */
export async function stopProcess(id: string): Promise<ManagedProcess | null> {
  const slot = live.get(id);
  if (!slot) return null;
  const { entry, child } = slot;

  if (!child || child.pid === undefined) {
    // Spawned but not yet bound, or already gone. Either way there is nothing
    // left to signal; drop it so it stops appearing as running.
    finishProcess(id, null);
    return entry;
  }

  if (entry.state !== 'stopping') {
    entry.state = 'stopping';
    entry.stoppedByUser = true;
    try {
      child.kill('SIGTERM');
    } catch {
      // Already reaped. The close handler will clean up the registry entry.
    }
    const timer = setTimeout(() => {
      const still = live.get(id);
      if (!still) return;
      try {
        still.child.kill('SIGKILL');
      } catch {
        /* gone between the check and the signal */
      }
    }, SIGKILL_GRACE_MS);
    timer.unref?.();
  }
  return entry;
}

export async function stopAllProcesses(): Promise<number> {
  let n = 0;
  for (const id of [...live.keys()]) {
    if (await stopProcess(id)) n += 1;
  }
  return n;
}

/** Everything the manager needs: what is running now, and what just stopped. */
export function listProcesses(): { active: ManagedProcess[]; recent: ManagedProcess[] } {
  const cutoff = Date.now() - RECENT_TTL_MS;
  while (recent.length && (recent[recent.length - 1].endedAt ?? 0) < cutoff) recent.pop();
  return {
    // Slower jobs first: the thing most likely to need stopping is the one the
    // user opened the manager to find.
    active: [...live.values()]
      .map((s) => ({ ...s.entry }))
      .sort((a, b) => b.startedAt - a.startedAt),
    recent: recent.filter((p) => p.state === 'exited').map((p) => ({ ...p })),
  };
}
