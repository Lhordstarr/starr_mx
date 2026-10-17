import { chmod, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * Which model is assigned to which job.
 *
 * The composer used to carry a speech-to-text picker, so the model for a task
 * was chosen in the one place you type rather than the one place you manage
 * models. Two things were wrong with that: it put task configuration in the
 * chat surface, and it only covered dictation — speech and images were not
 * assignable at all. Assignments belong to the machine, not to a browser tab,
 * so they live here and every surface reads the same answer.
 */
const prefsFile = path.resolve(process.env.PREFS_FILE ?? '.prefs.json');

export type TaskKey = 'stt' | 'tts' | 'image';

export interface Assignments {
  stt: string | null;
  tts: string | null;
  image: string | null;
}

const EMPTY: Assignments = { stt: null, tts: null, image: null };

let cache: Assignments = { ...EMPTY };

/** Values are model ids like `local:Systran/faster-whisper-tiny`. */
function isId(v: unknown): v is string {
  return typeof v === 'string' && v.length > 0 && v.length < 200;
}

export async function loadAssignments(): Promise<void> {
  try {
    const parsed = JSON.parse(await readFile(prefsFile, 'utf8')) as Record<string, unknown>;
    cache = {
      stt: isId(parsed.stt) ? parsed.stt : null,
      tts: isId(parsed.tts) ? parsed.tts : null,
      image: isId(parsed.image) ? parsed.image : null,
    };
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
      console.warn(`Could not read ${prefsFile}: ${(err as Error).message}`);
    }
    cache = { ...EMPTY };
  }
}

export function assignments(): Assignments {
  return { ...cache };
}

/**
 * Assigns a model to a task. An empty value clears it.
 *
 * Only the three known tasks are accepted, and only plain strings are stored,
 * so this file cannot be turned into a place to keep arbitrary data.
 */
export async function setAssignment(task: TaskKey, modelId: string | null): Promise<Assignments> {
  if (!(task in cache)) throw new Error(`Unknown task: ${task}`);
  const value = modelId && modelId.trim() ? modelId.trim() : null;
  if (value !== null && !isId(value)) throw new Error('Invalid model id');
  cache = { ...cache, [task]: value };
  await persist();
  return assignments();
}

async function persist(): Promise<void> {
  await writeFile(prefsFile, `${JSON.stringify(cache, null, 2)}\n`, 'utf8');
  // Holds model ids only, but keep it user-private out of habit.
  await chmod(prefsFile, 0o600).catch(() => {});
}

export function prefsFilePath(): string {
  return prefsFile;
}