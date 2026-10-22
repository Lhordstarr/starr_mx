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

export type TaskKey = 'vision' | 'stt' | 'tts' | 'image';

export interface Assignments {
  /**
   * The model that turns images and video into text for the active chat model.
   *
   * Kept separate from the chat model on purpose: a local text model may not
   * see, and a hosted chat model may charge per image. Assigning the
   * multimodal work here means the same description step works whichever chat
   * backend is selected, and the transcript never carries raw image bytes to a
   * model that cannot use them.
   */
  vision: string | null;
  stt: string | null;
  tts: string | null;
  image: string | null;
}

const EMPTY: Assignments = { vision: null, stt: null, tts: null, image: null };

let cache: Assignments = { ...EMPTY };

/** Values are model ids like `local:Systran/faster-whisper-tiny`. */
function isId(v: unknown): v is string {
  return typeof v === 'string' && v.length > 0 && v.length < 200;
}

/**
 * Vision ids were briefly written as `local:ollama/<model>` before the runner
 * was wired to the daemon by tag. Normalising here means a prefs file from that
 * build keeps working instead of silently disabling image handling.
 */
function normalizeVision(v: string | null): string | null {
  const legacy = 'local:ollama/';
  if (!v?.startsWith(legacy)) return v;
  const tag = v.slice(legacy.length);
  return `ollama:${tag.includes(':') ? tag : `${tag}:latest`}`;
}

export async function loadAssignments(): Promise<void> {
  try {
    const parsed = JSON.parse(await readFile(prefsFile, 'utf8')) as Record<string, unknown>;
    cache = {
      vision: isId(parsed.vision) ? normalizeVision(parsed.vision) : null,
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
 * Only the known tasks are accepted, and only plain strings are stored, so
 * this file cannot be turned into a place to keep arbitrary data.
 */
export async function setAssignment(task: TaskKey, modelId: string | null): Promise<Assignments> {
  if (!(task in cache)) throw new Error(`Unknown task: ${task}`);
  const raw = modelId && modelId.trim() ? modelId.trim() : null;
  if (raw !== null && !isId(raw)) throw new Error('Invalid model id');
  const value = task === 'vision' ? normalizeVision(raw) : raw;
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