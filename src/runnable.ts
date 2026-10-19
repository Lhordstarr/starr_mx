/**
 * What this machine can actually run right now.
 *
 * Three separate questions used to be answered in three places, and each one
 * guessed: the hub grouped downloads by a hardcoded runtime map, the catalog
 * showed a Run button keyed off catalog metadata alone, and dictation only ever
 * looked at cloud providers. The result was that a downloaded model could be
 * on disk, described one way on the card, grouped another way on the hub, and
 * still be impossible to actually invoke.
 *
 * This module is the single answer: given the catalog, what is on disk and what
 * is installed, it returns the models that can *genuinely* run, and for each
 * one the exact reason when it cannot. Every surface reads from here, so they
 * cannot disagree.
 *
 * The catalog is the curated path, not the only one. The Hub browser can fetch
 * any repository, and a download whose layout a bundled runner loads (a
 * transformers Whisper, a diffusers tree) is as runnable as a catalog entry —
 * it used to be on disk, runtime installed, and invisible to every picker
 * purely because nobody had hand-written it down. Those downloads are
 * discovered by layout below, with the same ready/reason discipline.
 */
import './net.js';
import { readdir, readFile } from 'node:fs/promises';
import * as path from 'node:path';
import { CATALOG, catalogRepoIds, type CatalogEntry, type CatalogTask } from './model-catalog.js';
import { hubCacheDir, listDownloads, modelsDir, type HFDownload } from './huggingface.js';
import { runtimeStatus } from './runtimes.js';

export type RunTask = 'stt' | 'tts' | 'image';

export interface RunnableModel {
  /** Catalog identity, exactly as the downloader and runners use it. */
  repo: string;
  label: string;
  task: RunTask;
  /** Which of the entry's files identifies the model to the runner. */
  file: string;
  /** Every file this model needs; a partial download is not runnable. */
  requiredFiles: string[];
  /** Which of those are already on disk. */
  presentFiles: string[];
  /** Installed runtime group -> version. */
  runtimes: Record<string, string>;
  missing: string[];
  /**
   * Which bundled runner would handle this, so the browser can ask for audio in
   * a form that runner accepts.
   *
   * Not something the client can work out: whisper-cli reads wav/flac/mp3/ogg
   * and nothing else, so a browser recording has to be decoded to 16 kHz WAV
   * before it is uploaded — but only for that runner. The two Python ones take
   * the original bytes and their own audio libraries.
   */
  driver?: 'whisper-cpp' | 'faster-whisper' | 'transformers';
  /** False when a required runtime is blocked on this interpreter. */
  blocked: boolean;
  /** Why it cannot run, or '' when it can. */
  reason: string;
  ready: boolean;
}

/**
 * The catalog task label for a runner task. The two vocabularies differ
 * ('speech-to-text' vs 'stt'), and conflating them is how entries end up
 * filed under the wrong section.
 */
const CATALOG_TASK: Record<RunTask, CatalogTask> = {
  stt: 'speech-to-text',
  tts: 'text-to-speech',
  image: 'image-generation',
};

/** One catalog entry as a candidate model. */
function candidatesFor(task: RunTask): CatalogEntry[] {
  return CATALOG.flatMap((s) => s.entries).filter(
    (e) => e.task === CATALOG_TASK[task] && e.run?.kind === task,
  );
}

/**
 * A model needs *every* file of a bundle, not just its weights: a CTranslate2
 * Whisper without `config.json` next to `model.bin` is not a model.
 */
function requiredFiles(entry: CatalogEntry): string[] {
  return (entry.files ?? []).map((f) => f.name);
}

/** The file the runner is pointed at — the weights, not a sidecar. */
function entryAnchor(entry: CatalogEntry): string {
  const files = entry.files ?? [];
  // The largest file is the weights in every format here (model.bin, *.safetensors,
  // *.onnx, *.ckpt), so it identifies the model reliably.
  const biggest = files.slice().sort((a, b) => (b.bytes ?? 0) - (a.bytes ?? 0))[0];
  return biggest?.name ?? files[0]?.name ?? '';
}

/**
 * A downloaded repository the catalog does not know, whose layout is still
 * something a bundled runner can load.
 *
 * Detection is by layout, never by name: a repo called "whisper-thing" proves
 * nothing, while `config.json` whose `model_type` is `whisper` is exactly what
 * the transformers pipeline will accept. Shapes the runners do not handle stay
 * unclaimed — being listed is a promise that running works.
 */
interface DetectedLayout {
  /** Short format tag shown beside the model name. */
  format: string;
  /** Files the runner needs, most important first — the first is the anchor. */
  needs: string[];
  /** Installed runtime groups this layout needs before it can run. */
  requires: string[];
}

/** `model_type` from a repo's config.json, or '' when absent or unreadable. */
async function modelTypeOf(dir: string): Promise<string> {
  try {
    const parsed = JSON.parse(await readFile(path.join(dir, 'config.json'), 'utf8')) as {
      model_type?: unknown;
    };
    return typeof parsed.model_type === 'string' ? parsed.model_type.toLowerCase() : '';
  } catch {
    return '';
  }
}

/**
 * What a downloaded repo is, when its files alone can say so.
 *
 * Every branch mirrors the runner that would execute the model, so the claim
 * and the execution cannot drift apart: CT2 pairs with whisper_ct2.py's marker
 * (`model.bin` + `config.json`), a `model_type` of whisper with the
 * transformers pipeline in transcribe.py, `model_index.json` with the
 * diffusers loader in generate.py.
 */
async function detectLayout(
  task: RunTask,
  dir: string,
  files: Set<string>,
): Promise<DetectedLayout | null> {
  if (task === 'stt') {
    // whisper.cpp's own models are single GGML files with a fixed naming
    // convention, and the loader rejects anything else — so the name is the
    // format here, exactly as the CTranslate2 pairing is.
    //
    // A GGUF Whisper is explicitly *not* claimed. Those exist and are good
    // models, but they are built for transcribe.cpp and whisper.cpp refuses
    // them with "bad magic", so offering one would be a promise the runner
    // cannot keep. It was the reason two catalog entries sat here labelled
    // "whisper.cpp" and pointed at files nothing here could load.
    for (const f of files) {
      const name = f.split('/').pop() ?? f;
      if (!/^ggml-[a-z0-9.]+(-[a-z0-9_]+)*\.bin$/i.test(name)) continue;
      return {
        format: 'whisper.cpp',
        needs: [f],
        requires: ['whisper-cpp'],
      };
    }
    // CTranslate2 keeps no `model_type` for Whisper conversions — the pairing
    // of model.bin with config.json is the format, and is the same signal
    // transcribe() uses to choose the CTranslate2 runner. A conversion that
    // kept a `model_type` naming another family is not a Whisper, and
    // whisper_ct2.py would only fail on it, so it is not claimed.
    if (files.has('model.bin') && files.has('config.json')) {
      const declared = await modelTypeOf(dir);
      if (!declared || declared === 'whisper') {
        return {
          format: 'CTranslate2',
          needs: ['model.bin', 'config.json'],
          requires: ['faster-whisper'],
        };
      }
    }
    // Transformers repos declare their architecture. Only Whisper is claimed:
    // it is the one the bundled runner is exercised against, and guessing past
    // the config would offer directories that then fail to load.
    if (files.has('config.json') && (await modelTypeOf(dir)) === 'whisper') {
      const weight = files.has('model.safetensors')
        ? 'model.safetensors'
        : files.has('pytorch_model.bin')
          ? 'pytorch_model.bin'
          : null;
      if (weight) {
        return {
          format: 'transformers',
          needs: [weight, 'config.json'],
          requires: ['torch', 'transformers'],
        };
      }
    }
    return null;
  }
  if (task === 'image') {
    /* A diffusers pipeline is a tree rooted at model_index.json, but the root
     * alone is not a model. The index names the components it wires together —
     * text_encoder, tokenizer, unet, vae, scheduler — each as a folder in the
     * same tree. Claiming a bundle on the index alone is what let a
     * half-finished download report itself as ready and then fail inside the
     * loader, where the message names a tensor rather than the file that was
     * never downloaded. So the components are required too, and a partial tree
     * says which ones are absent instead of "ready". */
    if (!files.has('model_index.json')) return null;
    return {
      format: 'diffusers',
      needs: ['model_index.json', ...(await diffusersComponents(dir))],
      requires: ['torch', 'diffusers'],
    };
  }
  // Text-to-speech has exactly one runner and it is Kokoro-specific; other TTS
  // architectures have nothing here that loads them, so nothing is claimed.
  return null;
}

/**
 * The component folders a diffusers `model_index.json` wires together.
 *
 * Each entry is a `[library, class, path]` triple with the folder last, so the
 * folder is the final element. Entries that name no folder at all — the
 * `"safety_checker": [null, null]` placeholders — are skipped, and so is
 * anything that is not that shape: a tree this cannot read leaves the index as
 * the only requirement, which is where this started and no worse.
 */
async function diffusersComponents(dir: string): Promise<string[]> {
  let index: unknown;
  try {
    index = JSON.parse(await readFile(path.join(dir, 'model_index.json'), 'utf8'));
  } catch {
    return [];
  }
  if (!index || typeof index !== 'object') return [];
  const out: string[] = [];
  for (const [key, value] of Object.entries(index as Record<string, unknown>)) {
    if (key.startsWith('_')) continue;
    /* Three elements, not two. A component that lives in its own folder is a
       `[library, class, path]` triple; a two-element entry names the class with
       no path at all (`"scheduler": ["diffusers", "DDIMScheduler"]`) and sits at
       the tree root. Taking the last element of one of those reads the class
       name as a folder and demands a directory called "DDIMScheduler". */
    if (!Array.isArray(value) || value.length < 3) continue;
    if (value.some((v) => v === null)) continue;
    const last = value[value.length - 1];
    if (typeof last !== 'string' || !last.trim()) continue;
    out.push(last.replace(/^\.?\/+/, '').replace(/\/+$/, ''));
  }
  return [...new Set(out)].filter(Boolean);
}

/**
 * What is actually in a directory, as repo-relative paths.
 *
 * The download records are not this. They say what *this app* fetched, so a repo
 * pulled by `huggingface-cli`, `git clone` or a colleague's tarball arrives as a
 * single record and every other file is counted missing — which is how a
 * complete diffusers tree reads as an empty one. Both are consulted: the records
 * for what the app did, this for what is there.
 */
async function treeOf(dir: string, depth = 3): Promise<string[]> {
  const out: string[] = [];
  const walk = async (prefix: string, level: number): Promise<void> => {
    if (level > depth) return;
    let entries;
    try {
      entries = await readdir(path.join(dir, prefix), { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      const rel = prefix ? `${prefix}/${e.name}` : e.name;
      if (e.isDirectory()) {
        out.push(`${rel}/`);
        await walk(rel, level + 1);
      } else if (e.isFile()) {
        out.push(rel);
      }
    }
  };
  await walk('', 0);
  return out;
}

/**
 * Is one `needs` entry satisfied?
 *
 * A component is a folder rather than a single file, so it counts as present
 * when anything lives under it. Without that, every diffusers bundle would
 * report all of its components missing — the check has to mean "the folder is
 * there", not "this exact path is a file".
 */
function needsSatisfied(files: Set<string>, entry: string): boolean {
  if (files.has(entry)) return true;
  for (const f of files) {
    if (f.startsWith(`${entry}/`)) return true;
  }
  return false;
}

/**
 * The runner implied by a set of required runtime groups.
 *
 * Read off `requires` rather than guessed from the layout, so a model can never
 * be handed to a runner whose runtime it does not list. An unrecognised set
 * returns undefined and the browser keeps the audio as-is.
 */
function driverFor(requires: string[]): RunnableModel['driver'] {
  if (requires.includes('whisper-cpp')) return 'whisper-cpp';
  if (requires.includes('faster-whisper')) return 'faster-whisper';
  if (requires.includes('transformers')) return 'transformers';
  return undefined;
}

/** Everything the live probe feeds in, overridable so tests need no venv. */
export interface RunnableDeps {
  /** Files already on disk, replacing the walk of the models directory. */
  downloads?: HFDownload[];
  /** Installed runtime versions, replacing the live probe. */
  installed?: Record<string, string | null>;
}

export async function runnableModels(task: RunTask, deps: RunnableDeps = {}): Promise<RunnableModel[]> {
  const [downloads, installed] = await Promise.all([
    deps.downloads ?? listDownloads(),
    // RuntimeStatus always carries the map; blank statuses carry an empty one.
    deps.installed ?? runtimeStatus(false).then((s) => s.installed),
  ]);
  const onDisk = new Set(downloads.map((d) => `${d.repo}::${d.file}`));

  const out: RunnableModel[] = [];
  for (const entry of candidatesFor(task)) {
    const needed = requiredFiles(entry);
    if (!needed.length) continue;
    const present = needed.filter((f) => onDisk.has(`${entry.repo}::${f}`));
    const missingFiles = needed.filter((f) => !onDisk.has(`${entry.repo}::${f}`));

    const requires = entry.run?.requires ?? [];
    const missingRuntimes = requires.filter((g) => !installed[g]);
    const runtimes: Record<string, string> = {};
    for (const g of requires) {
      const v = installed[g];
      if (v) runtimes[g] = v;
    }

    let reason = '';
    if (!present.length) {
      reason = 'Not downloaded yet.';
    } else if (missingFiles.length) {
      // Half a bundle is the most confusing state to be in: the model is
      // "on disk" but will not load, and the failure surfaces as a loader
      // error deep in a runtime.
      reason =
        missingFiles.length === 1
          ? `Missing ${missingFiles[0]} — download the whole set.`
          : `Missing ${missingFiles.length} of ${needed.length} files — download the whole set.`;
    } else if (missingRuntimes.length) {
      reason = `Needs ${missingRuntimes.join(' + ')}.`;
    }

    out.push({
      repo: entry.repo,
      label: entry.label,
      task,
      file: entryAnchor(entry),
      requiredFiles: needed,
      presentFiles: present,
      runtimes,
      missing: missingRuntimes,
      driver: driverFor(requires),
      blocked: false,
      reason,
      ready: !reason,
    });
  }

  // Downloads the catalog does not carry. Grouped by repo because a model can
  // be several files; the catalog's entries are skipped because they already
  // reported above, with their own bundle rules, and a second row for the same
  // repo would make the two disagree.
  const known = new Set(catalogRepoIds());
  const byRepo = new Map<string, { root: string; files: Set<string> }>();
  for (const d of downloads) {
    if (known.has(d.repo)) continue;
    const group = byRepo.get(d.repo) ?? { root: d.root, files: new Set<string>() };
    group.files.add(d.file);
    byRepo.set(d.repo, group);
  }
  // A layout scan may only read the two roots a download can live under: this
  // app's models dir and the shared Hugging Face cache. Both were walked by
  // listDownloads; a crafted download record still cannot steer a read.
  const scanRoots = [modelsDir(), hubCacheDir()].map((r) => path.resolve(r));
  for (const [repo, group] of byRepo) {
    if (!scanRoots.some((root) => group.root.startsWith(root + path.sep))) continue;
    const layout = await detectLayout(task, group.root, group.files);
    if (!layout) continue;

    /* The records say what this app fetched; the tree says what is there. A repo
       that arrived by any other route has one record and a full directory, and
       believing the records alone marks every real file missing. */
    const onDisk = new Set(group.files);
    for (const rel of await treeOf(group.root)) {
      onDisk.add(rel);
    }

    const present = layout.needs.filter((f) => needsSatisfied(onDisk, f));
    const missingFiles = layout.needs.filter((f) => !needsSatisfied(onDisk, f));
    const missingRuntimes = layout.requires.filter((g) => !installed[g]);
    const runtimes: Record<string, string> = {};
    for (const g of layout.requires) {
      const v = installed[g];
      if (v) runtimes[g] = v;
    }

    let reason = '';
    if (missingFiles.length) {
      reason =
        missingFiles.length === 1
          ? `Missing ${missingFiles[0]} — download the whole set.`
          : `Missing ${missingFiles.length} of ${layout.needs.length} files — download the whole set.`;
    } else if (missingRuntimes.length) {
      reason = `Needs ${missingRuntimes.join(' + ')}.`;
    }

    out.push({
      repo,
      // The repository id is kept whole: two "whisper-small" repositories are
      // different models, and the picker only ever shows this string.
      label: `${repo} (${layout.format})`,
      task,
      file: layout.needs[0] ?? '',
      requiredFiles: layout.needs,
      presentFiles: present,
      runtimes,
      missing: missingRuntimes,
      driver: driverFor(layout.requires),
      blocked: false,
      reason,
      ready: !reason,
    });
  }
  return out;
}