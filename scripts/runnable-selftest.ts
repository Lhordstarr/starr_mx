/**
 * Unit checks for runnable-model discovery.
 *
 * The catalog's own entries need a real machine (files on disk, runtimes
 * probed), but the layout detection that makes *downloaded, uncatalogued*
 * repos selectable is pure: give it a fixture tree and an installed-runtime
 * map. That is what this exercises — the shapes that must be found, the
 * lookalikes that must not, and the catalog/repo dedup.
 *
 * Run with: npm run test:runnable
 */
import '../src/net.js';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { runnableModels, type RunnableDeps } from '../src/runnable.js';
import type { HFDownload } from '../src/huggingface.js';

let passed = 0;
const failures: string[] = [];

async function check(name: string, fn: () => void | Promise<void>): Promise<void> {
  try {
    await fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures.push(name);
    console.log(`  FAIL  ${name}\n        ${(err as Error).message}`);
  }
}

/** Creates models/<repo-with-dashes>/<files…> under the fixture root. */
function seed(root: string, repo: string, files: Record<string, string>): void {
  const dir = join(root, repo.replace(/\//g, '--'));
  for (const [name, body] of Object.entries(files)) {
    const full = join(dir, name);
    mkdirSync(dirname(full), { recursive: true });
    writeFileSync(full, body);
  }
}

/** The downloads listing a walk of the fixture root would produce. */
function downloadsOf(root: string, repos: string[]): HFDownload[] {
  const out: HFDownload[] = [];
  for (const repo of repos) {
    const dir = join(root, repo.replace(/\//g, '--'));
    const walk = (rel: string) => {
      for (const entry of readdirSync(join(dir, rel), { withFileTypes: true })) {
        const sub = rel ? `${rel}/${entry.name}` : entry.name;
        if (entry.isDirectory()) walk(sub);
        else
          out.push({
            repo,
            file: sub,
            size: 10,
            modifiedAt: '2026-01-01T00:00:00Z',
            location: 'models',
            root: dir,
          });
      }
    };
    walk('');
  }
  return out;
}

const root = mkdtempSync(join(tmpdir(), 'runnable-test-'));
// The layout scan only reads under the two roots downloads can live in, and
// both are env-driven, so the fixture root has to be the models dir.
process.env.HF_MODELS_DIR = root;

// --- fixtures -----------------------------------------------------------------
// A CTranslate2 Whisper: model.bin + config.json, no model_type anywhere.
seed(root, 'Acme/ct2-whisper', {
  'model.bin': 'x',
  'config.json': JSON.stringify({ alignment_heads: [[2, 2]] }),
  'tokenizer.json': 'x',
  'vocabulary.txt': 'x',
});
// A CTranslate2 conversion of another family: same file pair, not a Whisper.
seed(root, 'Acme/ct2-not-whisper', {
  'model.bin': 'x',
  'config.json': JSON.stringify({ model_type: 'm2m_100' }),
});
// A transformers Whisper: architecture declared in config.json.
seed(root, 'Acme/tf-whisper', {
  'model.safetensors': 'x',
  'config.json': JSON.stringify({ model_type: 'whisper' }),
  'tokenizer.json': 'x',
});
// A transformers LLM — same file names, wrong architecture.
seed(root, 'Acme/tf-llm', {
  'model.safetensors': 'x',
  'config.json': JSON.stringify({ model_type: 'qwen2' }),
});
// Weights with no config at all: nothing can be verified, nothing is claimed.
seed(root, 'Acme/bare-weights', { 'model.safetensors': 'x' });
// A diffusers pipeline tree.
seed(root, 'Acme/diff-pipe', {
  'model_index.json': '{}',
  'unet/diffusion_pytorch_model.fp16.safetensors': 'x',
  'vae/config.json': '{}',
});
// A repo that matches no layout (only a config with an unrelated type).
seed(root, 'Acme/clip-thing', { 'config.json': JSON.stringify({ model_type: 'clip' }) });
// A catalog repo on disk: must be reported once, by the catalog, not twice.
seed(root, 'Systran/faster-whisper-tiny', {
  'model.bin': 'x',
  'config.json': '{}',
  'tokenizer.json': 'x',
  'vocabulary.txt': 'x',
});

const repos = [
  'Acme/ct2-whisper',
  'Acme/ct2-not-whisper',
  'Acme/tf-whisper',
  'Acme/tf-llm',
  'Acme/bare-weights',
  'Acme/diff-pipe',
  'Acme/clip-thing',
  'Systran/faster-whisper-tiny',
];
const downloads = downloadsOf(root, repos);

const ALL: Record<string, string | null> = {
  'faster-whisper': 'installed',
  torch: '2.14.1+cpu',
  transformers: 'installed',
  diffusers: 'installed',
  kokoro: 'installed',
  onnxruntime: 'installed',
  ollama: 'installed',
};

const deps = (
  installed: Record<string, string | null>,
  extraRepos: string[] = [],
): RunnableDeps => ({
  downloads: [...downloads, ...downloadsOf(root, extraRepos)],
  installed,
});

const byRepo = (list: { repo: string }[]) => new Map(list.map((m) => [m.repo, m]));

// --- discovery ----------------------------------------------------------------
console.log('\n=== downloaded models, by layout ===');

await check('a CT2 pair is discovered for stt and runnable with faster-whisper', async () => {
  const stt = byRepo(await runnableModels('stt', deps(ALL)));
  const m = stt.get('Acme/ct2-whisper');
  assert.ok(m, 'CT2 repo listed');
  assert.equal(m.ready, true);
  assert.equal(m.reason, '');
  assert.equal(m.file, 'model.bin');
  assert.deepEqual(m.requiredFiles, ['model.bin', 'config.json']);
  assert.deepEqual(m.runtimes, { 'faster-whisper': 'installed' });
  assert.equal(m.label, 'Acme/ct2-whisper (CTranslate2)');
});

await check('a transformers Whisper is discovered and anchored on its weights', async () => {
  const stt = byRepo(await runnableModels('stt', deps(ALL)));
  const m = stt.get('Acme/tf-whisper');
  assert.ok(m, 'whisper repo listed');
  assert.equal(m.ready, true);
  assert.equal(m.file, 'model.safetensors');
  assert.deepEqual(m.runtimes, { torch: '2.14.1+cpu', transformers: 'installed' });
});

await check('a same-shaped LLM (model_type qwen2) is not claimed as speech-to-text', async () => {
  const stt = byRepo(await runnableModels('stt', deps(ALL)));
  assert.equal(stt.has('Acme/tf-llm'), false);
});

await check('a CT2 pair declaring another family is not claimed', async () => {
  const stt = byRepo(await runnableModels('stt', deps(ALL)));
  assert.equal(stt.has('Acme/ct2-not-whisper'), false);
});

await check('weights with no config are not claimed', async () => {
  const stt = byRepo(await runnableModels('stt', deps(ALL)));
  assert.equal(stt.has('Acme/bare-weights'), false);
});

await check('a diffusers tree is discovered for image generation', async () => {
  const image = byRepo(await runnableModels('image', deps(ALL)));
  const m = image.get('Acme/diff-pipe');
  assert.ok(m, 'diffusers repo listed');
  assert.equal(m.ready, true);
  assert.equal(m.file, 'model_index.json');
  assert.deepEqual(m.runtimes, { torch: '2.14.1+cpu', diffusers: 'installed' });
  assert.equal(m.label, 'Acme/diff-pipe (diffusers)');
});

await check('a diffusers tree never leaks into stt or tts', async () => {
  const stt = byRepo(await runnableModels('stt', deps(ALL)));
  const tts = byRepo(await runnableModels('tts', deps(ALL)));
  assert.equal(stt.has('Acme/diff-pipe'), false);
  assert.equal(tts.has('Acme/diff-pipe'), false);
});

/* A diffusers repo is a tree of component folders described by model_index.json.
   Claiming one on the index alone let a download that had fetched nothing but
   the index report itself as ready, and then fail inside the loader naming a
   tensor rather than the file that was never fetched. This is that case. */
const INDEX = JSON.stringify({
  _class_name: 'AutoPipelineForText2Image',
  feature_extractor: [null, null],
  safety_checker: [null, null],
  scheduler: ['diffusers', 'DDIMScheduler'],
  text_encoder: ['transformers', 'CLIPTextModel', 'text_encoder'],
  tokenizer: ['transformers', 'CLIPTokenizer', 'tokenizer'],
  unet: ['diffusers', 'UNet2DConditionModel', 'unet'],
  vae: ['diffusers', 'AutoencoderKL', 'vae'],
});
await check('a diffusers tree with only its index is not runnable', async () => {
  seed(root, 'Acme/index-only', { 'model_index.json': INDEX });
  const image = byRepo(await runnableModels('image', deps(ALL, ['Acme/index-only'])));
  const m = image.get('Acme/index-only');
  assert.ok(m, 'repo is still listed, so the reason is visible');
  assert.equal(m.ready, false, 'an index alone is not a model');
  assert.match(m.reason, /Missing/, `reason named nothing useful: ${m.reason}`);
});

await check('and it says which components are absent', async () => {
  seed(root, 'Acme/index-only', { 'model_index.json': INDEX });
  const image = byRepo(await runnableModels('image', deps(ALL, ['Acme/index-only'])));
  const m = image.get('Acme/index-only')!;
  for (const part of ['text_encoder', 'tokenizer', 'unet', 'vae']) {
    assert.ok(
      m.requiredFiles.includes(part),
      `${part} should be required; needs=${JSON.stringify(m.requiredFiles)}`,
    );
  }
});

await check('a two-element scheduler entry is a class name, not a folder', async () => {
  seed(root, 'Acme/index-only', { 'model_index.json': INDEX });
  const image = byRepo(await runnableModels('image', deps(ALL, ['Acme/index-only'])));
  const m = image.get('Acme/index-only')!;
  assert.equal(
    m.requiredFiles.includes('DDIMScheduler'),
    false,
    'the scheduler triple has no path, so its class name must not be required',
  );
});

await check('a complete diffusers tree is still runnable', async () => {
  seed(root, 'Acme/index-only', {
    'model_index.json': INDEX,
    'unet/diffusion_pytorch_model.safetensors': 'w',
    'vae/diffusion_pytorch_model.safetensors': 'w',
    'text_encoder/model.safetensors': 'w',
    'tokenizer/vocab.json': '{}',
  });
  const image = byRepo(await runnableModels('image', deps(ALL, ['Acme/index-only'])));
  const m = image.get('Acme/index-only')!;
  assert.equal(m.ready, true, `a complete tree should run; reason=${m.reason}`);
});

await check('a diffusers tree fetched outside this app is still seen as complete', async () => {
  // The records only know what this app fetched; the directory is the truth.
  seed(root, 'Acme/index-only', {
    'model_index.json': INDEX,
    'unet/diffusion_pytorch_model.safetensors': 'w',
    'vae/diffusion_pytorch_model.safetensors': 'w',
    'text_encoder/model.safetensors': 'w',
    'tokenizer/vocab.json': '{}',
  });
  const onlyIndex = downloadsOf(root, ['Acme/index-only']).filter((d) => d.file === 'model_index.json');
  const image = byRepo(await runnableModels('image', { downloads: onlyIndex, installed: ALL }));
  const m = image.get('Acme/index-only')!;
  assert.equal(m.ready, true, `on-disk files should count; reason=${m.reason}`);
});

await check('an unrecognised repo is not claimed by any task', async () => {
  for (const task of ['stt', 'tts', 'image'] as const) {
    const list = byRepo(await runnableModels(task, deps(ALL)));
    assert.equal(list.has('Acme/clip-thing'), false, `clip-thing listed for ${task}`);
  }
});

await check('tts stays catalog-only: no layout claims exist for it', async () => {
  const tts = await runnableModels('tts', deps(ALL));
  assert.equal(tts.some((m) => m.repo.startsWith('Acme/')), false);
});

// --- runtime reasons ----------------------------------------------------------
console.log('\n=== a download waiting on its runtime says so ===');

await check('with nothing installed, every discovered model reports what it needs', async () => {
  const stt = byRepo(await runnableModels('stt', deps({})));
  assert.equal(stt.get('Acme/ct2-whisper')?.ready, false);
  assert.equal(stt.get('Acme/ct2-whisper')?.reason, 'Needs faster-whisper.');
  assert.equal(stt.get('Acme/tf-whisper')?.ready, false);
  assert.equal(stt.get('Acme/tf-whisper')?.reason, 'Needs torch + transformers.');
  const image = byRepo(await runnableModels('image', deps({})));
  assert.equal(image.get('Acme/diff-pipe')?.reason, 'Needs torch + diffusers.');
});

await check('a partially installed runtime set names only what is missing', async () => {
  const stt = byRepo(await runnableModels('stt', deps({ torch: '2.14.1+cpu' })));
  assert.equal(stt.get('Acme/tf-whisper')?.reason, 'Needs transformers.');
  assert.equal(stt.get('Acme/tf-whisper')?.ready, false);
});

// --- catalog interplay ---------------------------------------------------------
console.log('\n=== catalog and download interplay ===');

await check('a catalog repo is reported once, by the catalog, with its bundle rules', async () => {
  const stt = await runnableModels('stt', deps(ALL));
  const rows = stt.filter((m) => m.repo === 'Systran/faster-whisper-tiny');
  assert.equal(rows.length, 1, `got ${rows.length} rows`);
  assert.equal(rows[0].label, 'Whisper tiny (CTranslate2)');
  assert.equal(rows[0].ready, true);
  assert.deepEqual(rows[0].requiredFiles, ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.txt']);
});

await check('catalog entries appear even when their files are absent', async () => {
  const stt = await runnableModels('stt', deps(ALL));
  const openai = byRepo(stt).get('openai/whisper-large-v3-turbo');
  assert.ok(openai, 'undownloaded catalog entry still listed');
  assert.equal(openai.ready, false);
  assert.equal(openai.reason, 'Not downloaded yet.');
});

await check('a repo with an empty file set is skipped', async () => {
  const stt = await runnableModels('stt', { downloads: [], installed: ALL });
  assert.equal(stt.some((m) => m.repo.startsWith('Acme/')), false);
});

// --- the shared Hugging Face cache ---------------------------------------------
console.log('\n=== a download living in the shared Hugging Face cache ===');

const cacheRoot = mkdtempSync(join(tmpdir(), 'runnable-cache-'));
process.env.HF_HUB_CACHE = cacheRoot;
seed(cacheRoot, 'Acme/cached-whisper', {
  'model.bin': 'x',
  'config.json': '{}',
});
const cachedDownloads = downloadsOf(cacheRoot, ['Acme/cached-whisper']).map((d) => ({
  ...d,
  location: 'cache' as const,
}));

await check('a cache download is discovered from its own root', async () => {
  const stt = byRepo(await runnableModels('stt', { downloads: cachedDownloads, installed: ALL }));
  const m = stt.get('Acme/cached-whisper');
  assert.ok(m, 'cache repo listed');
  assert.equal(m.ready, true);
  assert.equal(m.file, 'model.bin');
});

await check('a download root outside both roots is not scanned', async () => {
  const outside = downloadsOf(root, ['Acme/ct2-whisper']).map((d) => ({ ...d, root: '/etc' }));
  const stt = await runnableModels('stt', { downloads: outside, installed: ALL });
  assert.equal(stt.some((m) => m.repo === 'Acme/ct2-whisper'), false);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log(`\nFailing: ${failures.join(', ')}`);
  process.exit(1);
}
