/**
 * Unit checks for the Hugging Face browser + downloader. The Hub API and file
 * CDN are stubbed locally (pointed at via HF_API_BASE / HF_MODELS_DIR, which
 * the module reads per call), so no real models are touched.
 *
 * Run with: npm run test:hf
 */
import '../src/net.js';
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  deleteCacheRepo,
  deleteDownload,
  downloadFile,
  listDownloads,
  modelDetail,
  requiredFiles,
  searchModels,
  validateFilePath,
  validateRepoId,
} from '../src/huggingface.js';

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

/** Stub Hub: search, one detailed repo (blobs), byte-serving resolve. */
function stubHub() {
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://x');
    if (url.pathname === '/api/models') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify([
          {
            id: 'acme/Quant-7B-GGUF',
            likes: 42,
            downloads: 1000,
            pipeline_tag: 'text-generation',
            library_name: 'transformers',
            tags: ['gguf', 'quantized'],
            private: false,
          },
          { id: 'acme/other', likes: 1 },
        ]),
      );
    } else if (url.pathname === '/api/models/acme/Quant-7B-GGUF') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          id: 'acme/Quant-7B-GGUF',
          likes: 42,
          downloads: 1000,
          pipeline_tag: 'text-generation',
          tags: ['gguf'],
          gated: false,
          private: false,
          usedStorage: 5000000000,
          siblings: [
            { rfilename: 'model-Q4_K_M.gguf', blobId: 'a', size: 4000000000 },
            { rfilename: 'model-Q8_0.gguf', blobId: 'b', size: 7000000000 },
            { rfilename: 'config.json', blobId: 'c', size: 800 },
            { rfilename: '.gitattributes', blobId: 'd', size: 100 },
          ],
          safetensors: null,
        }),
      );
    } else if (url.pathname === '/api/models/acme/Tuner-82M') {
      // A voice-clone tuner as they exist on the Hub: weights plus demo wavs,
      // a generator script and trainer leftovers — none of it runnable.
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          id: 'acme/Tuner-82M',
          likes: 3,
          downloads: 90,
          pipeline_tag: 'text-to-speech',
          tags: ['safetensors'],
          gated: false,
          private: false,
          usedStorage: 50000000,
          siblings: [
            { rfilename: 'model.safetensors', blobId: 'a', size: 23000000 },
            { rfilename: 'config.json', blobId: 'b', size: 1200 },
            { rfilename: 'pytorch_model.bin', blobId: 'c', size: 23000000 },
            { rfilename: 'training_args.bin', blobId: 'd', size: 2000 },
            { rfilename: 'trainer_state.json', blobId: 'e', size: 3000 },
            { rfilename: 'samples/demo.wav', blobId: 'f', size: 5000000 },
            { rfilename: 'samples/ref_voice.wav', blobId: 'g', size: 4000000 },
            { rfilename: 'make_samples.py', blobId: 'h', size: 900 },
            { rfilename: 'README.md', blobId: 'i', size: 4000 },
            { rfilename: '.gitattributes', blobId: 'j', size: 100 },
          ],
          safetensors: null,
        }),
      );
    } else if (url.pathname === '/api/models/acme/missing') {
      res.writeHead(404, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'Not found' }));
    } else if (url.pathname === '/api/models/acme/gated-model') {
      res.writeHead(401, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'Invalid username or password.' }));
    } else if (url.pathname.startsWith('/acme/Quant-7B-GGUF/resolve/')) {
      const body = 'x'.repeat(5000);
      res.writeHead(200, {
        'content-type': 'application/octet-stream',
        'content-length': String(body.length),
      });
      res.end(body);
    } else {
      res.writeHead(404, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'nope' }));
    }
  });
  return new Promise<{ base: string; close: () => Promise<void> }>((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as AddressInfo;
      resolve({ base: `http://127.0.0.1:${port}`, close: () => new Promise((r) => server.close(() => r())) });
    });
  });
}

console.log('\n=== validation ===');
await check('repo ids accept author/model only', () => {
  assert.equal(validateRepoId('acme/Quant-7B-GGUF'), null);
  assert.match(validateRepoId('')!, /required/);
  assert.match(validateRepoId('noslash')!, /Invalid/);
  assert.match(validateRepoId('a/b/c')!, /Invalid/);
  assert.match(validateRepoId('a/b;c')!, /Invalid/);
});
await check('file paths allow subdirs, never traversal', () => {
  assert.equal(validateFilePath('model-Q4_K_M.gguf'), null);
  assert.equal(validateFilePath('onnx/model.onnx'), null);
  for (const bad of ['', '../x', 'a/../../x', '/abs', 'a\\b', 'a//b', '.']) {
    assert.match(validateFilePath(bad)!, /Invalid|required/, bad);
  }
});

console.log('\n=== against the stub Hub ===');
const stub = await stubHub();
process.env.HF_API_BASE = stub.base;
process.env.HF_MODELS_DIR = mkdtempSync(join(tmpdir(), 'hf-test-'));
// The real machine cache must never leak into these checks; the cache cases
// below build their own tree under this temp root.
process.env.HF_HUB_CACHE = mkdtempSync(join(tmpdir(), 'hf-cache-test-'));
const noSignal = AbortSignal.timeout(15000);

await check('search maps the list shape', async () => {
  const hits = await searchModels('quant', { signal: noSignal });
  assert.equal(hits.length, 2);
  assert.equal(hits[0].id, 'acme/Quant-7B-GGUF');
  assert.equal(hits[0].likes, 42);
  assert.ok(hits[0].tags.includes('gguf'));
  assert.equal(hits[1].downloads, 0);
});
await check('empty query lists trending', async () => {
  const hits = await searchModels('   ', { signal: noSignal });
  assert.ok(hits.length > 0, 'trending returned');
});

await check('detail carries per-file sizes', async () => {
  const d = await modelDetail('acme/Quant-7B-GGUF', noSignal);
  assert.equal(d.usedStorage, 5000000000);
  assert.equal(d.files.length, 4);
  const q4 = d.files.find((f) => f.name === 'model-Q4_K_M.gguf');
  assert.equal(q4?.size, 4000000000);
  assert.equal(d.gated, false);
});
await check('required files skip demo clips, scripts and trainer residue', async () => {
  const wanted = await requiredFiles('acme/Tuner-82M', noSignal);
  assert.deepEqual(wanted.map((f) => f.path), ['model.safetensors', 'config.json']);
  assert.equal(wanted[0].size, 23000000);
});
await check('missing models 404 plainly', async () => {
  await assert.rejects(() => modelDetail('acme/missing', noSignal), /No such model/);
});
await check('gated models name the token fix', async () => {
  await assert.rejects(() => modelDetail('acme/gated-model', noSignal), /HF token/);
});

await check('download streams with progress to the hashed dir', async () => {
  const seen: { received: number; total: number | null }[] = [];
  const done = await downloadFile('acme/Quant-7B-GGUF', 'model-Q4_K_M.gguf', {
    signal: noSignal,
    onProgress: (p) => seen.push(p),
  });
  assert.equal(done.bytes, 5000);
  assert.ok(seen.length > 0 && seen[seen.length - 1].received === 5000);
  assert.equal(seen[0].total, 5000);
  const listed = await listDownloads();
  assert.equal(listed.length, 1);
  assert.equal(listed[0].repo, 'acme/Quant-7B-GGUF');
  assert.equal(listed[0].file, 'model-Q4_K_M.gguf');
  assert.equal(listed[0].size, 5000);
  assert.equal(listed[0].location, 'models');
  assert.equal(listed[0].root, join(process.env.HF_MODELS_DIR!, 'acme--Quant-7B-GGUF'));
});

await check('downloaded files delete and prune', async () => {
  await downloadFile('acme/Quant-7B-GGUF', 'onnx/model.onnx', { signal: noSignal });
  assert.equal((await listDownloads()).length, 2);
  await deleteDownload('acme/Quant-7B-GGUF', 'onnx/model.onnx');
  const left = await listDownloads();
  assert.equal(left.length, 1);
  await deleteDownload('acme/Quant-7B-GGUF', 'model-Q4_K_M.gguf');
  assert.deepEqual(await listDownloads(), []);
});

await check('delete traversal is refused', async () => {
  await assert.rejects(() => deleteDownload('acme/x', '../evil'), /Invalid/);
  await assert.rejects(() => deleteDownload('acme/x', 'nope.bin'), /no such file/i);
});

console.log('\n=== the shared HF cache ===');

await check('a repo another tool cached is listed — links followed, dangling ones skipped', async () => {
  const cache = process.env.HF_HUB_CACHE!;
  const repoDir = join(cache, 'models--acme--Cached-TTS');
  const snap = join(repoDir, 'snapshots', 'abc123');
  const blobs = join(repoDir, 'blobs');
  mkdirSync(snap, { recursive: true });
  mkdirSync(blobs, { recursive: true });
  writeFileSync(join(blobs, 'voice.onnx'), 'x'.repeat(1200));
  symlinkSync(join('..', '..', 'blobs', 'voice.onnx'), join(snap, 'voice.onnx'));
  writeFileSync(join(snap, 'config.json'), '{}');
  // Half a download: the blob never landed, so the link dangles.
  symlinkSync(join('..', '..', 'blobs', 'never.bin'), join(snap, 'stub.bin'));

  const rows = (await listDownloads()).filter((f) => f.repo === 'acme/Cached-TTS');
  assert.equal(rows.length, 2, `got ${rows.length} files`);
  const weights = rows.find((f) => f.file === 'voice.onnx');
  assert.ok(weights, 'symlinked weight listed');
  assert.equal(weights.size, 1200, 'size reads through the link');
  assert.equal(weights.location, 'cache');
  assert.equal(weights.root, snap);
  assert.equal(rows.some((f) => f.file === 'stub.bin'), false, 'dangling link skipped');
});

await check('the app’s own copy of a repo wins over its cache copy', async () => {
  await downloadFile('acme/Quant-7B-GGUF', 'model-Q4_K_M.gguf', { signal: noSignal });
  const cache = process.env.HF_HUB_CACHE!;
  const snap = join(cache, 'models--acme--Quant-7B-GGUF', 'snapshots', 'deadbee');
  mkdirSync(snap, { recursive: true });
  writeFileSync(join(snap, 'model-Q4_K_M.gguf'), 'x'.repeat(5000));

  const rows = (await listDownloads()).filter((f) => f.repo === 'acme/Quant-7B-GGUF');
  assert.equal(rows.length, 1, `got ${rows.length} rows`);
  assert.equal(rows[0].location, 'models');
  await deleteDownload('acme/Quant-7B-GGUF', 'model-Q4_K_M.gguf');
});

await check('a cache repo removes as one unit, and only its own folder', async () => {
  const cache = process.env.HF_HUB_CACHE!;
  await assert.rejects(() => deleteCacheRepo('a/b/c'), /Invalid/);
  await assert.rejects(() => deleteCacheRepo('acme/nothing-here'), /not in the Hugging Face cache/);
  await deleteCacheRepo('acme/Cached-TTS');
  assert.equal(existsSync(join(cache, 'models--acme--Cached-TTS')), false, 'repo folder gone');
  assert.equal(existsSync(join(cache, 'models--acme--Quant-7B-GGUF')), true, 'sibling untouched');
  assert.equal(
    (await listDownloads()).some((f) => f.repo === 'acme/Cached-TTS'),
    false,
    'no rows left',
  );
});

await stub.close();
delete process.env.HF_API_BASE;
delete process.env.HF_MODELS_DIR;
delete process.env.HF_HUB_CACHE;

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log(`\nFailing: ${failures.join(', ')}`);
  process.exit(1);
}
