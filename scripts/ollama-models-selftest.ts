/**
 * Unit checks for local Ollama model management. The daemon is stubbed with a
 * local HTTP server (pointed at via OLLAMA_BASE_URL, which the module reads
 * per call), so no real models are pulled or deleted.
 *
 * Run with: npm run test:ollama-models
 */
import '../src/net.js';
import assert from 'node:assert/strict';
import { lutimes, mkdir, mkdtemp, rm, symlink, utimes, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';

import {
  acquirePull,
  deleteLocalModel,
  listLocalModels,
  modelsOnDisk,
  pullModel,
  releasePull,
  validateModelName,
} from '../src/ollama-models.js';

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

const noSignal = AbortSignal.timeout(15000);

/** Stub daemon: tags, a scripted pull stream, delete capture. */
function stubDaemon() {
  const seen: { method: string; url: string; body: string }[] = [];
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      seen.push({ method: req.method ?? '', url: req.url ?? '', body });
      if (req.url === '/api/tags') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(
          JSON.stringify({
            models: [
              {
                name: 'qwen3:8b',
                size: 5200000000,
                modified_at: '2026-01-02T00:00:00Z',
                details: { parameter_size: '8.2B', quantization_level: 'Q4_K_M', family: 'qwen3' },
              },
              { name: 'tiny:latest', size: 1000, modified_at: '2026-01-01T00:00:00Z' },
            ],
          }),
        );
      } else if (req.url === '/api/pull') {
        res.writeHead(200, { 'content-type': 'application/x-ndjson' });
        for (const line of [
          { status: 'pulling manifest' },
          { status: 'downloading', digest: 'sha256:abc', total: 100, completed: 40 },
          { status: 'downloading', digest: 'sha256:abc', total: 100, completed: 100 },
          { status: 'verifying sha256 digest' },
          { status: 'writing manifest' },
          { status: 'success' },
        ]) {
          res.write(`${JSON.stringify(line)}\n`);
        }
        res.end();
      } else if (req.url === '/api/delete') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end('{}');
      } else {
        res.writeHead(404, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'no such endpoint' }));
      }
    });
  });
  return new Promise<{ base: string; seen: typeof seen; close: () => Promise<void> }>(
    (resolve) => {
      server.listen(0, '127.0.0.1', () => {
        const { port } = server.address() as AddressInfo;
        resolve({
          base: `http://127.0.0.1:${port}`,
          seen,
          close: () => new Promise((r) => server.close(() => r())),
        });
      });
    },
  );
}

console.log('\n=== validation ===');
await check('blank and absurd names rejected', () => {
  assert.match(validateModelName('')!, /required/);
  assert.match(validateModelName(null)!, /required/);
  assert.match(validateModelName('x'.repeat(201))!, /too long/);
  // Shell metacharacters and spaces can never be part of a model id.
  assert.match(validateModelName('a;b |c')!, /Invalid/);
  assert.match(validateModelName('$HOME/x')!, /Invalid/);
  assert.match(validateModelName('"quoted"')!, /Invalid/);
});
await check('real-world names pass', () => {
  for (const n of ['qwen3:8b', 'hf.co/bartowski/Qwen2.5-7B-GGUF', 'llama3.1_8b', 'a-b_c.d:e/f']) {
    assert.equal(validateModelName(n), null);
  }
});

console.log('\n=== against the stub daemon ===');
const stub = await stubDaemon();
process.env.OLLAMA_BASE_URL = stub.base;

await check('list parses tags newest-first', async () => {
  const models = await listLocalModels(noSignal);
  assert.equal(models.length, 2);
  assert.equal(models[0].name, 'qwen3:8b');
  assert.equal(models[0].size, 5200000000);
  assert.equal(models[0].parameterSize, '8.2B');
  assert.equal(models[0].quantization, 'Q4_K_M');
  assert.equal(models[1].name, 'tiny:latest');
});

await check('pull streams progress to success', async () => {
  const events: string[] = [];
  for await (const p of pullModel('qwen3:8b', noSignal)) {
    events.push(p.total ? `${p.status} ${p.completed}/${p.total}` : p.status);
  }
  assert.deepEqual(events, [
    'pulling manifest',
    'downloading 40/100',
    'downloading 100/100',
    'verifying sha256 digest',
    'writing manifest',
    'success',
  ]);
});

await check('pull error lines throw', async () => {
  const failing = await (async () => {
    const { createServer: cs } = await import('node:http');
    const srv = cs((_req, res) => {
      res.writeHead(200, { 'content-type': 'application/x-ndjson' });
      res.end(JSON.stringify({ error: 'model not found' }) + '\n');
    });
    await new Promise<void>((r) => srv.listen(0, '127.0.0.1', () => r()));
    return srv;
  })();
  const addr = failing.address() as AddressInfo;
  process.env.OLLAMA_BASE_URL = `http://127.0.0.1:${addr.port}`;
  try {
    await assert.rejects(
      (async () => {
        for await (const _ of pullModel('nope:latest', noSignal)) {
          /* drain */
        }
      })(),
      /model not found/,
    );
  } finally {
    await new Promise<void>((r) => failing.close(() => r()));
    process.env.OLLAMA_BASE_URL = stub.base;
  }
});

await check('delete sends the name', async () => {
  await deleteLocalModel('tiny:latest', noSignal);
  const last = stub.seen.filter((s) => s.url === '/api/delete').pop();
  assert.deepEqual(JSON.parse(last!.body), { name: 'tiny:latest' });
});

await check('one pull at a time', () => {
  assert.equal(acquirePull('a'), true);
  assert.equal(acquirePull('b'), false);
  releasePull();
  assert.equal(acquirePull('b'), true);
  releasePull();
});

await check('a long pull stream is not killed by the 30-second request timeout', async () => {
  // A real model pull is an NDJSON response kept open for minutes. `daemonFetch`
  // used to wrap the whole fetch in `AbortSignal.timeout(30_000)`, which aborts
  // the body mid-download even though the daemon is healthy. Shorten that timer
  // and hold the stream open past it: the pull must not create the timer at all.
  const { createServer: cs } = await import('node:http');
  const slow = cs((req, res) => {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      if (req.url !== '/api/pull') {
        res.writeHead(404, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'no such endpoint' }));
        return;
      }
      res.writeHead(200, { 'content-type': 'application/x-ndjson' });
      res.write(JSON.stringify({ status: 'pulling manifest' }) + '\n');
      setTimeout(() => {
        res.write(JSON.stringify({ status: 'downloading', total: 100, completed: 50 }) + '\n');
        res.end(JSON.stringify({ status: 'success' }) + '\n');
      }, 250);
    });
  });
  await new Promise<void>((resolve) => slow.listen(0, '127.0.0.1', () => resolve()));
  const addr = slow.address() as AddressInfo;
  process.env.OLLAMA_BASE_URL = `http://127.0.0.1:${addr.port}`;

  const realTimeout = AbortSignal.timeout;
  (AbortSignal as any).timeout = (ms: number) => realTimeout(Math.min(ms, 80));
  try {
    const events: string[] = [];
    for await (const progress of pullModel('slow:latest')) events.push(progress.status);
    assert.deepEqual(events, ['pulling manifest', 'downloading', 'success']);
  } finally {
    (AbortSignal as any).timeout = realTimeout;
    process.env.OLLAMA_BASE_URL = stub.base;
    await new Promise<void>((resolve) => slow.close(() => resolve()));
  }
});

console.log('\n=== a dead daemon is not an empty library ===');

// Ollama's own store, laid out the way the daemon leaves it: one model in the
// older `manifests/` files and one in the 0.12 `manifests-v2/` symlinks.
const store = await mkdtemp(path.join(tmpdir(), 'starr-ollama-'));
const blobs = path.join(store, 'blobs');
await mkdir(blobs, { recursive: true });
const hex = (c: string) => c.repeat(64);
const blob = (digest: string, body: unknown) =>
  writeFile(path.join(blobs, `sha256-${digest}`), JSON.stringify(body));
const oldLib = path.join(store, 'manifests', 'registry.ollama.ai', 'library');
await mkdir(path.join(oldLib, 'legacy'), { recursive: true });
await mkdir(path.join(oldLib, 'fresh'), { recursive: true });

await blob(hex('a'), { model_type: '7.1B', file_type: 'Q4_0', model_family: 'llama' });
await writeFile(
  path.join(oldLib, 'legacy', '7b'),
  JSON.stringify({
    config: { digest: `sha256:${hex('a')}`, size: 100 },
    layers: [{ size: 1000 }, { size: 2000 }],
  }),
);

await blob(hex('b'), { model_type: '1.5B', file_type: 'Q8_0', model_family: 'qwen2' });
await blob(hex('c'), {
  config: { digest: `sha256:${hex('b')}`, size: 50 },
  layers: [{ size: 5000 }],
});
const v2dir = path.join(store, 'manifests-v2', 'ollama.com', 'library', 'fresh');
await mkdir(v2dir, { recursive: true });
await symlink(`../../../../blobs/sha256-${hex('c')}`, path.join(v2dir, '1.5b'));

// The same name also sitting in the old layout: it must neither duplicate the
// entry nor win the size (the daemon reads the v2 manifest).
await writeFile(
  path.join(oldLib, 'fresh', '1.5b'),
  JSON.stringify({ config: { size: 1 }, layers: [{ size: 1 }] }),
);

const older = new Date('2026-01-01T00:00:00Z');
const newer = new Date('2026-01-02T00:00:00Z');
await utimes(path.join(oldLib, 'legacy', '7b'), older, older);
await lutimes(path.join(v2dir, '1.5b'), newer, newer);

process.env.OLLAMA_MODELS = store;
process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:1';

await check('installed models list without the daemon', async () => {
  const models = await listLocalModels(noSignal);
  assert.deepEqual(
    models.map((m) => m.name),
    ['fresh:1.5b', 'legacy:7b'],
  );
  assert.equal(models.filter((m) => m.name === 'fresh:1.5b').length, 1);
  assert.equal(models[0].size, 5050);
  assert.equal(models[0].parameterSize, '1.5B');
  assert.equal(models[0].quantization, 'Q8_0');
  assert.equal(models[0].family, 'qwen2');
  assert.equal(models[1].size, 3100);
  assert.equal(models[1].parameterSize, '7.1B');
  assert.deepEqual(await modelsOnDisk(), models);
});

await check('dead daemon explains itself', async () => {
  const empty = await mkdtemp(path.join(tmpdir(), 'starr-ollama-empty-'));
  process.env.OLLAMA_MODELS = empty;
  await assert.rejects(() => listLocalModels(noSignal), /not reachable/);
  await assert.rejects(
    (async () => {
      for await (const _ of pullModel('x:y', noSignal)) {
        /* drain */
      }
    })(),
    /not reachable/,
  );
  process.env.OLLAMA_BASE_URL = stub.base;
  delete process.env.OLLAMA_MODELS;
  await rm(empty, { recursive: true, force: true });
});

await stub.close();
await rm(store, { recursive: true, force: true });
delete process.env.OLLAMA_BASE_URL;
delete process.env.OLLAMA_MODELS;

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log(`\nFailing: ${failures.join(', ')}`);
  process.exit(1);
}
