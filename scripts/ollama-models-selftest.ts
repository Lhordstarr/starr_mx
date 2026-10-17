/**
 * Unit checks for local Ollama model management. The daemon is stubbed with a
 * local HTTP server (pointed at via OLLAMA_BASE_URL, which the module reads
 * per call), so no real models are pulled or deleted.
 *
 * Run with: npm run test:ollama-models
 */
import '../src/net.js';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';

import {
  acquirePull,
  deleteLocalModel,
  listLocalModels,
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

await check('dead daemon explains itself', async () => {
  process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:1';
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
});

await stub.close();
delete process.env.OLLAMA_BASE_URL;

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log(`\nFailing: ${failures.join(', ')}`);
  process.exit(1);
}
