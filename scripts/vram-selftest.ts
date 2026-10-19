/**
 * Self-test for the VRAM estimate behind the models page.
 *
 * Run with: npm run test:vram
 *
 * The header parser is checked against the real SmolLM2 GGUFs this repo's
 * smoke tests already use — a file whose metadata was written by llama.cpp's
 * converter, not a hand-made approximation — plus a spec-shaped fixture for the
 * one shape no local model happens to use (a per-layer kv-head array).
 *
 * The arithmetic has its own checks: the KV cache must scale with layers, kv
 * heads and context, because leaving any of those out makes a 30-layer model
 * look 30 times cheaper than it is, which is exactly the mistake a fit badge
 * cannot afford.
 */
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { estimateVram, fileVramEstimate, readGgufMeta, vramFit } from '../src/vram.js';

let passed = 0;
const failures: string[] = [];

async function check(name: string, fn: () => Promise<void> | void) {
  try {
    await fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures.push(name);
    console.log(`  FAIL  ${name}\n        ${(err as Error).message}`);
  }
}

const REAL = 'models/ggml-org--SmolLM2-135M-GGUF/SmolLM2-135M-Q4_K_M.gguf';
const REAL_META = {
  architecture: 'llama',
  blockCount: 30,
  embeddingLength: 576,
  headCount: 9,
  headCountKV: 3,
  contextLength: 8192,
};

/** Minimal GGUF v3 writer: metadata only, no tensors. */
function writeGguf(
  path: string,
  entries: { key: string; type: number; value: string | number | number[] }[],
): void {
  const str = (s: string) => {
    const body = Buffer.from(s, 'utf8');
    const len = Buffer.alloc(8);
    len.writeBigUInt64LE(BigInt(body.length));
    return Buffer.concat([len, body]);
  };
  const u32 = (n: number) => {
    const b = Buffer.alloc(4);
    b.writeUInt32LE(n);
    return b;
  };
  const i32 = (n: number) => {
    const b = Buffer.alloc(4);
    b.writeInt32LE(n);
    return b;
  };
  const u64 = (n: number) => {
    const b = Buffer.alloc(8);
    b.writeBigUInt64LE(BigInt(n));
    return b;
  };
  const chunks: Buffer[] = [Buffer.from('GGUF', 'latin1'), u32(3), u64(0), u64(entries.length)];
  for (const e of entries) {
    chunks.push(str(e.key), u32(e.type));
    if (typeof e.value === 'string') chunks.push(str(e.value));
    else if (Array.isArray(e.value)) {
      chunks.push(u32(5), u64(e.value.length), ...e.value.map(i32)); // INT32 array
    } else if (e.type === 4) chunks.push(u32(e.value));
    else chunks.push(i32(e.value));
  }
  writeFileSync(path, Buffer.concat(chunks));
}

const tmp = mkdtempSync(join(tmpdir(), 'starr-vram-'));

try {
  console.log('\n=== a real GGUF header ===');

  if (!existsSync(REAL)) {
    console.log(`  SKIP  ${REAL} is not on disk`);
  } else {
    await check('reads the hyperparameters llama.cpp wrote', async () => {
      const meta = await readGgufMeta(REAL);
      assert.deepEqual(meta, REAL_META);
    });

    await check('the KV cache scales with layers, kv heads and context', async () => {
      const meta = (await readGgufMeta(REAL)) as typeof REAL_META;
      const est = estimateVram(101_000_000, meta, 16_384);
      const headDim = Math.round(meta.embeddingLength / meta.headCount);
      const expected =
        2 * meta.blockCount * meta.headCountKV * headDim * 16_384 * 2;
      assert.equal(est.kvBytes, expected);
      assert.ok(est.kvBytes > est.weightsBytes, 'the cache outgrows a 135M model');
    });

    await check('the total is weights + cache + overhead, and nothing else', async () => {
      const meta = await readGgufMeta(REAL);
      const est = await fileVramEstimate(REAL, 8192);
      assert.equal(
        est.totalBytes,
        est.weightsBytes + est.kvBytes + est.overheadBytes,
      );
      assert.equal(est.overheadBytes, Math.round(est.weightsBytes * 0.02 + 150 * 1024 * 1024));
      assert.ok(est.totalBytes > est.weightsBytes);
    });

    await check('doubling the context adds exactly one cache', async () => {
      const meta = await readGgufMeta(REAL);
      const a = estimateVram(150_000_000, meta, 4096);
      const b = estimateVram(150_000_000, meta, 8192);
      assert.equal(b.kvBytes - a.kvBytes, a.kvBytes);
      assert.equal(b.totalBytes - a.totalBytes, a.kvBytes);
    });
  }

  console.log('\n=== fallbacks and verdicts ===');

  await check('without metadata the estimate is weights plus overhead only', () => {
    const est = estimateVram(4_000_000_000, null, 8192);
    assert.equal(est.kvBytes, 0);
    assert.equal(est.totalBytes, 4_000_000_000 + Math.round(4_000_000_000 * 0.02 + 150 * 1024 * 1024));
  });

  await check('a zero-byte file estimates nothing rather than a fake model', () => {
    const est = estimateVram(0, null, 8192);
    assert.equal(est.weightsBytes, 0);
    assert.equal(est.totalBytes, 150 * 1024 * 1024);
  });

  await check('fit is a band, not a line', () => {
    // The verdict reads totalBytes only; give it exact totals so the band
    // boundaries are what is being tested, not the overhead arithmetic.
    const est = (totalMB: number) => ({
      weightsBytes: 0,
      kvBytes: 0,
      overheadBytes: 0,
      totalBytes: totalMB * 1024 * 1024,
      ctx: 0,
      meta: null,
    });
    assert.equal(vramFit(est(700), 1000 * 1024 * 1024), 'fits');
    assert.equal(vramFit(est(800), 1000 * 1024 * 1024), 'fits');
    assert.equal(vramFit(est(900), 1000 * 1024 * 1024), 'tight');
    assert.equal(vramFit(est(1200), 1000 * 1024 * 1024), 'over');
    assert.equal(vramFit(est(700), null), null);
    assert.equal(vramFit(est(700), 0), null);
  });

  console.log('\n=== malformed and unusual headers ===');

  await check('a non-GGUF file is not a model', async () => {
    assert.equal(await readGgufMeta('package.json'), null);
  });

  await check('a truncated header answers null instead of throwing', async () => {
    const p = join(tmp, 'truncated.gguf');
    writeFileSync(p, Buffer.concat([Buffer.from('GGUF', 'latin1'), Buffer.alloc(6)]));
    assert.equal(await readGgufMeta(p), null);
  });

  await check('a per-layer kv-head array uses its largest entry', async () => {
    const p = join(tmp, 'array.gguf');
    writeGguf(p, [
      { key: 'general.architecture', type: 8, value: 'llama' },
      { key: 'llama.block_count', type: 4, value: 24 },
      { key: 'llama.embedding_length', type: 4, value: 2048 },
      { key: 'llama.attention.head_count', type: 4, value: 16 },
      { key: 'llama.attention.head_count_kv', type: 9, value: [2, 4, 4, 2] },
      { key: 'llama.context_length', type: 4, value: 32768 },
    ]);
    const meta = await readGgufMeta(p);
    assert.equal(meta?.headCountKV, 4);
    assert.equal(meta?.blockCount, 24);
    assert.equal(meta?.contextLength, 32768);
    const est = estimateVram(1_000_000, meta, 4096);
    assert.equal(est.kvBytes, 2 * 24 * 4 * 128 * 4096 * 2);
  });

  await check('missing head counts fall back to the geometry defaults', async () => {
    const p = join(tmp, 'sparse.gguf');
    writeGguf(p, [
      { key: 'general.architecture', type: 8, value: 'llama' },
      { key: 'llama.block_count', type: 4, value: 12 },
      { key: 'llama.embedding_length', type: 4, value: 1024 },
    ]);
    const meta = await readGgufMeta(p);
    assert.equal(meta?.headCount, 0);
    const est = estimateVram(1_000_000, meta, 2048);
    // headCount 0 → 32 heads → head dim 1024/32 = 32; kv heads fall back to
    // the head count.
    assert.equal(est.kvBytes, 2 * 12 * 32 * 32 * 2048 * 2);
  });
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

console.log(`\n${passed}/${passed + failures.length} checks passed.`);
if (failures.length) {
  console.log('failed:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
