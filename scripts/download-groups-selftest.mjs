import assert from 'node:assert/strict';
import {
  fileCandidates,
  groupFilesBySource,
  missingCatalogFiles,
  transferSources,
} from '../public/transfer-groups.js';

let passed = 0;
const failures = [];

async function check(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures.push({ name, err });
    console.log(`  FAIL  ${name}`);
    console.log(`        ${err instanceof Error ? err.message : String(err)}`);
  }
}

console.log('=== downloads: one card, several source repos ===');

await check('a card and its auxiliary VAE stay two sources', () => {
  const main = 'second-state/stable-diffusion-3.5-medium-GGUF';
  const vae = 'adamo1139/stable-diffusion-3.5-medium-ungated';
  const sources = transferSources({
    repo: main,
    files: [
      { repo: main, sourceRepo: main, file: 'sd3.5_medium-Q4_0.gguf' },
      { repo: main, sourceRepo: vae, file: 'vae/diffusion_pytorch_model.safetensors' },
    ],
  });
  assert.deepEqual([...sources.keys()], [main, vae]);
  assert.deepEqual([...sources.get(main)], ['sd3.5_medium-Q4_0.gguf']);
  assert.deepEqual([...sources.get(vae)], ['vae/diffusion_pytorch_model.safetensors']);
});

await check('a transfer without an explicit source falls back to its repo', () => {
  const sources = transferSources({
    repo: 'owner/card',
    files: [
      { repo: 'owner/quants', file: 'model.gguf' },
      { repo: 'owner/text', file: 'text.gguf' },
    ],
  });
  assert.deepEqual([...sources.keys()], ['owner/quants', 'owner/text']);
  assert.deepEqual([...sources.get('owner/quants')], ['model.gguf']);
});

await check('the displayed card repo is the last fallback', () => {
  const sources = transferSources({
    repo: 'owner/card',
    files: [{ file: 'weights.bin' }],
  });
  assert.deepEqual([...sources.keys()], ['owner/card']);
  assert.deepEqual([...sources.get('owner/card')], ['weights.bin']);
});

await check('duplicate entries do not duplicate files or sources', () => {
  const main = 'owner/quants';
  const sources = transferSources({
    repo: main,
    files: [
      { repo: main, sourceRepo: main, file: 'a.gguf' },
      { repo: main, sourceRepo: main, file: 'a.gguf' },
      { repo: main, sourceRepo: main, file: 'b.gguf' },
    ],
  });
  assert.equal(sources.size, 1);
  assert.deepEqual([...sources.get(main)].sort(), ['a.gguf', 'b.gguf']);
});

await check('an empty group has no sources', () => {
  assert.equal(transferSources({ repo: 'owner/card' }).size, 0);
  assert.equal(transferSources({ repo: 'owner/card', files: [] }).size, 0);
});

console.log('\n=== hub delete: files go back to the repo that owns them ===');

await check('a bundled VAE is deleted from its own mirror', () => {
  const main = 'second-state/stable-diffusion-3.5-medium-GGUF';
  const vae = 'adamo1139/stable-diffusion-3.5-medium-ungated';
  const bySource = groupFilesBySource(
    main,
    ['sd3.5_medium-Q4_0.gguf', 'vae/diffusion_pytorch_model.safetensors'],
    [
      { name: 'sd3.5_medium-Q4_0.gguf' },
      { name: 'vae/diffusion_pytorch_model.safetensors', repo: vae },
    ],
  );
  assert.deepEqual([...bySource.keys()], [main, vae]);
  assert.deepEqual(bySource.get(main), ['sd3.5_medium-Q4_0.gguf']);
  assert.deepEqual(bySource.get(vae), ['vae/diffusion_pytorch_model.safetensors']);
});

await check('files outside the catalog stay with the card repo', () => {
  const main = 'owner/card';
  const bySource = groupFilesBySource(main, ['model.gguf', 'README.md'], []);
  assert.deepEqual([...bySource.keys()], [main]);
  assert.deepEqual(bySource.get(main), ['model.gguf', 'README.md']);
});

await check('an empty row asks for nothing', () => {
  assert.equal(groupFilesBySource('owner/card', [], []).size, 0);
});

console.log('\n=== catalog slots with interchangeable exports ===');

const kokoroSlot = {
  name: 'onnx/model_q4.onnx',
  oneOf: ['onnx/model_q4f16.onnx'],
};

await check('the q4f16 export satisfies the preferred q4 slot', () => {
  assert.deepEqual(fileCandidates(kokoroSlot), [
    'onnx/model_q4.onnx',
    'onnx/model_q4f16.onnx',
  ]);
  assert.deepEqual(
    missingCatalogFiles([kokoroSlot], ['onnx/model_q4f16.onnx', 'config.json']),
    [],
  );
});

await check('a truly absent slot is still reported by its preferred name', () => {
  assert.deepEqual(missingCatalogFiles([kokoroSlot], ['config.json']), [
    'onnx/model_q4.onnx',
  ]);
});

await check('a slot with no alternatives still has to match itself', () => {
  assert.deepEqual(
    missingCatalogFiles([{ name: 'vae/diffusion_pytorch_model.safetensors' }], [
      'vae/diffusion_pytorch_model.safetensors',
    ]),
    [],
  );
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
