#!/usr/bin/env tsx
/**
 * ONNX speech is a real runner, not a format label.
 *
 * The catalog used to carry the Kokoro ONNX export with no `run` descriptor and
 * two alternative quantisations listed as if both were required; the runtime
 * page said ONNX Runtime was unused, and there was no ONNX speech-to-text entry
 * at all. This suite pins the three things that made that state visible:
 *
 *   * the catalog entries are runnable and name one graph file, not two quants;
 *   * downloaded ONNX Whisper and Kokoro layouts are discovered by shape;
 *   * the server selects `speak_onnx.py` / `transcribe_onnx.py` and the real
 *     models produce audio and text through the API-facing functions.
 *
 * The live runs need the models this repository's catalog downloads. They are
 * skipped only when those files are absent, which is the same rule the other
 * local-runner suites follow.
 */
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';

import { CATALOG, ONNX_WHISPER_SIDECARS } from '../src/model-catalog.ts';
import { runnableModels } from '../src/runnable.ts';
import { speak, transcribe } from '../src/infer.ts';
import { SCRIPT_RUNTIMES } from '../src/processes.ts';
import { modelsDir, type HFDownload } from '../src/huggingface.ts';

let passed = 0;
const failures: string[] = [];
function check(name: string, fn: () => unknown | Promise<unknown>): Promise<void> {
  return Promise.resolve()
    .then(fn)
    .then(() => {
      passed += 1;
      console.log(`  PASS  ${name}`);
    })
    .catch((err: unknown) => {
      failures.push(name);
      console.log(`  FAIL  ${name}${err instanceof Error ? ` — ${err.message}` : ''}`);
    });
}

const catalogEntry = (repo: string) =>
  CATALOG.flatMap((section) => section.entries).find((entry) => entry.repo === repo);

const downloadsFor = (repo: string): HFDownload[] => {
  const entry = catalogEntry(repo);
  assert.ok(entry, `catalog entry ${repo}`);
  return entry.files.map((file) => ({
    repo,
    file: file.name,
    size: file.bytes,
    modifiedAt: new Date().toISOString(),
    location: 'models' as const,
    root: path.resolve('/tmp/onnx-selftest-fixture'),
  }));
};

const installed = {
  kokoro: '0.9.4',
  onnxruntime: '1.31.0',
  transformers: '5.20.0',
  torch: '2.14.1+cpu',
};

console.log('The catalog must offer ONNX speech as a job, not just a file');
await check('Kokoro ONNX lists one graph file, not two quants', () => {
  const entry = catalogEntry('onnx-community/Kokoro-82M-v1.0-ONNX');
  assert.ok(entry, 'Kokoro ONNX entry exists');
  const graphs = entry.files.filter((file) => /\.onnx$/i.test(file.name));
  assert.equal(graphs.length, 1, `found ${graphs.map((f) => f.name).join(', ')}`);
  assert.equal(graphs[0]?.name, 'onnx/model_q4f16.onnx');
});

await check('Kokoro ONNX is runnable and carries its voice and vocabulary', async () => {
  const entry = catalogEntry('onnx-community/Kokoro-82M-v1.0-ONNX');
  assert.ok(entry, 'Kokoro ONNX entry exists');
  assert.equal(entry.run?.kind, 'tts');
  assert.ok(entry.files.some((file) => file.name === 'voices/af_heart.bin'));
  assert.ok(entry.files.some((file) => file.name === 'tokenizer.json'));

  const models = await runnableModels('tts', {
    downloads: downloadsFor(entry.repo),
    installed,
  });
  const model = models.find((m) => m.repo === entry.repo);
  assert.ok(model, 'Kokoro ONNX appears in the TTS picker');
  assert.equal(model.ready, true, model.reason);
  assert.equal(model.file, 'onnx/model_q4f16.onnx');
  assert.equal(model.runtimes.kokoro, '0.9.4');
});

await check('ONNX Whisper is a catalog STT entry with the ONNX driver', async () => {
  const entry = catalogEntry('onnx-community/whisper-tiny.en');
  assert.ok(entry, 'ONNX Whisper entry exists');
  assert.equal(entry.run?.kind, 'stt');
  assert.equal(entry.bundle, true);

  const models = await runnableModels('stt', {
    downloads: downloadsFor(entry.repo),
    installed,
  });
  const model = models.find((m) => m.repo === entry.repo);
  assert.ok(model, 'ONNX Whisper appears in the STT picker');
  assert.equal(model.ready, true, model.reason);
  assert.equal(model.driver, 'onnx');
  assert.ok(model.requiredFiles.includes('onnx/encoder_model_quantized.onnx'));
  assert.ok(model.requiredFiles.includes('onnx/decoder_model_merged_quantized.onnx'));
});

console.log('\nA downloaded ONNX layout must be discovered, not left with no job');
const fixtureRoots: string[] = [];
async function fixture(repo: string, files: Record<string, string>): Promise<HFDownload[]> {
  const root = mkdtempSync(path.join(modelsDir(), 'onnx-selftest-'));
  fixtureRoots.push(root);
  for (const [rel, body] of Object.entries(files)) {
    const full = path.join(root, ...rel.split('/'));
    mkdirSync(path.dirname(full), { recursive: true });
    writeFileSync(full, body);
  }
  return Object.entries(files).map(([rel, body]) => ({
    repo,
    file: rel,
    size: body.length,
    modifiedAt: new Date().toISOString(),
    location: 'models' as const,
    root,
  }));
}

try {
  await check('an ONNX Whisper pair is claimed for STT', async () => {
    const repo = 'Acme/onnx-whisper';
    const files: Record<string, string> = {
      'config.json': JSON.stringify({
        model_type: 'whisper',
        architectures: ['WhisperForConditionalGeneration'],
      }),
      'onnx/encoder_model.onnx': 'graph',
      'onnx/decoder_model_merged.onnx': 'graph',
      'generation_config.json': '{}',
      'preprocessor_config.json': '{}',
      'tokenizer.json': '{}',
      'tokenizer_config.json': '{}',
      'special_tokens_map.json': '{}',
      'vocab.json': '{}',
      'merges.txt': '',
      'normalizer.json': '{}',
    };
    for (const sidecar of ONNX_WHISPER_SIDECARS) {
      files[sidecar] ??= '{}';
    }
    const downloads = await fixture(repo, files);
    const models = await runnableModels('stt', { downloads, installed });
    const model = models.find((m) => m.repo === repo);
    assert.ok(model, 'ONNX Whisper discovered');
    assert.equal(model.ready, true, model.reason);
    assert.equal(model.driver, 'onnx');
    assert.equal(model.file, 'onnx/encoder_model.onnx');
    assert.match(model.label, /ONNX/);
  });

  await check('an ONNX Kokoro export is claimed for TTS', async () => {
    const repo = 'Acme/onnx-kokoro';
    const downloads = await fixture(repo, {
      'config.json': JSON.stringify({ model_type: 'style_text_to_speech_2' }),
      'onnx/model_q4f16.onnx': 'graph',
      'voices/af_heart.bin': 'voice',
      'tokenizer.json': '{"model":{"vocab":{}}}',
    });
    const models = await runnableModels('tts', { downloads, installed });
    const model = models.find((m) => m.repo === repo);
    assert.ok(model, 'ONNX Kokoro discovered');
    assert.equal(model.ready, true, model.reason);
    assert.equal(model.file, 'onnx/model_q4f16.onnx');
    assert.equal(model.runtimes.kokoro, '0.9.4');
    assert.match(model.label, /ONNX/);
  });
} finally {
  for (const root of fixtureRoots) rmSync(root, { recursive: true, force: true });
}

console.log('\nThe server must choose the ONNX scripts and they must produce output');
const speakSource = path.resolve('scripts/py/speak_onnx.py');
const transcribeSource = path.resolve('scripts/py/transcribe_onnx.py');
await check('the ONNX TTS runner exists and uses the graph plus Kokoro G2P', () => {
  assert.ok(existsSync(speakSource));
  const source = readFileSync(speakSource, 'utf8');
  assert.match(source, /onnxruntime/);
  assert.match(source, /KPipeline/);
  assert.match(source, /input_ids/);
  assert.match(source, /voices/);
  assert.doesNotMatch(source, /KModel\s*\(/);
});
await check('the ONNX STT runner exists and keeps the decoder on its no-cache branch', () => {
  assert.ok(existsSync(transcribeSource));
  const source = readFileSync(transcribeSource, 'utf8');
  assert.match(source, /decoder_model_merged/);
  assert.match(source, /encoder_model/);
  assert.match(source, /use_cache_branch/);
  assert.match(source, /argmax/);
});

await check('the resource manager credits the ONNX runners with their packages', () => {
  assert.deepEqual(SCRIPT_RUNTIMES['transcribe_onnx.py'], ['onnxruntime', 'transformers']);
  assert.deepEqual(SCRIPT_RUNTIMES['speak_onnx.py'], ['onnxruntime', 'kokoro']);
  // The generic transformers TTS runner was missing from this table before the
  // ONNX work; a process row that names nothing is still a wrong row.
  assert.deepEqual(SCRIPT_RUNTIMES['speak_hf.py'], ['torch', 'transformers']);
});

const onnxKokoroDir = path.resolve('models/onnx-community--Kokoro-82M-v1.0-ONNX');
const onnxWhisperDir = path.resolve('models/onnx-community--whisper-tiny.en');
const foxClip = '_uploads/clip-1792421379226.wav';

await check('the real Kokoro ONNX export synthesises audio through speak()', async () => {
  if (
    !existsSync(path.resolve('.runtimes/venv-kokoro/bin/python')) ||
    !existsSync(path.join(onnxKokoroDir, 'onnx/model_q4f16.onnx')) ||
    !existsSync(path.join(onnxKokoroDir, 'voices/af_heart.bin'))
  ) {
    console.log('  SKIP  no Kokoro ONNX venv/weights on this machine');
    return;
  }
  const { result, outPath } = await speak({
    repo: 'onnx-community/Kokoro-82M-v1.0-ONNX',
    file: 'onnx/model_q4f16.onnx',
    voice: 'af_heart',
    speed: 1,
    device: 'cpu',
    text: 'ONNX speech is a real job in this app.',
  });
  assert.equal(result.ok, true, JSON.stringify(result.data));
  assert.ok(existsSync(outPath), 'wav written');
  assert.ok(statSync(outPath).size > 20_000, `wav is ${statSync(outPath).size} bytes`);
  assert.ok(Number(result.data?.seconds) > 1, JSON.stringify(result.data));
  assert.equal(result.data?.device, 'cpu');
});

await check('the real ONNX Whisper export transcribes the fox clip through transcribe()', async () => {
  if (
    !existsSync(path.resolve('.runtimes/venv/bin/python')) ||
    !existsSync(path.join(onnxWhisperDir, 'onnx/encoder_model_quantized.onnx')) ||
    !existsSync(path.join(onnxWhisperDir, 'onnx/decoder_model_merged_quantized.onnx')) ||
    !existsSync(path.resolve('models', foxClip))
  ) {
    console.log('  SKIP  no ONNX Whisper weights or fox clip on this machine');
    return;
  }
  const result = await transcribe({
    repo: 'onnx-community/whisper-tiny.en',
    file: 'onnx/decoder_model_merged_quantized.onnx',
    audio: foxClip,
    device: 'cpu',
  });
  assert.equal(result.ok, true, JSON.stringify(result.data));
  const text = String(result.data?.text ?? '');
  assert.match(text, /quick brown fox/i, text);
  assert.equal(result.data?.device, 'cpu');
  assert.ok(Array.isArray(result.data?.chunks), 'chunks returned');
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((failure) => `  - ${failure}`).join('\n'));
process.exit(failures.length ? 1 : 0);
