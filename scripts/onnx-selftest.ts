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
import { spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';

import { CATALOG, ONNX_WHISPER_SIDECARS } from '../src/model-catalog.ts';
import { runnableModels } from '../src/runnable.ts';
import { speak, transcribe } from '../src/infer.ts';
import { SCRIPT_RUNTIMES } from '../src/processes.ts';
import { RUNTIME_GROUPS, installRuntimes, venvPython } from '../src/runtimes.ts';
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
await check('Kokoro ONNX lists one graph file, and it is the q4 export', () => {
  const entry = catalogEntry('onnx-community/Kokoro-82M-v1.0-ONNX');
  assert.ok(entry, 'Kokoro ONNX entry exists');
  const graphs = entry.files.filter((file) => /\.onnx$/i.test(file.name));
  assert.equal(graphs.length, 1, `found ${graphs.map((f) => f.name).join(', ')}`);
  // q4f16 and fp16 compute on the CPU but return NaNs under the CUDA
  // provider, which produced a valid-looking silent WAV on an NVIDIA box.
  // The bundle therefore ships the q4 graph (fp32 activations), which was
  // verified on both devices.
  assert.equal(graphs[0]?.name, 'onnx/model_q4.onnx');
  assert.doesNotMatch(graphs[0]?.name ?? '', /f16/i);
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
  assert.equal(model.file, 'onnx/model_q4.onnx');
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

  await check('an old bundle with only the fp16 graph is still claimed for TTS', async () => {
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

  await check('a bundle with both graphs picks the q4 export, not q4f16', async () => {
    const repo = 'Acme/onnx-kokoro-mixed';
    const downloads = await fixture(repo, {
      'config.json': JSON.stringify({ model_type: 'style_text_to_speech_2' }),
      'onnx/model_q4.onnx': 'graph',
      'onnx/model_q4f16.onnx': 'graph',
      'voices/af_heart.bin': 'voice',
      'tokenizer.json': '{"model":{"vocab":{}}}',
    });
    const models = await runnableModels('tts', { downloads, installed });
    const model = models.find((m) => m.repo === repo);
    assert.ok(model, 'ONNX Kokoro discovered');
    assert.equal(model.ready, true, model.reason);
    // The old preference order returned q4f16 here, which is exactly the
    // file that runs silently on a CUDA machine.
    assert.equal(model.file, 'onnx/model_q4.onnx');
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
  // The call has to happen before the file is written: libsndfile happily
  // turns a NaN array into a valid silent WAV.
  const guarded = source.indexOf('check_audio(merged');
  assert.ok(guarded > 0, 'render never calls check_audio');
  assert.ok(guarded < source.indexOf('sf.write(out_path'), 'check_audio runs after sf.write');
});
await check('the ONNX runner refuses to write a silent WAV', () => {
  // fp16 graphs under the CUDA provider return a full-length array of NaNs;
  // the runner used to write it out and report success. This drives the guard
  // directly, with no numpy or model download, so it runs everywhere.
  const local = readFileSync(speakSource, 'utf8');
  assert.match(local, /def check_audio\(/);
  const script = `
import importlib.util, sys
spec = importlib.util.spec_from_file_location('speak_onnx', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for bad in ([float('nan')] * 64, [float('inf')] * 64, [0.0] * 64, [-1.0] * 64):
    try:
        module.check_audio(bad, 'cuda', 'onnx/model_q4f16.onnx')
    except RuntimeError as exc:
        message = str(exc)
        assert 'did not compute' in message, message
        assert 'q4' in message, message
    else:
        raise SystemExit('check_audio accepted a silent signal')
module.check_audio([0.0, 0.25, -0.25, 0.0], 'cpu', 'onnx/model_q4.onnx')
print('guard ok')
`;
  const run = spawnSync('python3', ['-c', script, speakSource], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);
  assert.match(run.stdout, /guard ok/);
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

console.log('\nThe ONNX runners have to be able to use the card');
await check('the ONNX runners ask for CUDA when the device is not the CPU', () => {
  for (const file of ['speak_onnx.py', 'transcribe_onnx.py']) {
    const source = readFileSync(path.resolve('scripts/py', file), 'utf8');
    assert.match(source, /CUDAExecutionProvider/, `${file} never asks for CUDA`);
    assert.match(source, /ROCMExecutionProvider/, `${file} never asks for ROCm`);
    // The request is not the answer: the provider reported is the one the
    // session actually got.
    assert.match(
      source,
      /provider_label\((?:enc_)?sess\.get_providers\(\)\)/,
      `${file} reports the request`,
    );
  }
});
await check('they load torch first, and preload the CUDA libraries', () => {
  for (const file of ['speak_onnx.py', 'transcribe_onnx.py']) {
    const source = readFileSync(path.resolve('scripts/py', file), 'utf8');
    assert.match(source, /import torch/, `${file} does not import torch`);
    assert.match(source, /preload_dlls/, `${file} does not preload the CUDA libraries`);
    // torch has to be imported before the runtime: the CUDA libraries are
    // loaded when torch is, and a runtime that looks for them before that
    // finds nothing.
    const torchAt = source.indexOf('import torch');
    const runtimeAt = source.indexOf('import onnxruntime');
    assert.ok(torchAt >= 0 && runtimeAt > torchAt, `${file} imports the runtime before torch`);
  }
});
await check('and a GPU that will not start falls back to the CPU, and says so', () => {
  for (const file of ['speak_onnx.py', 'transcribe_onnx.py']) {
    const source = readFileSync(path.resolve('scripts/py', file), 'utf8');
    assert.match(source, /def open_session\(/, `${file} has no session helper`);
    assert.match(source, /"stage": "fallback"/, `${file} does not say it fell back`);
    assert.match(
      source,
      /InferenceSession\(model_path, providers=\["CPUExecutionProvider"\]\)/,
      `${file} does not fall back to the CPU`,
    );
  }
});

console.log('\nThe installer has to bring the build the machine needs');
{
  // Driven against a real installRuntimes and a stand-in interpreter that
  // records what pip was asked to do, because the choice of wheel is the whole
  // of this: the CPU wheel and the GPU wheel import under the same name, so
  // getting it wrong means ONNX silently stays on the processor.
  const home = mkdtempSync(path.join(tmpdir(), 'starr-onnx-gpu-'));
  const root = path.join(home, '.runtimes');
  const log = path.join(home, 'pip-calls.log');
  const saved = { dir: process.env.RUNTIMES_DIR, py: process.env.RUNTIMES_PYTHON };
  process.env.RUNTIMES_DIR = root;
  const fake = venvPython('venv');
  process.env.RUNTIMES_PYTHON = fake;
  mkdirSync(path.dirname(fake), { recursive: true });
  writeFileSync(
    fake,
    [
      '#!/bin/bash',
      'case "$1" in',
      '  -m)',
      '    if [ "$2" = "pip" ] && [ "$3" = "--version" ]; then',
      '      echo "pip 24.0 from /fake/lib (python 3.12)"; exit 0; fi',
      '    if [ "$2" = "pip" ] && [ "$3" = "install" ]; then',
      `      echo "install $@" >> ${JSON.stringify(log)}`,
      '      echo "Successfully installed"; exit 0; fi',
      '    if [ "$2" = "pip" ] && [ "$3" = "uninstall" ]; then',
      `      echo "uninstall $@" >> ${JSON.stringify(log)}`,
      '      exit 0; fi',
      '    exit 0 ;;',
      '  -c)',
      '    echo \'{"packages": {"onnxruntime": true}, "torch": null, "onnx_providers": ["CPUExecutionProvider"]}\'',
      '    exit 0 ;;',
      'esac',
      'exit 0',
      '',
    ].join('\n'),
  );
  chmodSync(fake, 0o755);

  const group = RUNTIME_GROUPS.find((g) => g.id === 'onnxruntime');
  const calls = (): string => (existsSync(log) ? readFileSync(log, 'utf8') : '');

  try {
    await check('an NVIDIA machine gets the GPU build', async () => {
      assert.ok(group, 'the ONNX Runtime group exists');
      writeFileSync(log, '');
      await installRuntimes({
        groups: [group!],
        device: 'nvidia',
        onProgress: () => {},
        onNeedsPrivilege: () => {},
      });
      const said = calls();
      assert.match(said, /^install .*onnxruntime-gpu/m, said || '(pip was not asked for anything)');
      // And the build it is replacing goes first: the two install the same
      // import package, and left in place they overwrite each other.
      assert.match(said, /^uninstall .*onnxruntime\b/m, said || '(nothing was uninstalled)');
    });

    await check('a machine without a card gets the CPU build', async () => {
      writeFileSync(log, '');
      await installRuntimes({
        groups: [group!],
        device: 'cpu',
        onProgress: () => {},
        onNeedsPrivilege: () => {},
      });
      const said = calls();
      // The lines are prefixed, because "uninstall" contains "install" and a
      // looser match would read the removal of the GPU build as its install.
      assert.match(said, /^install .*onnxruntime\b(?!-gpu)/m, said || '(pip was not asked for anything)');
      assert.doesNotMatch(said, /^install .*onnxruntime-gpu/m, said);
      // And the build it is replacing goes first.
      assert.match(said, /^uninstall .*onnxruntime-gpu/m, said || '(nothing was uninstalled)');
    });

    await check('and the GPU wheel is the one that matches the CUDA build torch uses', () => {
      const gpu = group?.deviceInstall?.('nvidia', {
        vendor: 'nvidia',
        detail: '',
        needsGroup: false,
      });
      assert.ok(gpu, 'the group resolves a device-specific install');
      assert.equal(gpu.pkgs.length, 1);
      assert.match(gpu.pkgs[0], /^onnxruntime-gpu==/);
      // The wheel has to be a CUDA 12 build, because that is the CUDA the
      // torch build comes from — a CUDA 13 runtime does not run on a CUDA 12
      // driver.
      const version = gpu.pkgs[0].split('==')[1] ?? '';
      const [major, minor] = version.split('.').map((n) => Number(n));
      assert.ok(major < 27, `${version} is a CUDA 13 build`);
      assert.ok(Number.isFinite(minor));
    });

    await check('the size the page shows is the one that will be fetched', () => {
      const gpu = group?.deviceInstall?.('nvidia', {
        vendor: 'nvidia',
        detail: '',
        needsGroup: false,
      });
      const cpu = group?.deviceInstall?.('cpu', { vendor: null, detail: '', needsGroup: false });
      assert.ok(gpu && cpu);
      assert.ok(gpu.approxMB > cpu.approxMB, 'the GPU wheel is the bigger download');
      assert.ok(cpu.approxMB > 0);
    });
  } finally {
    if (saved.dir === undefined) delete process.env.RUNTIMES_DIR;
    else process.env.RUNTIMES_DIR = saved.dir;
    if (saved.py === undefined) delete process.env.RUNTIMES_PYTHON;
    else process.env.RUNTIMES_PYTHON = saved.py;
  }
}

const onnxKokoroDir = path.resolve('models/onnx-community--Kokoro-82M-v1.0-ONNX');
const onnxWhisperDir = path.resolve('models/onnx-community--whisper-tiny.en');
const foxClip = '_uploads/clip-1792421379226.wav';

// The current bundle's graph, or the fp16 one an older bundle left behind.
const onnxKokoroGraph = ['onnx/model_q4.onnx', 'onnx/model_q4f16.onnx'].find((file) =>
  existsSync(path.join(onnxKokoroDir, file)),
);

await check('the real Kokoro ONNX export synthesises audio through speak()', async () => {
  if (
    !existsSync(path.resolve('.runtimes/venv-kokoro/bin/python')) ||
    !onnxKokoroGraph ||
    !existsSync(path.join(onnxKokoroDir, 'voices/af_heart.bin'))
  ) {
    console.log('  SKIP  no Kokoro ONNX venv/weights on this machine');
    return;
  }
  const { result, outPath } = await speak({
    repo: 'onnx-community/Kokoro-82M-v1.0-ONNX',
    file: onnxKokoroGraph,
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
  // A silent WAV has the same sample count and passes the size check, so
  // count the values that actually differ.
  const audible = spawnSync(
    'python3',
    [
      '-c',
      'import array,sys,wave; w=wave.open(sys.argv[1]); a=array.array("h", w.readframes(w.getnframes())); print(len(set(a)))',
      outPath,
    ],
    { encoding: 'utf8' },
  );
  assert.equal(audible.status, 0, audible.stderr);
  assert.ok(Number(audible.stdout.trim()) > 100, `only ${audible.stdout.trim()} distinct samples`);
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
