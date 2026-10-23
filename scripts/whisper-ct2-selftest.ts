#!/usr/bin/env tsx
/**
 * Faster Whisper has two ways to fail on a GPU box even though CTranslate2 can
 * see the card:
 *
 *   * the CUDA libraries live in the venv (torch ships cuBLAS and cuDNN) but
 *     are not on the dynamic loader's path, so the model raises "Library
 *     libcublas.so.12 is not found or cannot be loaded";
 *   * the loader path is fixed but the CUDA backend still cannot run, in which
 *     case the runner must fall back to CPU rather than fail the transcription.
 *
 * This suite runs the real runner against fake CTranslate2/faster-whisper
 * modules. One case provides the CUDA library layout and asserts the process
 * re-execs with it on LD_LIBRARY_PATH. The other omits it and asserts the
 * runner reports CPU and still returns the text.
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { venvPython } from '../src/runtimes.ts';

let passed = 0;
const failures: string[] = [];

function check(name: string, fn: () => void) {
  try {
    fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures.push(name);
    console.log(`  FAIL  ${name}${err instanceof Error ? ` — ${err.message}` : ''}`);
  }
}

const here = path.dirname(fileURLToPath(import.meta.url));
const SCRIPT = path.resolve(here, 'py', 'whisper_ct2.py');
const PY = existsSync(venvPython('venv')) ? venvPython('venv') : 'python3';

const FAKE_CT2 = `\
import os

_log = os.environ.get("CT2_TEST_LOG")
if _log:
    with open(_log, "a", encoding="utf-8") as fh:
        fh.write(os.environ.get("LD_LIBRARY_PATH", "") + "\\n")

def get_cuda_device_count():
    return 1

def get_supported_compute_types(device):
    return {"float16"}
`;

const FAKE_FASTER_WHISPER = `\
import os

class Segment:
    def __init__(self):
        self.text = " hello"
        self.start = 0.0
        self.end = 0.1

class Info:
    language = "en"
    language_probability = 1.0
    duration = 0.1

class WhisperModel:
    def __init__(self, model_dir, device, device_index, compute_type):
        if device == "cuda":
            expected = os.environ.get("CT2_TEST_LIB_DIR")
            paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if not expected or expected not in paths:
                raise RuntimeError(
                    "Library libcublas.so.12 is not found or cannot be loaded"
                )
        self.device = device

    def transcribe(self, samples, beam_size=1):
        if (
            self.device == "cuda"
            and os.environ.get("CT2_TEST_FAIL_TRANSCRIBE") == "1"
        ):
            raise RuntimeError(
                "Library libcublas.so.12 is not found or cannot be loaded"
            )
        return [Segment()], Info()
`;

const FAKE_NUMPY = `\
float32 = "float32"

def asarray(values, dtype=None):
    return list(values)
`;

/** A mono 16 kHz PCM WAV the standard-library decoder can read. */
function wavBytes(samples: number): Buffer {
  const bytes = Buffer.alloc(44 + samples * 2);
  bytes.write('RIFF', 0);
  bytes.writeUInt32LE(36 + samples * 2, 4);
  bytes.write('WAVE', 8);
  bytes.write('fmt ', 12);
  bytes.writeUInt32LE(16, 16);
  bytes.writeUInt16LE(1, 20);
  bytes.writeUInt16LE(1, 22);
  bytes.writeUInt32LE(16_000, 24);
  bytes.writeUInt32LE(32_000, 28);
  bytes.writeUInt16LE(2, 32);
  bytes.writeUInt16LE(16, 34);
  bytes.write('data', 36);
  bytes.writeUInt32LE(samples * 2, 40);
  return bytes;
}

function jsonLines(stdout: string): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  for (const line of stdout.split('\n')) {
    const text = line.trim();
    if (!text.startsWith('{')) continue;
    try {
      out.push(JSON.parse(text) as Record<string, unknown>);
    } catch {
      /* progress line */
    }
  }
  return out;
}

function runCase(withCudaLibs: boolean, failTranscribe = false, requested = 'cuda') {
  const root = mkdtempSync(path.join(tmpdir(), 'starr-whisper-ct2-'));
  const site = path.join(root, 'site-packages');
  const torchLib = path.join(site, 'torch', 'lib');
  const modelDir = path.join(root, 'model');
  const logPath = path.join(root, 'ct2.log');
  mkdirSync(site, { recursive: true });
  mkdirSync(modelDir, { recursive: true });
  writeFileSync(path.join(site, 'ctranslate2.py'), FAKE_CT2);
  writeFileSync(path.join(site, 'faster_whisper.py'), FAKE_FASTER_WHISPER);
  writeFileSync(path.join(site, 'numpy.py'), FAKE_NUMPY);
  if (withCudaLibs) {
    mkdirSync(torchLib, { recursive: true });
    writeFileSync(path.join(torchLib, 'libcublas.so.12'), '');
    writeFileSync(path.join(torchLib, 'libcudnn.so.9'), '');
  }
  const wav = path.join(root, 'clip.wav');
  writeFileSync(wav, wavBytes(1600));

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    PYTHONPATH: [site, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
    LD_LIBRARY_PATH: '/opt/example-lib',
    CT2_TEST_LOG: logPath,
  };
  if (withCudaLibs) env.CT2_TEST_LIB_DIR = torchLib;
  else delete env.CT2_TEST_LIB_DIR;
  if (failTranscribe) env.CT2_TEST_FAIL_TRANSCRIBE = '1';
  else delete env.CT2_TEST_FAIL_TRANSCRIBE;

  const run = spawnSync(PY, [SCRIPT, modelDir, wav, requested], {
    encoding: 'utf8',
    env,
    timeout: 30_000,
  });
  const stdout = run.stdout ?? '';
  const stderr = run.stderr ?? '';
  const lines = jsonLines(stdout);
  const deviceLines = lines.filter((line) => typeof line.device === 'string');
  const device = deviceLines[deviceLines.length - 1]?.device;
  const result = lines[lines.length - 1];
  const cudaLog = existsSync(logPath) ? readFileSync(logPath, 'utf8') : '';

  rmSync(root, { recursive: true, force: true });
  return { run, stdout, stderr, device, result, cudaLog, torchLib };
}

console.log('\n=== CUDA library path ===');

let withLibs: ReturnType<typeof runCase> | null = null;
check('a venv torch/lib is put on the loader path before CTranslate2 starts', () => {
  withLibs = runCase(true);
  assert.equal(withLibs.run.status, 0, withLibs.stderr);
  assert.equal(withLibs.device, 'cuda', withLibs.stdout + withLibs.stderr);
  assert.equal(withLibs.result?.text, 'hello');
  assert.match(withLibs.stderr, /adding CUDA libraries from/);
  // The fake CTranslate2 records what the loader will see at import time.
  // A Python assignment after start would leave this without torch/lib.
  assert.ok(withLibs.cudaLog.includes(withLibs.torchLib), withLibs.cudaLog);
  assert.ok(withLibs.cudaLog.includes('/opt/example-lib'), withLibs.cudaLog);
  assert.doesNotMatch(withLibs.stderr, /retrying on CPU/);
});

check('the server\'s "nvidia" device request reaches the CUDA backend', () => {
  const nvidia = runCase(true, false, 'nvidia');
  assert.equal(nvidia.run.status, 0, nvidia.stderr);
  assert.equal(nvidia.device, 'cuda', nvidia.stdout + nvidia.stderr);
  assert.equal(nvidia.result?.text, 'hello');
});

console.log('\n=== CPU fallback ===');

check('a missing cuBLAS is reported as CPU and still transcribes', () => {
  const without = runCase(false);
  assert.equal(without.run.status, 0, without.stderr);
  assert.equal(without.device, 'cpu', without.stdout + without.stderr);
  assert.equal(without.result?.text, 'hello');
  assert.match(without.stderr, /libcublas\.so\.12/);
  assert.match(without.stderr, /retrying on CPU/);
  assert.ok(!without.cudaLog.includes(path.join('torch', 'lib')), without.cudaLog);
});

check('CUDA that loads but fails on the first inference also retries on CPU', () => {
  const late = runCase(true, true);
  assert.equal(late.run.status, 0, late.stderr);
  assert.equal(late.device, 'cpu', late.stdout + late.stderr);
  assert.equal(late.result?.text, 'hello');
  assert.match(late.stderr, /CUDA could not transcribe/);
  assert.match(late.stderr, /retrying on CPU/);
});

if (failures.length) {
  console.log(`\n${failures.length} failed; ${passed} passed`);
  for (const name of failures) console.log(`  - ${name}`);
  process.exit(1);
}
console.log(`\n${passed} passed`);
