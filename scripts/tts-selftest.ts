#!/usr/bin/env tsx
/**
 * Speech-to-text and speech device honesty.
 *
 * Kokoro ran on the CPU on every machine while its process row said GPU, and
 * was slow for that reason. The cause is a quiet one: `KPipeline` accepts a
 * `device`, but only honours it on the branch where it constructs the model
 * itself. Given a `KModel` instance — which `speak.py` has to pass, so it can
 * point at downloaded weights — it takes it verbatim and never moves it. The
 * script resolved a device, reported it, and then never used it.
 *
 * So the check here is not "does it run on the GPU", which cannot be answered
 * on a machine without a usable card. It is the invariant that makes the lie
 * impossible: **the device the script reports is read off the model, and equals
 * the device the model's parameters are actually on.** Revert the fix and that
 * stops being true on any machine with a GPU; leave it in and it is true
 * everywhere, including on the machine running the suite.
 */
import { spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';

import { RUNTIME_GROUPS, installRuntimes, runtimeStatus, venvPython } from '../src/runtimes.ts';
import { speak as speakWith } from '../src/infer.ts';

let passed = 0;
const failures: string[] = [];
function report(name: string, ok: boolean, detail = '') {
  if (ok) {
    passed += 1;
    console.log(`  PASS  ${name}`);
  } else {
    failures.push(name);
    console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

const speak = readFileSync(path.resolve('scripts/py/speak.py'), 'utf8');

console.log('Kokoro must be moved onto the device it claims');
report(
  'the model is moved with .to() before the pipeline is built',
  /\.to\(\s*kind\s*\)/.test(speak) && /model\s*=\s*model\.to\(/.test(speak),
  'KPipeline takes a KModel verbatim and never moves it',
);
report(
  'the device reported is read off the model, not off the request',
  /landed\s*=\s*next\(model\.parameters\(\)\)\.device\.type/.test(speak) &&
    /json\.dumps\(\{\s*"device":\s*landed\s*\}\)/.test(speak),
  'reporting the request is how a CPU run gets labelled GPU',
);
report(
  'the old spelling — reporting what was asked for — is gone',
  !/vendor_name\(\)\s*if\s+use_gpu\s+else\s+["']cpu["']/.test(speak),
);
report(
  'the pipeline is told the device too, for its own components',
  /KPipeline\([^)]*device=kind/.test(speak),
);
report(
  'the model is put in eval mode as well as on the device',
  /\.eval\(\)/.test(speak),
  'a training-mode dropout would change the output, not just slow it',
);

console.log('\nAnd the device it reports must be the one it is on');
{
  // Only runs where the Kokoro venv and its weights are actually present; the
  // reasoning above does not need them, but this is the part that would catch a
  // report lying on a machine that has a card.
  const py = path.resolve('.runtimes/venv-kokoro/bin/python');
  const modelDir = path.resolve('models/hexgrad--Kokoro-82M');
  if (!existsSync(py) || !existsSync(path.join(modelDir, 'kokoro-v1_0.pth'))) {
    console.log('  SKIP  no Kokoro venv or weights on this machine');
  } else {
    const out = spawnSync(
      py,
      [
        path.resolve('scripts/py/speak.py'),
        modelDir,
        'af_heart',
        path.resolve('.runtimes/out/kokoro-selftest.wav'),
        'auto',
        '1.0',
        'hexgrad/Kokoro-82M',
      ],
      { input: 'The quick brown fox jumps over the lazy dog.', encoding: 'utf8' },
    );
    let reported: string | null = null;
    for (const line of String(out.stdout).split('\n')) {
      const t = line.trim();
      if (!t.startsWith('{')) continue;
      try {
        const doc = JSON.parse(t) as { device?: string };
        if (typeof doc.device === 'string') reported = doc.device;
      } catch {
        /* a progress line */
      }
    }
    report('the runner reported a device at all', reported !== null, String(out.stdout).slice(-200));
    // What torch would answer for the same interpreter: the yardstick the row
    // is measured against.
    const probe = spawnSync(py, ['-c', 'import torch;print(torch.cuda.is_available())'], {
      encoding: 'utf8',
    });
    const cudaAvailable = String(probe.stdout).trim() === 'true';
    report(
      'and it matches what this machine can actually do',
      reported === (cudaAvailable ? 'cuda' : 'cpu'),
      `reported ${reported}, torch says cuda is ${cudaAvailable ? 'available' : 'unavailable'}`,
    );
    report('it also produced audio', existsSync(path.resolve('.runtimes/out/kokoro-selftest.wav')));
  }
}

console.log('\nAn environment whose PyTorch is a CPU-only build must be repairable');
{
  // The state a machine ends up in when the runtimes are installed before the
  // GPU driver: Compute defaults to CPU until a usable card is seen, so torch
  // arrives from the CPU index and `torch.cuda` does not exist in the venv at
  // all. Reinstalling used to change nothing — pip answers "Requirement already
  // satisfied" for a wheel it cannot tell apart by version, so the local tag
  // that says `+cpu` was the one thing pip would never revisit.
  //
  // Driven against a real `installRuntimes`, a real probe and a real pip call,
  // with a stand-in interpreter that reports a CPU-only torch and records what
  // it was asked to install.
  const home = mkdtempSync(path.join(tmpdir(), 'starr-tts-'));
  const root = path.join(home, '.runtimes');
  const log = path.join(home, 'pip-calls.log');

  const saved = { dir: process.env.RUNTIMES_DIR, py: process.env.RUNTIMES_PYTHON };
  process.env.RUNTIMES_DIR = root;
  process.env.RUNTIMES_PYTHON = '';
  // The app's own layout, asked for rather than guessed, so a change to where
  // venvs live cannot leave this fixture passing against nothing.
  const fake = venvPython('venv-kokoro');
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
      `      echo "$@" >> ${JSON.stringify(log)}`,
      '      echo "Successfully installed"; exit 0; fi',
      '    exit 0 ;;',
      '  -c)',
      // One line: the probe reads the last line of stdout and parses it, so a
      // pretty-printed answer is silently a failed probe.
      '    echo \'{"packages": {"kokoro": true}, "torch": {"version": "2.14.1+cpu", "cuda": false, "count": 0, "name": null, "cuda_version": null}}\'',
      '    exit 0 ;;',
      'esac',
      'exit 0',
      '',
    ].join('\n'),
  );
  chmodSync(fake, 0o755);

  const pipArgs = (): string[] => (existsSync(log) ? readFileSync(log, 'utf8').split('\n') : []);
  const torchCall = (): string => pipArgs().find((l) => /\btorch\b/.test(l) && /--index-url/.test(l)) ?? '';

  const kokoro = RUNTIME_GROUPS.find((g) => g.id === 'kokoro')!;
  const lines: string[] = [];

  try {
    await installRuntimes({
      groups: [kokoro],
      // Forced rather than detected, so the GPU index is reached on a machine
      // with no card — which is what makes this runnable in the suite.
      device: 'nvidia',
      onProgress: (l) => lines.push(l),
      onNeedsPrivilege: () => {},
    });

    const call = torchCall();
    report(
      'installing with Compute on GPU asks pip to upgrade torch',
      /--upgrade/.test(call),
      call || '(no torch pip call recorded)',
    );
    report(
      'and it asks the CUDA index for it, not the CPU one',
      /download\.pytorch\.org\/whl\/cu128/.test(call),
      call,
    );
    report(
      'it says out loud that a CPU-only build is being replaced',
      lines.some((l) => /CPU-only build/.test(l) && /Replacing/.test(l)),
      lines.filter((l) => /CPU-only/.test(l)).join(' | ') || '(nothing said)',
    );

    // And the status names each environment's torch, rather than reporting one
    // environment's answer as the machine's.
    const status = await runtimeStatus(false);
    const torches = status.torches ?? [];
    const entry = torches.find((t) => t.venv === 'venv-kokoro');
    report(
      'the status reports Kokoro’s own environment separately',
      entry?.label === 'Kokoro',
      JSON.stringify(torches),
    );
    report(
      'with the CPU-only build it really has',
      entry?.cudaAvailable === false && entry?.version === '2.14.1+cpu',
      JSON.stringify(entry),
    );

    // The flag must not be unconditional: a CPU install re-downloading a
    // gigabyte of identical wheels every time is its own bug.
    writeFileSync(log, '');
    lines.length = 0;
    await installRuntimes({
      groups: [kokoro],
      device: 'cpu',
      onProgress: (l) => lines.push(l),
      onNeedsPrivilege: () => {},
    });
    const cpuCall = torchCall();
    report(
      'a CPU install does not pass --upgrade',
      !/--upgrade/.test(cpuCall) && /whl\/cpu/.test(cpuCall),
      cpuCall || '(no torch pip call recorded)',
    );
  } finally {
    if (saved.dir === undefined) delete process.env.RUNTIMES_DIR;
    else process.env.RUNTIMES_DIR = saved.dir;
    if (saved.py === undefined) delete process.env.RUNTIMES_PYTHON;
    else process.env.RUNTIMES_PYTHON = saved.py;
  }
}

console.log('\nSpeech has more than one runner, and the weights must choose between them');
{
  // Kokoro is a styletts2 checkpoint that only the `kokoro` package loads;
  // every other speech model the app offers is a transformers pipeline. The two
  // live in different Python environments, so choosing wrong is not a worse
  // answer, it is "No module named 'kokoro'" or "no pipeline for this model" —
  // and before the generic runner existed the second case could not run at all.
  //
  // Driven against the real speak() and the real venv resolution, with stand-in
  // interpreters that record which one ran and what it was asked to run.
  const home = mkdtempSync(path.join(tmpdir(), 'starr-tts-dispatch-'));
  const models = path.join(home, 'models');
  const runtimes = path.join(home, 'runtimes');
  const out = path.join(home, 'out');
  const log = path.join(home, 'interp.log');

  const seedModel = (repo: string, files: Record<string, string>): void => {
    for (const [name, body] of Object.entries(files)) {
      const full = path.join(models, repo.replace('/', '--'), name);
      mkdirSync(path.dirname(full), { recursive: true });
      writeFileSync(full, body);
    }
  };
  // A VITS model: the shape every MMS-TTS checkpoint has.
  seedModel('Acme/vits-tts', {
    'model.safetensors': 'x',
    'config.json': JSON.stringify({ model_type: 'vits', architectures: ['VitsModel'] }),
  });
  // Kokoro: a .pth checkpoint beside its voices/, and a config that names no
  // transformers architecture at all.
  seedModel('Acme/kokoro-82m', {
    'kokoro-v1_0.pth': 'x',
    'config.json': JSON.stringify({ istftnet: {}, vocab: {} }),
    'voices/af_heart.pt': 'x',
  });

  const interpreter = (venv: string): void => {
    const py = path.join(runtimes, venv, 'bin', 'python');
    mkdirSync(path.dirname(py), { recursive: true });
    writeFileSync(
      py,
      [
        '#!/bin/bash',
        `echo "$0 $*" >> ${JSON.stringify(log)}`,
        `echo '{"path": ${JSON.stringify(path.join(out, 'said.wav'))}, "samples": 10, "seconds": 0.5, "device": "cpu"}'`,
        '',
      ].join('\n'),
    );
    chmodSync(py, 0o755);
  };
  interpreter('venv');
  interpreter('venv-kokoro');

  const saved = {
    models: process.env.HF_MODELS_DIR,
    dir: process.env.RUNTIMES_DIR,
    py: process.env.RUNTIMES_PYTHON,
    out: process.env.RUNTIME_OUT_DIR,
  };
  process.env.HF_MODELS_DIR = models;
  process.env.RUNTIMES_DIR = runtimes;
  process.env.RUNTIMES_PYTHON = '';
  process.env.RUNTIME_OUT_DIR = out;

  const calls = (): string[] => (existsSync(log) ? readFileSync(log, 'utf8').trim().split('\n').filter(Boolean) : []);

  try {
    await speakWith({
      repo: 'Acme/vits-tts',
      file: 'model.safetensors',
      voice: 'af_heart',
      speed: 1,
      device: 'cpu',
      text: 'Hello there.',
    });
    await speakWith({
      repo: 'Acme/kokoro-82m',
      file: 'kokoro-v1_0.pth',
      voice: 'af_heart',
      speed: 1,
      device: 'cpu',
      text: 'Hello there.',
    });

    const seen = calls();
    const vits = seen.find((l) => l.includes('Acme--vits-tts')) ?? '';
    const kokoro = seen.find((l) => l.includes('Acme--kokoro-82m')) ?? '';

    report('a VITS model is spoken by the generic runner', /speak_hf\.py/.test(vits), vits || '(nothing ran)');
    report(
      'and in the shared environment, where transformers is',
      vits.includes(path.join(runtimes, 'venv', 'bin', 'python')) && !vits.includes('venv-kokoro'),
      vits || '(nothing ran)',
    );
    report(
      'Kokoro still runs its own script in its own environment',
      kokoro.includes(path.join(runtimes, 'venv-kokoro', 'bin', 'python')) && /speak\.py/.test(kokoro),
      kokoro || '(nothing ran)',
    );
    report(
      'the generic runner is not asked to load a Kokoro checkpoint',
      !/speak_hf\.py/.test(kokoro) && !/speak\.py/.test(vits.replace('speak_hf.py', '')),
      `${vits} | ${kokoro}`,
    );

    // A runner that dies without ever reading its input must not take the
    // process with it. The text is written to the child's stdin, and when the
    // child is already gone that write fails with EPIPE — an 'error' event on a
    // stream nobody is listening to, which in Node is an unhandled exception.
    // The whole server exits, and the runner's stderr, the only place that says
    // why it died, is never read. A payload larger than the pipe buffer makes
    // the race deterministic instead of occasional: the write cannot finish
    // into the buffer, so it is still pending when the child is gone.
    const dead = path.join(runtimes, 'venv', 'bin', 'python');
    writeFileSync(
      dead,
      [
        '#!/bin/bash',
        '# Never reads its stdin: it reports why it is dying and is gone.',
        `echo '{"error": "the runner died before it read its input"}'`,
        'exit 3',
        '',
      ].join('\n'),
    );
    chmodSync(dead, 0o755);
    let died: { ok: boolean; log: string } | null = null;
    try {
      const gone = await speakWith({
        repo: 'Acme/vits-tts',
        file: 'model.safetensors',
        voice: 'af_heart',
        speed: 1,
        device: 'cpu',
        text: 'x'.repeat(1_000_000),
      });
      died = { ok: gone.result.ok, log: String(gone.result.log ?? '') };
    } catch (e) {
      report(
        'a runner that dies without reading its input fails, and does not crash the process',
        false,
        (e as Error).message,
      );
    }
    if (died) {
      report(
        'a runner that dies without reading its input fails, and does not crash the process',
        died.ok === false,
        died.ok
          ? 'it reported success with nothing spoken'
          : `ok=false, log: ${died.log.trim().split('\n')[0] || '(nothing on stderr)'}`,
      );
    }
    interpreter('venv');
  } finally {
    if (saved.models === undefined) delete process.env.HF_MODELS_DIR;
    else process.env.HF_MODELS_DIR = saved.models;
    if (saved.dir === undefined) delete process.env.RUNTIMES_DIR;
    else process.env.RUNTIMES_DIR = saved.dir;
    if (saved.py === undefined) delete process.env.RUNTIMES_PYTHON;
    else process.env.RUNTIMES_PYTHON = saved.py;
    if (saved.out === undefined) delete process.env.RUNTIME_OUT_DIR;
    else process.env.RUNTIME_OUT_DIR = saved.out;
  }
}

console.log('\nThe generic runner must survive text longer than an utterance');
{
  const generic = readFileSync(path.resolve('scripts/py/speak_hf.py'), 'utf8');
  report(
    'long text is split into utterance-sized pieces',
    /def split_text\(/.test(generic) && /split_text\(text\)/.test(generic),
    'VITS predicts a duration for the whole input; 20k characters is outside it',
  );
  report(
    'the device reported is read off the model, not off the request',
    /landed\s*=\s*next\(pipe\.model\.parameters\(\)\)\.device\.type/.test(generic) &&
      /json\.dumps\(\{\s*"device":\s*landed\s*\}\)/.test(generic),
    'reporting the request is how a CPU run gets labelled GPU',
  );
  report(
    'the voice and speed it cannot honour are said out loud, on stderr',
    /do not apply/.test(generic) && /file=sys\.stderr/.test(generic),
    'a setting that does nothing and says nothing reads as a broken feature',
  );
  report(
    'the wav is written with the standard library, not a new dependency',
    !/import soundfile/.test(generic) && /import wave/.test(generic),
  );

  // And it must actually speak, through the real interpreter and real weights.
  const py = path.resolve('.runtimes/venv/bin/python');
  const modelDir = path.resolve('models/facebook--mms-tts-eng');
  const wav = path.resolve('.runtimes/out/mms-tts-selftest.wav');
  if (!existsSync(py) || !existsSync(path.join(modelDir, 'model.safetensors'))) {
    console.log('  SKIP  no shared venv or MMS-TTS weights on this machine');
  } else {
    const run = spawnSync(
      py,
      [path.resolve('scripts/py/speak_hf.py'), modelDir, 'af_heart', wav, 'cpu', '1', 'facebook/mms-tts-eng'],
      {
        // Two sentences: one piece each, so the split is exercised rather than
        // assumed, and the answer says how many pieces it made.
        input: 'The quick brown fox jumps over the lazy dog. This second sentence proves the text is split.',
        encoding: 'utf8',
      },
    );
    const lines = String(run.stdout ?? '')
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    let chunks = 0;
    let answer: { seconds?: unknown; error?: unknown } | null = null;
    for (const line of lines) {
      if (!line.startsWith('{')) continue;
      try {
        const doc = JSON.parse(line) as { chunks?: unknown; seconds?: unknown; error?: unknown };
        if (typeof doc.chunks === 'number') chunks = doc.chunks;
        if ('seconds' in doc || 'error' in doc) answer = doc;
      } catch {
        /* progress chatter */
      }
    }
    report('it exits cleanly', run.status === 0, `exit ${run.status}`);
    report(
      'it returns audio rather than an error',
      answer !== null && typeof answer.seconds === 'number' && !('error' in answer!),
      JSON.stringify(answer) || lines.slice(-2).join(' '),
    );
    report('it produced a file', existsSync(wav));
    report('two sentences became two pieces', chunks === 2, `reported ${chunks}`);
  }
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);