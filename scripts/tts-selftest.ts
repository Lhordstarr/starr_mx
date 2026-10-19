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

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);