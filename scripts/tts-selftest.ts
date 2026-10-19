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
import { existsSync, readFileSync } from 'node:fs';
import * as path from 'node:path';

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

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);