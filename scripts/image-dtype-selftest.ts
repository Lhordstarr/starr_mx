#!/usr/bin/env tsx
/**
 * The image runner chooses a CPU dtype the machine can actually run in.
 *
 * Every image pipeline used to load in bfloat16 on CPU to save memory. On a
 * CPU without AVX512-BF16 or AMX, PyTorch has no bf16 kernels and converts
 * for every operation — measured here at 0.3 GFLOPS against 55 for plain
 * fp32. One 64x64 step took four and a half minutes, and a 256x256, one-step
 * image was still running after twenty. That is not "slow on CPU", it is a
 * broken choice of dtype.
 *
 * The policy is now: native bf16, or fp32 when the machine has the room for
 * it, or bf16 only where fp32 would not fit at all. This suite asks the
 * runner for its answer and compares it with the machine facts read
 * independently, so a future edit that hard-codes one dtype fails here.
 *
 * Needs the Python environment (torch is imported by the runner); skipped when
 * there is none, like the other local-runner suites.
 */
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import * as path from 'node:path';

import { venvPython } from '../src/runtimes.ts';

let passed = 0;
const failures: string[] = [];
function check(name: string, fn: () => unknown): void {
  try {
    fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err: unknown) {
    failures.push(name);
    console.log(`  FAIL  ${name} — ${(err as Error).message}`);
  }
}

const python = venvPython();
const runner = path.resolve('scripts/py/generate.py');
if (!existsSync(python) || !existsSync(runner)) {
  console.log('SKIP — no Python environment or generate.py');
  process.exit(0);
}

/* The runner's own answer, plus the machine facts it is supposed to be
   derived from — read by the same script, but reported separately so the
   comparison below is not the runner marking its own homework. */
const probe = `
import importlib.util, json
spec = importlib.util.spec_from_file_location("generate", ${JSON.stringify(runner)})
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
dtype, why = gen.cpu_dtype()
print(json.dumps({
    "native": gen.cpu_has_native_bf16(),
    "total": gen.physical_ram_bytes(),
    "dtype": str(dtype).replace("torch.", ""),
    "why": why,
}))
`;
const out = spawnSync(python, ['-c', probe], { encoding: 'utf8', timeout: 180_000 });
if (out.status !== 0) {
  console.log('SKIP — the runner could not be imported');
  console.log((out.stderr ?? '').trim().split('\n').slice(-3).join('\n'));
  process.exit(0);
}
const report = JSON.parse(out.stdout.trim().split('\n').at(-1) ?? '{}') as {
  native?: boolean;
  total?: number;
  dtype?: string;
  why?: string;
};

const GiB = 1024 ** 3;
const enoughRam = (report.total ?? 0) >= 7 * GiB;

console.log('\nCPU dtype');
check('the runner reports a dtype and a sentence for it', () => {
  if (report.dtype !== 'float32' && report.dtype !== 'bfloat16') {
    throw new Error(`unexpected dtype ${String(report.dtype)}`);
  }
  if (!report.why || report.why.length < 15) throw new Error(`no reason: ${String(report.why)}`);
});
check('it reads whether this CPU runs bf16 natively', () => {
  if (typeof report.native !== 'boolean') throw new Error(`native is ${String(report.native)}`);
  if (typeof report.total !== 'number') throw new Error(`total RAM is ${String(report.total)}`);
});
check('a CPU without native bf16 and with the RAM for fp32 gets fp32', () => {
  if (report.native || !enoughRam) {
    console.log(`        (this machine: native=${report.native}, ${((report.total ?? 0) / GiB).toFixed(1)} GiB)`);
    return;
  }
  if (report.dtype !== 'float32') {
    throw new Error(`picked ${String(report.dtype)} on a ${((report.total ?? 0) / GiB).toFixed(1)} GiB CPU without bf16 kernels`);
  }
});
check('bf16 is kept only where it is native or the only build that fits', () => {
  if (!report.native && enoughRam) {
    console.log(`        (this machine runs fp32: ${report.why})`);
    return;
  }
  if (report.dtype !== 'bfloat16') {
    throw new Error(`picked ${String(report.dtype)} where bf16 is the supported choice`);
  }
});

console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length ? 1 : 0);
