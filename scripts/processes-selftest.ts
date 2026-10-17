/**
 * Self-test for the process registry behind the resource manager.
 *
 * Run with: npm run test:processes
 *
 * The thing worth proving is that a job this app spawned can actually be
 * stopped, and that the registry cannot be talked into stopping something it
 * did not start. Both matter: a stuck image generation can hold a core for an
 * hour, and this server has no login, so the registry is the whole boundary.
 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';

import {
  attachChild,
  finishProcess,
  listProcesses,
  processMemoryBytes,
  registerProcess,
  setProcessDevice,
  stopAllProcesses,
  stopProcess,
} from '../src/processes.js';

let passed = 0;
const failures: string[] = [];

async function test(name: string, fn: () => Promise<void> | void) {
  try {
    await fn();
    passed += 1;
    console.log(`  PASS ${name}`);
  } catch (err) {
    failures.push(name);
    console.log(`  FAIL ${name}\n        ${(err as Error).message}`);
  }
}

/** Waits for `check` to hold, or gives up. */
async function until(check: () => boolean, ms = 5000): Promise<void> {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (check()) return;
    await sleep(25);
  }
  throw new Error('condition not reached in time');
}

/** A child that outlives a single test if nothing stops it. */
function sleeper(seconds: number) {
  const child = spawn(process.execPath, ['-e', `setTimeout(()=>{}, ${seconds * 1000})`], {
    stdio: 'ignore',
  });
  return child;
}

console.log('\n=== registry ===');

await test('a registered job is listed as running', () => {
  const job = registerProcess({ kind: 'image', script: 'generate.py', pid: 4242, device: 'cpu' });
  const { active } = listProcesses();
  assert.ok(active.some((p) => p.id === job.id), 'job missing from the active list');
  assert.equal(job.state, 'running');
  assert.ok(job.runtimes.includes('diffusers'), 'generate.py should report diffusers');
});

await test('each job gets a distinct id, never the pid', () => {
  const a = registerProcess({ kind: 'stt', script: 'whisper_ct2.py', pid: 1111, device: 'cpu' });
  const b = registerProcess({ kind: 'stt', script: 'whisper_ct2.py', pid: 1111, device: 'cpu' });
  assert.notEqual(a.id, b.id);
  assert.notEqual(a.id, String(a.pid), 'the id must not be the pid');
  assert.notEqual(a.id, '1111');
});

await test('runtimes come from what the script imports', () => {
  // whisper_ct2.py is the whole reason CTranslate2 is offered: it must not
  // claim torch, which is the 529MB alternative.
  const ct2 = registerProcess({ kind: 'stt', script: 'whisper_ct2.py', pid: 1, device: 'cpu' });
  assert.deepEqual(ct2.runtimes, ['faster-whisper']);
  const tts = registerProcess({ kind: 'tts', script: 'speak.py', pid: 1, device: 'cpu' });
  assert.deepEqual(tts.runtimes, ['torch', 'kokoro']);
  const hf = registerProcess({ kind: 'stt', script: 'transcribe.py', pid: 1, device: 'nvidia' });
  assert.deepEqual(hf.runtimes, ['torch', 'transformers']);
});

await test('an unknown script reports no runtimes rather than inventing them', () => {
  const job = registerProcess({ kind: 'image', script: 'not-a-script.py', pid: 1, device: 'cpu' });
  assert.deepEqual(job.runtimes, []);
});

console.log('\n=== stopping ===');

await test('stopping a job terminates the child process', async () => {
  const child = sleeper(60);
  const job = registerProcess({ kind: 'image', script: 'generate.py', pid: child.pid ?? -1, device: 'cpu' });
  attachChild(job.id, child);

  const exited = new Promise<void>((resolve) => child.on('close', () => resolve()));
  const stopped = await stopProcess(job.id);
  assert.ok(stopped, 'stopProcess should report the job it stopped');
  assert.equal(stopped.stoppedByUser, true);

  await exited;
  finishProcess(job.id, null, 'SIGTERM');
  assert.ok(!listProcesses().active.some((p) => p.id === job.id), 'job should leave the active list');
});

await test('stopping an unknown id is a no-op, not an error', async () => {
  assert.equal(await stopProcess('job-does-not-exist'), null);
  // A pid-looking string is just an unknown id: there is no path from a pid.
  assert.equal(await stopProcess('31337'), null);
});

await test('stopping twice does not double-signal or throw', async () => {
  const child = sleeper(60);
  const job = registerProcess({ kind: 'tts', script: 'speak.py', pid: child.pid ?? -1, device: 'cpu' });
  attachChild(job.id, child);
  const exited = new Promise<void>((resolve) => child.on('close', () => resolve()));
  await stopProcess(job.id);
  await stopProcess(job.id);
  await exited;
  finishProcess(job.id, null, 'SIGTERM');
});

await test('stop-all clears every active job', async () => {
  // The registry is module state and earlier tests leave jobs in it, so drain
  // it first — otherwise this measures the leftovers, not the three jobs below.
  await stopAllProcesses();
  const children = [sleeper(60), sleeper(60), sleeper(60)];
  const jobs = children.map((c, i) => {
    const job = registerProcess({
      kind: i === 0 ? 'stt' : i === 1 ? 'tts' : 'image',
      script: ['whisper_ct2.py', 'speak.py', 'generate.py'][i],
      pid: c.pid ?? -1,
      device: 'cpu',
    });
    attachChild(job.id, c);
    return { job, child: c };
  });
  const exits = jobs.map(({ child }) => new Promise<void>((r) => child.on('close', () => r())));

  const n = await stopAllProcesses();
  assert.equal(n, jobs.length, 'every active job should be signalled');
  await Promise.all(exits);
  for (const { job } of jobs) finishProcess(job.id, null, 'SIGTERM');
  assert.equal(listProcesses().active.length, 0);
});

console.log('\n=== bookkeeping ===');

await test('a finished job moves to recent and leaves active', () => {
  const job = registerProcess({ kind: 'stt', script: 'transcribe.py', pid: 7, device: 'cpu' });
  finishProcess(job.id, 0);
  const { active, recent } = listProcesses();
  assert.ok(!active.some((p) => p.id === job.id));
  assert.ok(recent.some((p) => p.id === job.id), 'the finished job should be visible briefly');
  assert.equal(recent.find((p) => p.id === job.id)?.state, 'exited');
});

await test('finishing an unknown job is harmless', () => {
  finishProcess('nope', 0);
  assert.ok(listProcesses().active.every((p) => p.id !== 'nope'));
});

await test('a registered-but-unattached job is droppable', async () => {
  // The gap between spawn returning a pid and the handle being bound. Signalling
  // nothing is the only correct answer; hanging there as "running" forever is
  // what would stop the manager ever showing an empty list.
  const job = registerProcess({ kind: 'image', script: 'generate.py', pid: 999, device: 'cpu' });
  const stopped = await stopProcess(job.id);
  assert.ok(stopped);
  await until(() => !listProcesses().active.some((p) => p.id === job.id));
});

console.log('\n=== labels and memory ===');

await test('the device a runner reported is what the entry shows', () => {
  // The request ("auto") is kept as made, and the answer arrives separately —
  // the manager labels a job with the answer, because on a CPU-only machine
  // "auto" is not a GPU job.
  const job = registerProcess({ kind: 'image', script: 'generate.py', pid: 1, device: 'auto' });
  assert.equal(job.device, 'auto');
  assert.equal(job.resolvedDevice, undefined, 'nothing should be invented before the runner says');
  setProcessDevice(job.id, 'cpu');
  const listed = listProcesses().active.find((p) => p.id === job.id);
  assert.equal(listed?.resolvedDevice, 'cpu');
});

await test('a device report for a finished job is discarded, not an error', () => {
  const job = registerProcess({ kind: 'stt', script: 'whisper_ct2.py', pid: 3, device: 'auto' });
  finishProcess(job.id, 0);
  setProcessDevice(job.id, 'cuda');
  assert.ok(!listProcesses().active.some((p) => p.id === job.id));
  const gone = listProcesses().recent.find((p) => p.id === job.id);
  assert.equal(gone?.resolvedDevice, undefined, 'the recent copy is a snapshot, not a live view');
});

await test('memory is read from a live process', async () => {
  const bytes = await processMemoryBytes(process.pid);
  assert.ok(typeof bytes === 'number' && bytes > 0, `expected a figure, got ${bytes}`);
});

await test('memory of a gone process is null, not an invented figure', async () => {
  const child = spawn(process.execPath, ['-e', ''], { stdio: 'ignore' });
  await new Promise<void>((resolve) => child.on('close', () => resolve()));
  assert.equal(await processMemoryBytes(child.pid ?? -1), null);
});

await test('memory of a nonsense pid is null', async () => {
  assert.equal(await processMemoryBytes(-1), null);
});

console.log(
  `\n${passed} passed, ${failures.length} failed` +
    (failures.length ? `\n  failing: ${failures.join(', ')}` : ''),
);
process.exit(failures.length ? 1 : 0);
