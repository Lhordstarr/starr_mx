#!/usr/bin/env tsx
/**
 * A voice that stays loaded.
 *
 * The complaint this answers: the TTS engine appeared in the resource manager
 * and was gone before it could be read. It was not a bug — a speech runner was
 * a process that lived for exactly as long as the utterance, so it loaded the
 * weights, said one thing and exited, and the next reply paid for the load
 * again.
 *
 * The rule the fix is built on:
 *
 *   - the engine outlives the utterance, and the next request is served from
 *     the model that is already in memory;
 *   - it does not outlive its welcome — after ten idle minutes it lets go of
 *     the memory it is holding;
 *   - it can be let go of at any time, from the resource manager;
 *   - and it is never the only way to speak: a request that cannot have an
 *     engine falls back to the one-shot runner, which is slower and always
 *     worked.
 *
 * The live part needs the Kokoro weights this repository's catalog downloads.
 * It is skipped when they are absent, the same rule the other local-runner
 * suites follow.
 */
import assert from 'node:assert/strict';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';

import { listProcesses, stopProcess } from '../src/processes.ts';
import { speakViaEngine, ttsEngines } from '../src/tts-engine.ts';
import { speak } from '../src/infer.ts';

let passed = 0;
const failures: string[] = [];
async function check(name: string, fn: () => unknown | Promise<unknown>): Promise<void> {
  try {
    await fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures.push(name);
    console.log(`  FAIL  ${name}${err instanceof Error ? ` — ${err.message}` : ''}`);
  }
}

const kokoroDir = path.resolve('models/hexgrad--Kokoro-82M');
const kokoroWeights = path.join(kokoroDir, 'kokoro-v1_0.pth');
const kokoroVenv = path.resolve('.runtimes/venv-kokoro/bin/python');
const outDir = path.resolve('.runtimes/out');

console.log('The runners must be able to stay loaded');
await check('the speech runners can be loaded once and used again', () => {
  for (const script of ['speak.py', 'speak_hf.py', 'speak_onnx.py']) {
    const source = readFileSync(path.resolve('scripts/py', script), 'utf8');
    assert.match(source, /def build_engine\(/, `${script} has no build_engine`);
    assert.match(source, /def render\(/, `${script} has no render`);
  }
});
await check('there is an engine that keeps them alive and speaks to them', () => {
  const source = readFileSync(path.resolve('scripts/py/tts_engine.py'), 'utf8');
  assert.match(source, /build_engine/);
  assert.match(source, /for line in sys\.stdin/);
  assert.match(source, /"ready": True/);
});
await check('the engine waits ten minutes before it lets go of the model', () => {
  const source = readFileSync(path.resolve('src/tts-engine.ts'), 'utf8');
  assert.match(source, /TTS_ENGINE_IDLE_MS\s*\?\?\s*10 \* 60_000/);
});
await check('the resource manager can tell a warm engine from a working job', () => {
  const source = readFileSync(path.resolve('public/resources.js'), 'utf8');
  assert.match(source, /j\.warm/);
  assert.match(source, /Unload/);
  assert.match(source, /until it is released/);
});

console.log('\nAnd a loaded voice must actually be reused');
const haveWeights = existsSync(kokoroVenv) && existsSync(kokoroWeights);
if (!haveWeights) {
  console.log('  SKIP  no Kokoro environment or weights on this machine');
} else {
  const spec = {
    runner: 'kokoro' as const,
    dir: kokoroDir,
    device: 'auto' as const,
    repo: 'hexgrad/Kokoro-82M',
    venv: 'venv-kokoro',
    script: 'speak.py',
  };
  const first = path.join(outDir, 'tts-engine-selftest-1.wav');
  const second = path.join(outDir, 'tts-engine-selftest-2.wav');

  let firstPid = -1;
  await check('the first request loads the model and says the words', async () => {
    const result = await speakViaEngine(spec, {
      text: 'The engine stays loaded between replies.',
      voice: 'af_heart',
      speed: 1,
      outPath: first,
    });
    assert.ok(result, 'the engine path was used');
    assert.equal(result.ok, true, JSON.stringify(result.data));
    assert.ok(existsSync(first), 'a wav was written');
    assert.ok(statSync(first).size > 20_000, `the wav is ${statSync(first).size} bytes`);
    assert.equal(result.data?.device, 'cpu');
    const engines = ttsEngines();
    assert.equal(engines.length, 1, 'one engine is held');
    firstPid = engines[0].pid ?? -1;
    assert.ok(firstPid > 0, 'the engine has a pid');
  });

  await check('and the engine is still there afterwards, warm, with a deadline', () => {
    const engines = ttsEngines();
    assert.equal(engines.length, 1, 'the engine was not thrown away with the request');
    const row = listProcesses().active.find((p) => p.pid === firstPid);
    assert.ok(row, 'the engine is in the process registry');
    assert.equal(row?.warm, true, 'the row says it is warm, not working');
    assert.ok((row?.idleUntil ?? 0) > Date.now(), 'it has a deadline in the future');
  });

  await check('the second request is served by the same engine, without a reload', async () => {
    const result = await speakViaEngine(spec, {
      text: 'So the second reply does not wait for the weights.',
      voice: 'af_heart',
      speed: 1,
      outPath: second,
    });
    assert.ok(result, 'the engine path was used again');
    assert.equal(result.ok, true, JSON.stringify(result.data));
    assert.ok(existsSync(second), 'a wav was written');
    const engines = ttsEngines();
    assert.equal(engines.length, 1, 'still exactly one engine');
    assert.equal(engines[0].pid, firstPid, 'the same process answered the second request');
  });

  await check('and it can be let go of on demand', async () => {
    const engines = ttsEngines();
    const job = listProcesses().active.find((p) => p.pid === firstPid);
    assert.ok(job, 'the engine is listed');
    const stopped = await stopProcess(job!.id);
    assert.ok(stopped, 'the engine was stopped');
    await new Promise((r) => setTimeout(r, 1200));
    assert.equal(ttsEngines().length, 0, 'the engine is gone');
    assert.equal(
      listProcesses().active.filter((p) => p.pid === firstPid).length,
      0,
      'and it is no longer listed as running',
    );
  });

  console.log('\nAn engine nobody is using must not sit in memory for ever');
  await check('it lets go of the model after the idle time', async () => {
    const previous = process.env.TTS_ENGINE_IDLE_MS;
    process.env.TTS_ENGINE_IDLE_MS = '1500';
    try {
      const out = path.join(outDir, 'tts-engine-selftest-idle.wav');
      const result = await speakViaEngine(spec, {
        text: 'And then it goes quiet.',
        voice: 'af_heart',
        speed: 1,
        outPath: out,
      });
      assert.ok(result?.ok, 'the utterance was made');
      const pid = ttsEngines()[0]?.pid;
      assert.ok(pid, 'an engine is held');
      const deadline = Date.now() + 15_000;
      while (ttsEngines().length && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 250));
      }
      assert.equal(ttsEngines().length, 0, 'the engine released itself');
      assert.equal(
        listProcesses().active.filter((p) => p.pid === pid).length,
        0,
        'and it left the process registry',
      );
    } finally {
      if (previous === undefined) delete process.env.TTS_ENGINE_IDLE_MS;
      else process.env.TTS_ENGINE_IDLE_MS = previous;
    }
  });

  await check('a model that cannot be loaded falls back to the one-shot runner', async () => {
    // A directory with no weights in it, outside the models folder: the
    // engine cannot start, so the request has to be served another way rather
    // than lost. (Outside, because an engine that reaches for a missing file
    // creates the folder it was going to look in, and the models folder is
    // not the place for a test's leftovers.)
    const broken = await speakViaEngine(
      { ...spec, dir: path.join(tmpdir(), 'starr-no-such-model'), repo: 'nobody/nothing' },
      {
        text: 'This one cannot be loaded.',
        voice: 'af_heart',
        speed: 1,
        outPath: path.join(outDir, 'tts-engine-selftest-broken.wav'),
      },
    );
    assert.equal(broken, null, 'the engine path was declined, not faked');
  });

  await check('and the voice still speaks through the public entry point', async () => {
    const { result, outPath } = await speak({
      repo: 'hexgrad/Kokoro-82M',
      file: 'kokoro-v1_0.pth',
      voice: 'af_heart',
      speed: 1,
      device: 'cpu',
      text: 'The voice still speaks.',
    });
    assert.equal(result.ok, true, JSON.stringify(result.data));
    assert.ok(existsSync(outPath), 'a wav was written');
    assert.ok(statSync(outPath).size > 20_000, `the wav is ${statSync(outPath).size} bytes`);
  });
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
