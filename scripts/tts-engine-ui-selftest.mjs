#!/usr/bin/env node
/**
 * What the resource manager says about a voice that stays loaded.
 *
 * The engine exists so a reply does not pay for the model every time. The
 * other half of that is telling the person: that the voice is still there,
 * how long it will stay there, and how to let go of it now if they want to.
 *
 * The rule this checks is the difference between a loaded voice and a job:
 *
 *   - it is not labelled as working — it is not working, it is waiting;
 *   - it says how long it will wait, because "it is still there" and "it will
 *     still be there in four minutes" are different answers;
 *   - and it can be released from the panel, without going to a terminal.
 *
 * Run with the app up (`npm run dev`):
 *   npm run test:tts-engine-ui
 */
import { spawn, spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000/';
const CDP_PORT = Number(process.env.CDP_PORT ?? 9393);

const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
if (!browser) {
  console.log('SKIP — no chromium/chrome found on PATH');
  process.exit(0);
}

/** A voice on this machine that can be asked to speak. */
let model = null;
try {
  const res = await fetch(new URL('/api/runnable?task=tts', APP_URL), {
    signal: AbortSignal.timeout(15000),
  });
  const data = await res.json();
  const assigned = String(data?.assigned?.tts ?? '').replace(/^local:/, '');
  model =
    (data?.tts ?? []).find((m) => m.repo === assigned && m.ready) ??
    (data?.tts ?? []).find((m) => m.ready) ??
    null;
} catch {
  console.log(`SKIP — app not reachable at ${APP_URL} (start it with npm run dev)`);
  process.exit(0);
}
if (!model) {
  console.log('SKIP — no runnable voice model on this machine (Settings → Models)');
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PROFILE = join(tmpdir(), `starr-tts-engine-ui-${process.pid}`);
rmSync(PROFILE, { recursive: true, force: true });

const chrome = spawn(
  browser,
  [
    '--headless=new',
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    `--remote-debugging-port=${CDP_PORT}`,
    `--user-data-dir=${PROFILE}`,
    'about:blank',
  ],
  { stdio: 'ignore' },
);

let passed = 0;
const failures = [];
function report(name, ok, extra = '') {
  if (ok) {
    passed += 1;
    console.log(`  PASS  ${name}`);
  } else {
    failures.push(name);
    console.log(`  FAIL  ${name}${extra ? ` — ${extra}` : ''}`);
  }
}

async function getWsUrl() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)).json();
      const page = list.find((t) => t.type === 'page');
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch {
      /* chromium still starting */
    }
    await sleep(200);
  }
  throw new Error('chromium never came up');
}

let chromeKilled = false;
async function cleanup() {
  if (!chromeKilled) {
    chromeKilled = true;
    chrome.kill('SIGKILL');
    await new Promise((resolve) => {
      if (chrome.exitCode !== null) return resolve();
      chrome.once('exit', resolve);
      setTimeout(resolve, 3000);
    });
  }
  await rm(PROFILE, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => {});
}

try {
  console.log('\n=== a voice that stays loaded (headless chromium) ===');

  // Ask for a voice, so the engine exists and is warm by the time the panel
  // is opened. The first one pays for the load; the rest do not.
  const said = await fetch(new URL('/api/infer/tts', APP_URL), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      repo: model.repo,
      file: model.file,
      text: 'The engine is loaded and waiting.',
      voice: 'af_heart',
      speed: 1,
      device: 'auto',
    }),
    signal: AbortSignal.timeout(300_000),
  }).then(async (r) => r.text());
  report('the voice was made', /"samples"/.test(said), said.slice(-200));

  const ws = new WebSocket(await getWsUrl());
  await new Promise((res, rej) => {
    ws.onopen = res;
    ws.onerror = rej;
  });

  let idc = 0;
  const pending = new Map();
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  };
  const cdp = (method, params = {}) => {
    const id = ++idc;
    ws.send(JSON.stringify({ id, method, params }));
    return new Promise((res) => pending.set(id, res));
  };
  const evalJs = async (expression) => {
    const res = await cdp('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (res.result?.exceptionDetails) throw new Error(`eval failed: ${res.result.exceptionDetails.text}`);
    return res.result?.result?.value;
  };
  const waitFor = async (expr, timeoutMs = 30000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      try {
        if (await evalJs(expr)) return true;
      } catch {
        /* the page is still loading */
      }
      await sleep(250);
    }
    return false;
  };

  await cdp('Page.enable');
  await cdp('Runtime.enable');
  await cdp('Page.navigate', { url: APP_URL });
  await waitFor(`!!document.querySelector('#open-procman')`, 30000);

  // The manager is a pop-out behind a button.
  await evalJs(`document.querySelector('#open-procman').click(); true`);
  const opened = await waitFor(
    `!!document.querySelector('.procman__row') || !!(document.querySelector('#procman') && getComputedStyle(document.querySelector('#procman')).display !== 'none')`,
    15000,
  );
  report('the resource manager opens', opened);

  // The engine row: named by its state, not as a job that is working.
  //
  // It is found by the model it speaks, not by "the first engine on the
  // page": this machine may well have another voice warm from an earlier
  // reply, and a test that unloads somebody else's engine is not a test.
  const rowFor = `(() => {
    const r = [...document.querySelectorAll('.procman__row')].find((r) => /Warm/.test(r.textContent) && r.textContent.includes(${JSON.stringify(model.repo)}));
    return r ? r.textContent : '';
  })()`;
  const hasRow = await waitFor(`${rowFor}.length > 0`, 20000);
  const row = hasRow ? await evalJs(rowFor) : '';
  report('the engine is listed as warm, not as a working job', /Warm/.test(String(row)), String(row));
  report('and it says it is warm', /warm/.test(String(row)), String(row));
  report('and it says how long it will stay', /until it is released/.test(String(row)), String(row));
  report(
    'with a countdown, not a duration that has already passed',
    /\d+:\d\d until it is released/.test(String(row)),
    String(row),
  );

  const button = await evalJs(
    `(() => { const r = [...document.querySelectorAll('.procman__row')].find((r) => /Warm/.test(r.textContent) && r.textContent.includes(${JSON.stringify(model.repo)})); const b = r?.querySelector('button'); return b ? b.textContent : ''; })()`,
  );
  report('and it can be let go of from the panel', button === 'Unload', `the button said "${button}"`);

  // And letting go of it really does let go of it.
  await evalJs(
    `(() => { const r = [...document.querySelectorAll('.procman__row')].find((r) => /Warm/.test(r.textContent) && r.textContent.includes(${JSON.stringify(model.repo)})); r?.querySelector('button')?.click(); return true; })()`,
  );
  const gone = await waitFor(
    `![...document.querySelectorAll('.procman__row')].some((r) => /Warm/.test(r.textContent) && r.textContent.includes(${JSON.stringify(model.repo)}))`,
    20000,
  );
  report('the engine is gone once it is released', gone);
} catch (err) {
  report('the panel check ran to the end', false, err instanceof Error ? err.message : String(err));
} finally {
  await cleanup();
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
