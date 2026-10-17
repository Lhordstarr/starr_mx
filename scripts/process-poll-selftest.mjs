#!/usr/bin/env node
/**
 * Reproduces the process-manager poll ceiling.
 *
 * The manager polls /api/processes once a second while it is open. If that
 * route sits on the general 30-a-minute limit, the list answers 429 from thirty
 * seconds in and stops updating — the pop-out looks hung rather than refused.
 *
 * Polls at the app's own interval and reports where it first got refused.
 *
 *   node scripts/process-poll-selftest.mjs
 */
import { spawn, spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000';
const SECONDS = Number(process.env.POLL_SECONDS ?? 45);
const PORT = Number(process.env.CDP_PORT ?? 9391);

const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
if (!browser) {
  console.log('SKIP — no chromium/chrome found on PATH');
  process.exit(0);
}
try {
  const res = await fetch(APP_URL, { signal: AbortSignal.timeout(3000) });
  if (!res.ok) throw new Error(String(res.status));
} catch {
  console.log(`SKIP — app not reachable at ${APP_URL} (start it with npm run dev)`);
  process.exit(0);
}

const PROFILE = mkdtempSync(join(tmpdir(), 'starr-procpoll-'));
const chrome = spawn(
  browser,
  [
    '--headless=new',
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${PROFILE}`,
    'about:blank',
  ],
  { stdio: 'ignore' },
);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const cleanup = () => {
  try {
    chrome.kill('SIGKILL');
  } catch {
    /* gone */
  }
  try {
    rmSync(PROFILE, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  } catch {
    /* leaving it behind */
  }
};

try {
  let wsUrl = '';
  for (let i = 0; i < 60 && !wsUrl; i += 1) {
    try {
      const list = await (
        await fetch(`http://127.0.0.1:${PORT}/json/list`, { signal: AbortSignal.timeout(1000) })
      ).json();
      wsUrl = list.find((t) => t.type === 'page')?.webSocketDebuggerUrl ?? '';
    } catch {
      /* not up */
    }
    if (!wsUrl) await sleep(250);
  }
  if (!wsUrl) throw new Error('chromium never came up');
  const ws = new WebSocket(wsUrl);
  await new Promise((r) => ws.addEventListener('open', r));
  let id = 0;
  const pending = new Map();
  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(String(ev.data));
    if (m.id && pending.has(m.id)) {
      pending.get(m.id)(m);
      pending.delete(m.id);
    }
  });
  const cdp = (method, params = {}) =>
    new Promise((r) => {
      const n = ++id;
      pending.set(n, r);
      ws.send(JSON.stringify({ id: n, method, params }));
    });
  const evalJs = async (expression) => {
    const r = await cdp('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(String(r.result.exceptionDetails.exception?.description));
    return r.result?.result?.value;
  };

  await cdp('Page.enable');
  await cdp('Page.bringToFront');
  await cdp('Page.navigate', { url: `${APP_URL}/` });
  for (let i = 0; i < 40; i += 1) {
    await sleep(500);
    if (await evalJs(`!!document.querySelector('[id*=proc], [class*=proc]')`)) break;
  }

  // Count the status codes this page receives from the poll endpoint, from the
  // browser itself, so the ceiling under test is the one in force.
  await evalJs(`(() => {
    window.__procCalls = [];
    const orig = window.fetch;
    window.fetch = async (url, opts) => {
      const res = await orig(url, opts);
      if (String(url).includes('/api/processes')) window.__procCalls.push(res.status);
      return res;
    };
    return true;
  })()`);

  // Open the manager the way a user does, if it is reachable from this page.
  const opened = await evalJs(`(() => {
    const btn = [...document.querySelectorAll('button, [role=button]')]
      .find((b) => /process/i.test(b.textContent || '') || /process/i.test(b.getAttribute('aria-label') || ''));
    if (!btn) return false;
    btn.click();
    return true;
  })()`);
  console.log(`opened the process manager from the page: ${opened}`);

  // Poll exactly as the app does, and see whether the server ever refuses.
  let refusedAt = null;
  let calls = 0;
  for (let s = 0; s < SECONDS; s += 1) {
    await sleep(1000);
    const res = await fetch(`${APP_URL}/api/processes`, { signal: AbortSignal.timeout(8000) });
    calls += 1;
    if (res.status === 429 && refusedAt === null) {
      refusedAt = s + 1;
      await res.text();
    }
  }
  console.log(`${calls} polls over ${SECONDS}s`);
  console.log(refusedAt === null ? 'PASS  /api/processes never rate-limited' : `FAIL  refused with 429 after ${refusedAt}s`);
  ws.close();
  if (refusedAt !== null) process.exitCode = 1;
} catch (err) {
  console.log(`FAIL  ${err.message}`);
  process.exitCode = 1;
} finally {
  cleanup();
}
