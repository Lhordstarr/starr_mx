#!/usr/bin/env node
/**
 * Frontend checks for the thinking pane, driven in headless Chromium over the
 * DevTools protocol. The pane's job is to keep a model's scratchpad visibly
 * separate from its answer — open while the model is still thinking, closed
 * once the answer starts, and never silently mixed back into the transcript —
 * so that is what this exercises against a stubbed /api/chat SSE stream.
 *
 * Run with the app up (`npm run dev` or `npm start`), then:
 *   npm run test:reasoning-ui
 *
 * Skips (exit 0) when Chromium or the app server is absent — it is a check,
 * not a build gate.
 */
import { spawn, spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000/';
const PORT = Number(process.env.CDP_PORT ?? 9379);

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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PROFILE = join(tmpdir(), `starr-reasoning-ui-${process.pid}`);
rmSync(PROFILE, { recursive: true, force: true });

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
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
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
  console.log('\n=== thinking pane UI (headless chromium) ===');

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
    const res = await cdp('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (res.result?.exceptionDetails) {
      throw new Error(`eval failed: ${res.result.exceptionDetails.text} — ${expression}`);
    }
    return res.result?.result?.value;
  };
  const waitFor = async (expr, timeoutMs = 5000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      if (await evalJs(expr)) return true;
      await sleep(100);
    }
    return false;
  };

  await cdp('Page.enable');
  await cdp('Page.navigate', { url: APP_URL });
  // The composer refuses to send until the provider list has loaded; wait
  // for the topbar to name one rather than racing the first fetch.
  await waitFor(
    `document.readyState === 'complete' && !!document.getElementById('input') &&
     !!document.documentElement.dataset.provider`,
    10000,
  );

  report(
    'a fresh page has no thinking pane',
    await evalJs(`document.querySelectorAll('.thinking').length === 0`),
  );

  /* A chat request we control: a live SSE stream the test pushes into. */
  const installStub = `(() => {
    const stream = new ReadableStream({
      start(c) { window.__sseCtl = c; },
    });
    const enc = new TextEncoder();
    window.__ssePush = (event, data) =>
      window.__sseCtl.enqueue(enc.encode('event: ' + event + '\\ndata: ' + JSON.stringify(data) + '\\n\\n'));
    window.__sseEnd = () => { try { window.__sseCtl.close(); } catch {} };
    window.__origFetch = window.fetch;
    window.fetch = (url, init) => {
      if (String(url).includes('/api/chat')) {
        return Promise.resolve(
          new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
        );
      }
      return window.__origFetch(url, init);
    };
    return true;
  })()`;
  await evalJs(installStub);

  await evalJs(`(() => {
    const i = document.getElementById('input');
    i.value = 'think for me';
    i.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('composer').requestSubmit();
  })()`);
  report(
    'the assistant bubble is live before any token arrives',
    await waitFor(`document.querySelectorAll('.message.assistant').length === 1`),
  );

  await evalJs(`window.__ssePush('reasoning', { text: 'Step one: weigh it.' })`);
  report(
    'reasoning opens a thinking pane',
    await waitFor(`!!document.querySelector('.message.assistant .thinking')`),
  );
  report(
    'the pane is labelled Thinking',
    (await evalJs(`document.querySelector('.thinking-label')?.textContent`)) === 'Thinking',
  );
  report(
    'the pane is open while there is no answer yet',
    await evalJs(
      `document.querySelector('.thinking')?.getAttribute('data-open') === '1' &&
       document.querySelector('.thinking-body')?.offsetHeight > 0`,
    ),
  );
  report(
    'the scratchpad lands in the pane, not the answer',
    await evalJs(`(() => {
      const bubble = document.querySelector('.message.assistant .bubble');
      if (!bubble.textContent.includes('Step one: weigh it.')) return false;
      const copy = bubble.cloneNode(true);
      copy.querySelector('.thinking')?.remove();
      return copy.textContent.trim() === '';
    })()`),
  );

  await evalJs(`window.__ssePush('delta', { text: 'The answer is 42.' })`);
  report(
    'the answer paints into the bubble',
    await waitFor(
      `document.querySelector('.message.assistant .bubble').textContent.includes('The answer is 42.')`,
    ),
  );
  report(
    'the pane closes automatically once the answer starts',
    await evalJs(
      `document.querySelector('.thinking')?.getAttribute('data-open') !== '1' &&
       document.querySelector('.thinking-body')?.offsetHeight === 0`,
    ),
  );
  report(
    'the reasoning survives the collapse',
    await evalJs(
      `document.querySelector('.thinking-body')?.textContent.includes('Step one: weigh it.')`,
    ),
  );

  await evalJs(`document.querySelector('.thinking-toggle').click()`);
  report(
    'clicking the toggle opens the pane',
    await evalJs(
      `document.querySelector('.thinking')?.getAttribute('data-open') === '1' &&
       document.querySelector('.thinking-toggle')?.getAttribute('aria-expanded') === 'true' &&
       document.querySelector('.thinking-body')?.offsetHeight > 0`,
    ),
  );
  await evalJs(`document.querySelector('.thinking-toggle').click()`);
  report(
    'clicking again closes it',
    await evalJs(
      `document.querySelector('.thinking')?.getAttribute('data-open') !== '1' &&
       document.querySelector('.thinking-toggle')?.getAttribute('aria-expanded') === 'false'`,
    ),
  );

  /* An explicit choice wins over the automatic open/close behaviour. */
  await evalJs(`document.querySelector('.thinking-toggle')?.click()`);
  await evalJs(`window.__ssePush('reasoning', { text: ' Step two.' })`);
  await waitFor(`document.querySelector('.thinking-body')?.textContent.includes('Step two.')`);
  await evalJs(`window.__ssePush('delta', { text: ' More.' })`);
  await waitFor(
    `document.querySelector('.message.assistant .bubble').textContent.includes('More.')`,
  );
  report(
    'a user-opened pane stays open as the answer streams',
    await evalJs(`document.querySelector('.thinking')?.getAttribute('data-open') === '1'`),
  );

  await evalJs(`window.__ssePush('done', {})`);
  report(
    'the turn finishes with its actions',
    await waitFor(
      `[...document.querySelectorAll('.message.assistant .msg-actions button')].some(
         (b) => b.textContent === 'Copy')`,
    ),
  );
  report(
    'the answer bubble does not contain the reasoning',
    await evalJs(`(() => {
      const bubble = document.querySelector('.message.assistant .bubble');
      const copy = bubble.cloneNode(true);
      copy.querySelector('.thinking')?.remove();
      return !copy.textContent.includes('Step one') && !copy.textContent.includes('Step two');
    })()`),
  );

  const stored = await evalJs(`(() => {
    const s = JSON.parse(localStorage.getItem('llm-chat-state'));
    const c = s.chats.find((x) => x.id === s.active);
    const a = c.conversation.filter((m) => m.role === 'assistant').at(-1);
    return JSON.stringify({ content: a.content, reasoning: a.reasoning });
  })()`);
  report(
    'the saved turn keeps reasoning beside the answer',
    stored ===
      JSON.stringify({
        content: 'The answer is 42. More.',
        reasoning: 'Step one: weigh it. Step two.',
      }),
    stored,
  );

  /* A reload must bring the pane back — collapsed — from storage. */
  await cdp('Page.reload');
  await waitFor(
    `document.readyState === 'complete' && !!document.getElementById('input') &&
     !!document.documentElement.dataset.provider`,
    10000,
  );
  report(
    'reopening the chat restores the answer',
    await waitFor(`document.body.textContent.includes('The answer is 42. More.')`),
  );
  report(
    'the restored pane is collapsed and holds the reasoning',
    await evalJs(`(() => {
      const t = document.querySelector('.thinking');
      return !!t &&
        t.getAttribute('data-open') !== '1' &&
        t.querySelector('.thinking-body').textContent.includes('Step two.');
    })()`),
  );

  /* A plain answer must not grow a pane at all. */
  await evalJs(installStub);
  await evalJs(`(() => {
    const i = document.getElementById('input');
    i.value = 'plain please';
    i.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('composer').requestSubmit();
  })()`);
  await waitFor(`document.querySelectorAll('.message.assistant').length === 2`);
  await evalJs(`window.__ssePush('delta', { text: 'Just prose.' })`);
  await waitFor(
    `[...document.querySelectorAll('.message.assistant .bubble')].at(-1).textContent.includes('Just prose.')`,
  );
  report(
    'a plain answer grows no thinking pane',
    await evalJs(`document.querySelectorAll('.message.assistant .thinking').length === 1`),
  );
  await evalJs(`window.__ssePush('done', {})`);
  await sleep(200);

  ws.close();
} finally {
  await cleanup();
}

console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length ? 1 : 0);
