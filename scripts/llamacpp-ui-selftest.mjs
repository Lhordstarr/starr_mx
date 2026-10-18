#!/usr/bin/env node
/**
 * The two screens the llama.cpp work is judged on, driven in headless Chromium:
 *
 *  - the chat page's provider popover: llama.cpp sits in the local group, and
 *    the row no longer carries the "runs on this PC" tag;
 *  - the models page: a downloaded GGUF offers a Chat button that lands on the
 *    chat page with that provider and model already chosen.
 *
 * Run with the app up (`npm run dev`) on a machine that has the runtime and at
 * least one GGUF chat model, then:
 *   npm run test:llamacpp-ui
 *
 * Skips (exit 0) when Chromium is absent, the app is not reachable, or this
 * machine has no llama.cpp chat models to show — the checks are about a live
 * page, and an empty machine has no page to check.
 */
import { spawn, spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000/';
const CDP_PORT = Number(process.env.CDP_PORT ?? 9378);

const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
if (!browser) {
  console.log('SKIP — no chromium/chrome found on PATH');
  process.exit(0);
}

let providers;
try {
  const res = await fetch(new URL('/api/providers', APP_URL), { signal: AbortSignal.timeout(15000) });
  providers = (await res.json())?.providers ?? [];
} catch {
  console.log(`SKIP — app not reachable at ${APP_URL} (start it with npm run dev)`);
  process.exit(0);
}

const llama = providers.find((p) => p.id === 'llamacpp');
const models = llama?.models ?? [];
if (!llama?.configured || models.length === 0) {
  console.log('SKIP — this machine has no llama.cpp chat models installed yet');
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PROFILE = join(tmpdir(), `starr-llamacpp-ui-${process.pid}`);
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
  console.log('\n=== llama.cpp screens (headless chromium) ===');

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
  const waitFor = async (expr, timeoutMs = 8000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      // A navigation tears the execution context down mid-poll; that is not a
      // result, it is the page being replaced, so it counts as "not yet".
      try {
        if (await evalJs(expr)) return true;
      } catch {
        /* context destroyed by a navigation */
      }
      await sleep(120);
    }
    return false;
  };

  await cdp('Page.enable');

  /* ------- The provider popover ------- */
  await cdp('Page.navigate', { url: APP_URL });
  await waitFor(`document.readyState === 'complete' && !!document.getElementById('open-panel')`);
  await waitFor(`!!document.querySelector('#provider-list .popover__item[data-id=llamacpp]')`, 15000);

  await evalJs(`document.getElementById('open-panel').click()`);
  report('the provider popover opens', await waitFor(`document.getElementById('provider-pop').hidden === false`));

  const row = await evalJs(`(() => {
    const item = document.querySelector('#provider-list .popover__item[data-id=llamacpp]');
    return item ? { label: item.querySelector('.popover__item-label')?.textContent ?? '', text: item.textContent } : null;
  })()`);
  report('llama.cpp is listed as a provider', row?.label === 'llama.cpp (local)', JSON.stringify(row));
  report(
    'the "runs on this PC" tag is gone from the provider rows',
    !(await evalJs(`document.getElementById('provider-pop').textContent`)).includes('runs on this PC'),
  );

  await evalJs(`document.querySelector('#provider-list .popover__item[data-id=llamacpp]').click()`);
  report(
    'picking it makes llama.cpp the active provider',
    await waitFor(`document.getElementById('chip-provider').textContent === 'llama.cpp (local)'`),
    await evalJs(`document.getElementById('chip-provider').textContent`),
  );

  /* ------- The model picker on the chat page ------- */
  await evalJs(`document.getElementById('model-trigger').click()`);
  report('the model menu opens', await waitFor(`document.getElementById('model-menu').hidden === false`));
  const listed = await evalJs(
    `JSON.stringify([...document.querySelectorAll('#model-list .mm-option')].map(o => o.dataset.model))`,
  );
  const listedIds = JSON.parse(listed ?? '[]');
  report(
    'the GGUF models on this machine are offered in the picker',
    models.every((m) => listedIds.includes(m)),
    `provider says ${models.join(', ')}; picker shows ${listedIds.join(', ')}`,
  );
  report(
    'no non-chat GGUF is offered',
    !listedIds.some((id) => /whisper|mmproj|vocoder|diffusion|encore|turbo/i.test(id)),
    listedIds.join(', '),
  );
  const current = await evalJs(`document.getElementById('model-current').textContent`);
  report('a model from that list is selected by default', models.includes(current), current);

  /* ------- The deep link the Chat button uses ------- */
  const target = models[0];
  await cdp('Page.navigate', {
    url: `${new URL(APP_URL).origin}/?provider=llamacpp&model=${encodeURIComponent(target)}`,
  });
  await waitFor(`document.readyState === 'complete' && !!document.getElementById('model-current')`);
  report(
    'a deep link selects the provider and the model',
    await waitFor(
      `document.getElementById('chip-provider').textContent === 'llama.cpp (local)' &&
       document.getElementById('model-current').textContent === ${JSON.stringify(target)}`,
      15000,
    ),
    `${await evalJs(`document.getElementById('chip-provider').textContent`)} / ${await evalJs(`document.getElementById('model-current').textContent`)}`,
  );
  report(
    'the topbar chip reflects it',
    (await evalJs(`document.getElementById('chip-model').textContent`)) === target,
  );

  /* ------- The models page ------- */
  await cdp('Page.navigate', { url: `${new URL(APP_URL).origin}/settings#models` });
  await waitFor(`document.readyState === 'complete'`);
  // The hub's lists arrive from three endpoints; the rows are the last to paint.
  const rowsUp = await waitFor(`!!document.querySelector('.local-download-row')`, 20000);
  if (!rowsUp) {
    console.log('  SKIP  the models page is not showing any downloaded rows');
  } else {
    const ggufRow = await evalJs(`(() => {
      const rows = [...document.querySelectorAll('.local-download-row')];
      const row = rows.find((r) => /\\.gguf/i.test(r.querySelector('.model-name')?.textContent ?? ''));
      if (!row) return null;
      const btn = [...row.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Chat');
      return {
        name: row.querySelector('.model-name')?.textContent ?? '',
        note: row.querySelector('.local-meta__runtime')?.textContent ?? '',
        chat: Boolean(btn),
        disabled: btn ? btn.disabled : null,
      };
    })()`);
    if (!ggufRow) {
      console.log('  SKIP  no GGUF row on the models page for this machine');
    } else {
      report('a downloaded GGUF row is listed', ggufRow.name.length > 0, JSON.stringify(ggufRow));
      report('its runtime reads llama.cpp', /llama\.cpp/i.test(ggufRow.note), ggufRow.note);
      report('it offers a working Chat button', ggufRow.chat && ggufRow.disabled === false, JSON.stringify(ggufRow));

      if (ggufRow.chat) {
        await evalJs(`(() => {
          const rows = [...document.querySelectorAll('.local-download-row')];
          const row = rows.find((r) => /\\.gguf/i.test(r.querySelector('.model-name')?.textContent ?? ''));
          [...row.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Chat').click();
        })()`);
        const landed = await waitFor(
          `location.pathname === '/' && location.search.includes('provider=llamacpp')`,
          15000,
        );
        report('clicking Chat lands on the chat page with that model', landed, await evalJs(`location.href`));
        if (landed) {
          report(
            'and the chat page has it selected',
            await waitFor(
              `document.getElementById('chip-provider').textContent === 'llama.cpp (local)' &&
               ${JSON.stringify(models)}.includes(document.getElementById('model-current').textContent)`,
              15000,
            ),
            `${await evalJs(`document.getElementById('model-current').textContent`)}`,
          );
        }
      }
    }
  }
} catch (err) {
  report('the llama.cpp screens ran', false, err.message);
} finally {
  await cleanup();
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
