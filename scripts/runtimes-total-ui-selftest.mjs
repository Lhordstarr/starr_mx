#!/usr/bin/env node
/**
 * The Runtimes page's total estimate, including dependencies.
 *
 * "Install all" selected every installable group and started the install, but
 * never repainted the total, so the only number that answers "how big is this?"
 * stayed on "Nothing selected" for the whole run. The estimate also summed the
 * selected rows and then merely said "includes PyTorch": it never added the
 * wheel, and it counted one wheel even when Kokoro's separate environment made
 * a second copy necessary.
 *
 * This drives the page with a stubbed runtime catalog and a held-open install
 * request, so the pre-install state is what is checked. It fails against the
 * old page because the total never leaves "Nothing selected" and the log never
 * carries the estimate.
 *
 * Run with the app up (`npm run dev`):
 *   npm run test:runtimes-total-ui
 */
import { spawn, spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000/';
const CDP_PORT = Number(process.env.CDP_PORT ?? 9384);

const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
if (!browser) {
  console.log('SKIP — no chromium/chrome found on PATH');
  process.exit(0);
}

try {
  const res = await fetch(new URL('/api/runtimes', APP_URL), {
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
} catch {
  console.log(`SKIP — app not reachable at ${APP_URL} (start it with npm run dev)`);
  process.exit(0);
}

const group = (id, label, approxMB, extra = {}) => ({
  id,
  probe: id,
  label,
  blurb: `${label} stub`,
  unlocks: [],
  pkgs: [],
  needsTorch: false,
  approxMB,
  ...extra,
});

const GROUPS = [
  group('torch', 'PyTorch', 529),
  group('transformers', 'Transformers', 45, { needsTorch: true }),
  group('kokoro', 'Kokoro', 53, { needsTorch: true, venv: 'venv-kokoro', maxPython: '<3.13' }),
  group('ollama', 'Ollama', 1500, { installer: 'ollama' }),
  group('onnxruntime', 'ONNX Runtime', 23),
  group('llama-cpp', 'llama.cpp', 17, { installer: 'llama-cpp' }),
];

const INSTALLED = Object.fromEntries(GROUPS.map((g) => [g.id, false]));

const RUNTIMES = {
  runtimes: {
    venv: true,
    python: '/usr/bin/python3.13',
    installed: INSTALLED,
    ollamaVulkan: false,
    torch: { version: null, cudaAvailable: false, deviceCount: 0, deviceName: null },
    torches: [],
    onnx: [],
    busy: false,
  },
  groups: GROUPS,
  python: '3.13',
  blocked: [],
  warning: null,
  gpu: { vendor: 'nvidia', detail: 'NVIDIA GPU: Test T4', usable: true, needsGroup: false },
  upgrades: {},
};

// 45 + 53 + 1500 + 23 + 17 = 1638, plus torch twice (shared + Kokoro):
// 1638 + 2 * 529 = 2696.
const EXPECTED_MB = 2696;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PROFILE = join(tmpdir(), `starr-runtimes-total-ui-${process.pid}`);
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
  console.log('\n=== Runtimes total estimate (headless chromium) ===');

  const ws = new WebSocket(await getWsUrl());
  await new Promise((res, rej) => {
    ws.onopen = res;
    ws.onerror = rej;
  });

  let idc = 0;
  const pending = new Map();
  let cdp;

  async function fulfill(requestId, contentType, body) {
    await cdp('Fetch.fulfillRequest', {
      requestId,
      responseCode: 200,
      responseHeaders: [
        { name: 'content-type', value: contentType },
        { name: 'cache-control', value: 'no-store' },
      ],
      body: Buffer.from(body).toString('base64'),
    });
  }

  async function onPaused({ requestId, request }) {
    const url = request.url;
    if (/\/api\/runtimes\/install(?:\?|$)/.test(url)) {
      // Held open deliberately: the test is about what the page says before
      // the install finishes, and letting the server start one is unnecessary.
      return;
    }
    if (/\/api\/runtimes(?:\?|$)/.test(url)) {
      await fulfill(requestId, 'application/json', JSON.stringify(RUNTIMES));
      return;
    }
    await cdp('Fetch.continueRequest', { requestId });
  }

  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
      return;
    }
    if (msg.method === 'Fetch.requestPaused') {
      onPaused(msg.params).catch((err) => {
        console.error(`  (interception failed: ${err.message})`);
      });
    }
  };

  cdp = (method, params = {}) => {
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
    if (res.result?.exceptionDetails) throw new Error(`eval failed: ${res.result.exceptionDetails.text}`);
    return res.result?.result?.value;
  };
  const waitFor = async (expr, timeoutMs = 15000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      try {
        if (await evalJs(expr)) return true;
      } catch {
        /* context destroyed by a navigation */
      }
      await sleep(150);
    }
    return false;
  };

  await cdp('Page.enable');
  await cdp('Fetch.enable', {
    patterns: [
      { urlPattern: '*api/runtimes*', requestStage: 'Request' },
    ],
  });

  await cdp('Page.navigate', { url: new URL('/settings#models/runtimes', APP_URL).href });
  await waitFor(`document.readyState === 'complete'`);

  const rendered = await waitFor(`document.querySelectorAll('#rt-list .rt-row').length === 6`);
  report('the stubbed runtime rows render', rendered);
  if (!rendered) {
    console.error(
      await evalJs(`JSON.stringify({
        hash: location.hash,
        rows: document.querySelectorAll('.rt-row').length,
        list: document.getElementById('rt-list')?.textContent?.slice(0, 300),
        total: document.getElementById('rt-total')?.textContent,
      })`),
    );
    throw new Error('runtime rows never rendered');
  }

  const before = await evalJs(`document.getElementById('rt-total').textContent`);
  report('nothing is selected before Install all', before === 'Nothing selected', before);

  await evalJs(`document.getElementById('rt-install-all').click()`);

  const updated = await waitFor(
    `document.getElementById('rt-total').textContent.includes('${EXPECTED_MB} MB')`,
    5000,
  );
  const total = await evalJs(`document.getElementById('rt-total').textContent`);
  report(
    'Install all repaints the total before the first package is fetched',
    updated && /6 selected/.test(total ?? ''),
    total ?? '',
  );
  report(
    'the total counts PyTorch in both environments that need it',
    /≈ 2696 MB/.test(total ?? '') && /PyTorch ×2/.test(total ?? ''),
    total ?? '',
  );

  const logShown = await waitFor(
    `document.getElementById('rt-log-body').textContent.includes('Estimated download')`,
    5000,
  );
  const log = await evalJs(`document.getElementById('rt-log-body').textContent`);
  report('the install log states the estimate before the first pip line', logShown, log ?? '');
  report(
    'the log names both PyTorch environments rather than hiding the second wheel',
    /≈ 2696 MB/.test(log ?? '') && /PyTorch in 2 environments/.test(log ?? ''),
    log ?? '',
  );
} finally {
  await cleanup();
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);
