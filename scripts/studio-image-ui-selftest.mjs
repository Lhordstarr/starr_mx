#!/usr/bin/env node
/**
 * The Studio's local image runner, driven through the page.
 *
 * Two client bugs made a finished local generation look like nothing happened:
 *
 *  - `/api/infer/image` answers with the path of the PNG it wrote, but the
 *    Studio looked for a `name` field that the route never sends. The turn was
 *    never appended, so the status returned to "ready" with no image.
 *  - choosing "On this PC" returned from syncControls() before rebuilding the
 *    size list, leaving the previously selected cloud provider's 1024x1024 (or
 *    1536x640) in place for a 512px turbo checkpoint.
 *
 * This visits /studio with a stubbed catalog and runner, so it needs neither a
 * downloaded model nor a GPU. The interception is what makes it fail on the
 * old client: the runner really does answer with `path`, and the size list
 * really is whatever the page put there.
 *
 * Run with the app up (`npm run dev`):
 *   npm run test:studio-image-ui
 */
import { spawn, spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000/';
const CDP_PORT = Number(process.env.CDP_PORT ?? 9383);

const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
if (!browser) {
  console.log('SKIP — no chromium/chrome found on PATH');
  process.exit(0);
}

try {
  const res = await fetch(new URL('/api/image/providers', APP_URL), {
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
} catch {
  console.log(`SKIP — app not reachable at ${APP_URL} (start it with npm run dev)`);
  process.exit(0);
}

const FAKE_PROVIDER = {
  id: 'test-cloud',
  label: 'Test Cloud',
  keyless: true,
  configured: true,
  hint: 'stub',
  models: ['flux'],
  defaultModel: 'flux',
  sizes: [
    { w: 1024, h: 1024 },
    { w: 1536, h: 640 },
  ],
};

const FAKE_MODEL = {
  repo: 'stub/sd-turbo',
  label: 'Stub SD Turbo',
  file: 'unet/diffusion_pytorch_model.fp16.safetensors',
  ready: true,
  reason: '',
};

const DONE = {
  path: '/tmp/starr-studio-selftest/img-stub.png',
  width: 512,
  height: 512,
  device: 'cuda',
};

/** A 1x1 transparent PNG, so the appended <img> is a real successful load. */
const PNG_1X1 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PROFILE = join(tmpdir(), `starr-studio-image-ui-${process.pid}`);
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
  console.log('\n=== Studio local image generation (headless chromium) ===');

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

  /** Answers the requests that decide what the page sees. */
  async function onPaused({ requestId, request }) {
    const url = request.url;
    if (/\/api\/image\/providers(?:\?|$)/.test(url)) {
      await fulfill(requestId, 'application/json', JSON.stringify({ providers: [FAKE_PROVIDER] }));
      return;
    }
    if (/\/api\/runnable(?:\?|$)/.test(url)) {
      if (/[?&]task=image(?:&|$)/.test(url)) {
        await fulfill(
          requestId,
          'application/json',
          JSON.stringify({ image: [FAKE_MODEL], assigned: {} }),
        );
      } else {
        await cdp('Fetch.continueRequest', { requestId });
      }
      return;
    }
    if (/\/api\/infer\/image(?:\?|$)/.test(url)) {
      await fulfill(
        requestId,
        'text/event-stream',
        `event: done\ndata: ${JSON.stringify(DONE)}\n\n`,
      );
      return;
    }
    if (/\/api\/infer\/out\/img-stub\.png(?:\?|$)/.test(url)) {
      await fulfill(requestId, 'image/png', PNG_1X1);
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
      { urlPattern: '*api/image/providers*', requestStage: 'Request' },
      { urlPattern: '*api/runnable*', requestStage: 'Request' },
      { urlPattern: '*api/infer/image*', requestStage: 'Request' },
      { urlPattern: '*api/infer/out/*', requestStage: 'Request' },
    ],
  });

  await cdp('Page.navigate', { url: new URL('/studio', APP_URL).href });
  await waitFor(`document.readyState === 'complete'`);

  const localAdded = await waitFor(
    `!!document.querySelector('#studio-img-provider option[value="__local__"]')`,
  );
  report('the local diffusers entry is offered when a model is ready', localAdded);
  if (!localAdded) throw new Error('the local provider never appeared');

  const before = await evalJs(`document.getElementById('studio-img-provider').value`);
  report('with no assignment the Studio still opens on a cloud provider', before === 'test-cloud', before);

  await evalJs(`(() => {
    const sel = document.getElementById('studio-img-provider');
    sel.value = '__local__';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
  })()`);

  const sizesReady = await waitFor(
    `!!document.querySelector('#studio-img-size option[value="512x512"]')`,
  );
  const sizes = JSON.parse(
    (await evalJs(
      `JSON.stringify([...document.querySelectorAll('#studio-img-size option')].map(o => o.value))`,
    )) ?? '[]',
  );
  report(
    'choosing this PC replaces the cloud sizes with local ones',
    sizesReady && sizes.join(',') === '512x512,768x768,1024x1024',
    sizes.join(',') || '(empty)',
  );

  const selectedSize = await evalJs(`document.getElementById('studio-img-size').value`);
  report('local output starts at 512x512, the size turbo checkpoints are trained for', selectedSize === '512x512', selectedSize);

  const picked = await evalJs(`document.getElementById('model-current').textContent`);
  report('the first ready local model is chosen for the pill', /Stub SD Turbo/.test(picked ?? ''), picked ?? '');

  await evalJs(`(() => {
    const input = document.getElementById('input');
    input.value = 'a red apple on a wooden table';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('go').click();
  })()`);

  const rendered = await waitFor(
    `!!document.querySelector('#messages img[src="/api/infer/out/img-stub.png"]')`,
    20000,
  );
  report(
    'a done event carrying only a path still appends the generated image',
    rendered,
    await evalJs(`JSON.stringify([...document.querySelectorAll('#messages img')].map(i => i.getAttribute('src')))`),
  );

  if (rendered) {
    const meta = await evalJs(`(() => {
      const img = document.querySelector('#messages img[src="/api/infer/out/img-stub.png"]');
      return img?.closest('.turn, .message, article')?.textContent ?? img?.parentElement?.textContent ?? '';
    })()`);
    report('the turn labels the size the runner actually produced', /512\s*[×x]\s*512/.test(meta ?? ''), meta ?? '');
  }

  const status = await evalJs(`document.getElementById('status').textContent`);
  report('the composer returns to ready after the image is shown', status === 'ready', status ?? '');
} finally {
  await cleanup();
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);
