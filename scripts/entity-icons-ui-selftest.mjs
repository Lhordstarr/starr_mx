#!/usr/bin/env node
/**
 * Provider and model icons, checked on the real pages.
 *
 * The app's icon language is inline SVG, not glyphs, and the provider marks are
 * generic role shapes rather than brand logos. This exercises both halves: the
 * id-to-mark mapping in Node, then the chat provider popover/chip, the chat
 * model pill/menu, the settings provider rows, and the Studio's chips and
 * model menu in headless Chromium.
 *
 * Providers and models are stubbed, so the check is about what the page renders
 * rather than what happens to be installed on this machine. Every UI assertion
 * fails on the pre-icon page, where the rows have no `svg.entity-icon` at all.
 *
 * Run with the app up (`npm run dev`):
 *   npm run test:entity-icons-ui
 */
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  modelIcon,
  modelIconKind,
  providerIcon,
  providerIconKind,
} from '../public/entity-icons.js';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000/';
const CDP_PORT = Number(process.env.CDP_PORT ?? 9384);

const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
if (!browser) {
  console.log('SKIP — no chromium/chrome found on PATH');
  process.exit(0);
}

try {
  const res = await fetch(new URL('/api/providers', APP_URL), { signal: AbortSignal.timeout(15000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
} catch {
  console.log(`SKIP — app not reachable at ${APP_URL} (start it with npm run dev)`);
  process.exit(0);
}

const CHAT_PROVIDERS = {
  providers: [
    {
      id: 'ollama',
      label: 'Ollama',
      local: true,
      configured: true,
      hint: 'Local models',
      models: [
        'llama3.2:3b',
        'qwen2.5-coder:7b',
        'llava:latest',
        'nomic-embed-text',
        'black-forest-labs/FLUX.1-schnell',
        'openai/whisper-large-v3',
      ],
      stt: false,
      sttModels: [],
    },
    {
      id: 'openai',
      label: 'OpenAI',
      local: false,
      configured: true,
      hint: 'Cloud',
      models: ['gpt-4o'],
      stt: true,
      sttModels: ['whisper-1'],
    },
    {
      id: 'my-llm',
      label: 'My Local Server',
      local: false,
      configured: true,
      hint: 'Custom',
      models: ['custom-model'],
      stt: false,
      sttModels: [],
    },
  ],
  localServers: [],
};

const SETTINGS = {
  storagePath: '/tmp/keys.json',
  fields: [
    {
      envKey: 'OPENAI_API_KEY',
      providerId: 'openai',
      label: 'OpenAI',
      configured: true,
      source: 'settings',
      masked: 'sk-a…z',
      placeholder: 'sk-...',
    },
    {
      envKey: 'HF_API_TOKEN',
      providerId: null,
      label: 'Hugging Face',
      configured: false,
      source: 'none',
      masked: null,
      placeholder: 'hf_...',
    },
    {
      envKey: 'CLOUDFLARE_ACCOUNT_ID',
      providerId: null,
      label: 'Cloudflare account id',
      configured: false,
      source: 'none',
      masked: null,
      placeholder: 'account id',
    },
  ],
};

const CUSTOM_PROVIDERS = {
  storagePath: '/tmp/custom.json',
  providers: [
    {
      id: 'my-llm',
      label: 'My LLM',
      baseUrl: 'http://localhost:1234/v1',
      hasKey: false,
      masked: null,
      stt: true,
      createdAt: 0,
    },
  ],
};

const DICTATION = {
  providers: [
    { id: 'openai', label: 'OpenAI', custom: false, configured: true, hasKey: true },
    { id: 'my-llm', label: 'My LLM', custom: true, configured: false, hasKey: false },
  ],
  local: [],
};

const IMAGE_PROVIDERS = {
  providers: [
    {
      id: 'pollinations',
      label: 'Pollinations',
      keyless: true,
      configured: true,
      hint: 'Keyless',
      models: ['flux'],
      defaultModel: 'flux',
      sizes: [{ w: 1024, h: 1024 }],
    },
  ],
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const PROFILE = join(tmpdir(), `starr-entity-icons-${process.pid}`);
rmSync(PROFILE, { recursive: true, force: true });

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
function check(name, fn) {
  try {
    fn();
    report(name, true);
  } catch (err) {
    report(name, false, err instanceof Error ? err.message : String(err));
  }
}

console.log('\n=== provider/model icon mapping ===');
check('provider ids get distinct generic marks', () => {
  assert.equal(providerIconKind('ollama'), 'layers');
  assert.equal(providerIconKind('llamacpp'), 'cpu');
  assert.equal(providerIconKind('openrouter'), 'route');
  assert.equal(providerIconKind('openai'), 'spark');
  assert.equal(providerIconKind('HF_API_TOKEN'), 'box');
  assert.equal(providerIconKind('CLOUDFLARE_ACCOUNT_ID'), 'cloud');
  assert.equal(providerIconKind('my-llm'), 'plug');
});
check('model ids are classified by name shape', () => {
  assert.equal(modelIconKind('llama3.2:3b'), 'chat');
  assert.equal(modelIconKind('qwen2.5-coder:7b'), 'code');
  assert.equal(modelIconKind('llava:latest'), 'vision');
  assert.equal(modelIconKind('openai/whisper-large-v3'), 'stt');
  assert.equal(modelIconKind('hexgrad/Kokoro-82M'), 'tts');
  assert.equal(modelIconKind('black-forest-labs/FLUX.1-schnell'), 'image');
  assert.equal(modelIconKind('nomic-embed-text'), 'embedding');
  assert.equal(modelIconKind('deepseek-r1:8b'), 'reasoning');
});
check('marks are stroke SVG, not glyphs or brand artwork', () => {
  const provider = providerIcon('openai');
  assert.match(provider, /class="entity-icon"/);
  assert.match(provider, /data-icon="spark"/);
  assert.match(provider, /fill="none"/);
  assert.doesNotMatch(provider, /<text/);
  assert.doesNotMatch(modelIcon('qwen2.5-coder'), /<text/);
  assert.match(modelIcon('qwen2.5-coder'), /data-icon="code"/);
});

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

async function getWsUrl() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)).json();
      const page = list.find((target) => target.type === 'page');
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
  const ws = new WebSocket(await getWsUrl());
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
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

    // Specific before general: `/api/providers` is a substring of both custom
    // providers and image providers, and the wrong stub would make the page
    // look broken for a reason unrelated to icons.
    if (/\/api\/custom-providers(?:\?|$)/.test(url)) {
      await fulfill(requestId, 'application/json', JSON.stringify(CUSTOM_PROVIDERS));
      return;
    }
    if (/\/api\/image\/providers(?:\?|$)/.test(url)) {
      await fulfill(requestId, 'application/json', JSON.stringify(IMAGE_PROVIDERS));
      return;
    }
    if (/\/api\/providers(?:\?|$)/.test(url)) {
      await fulfill(requestId, 'application/json', JSON.stringify(CHAT_PROVIDERS));
      return;
    }
    if (/\/api\/settings(?:\?|$)/.test(url)) {
      await fulfill(requestId, 'application/json', JSON.stringify(SETTINGS));
      return;
    }
    if (/\/api\/dictation(?:\?|$)/.test(url)) {
      await fulfill(requestId, 'application/json', JSON.stringify(DICTATION));
      return;
    }
    if (/\/api\/runnable\?task=image(?:&|$)/.test(url)) {
      await fulfill(requestId, 'application/json', JSON.stringify({ image: [], assigned: {} }));
      return;
    }
    await cdp('Fetch.continueRequest', { requestId });
  }

  ws.onmessage = (message) => {
    const msg = JSON.parse(message.data);
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
    return new Promise((resolve) => pending.set(id, resolve));
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
  const waitFor = async (expression, timeoutMs = 15000) => {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
      try {
        if (await evalJs(expression)) return true;
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
      { urlPattern: '*api/custom-providers*', requestStage: 'Request' },
      { urlPattern: '*api/image/providers*', requestStage: 'Request' },
      { urlPattern: '*api/providers*', requestStage: 'Request' },
      { urlPattern: '*api/settings*', requestStage: 'Request' },
      { urlPattern: '*api/dictation*', requestStage: 'Request' },
      { urlPattern: '*api/runnable?task=image*', requestStage: 'Request' },
    ],
  });

  /* ---------------- Chat: providers ---------------- */
  console.log('\n=== chat providers ===');
  await cdp('Page.navigate', { url: new URL('/', APP_URL).href });
  await waitFor(`document.readyState === 'complete' && !!document.getElementById('open-panel')`);
  const providersReady = await waitFor(
    `!!document.querySelector('#provider-list .popover__item[data-id="ollama"]')`,
  );
  report('the stubbed provider list is on the page', providersReady);

  if (providersReady) {
    await evalJs(`document.getElementById('open-panel').click()`);
    await waitFor(`document.getElementById('provider-pop').hidden === false`);

    const rows = JSON.parse(
      (await evalJs(`JSON.stringify([...document.querySelectorAll('#provider-list .popover__item')].map((row) => ({
        id: row.dataset.id,
        label: row.querySelector('.popover__item-label')?.textContent ?? '',
        icon: row.querySelector('svg.entity-icon')?.dataset.icon ?? '',
      })))`)) ?? '[]',
    );
    const byId = Object.fromEntries(rows.map((row) => [row.id, row]));
    report(
      'every provider row carries an icon',
      rows.length === 3 && rows.every((row) => row.icon),
      JSON.stringify(rows),
    );
    report(
      'local, cloud and custom providers get distinct marks',
      byId.ollama?.icon === 'layers' && byId.openai?.icon === 'spark' && byId['my-llm']?.icon === 'plug',
      JSON.stringify({ ollama: byId.ollama?.icon, openai: byId.openai?.icon, custom: byId['my-llm']?.icon }),
    );
    report(
      'the provider chip has a mark and keeps only the label as its text',
      (await evalJs(`document.querySelector('#chip-provider-icon svg.entity-icon')?.dataset.icon`)) === 'layers' &&
        (await evalJs(`document.getElementById('chip-provider').textContent`)) === 'Ollama',
    );
    report(
      'the icon pass did not bring back the "runs on this PC" tag',
      !(await evalJs(`document.getElementById('provider-pop').textContent`)).includes('runs on this PC'),
    );

    /* ---------------- Chat: models ---------------- */
    console.log('\n=== chat models ===');
    await evalJs(`document.getElementById('model-trigger').click()`);
    const menuReady = await waitFor(
      `document.getElementById('model-menu').hidden === false && document.querySelectorAll('#model-list .mm-option').length >= 6`,
    );
    report('the stubbed model menu opens', menuReady);

    if (menuReady) {
      const options = JSON.parse(
        (await evalJs(`JSON.stringify([...document.querySelectorAll('#model-list .mm-option')].map((row) => ({
          id: row.dataset.model,
          icon: row.querySelector('svg.entity-icon')?.dataset.icon ?? '',
        })))`)) ?? '[]',
      );
      const optionByModel = Object.fromEntries(options.map((option) => [option.id, option]));
      report(
        'every model row carries an icon',
        options.length === 6 && options.every((option) => option.icon),
        JSON.stringify(options),
      );
      report(
        'model kinds are distinct in the picker',
        optionByModel['llama3.2:3b']?.icon === 'chat' &&
          optionByModel['qwen2.5-coder:7b']?.icon === 'code' &&
          optionByModel['llava:latest']?.icon === 'vision' &&
          optionByModel['nomic-embed-text']?.icon === 'embedding' &&
          optionByModel['black-forest-labs/FLUX.1-schnell']?.icon === 'image' &&
          optionByModel['openai/whisper-large-v3']?.icon === 'stt',
        JSON.stringify(optionByModel),
      );
      const currentModel = await evalJs(`document.getElementById('model-current').textContent`);
      const currentIcon = await evalJs(
        `document.querySelector('#model-icon svg.entity-icon')?.dataset.icon ?? ''`,
      );
      report(
        'the model pill shows the selected model kind',
        currentIcon === modelIconKind(currentModel),
        JSON.stringify({ current: currentModel, icon: currentIcon, expected: modelIconKind(currentModel) }),
      );

      await evalJs(`document.querySelector('#model-list .mm-option[data-model="qwen2.5-coder:7b"]').click()`);
      const codePicked = await waitFor(
        `document.getElementById('model-current').textContent === 'qwen2.5-coder:7b'`,
      );
      report(
        'picking a code model updates the pill mark',
        codePicked &&
          (await evalJs(`document.querySelector('#model-icon svg.entity-icon')?.dataset.icon`)) === 'code',
      );
    }

    // Select OpenAI after the model checks, so the provider chip can be shown
    // updating rather than only being right on first paint.
    await evalJs(`document.getElementById('open-panel').click()`);
    await waitFor(`document.getElementById('provider-pop').hidden === false`);
    await evalJs(`document.querySelector('#provider-list .popover__item[data-id="openai"]').click()`);
    const openAiSelected = await waitFor(
      `document.getElementById('chip-provider').textContent === 'OpenAI'`,
    );
    report(
      'switching providers swaps the chip mark',
      openAiSelected &&
        (await evalJs(`document.querySelector('#chip-provider-icon svg.entity-icon')?.dataset.icon`)) === 'spark',
    );
  }

  /* ---------------- Settings: provider rows ---------------- */
  console.log('\n=== settings providers ===');
  await cdp('Page.navigate', { url: new URL('/settings#keys', APP_URL).href });
  const keysReady = await waitFor(`document.readyState === 'complete' && !!document.querySelector('#keys .key-row')`);
  report('the stubbed key rows are on the page', keysReady);

  if (keysReady) {
    const keyRows = JSON.parse(
      (await evalJs(`JSON.stringify([...document.querySelectorAll('#keys .key-row')].map((row) => ({
        text: row.textContent,
        icon: row.querySelector('svg.entity-icon')?.dataset.icon ?? '',
      })))`)) ?? '[]',
    );
    const keyIcon = (envKey) => keyRows.find((row) => row.text.includes(envKey))?.icon ?? '';
    report(
      'key rows carry the provider mark, including image-only keys',
      keyRows.length === 3 && keyRows.every((row) => row.icon),
      JSON.stringify(keyRows),
    );
    report(
      'key marks distinguish chat and image providers',
      keyIcon('OPENAI_API_KEY') === 'spark' &&
        keyIcon('HF_API_TOKEN') === 'box' &&
        keyIcon('CLOUDFLARE_ACCOUNT_ID') === 'cloud',
      JSON.stringify({ openai: keyIcon('OPENAI_API_KEY'), hf: keyIcon('HF_API_TOKEN'), cf: keyIcon('CLOUDFLARE_ACCOUNT_ID') }),
    );

    const customReady = await waitFor(`!!document.querySelector('#custom-list .custom-row')`);
    report(
      'custom provider rows carry a mark',
      customReady &&
        (await evalJs(`document.querySelector('#custom-list .custom-row .provider svg.entity-icon')?.dataset.icon`)) === 'plug',
    );

    const dictationReady = await waitFor(`document.querySelectorAll('#dictation-list .dictation-row').length === 2`);
    const dictationIcons = dictationReady
      ? JSON.parse(
          (await evalJs(`JSON.stringify([...document.querySelectorAll('#dictation-list .dictation-row')].map((row) => ({
            text: row.textContent,
            icon: row.querySelector('svg.entity-icon')?.dataset.icon ?? '',
          })))`)) ?? '[]',
        )
      : [];
    report(
      'dictation rows use the provider mark rather than a repeated mic glyph',
      dictationIcons.length === 2 &&
        dictationIcons.every((row) => row.icon) &&
        dictationIcons.find((row) => row.text.includes('OpenAI'))?.icon === 'spark' &&
        dictationIcons.find((row) => row.text.includes('My LLM'))?.icon === 'plug' &&
        !(await evalJs(`!!document.querySelector('#dictation-list .mic-mini')`)),
      JSON.stringify(dictationIcons),
    );
  }

  /* ---------------- Studio: chips and model menu ---------------- */
  console.log('\n=== studio ===');
  await cdp('Page.navigate', { url: new URL('/studio', APP_URL).href });
  const studioReady = await waitFor(
    `document.readyState === 'complete' && !!document.querySelector('#studio-img-provider option[value="pollinations"]')`,
    20000,
  );
  report('the stubbed image provider is on the page', studioReady);

  if (studioReady) {
    report(
      'the Studio provider chip carries the image provider mark',
      (await evalJs(`document.getElementById('chip-provider').textContent`)) === 'Pollinations' &&
        (await evalJs(`document.querySelector('#chip-provider-icon svg.entity-icon')?.dataset.icon`)) === 'sparkles',
    );

    await evalJs(`document.getElementById('model-trigger').click()`);
    const imageMenuReady = await waitFor(
      `document.getElementById('model-menu').hidden === false && !!document.querySelector('#model-list .mm-option')`,
    );
    report(
      'Studio model rows carry the model mark',
      imageMenuReady &&
        (await evalJs(`document.querySelector('#model-list .mm-option svg.entity-icon')?.dataset.icon`)) === 'image' &&
        (await evalJs(`document.querySelector('#model-icon svg.entity-icon')?.dataset.icon`)) === 'image',
    );
  }
} finally {
  await cleanup();
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((failure) => `  - ${failure}`).join('\n'));
process.exit(failures.length ? 1 : 0);
