#!/usr/bin/env node
/**
 * Frontend checks for the /tool picker and its chips, driven in headless
 * Chromium over the DevTools protocol. There is no frontend test framework in
 * this project (the UI is plain modules with no build step), so this owns
 * that layer: picker open/close, selection, chips, keyboard, outside-click,
 * the exact tools payload on the wire, and failure-preservation.
 *
 * Run with the app up (`npm run dev` or `npm start`), then:
 *   npm run test:ui-tools
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
const PORT = Number(process.env.CDP_PORT ?? 9377);

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
const PROFILE = join(tmpdir(), `starr-ui-tools-${process.pid}`);
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
    // SIGKILL lands immediately, but Chromium's helper processes can outlive
    // the parent by a moment and still be writing into the profile — which is
    // how a fully passing run used to crash in the teardown with ENOTEMPTY.
    await new Promise((resolve) => {
      if (chrome.exitCode !== null) return resolve();
      chrome.once('exit', resolve);
      setTimeout(resolve, 3000);
    });
  }
  // fs.rm retries ENOTEMPTY/EBUSY for maxRetries; a leftover temp dir is still
  // better than crashing a suite whose checks all passed.
  await rm(PROFILE, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => {});
}

try {
  console.log('\n=== /tool picker UI (headless chromium) ===');

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
  await waitFor(`document.readyState === 'complete' && !!document.getElementById('input')`);

  report(
    'page loads with the picker closed and no chips',
    await evalJs(
      `document.getElementById('tool-menu').hidden && document.getElementById('tools-box').hidden`,
    ),
  );

  /* The top bar takes the nav glass recipe, not the panel one: a specular
     inset plus a separate offset shadow beneath. Split on top-level commas
     so the rgba() argument lists are not mistaken for shadow layers. */
  const navShadow = String(
    await evalJs(`getComputedStyle(document.querySelector('.topbar')).boxShadow`),
  );
  const navLayers = navShadow
    .split(/,(?![^(]*\))/)
    .map((s) => s.trim())
    .filter(Boolean);
  report(
    'the top bar carries both a specular rim and an ambient shadow',
    navLayers.some((l) => l.includes('inset')) &&
      navLayers.some((l) => !l.includes('inset')),
    navShadow,
  );

  /* "/tool" opens the picker without being submitted or left behind. */
  await evalJs(`(() => {
    const i = document.getElementById('input');
    i.value = '/tool';
    i.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
  report(
    'typing "/tool" opens the picker',
    await waitFor(`document.getElementById('tool-menu').hidden === false`),
  );
  report('"/tool" is not left in the input', (await evalJs(`document.getElementById('input').value`)) === '');
  report(
    'the picker lists all three built-ins with labels and icons',
    (await evalJs(
      `JSON.stringify([...document.querySelectorAll('.tm-row')].map(r => ({
         name: r.dataset.name,
         label: r.querySelector('.tm-name').textContent,
         icon: !!r.querySelector('svg'),
         disabled: r.disabled,
       })))`,
    )) ===
      JSON.stringify([
        { name: 'web_search', label: 'Web Search', icon: true, disabled: false },
        { name: 'fetch_page', label: 'Fetch Page', icon: true, disabled: false },
        { name: 'wikipedia', label: 'Wikipedia', icon: true, disabled: false },
      ]),
  );
  report(
    'the Web Search row carries the globe icon',
    await evalJs(`(() => {
      const svg = document.querySelector('.tm-row[data-name=web_search] svg');
      return !!svg && svg.innerHTML.includes('<circle') && svg.innerHTML.includes('M2 12h20');
    })()`),
  );

  /* Selecting paints a chip; clicking again removes it; no duplicates. */
  await evalJs(`document.querySelector('.tm-row[data-name=web_search]').click()`);
  report(
    'selecting Web Search shows the "Web Search ✕" chip',
    await evalJs(`(() => {
      const box = document.getElementById('tools-box');
      const chip = box.querySelector('.tool-chip');
      return !box.hidden && !!chip &&
        chip.textContent.includes('Web Search') &&
        !!chip.querySelector('svg') &&
        chip.querySelector('.tool-chip-remove')?.getAttribute('aria-label') === 'Remove Web Search';
    })()`),
  );
  report(
    'the row turns selected and the picker stays open',
    await evalJs(
      `document.querySelector('.tm-row[data-name=web_search]').getAttribute('aria-selected') === 'true' &&
       document.getElementById('tool-menu').hidden === false`,
    ),
  );
  await evalJs(`document.querySelector('.tm-row[data-name=web_search]').click()`);
  report(
    'clicking the row again removes the chip',
    await evalJs(
      `document.getElementById('tools-box').hidden && document.querySelectorAll('.tool-chip').length === 0`,
    ),
  );

  /* A partial command word is ordinary typing. */
  await evalJs(
    `document.getElementById('input').dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))`,
  );
  await evalJs(`(() => {
    const i = document.getElementById('input');
    i.value = '/tools';
    i.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
  report(
    '"/tools" is ordinary text: no picker, input intact',
    (await evalJs(`document.getElementById('tool-menu').hidden`)) === true &&
      (await evalJs(`document.getElementById('input').value`)) === '/tools',
  );
  await evalJs(`document.getElementById('input').value = ''`);

  /* Two tools, independent removal, no duplicates. */
  await evalJs(`document.querySelector('.tm-row[data-name=wikipedia]').click();
               document.querySelector('.tm-row[data-name=web_search]').click();`);
  report(
    'two selections produce two chips in click order',
    (await evalJs(
      `JSON.stringify([...document.querySelectorAll('.tool-chip .tc-name')].map(n => n.textContent))`,
    )) === JSON.stringify(['Wikipedia', 'Web Search']),
  );
  await evalJs(`document.querySelector('.tool-chip-remove[aria-label="Remove Web Search"]').click()`);
  report(
    'removing one chip leaves the other',
    (await evalJs(
      `JSON.stringify([...document.querySelectorAll('.tool-chip .tc-name')].map(n => n.textContent))`,
    )) === JSON.stringify(['Wikipedia']),
  );
  await evalJs(`document.querySelector('.tm-row[data-name=wikipedia]').click()`);
  await evalJs(`document.querySelector('.tm-row[data-name=wikipedia]').click()`);
  report(
    'toggling the same tool never leaves duplicates',
    await evalJs(
      `document.querySelectorAll('.tool-chip').length === 1 &&
       document.querySelector('.tool-chip .tc-name').textContent === 'Wikipedia'`,
    ),
  );

  /* Escape and outside-click close; picks survive. */
  await evalJs(
    `document.getElementById('input').dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))`,
  );
  report(
    'Escape closes the picker, chips survive',
    await evalJs(
      `document.getElementById('tool-menu').hidden && document.querySelectorAll('.tool-chip').length === 1`,
    ),
  );
  await evalJs(`(() => {
    const i = document.getElementById('input');
    i.value = '/tool';
    i.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
  await waitFor(`document.getElementById('tool-menu').hidden === false`);
  await evalJs(`document.body.click()`);
  report(
    'clicking outside closes the picker',
    await evalJs(`document.getElementById('tool-menu').hidden`),
  );

  /* The picker has to be closable by the person who opened it. Clicking away
     from a pop-up is not a way to close it: the click lands on whatever was
     under it, so the thing they were aiming at is the thing that gets
     clicked. */
  await evalJs(`(() => {
    const i = document.getElementById('input');
    i.value = '/tool';
    i.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
  await waitFor(`document.getElementById('tool-menu').hidden === false`);
  report(
    'the picker has a close button',
    await evalJs(
      `!!document.getElementById('tool-menu-close') &&
       document.getElementById('tool-menu-close').getAttribute('aria-label') === 'Close the tools list'`,
    ),
  );
  report(
    'and it sits at the top of the pop-up, beside its title',
    await evalJs(`(() => {
      const head = document.querySelector('.tm-head');
      const btn = document.getElementById('tool-menu-close');
      if (!head || !btn) return false;
      const h = head.getBoundingClientRect();
      const b = btn.getBoundingClientRect();
      return b.top >= h.top && b.bottom <= h.bottom + 1 && b.right <= h.right + 1;
    })()`),
  );
  await evalJs(`document.getElementById('tool-menu-close').click()`);
  report(
    'clicking it closes the picker',
    await evalJs(`document.getElementById('tool-menu').hidden`),
  );
  report(
    'and the focus goes back to the input, the way Escape does',
    await evalJs(`document.activeElement?.id === 'input'`),
  );
  report(
    'without losing what was picked',
    await evalJs(`document.querySelectorAll('.tool-chip').length === 1`),
    String(await evalJs(`document.querySelectorAll('.tool-chip').length`)),
  );

  /* Keyboard: arrows focus rows, Enter picks, Escape returns to the input. */
  await evalJs(`(() => {
    const i = document.getElementById('input');
    i.focus();
    i.value = '/tool';
    i.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
  await waitFor(`document.getElementById('tool-menu').hidden === false`);
  await evalJs(
    `document.getElementById('input').dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }))`,
  );
  report(
    'ArrowDown from the input focuses the first row',
    await evalJs(`document.activeElement?.dataset?.name === 'web_search'`),
  );
  await evalJs(
    `document.activeElement.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))`,
  );
  report(
    'Enter on a focused row picks it (picker stays open)',
    await evalJs(
      `document.querySelector('.tm-row[data-name=web_search]').getAttribute('aria-selected') === 'true' &&
       document.getElementById('tool-menu').hidden === false`,
    ),
  );
  await evalJs(
    `document.activeElement.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))`,
  );
  report(
    'Escape from a row closes and returns focus to the input',
    await evalJs(`document.getElementById('tool-menu').hidden && document.activeElement.id === 'input'`),
  );

  /* The wire: capture the next /api/chat body and stop it before it runs. */
  await evalJs(`document.querySelector('.tool-chip-remove[aria-label="Remove Web Search"]').click()`);
  await evalJs(`(() => {
    window.__bodies = [];
    const orig = window.fetch;
    window.__origFetch = orig;
    window.fetch = (...args) => {
      if (String(args[0]).includes('/api/chat')) {
        window.__bodies.push(args[1].body);
        return Promise.reject(new Error('stubbed for payload check'));
      }
      return orig(...args);
    };
  })()`);
  await evalJs(`(() => {
    const i = document.getElementById('input');
    i.value = 'hello there';
    i.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('composer').requestSubmit();
  })()`);
  await sleep(400);
  report(
    'the send carries the selected tool ids as an array',
    (await evalJs(`(() => {
      if (!window.__bodies.length) return 'no body captured';
      const body = JSON.parse(window.__bodies.at(-1));
      return JSON.stringify(body.tools);
    })()`)) === JSON.stringify(['wikipedia']),
  );
  report(
    'a failed send keeps the selection for Retry',
    await evalJs(`document.querySelectorAll('.tool-chip').length === 1`),
  );
  await evalJs(`window.fetch = window.__origFetch`);
  await evalJs(`document.querySelector('.tool-chip-remove').click()`);
  report(
    'the chip remove button disarms the tool',
    await evalJs(
      `document.getElementById('tools-box').hidden && document.querySelectorAll('.tool-chip').length === 0`,
    ),
  );

  /* ---- the always-on switch ----

     The complaint it answers: the tools were armed for one message and then
     gone, so every reply had to go through the picker again. The switch is
     the one control that says "not this time" — the model may reach for the
     tools on every message, until it is turned off. */
  report(
    'the composer carries a switch for it',
    await evalJs(
      `!!document.getElementById('tools-always') && document.getElementById('tools-always').getAttribute('role') === 'switch'`,
    ),
  );
  report(
    'and it starts off, like the tools themselves',
    await evalJs(
      `document.getElementById('tools-always').dataset.on !== '1' &&
       document.getElementById('tools-always').getAttribute('aria-checked') === 'false'`,
    ),
  );

  // A successful reply, so the "the selection is released when the reply
  // finishes" path is the one being exercised.
  await evalJs(`(() => {
    window.__bodies = [];
    const orig = window.fetch;
    window.__origFetch = orig;
    window.fetch = (...args) => {
      if (String(args[0]).includes('/api/chat')) {
        window.__bodies.push(args[1].body);
        const body = ${JSON.stringify('event: delta\ndata: {"text":"ok"}\n\nevent: done\ndata: {}\n\n')};
        return Promise.resolve(new Response(new ReadableStream({
          start(c) { c.enqueue(new TextEncoder().encode(body)); c.close(); },
        }), { status: 200, headers: { 'content-type': 'text/event-stream' } }));
      }
      return orig(...args);
    };
  })()`);

  await evalJs(`document.getElementById('tools-always').click()`);
  report(
    'turning it on arms the tools the registry has',
    await waitFor(`document.querySelectorAll('.tool-chip').length === 3`, 8000),
    String(await evalJs(`document.querySelectorAll('.tool-chip').length`)),
  );
  report(
    'and the switch says it is on',
    await evalJs(
      `document.getElementById('tools-always').dataset.on === '1' &&
       document.getElementById('tools-always').getAttribute('aria-checked') === 'true'`,
    ),
  );
  report(
    'and the chips say they are not just for this message',
    await evalJs(
      `!!document.querySelector('.tools-always') &&
       /every message/i.test(document.querySelector('.tools-always').textContent)`,
    ),
  );
  report(
    'and the button says what is armed',
    await evalJs(`/3 tools/.test(document.getElementById('tools-pick-label').textContent)`),
    String(await evalJs(`document.getElementById('tools-pick-label').textContent`)),
  );

  // Now a message, and a reply that finishes: the tools must still be armed.
  await evalJs(`(() => {
    const i = document.getElementById('input');
    i.value = 'hello there';
    i.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('composer').requestSubmit();
  })()`);
  await waitFor(`window.__bodies.length > 0`, 8000);
  await sleep(600);
  report(
    'the message carries the armed tools',
    (await evalJs(`(() => {
      if (!window.__bodies.length) return 'no body captured';
      const body = JSON.parse(window.__bodies.at(-1));
      return JSON.stringify(body.tools);
    })()`)) ===
      JSON.stringify(['web_search', 'fetch_page', 'wikipedia']),
  );
  report(
    'and they are still armed after the reply finished',
    await evalJs(`document.querySelectorAll('.tool-chip').length === 3`),
    String(await evalJs(`document.querySelectorAll('.tool-chip').length`)),
  );

  // A reload must not quietly disarm them: "until you turn it back off".
  await cdp('Page.navigate', { url: APP_URL });
  await waitFor(`document.readyState === 'complete' && !!document.getElementById('tools-always')`);
  report(
    'the switch is still on after a reload',
    await waitFor(`document.getElementById('tools-always').dataset.on === '1'`, 8000),
  );
  report(
    'and the tools are still armed',
    await waitFor(`document.querySelectorAll('.tool-chip').length === 3`, 8000),
    String(await evalJs(`document.querySelectorAll('.tool-chip').length`)),
  );
  report(
    'and the restored chips carry the tool names, not the registry ids',
    await waitFor(
      `[...document.querySelectorAll('.tool-chip .tc-name')].map(n => n.textContent).join(',') === 'Web Search,Fetch Page,Wikipedia'`,
      8000,
    ),
    String(
      await evalJs(`[...document.querySelectorAll('.tool-chip .tc-name')].map(n => n.textContent).join(',')`),
    ),
  );

  // Turning it off gives the old per-message behaviour back.
  await evalJs(`document.getElementById('tools-always').click()`);
  await waitFor(`document.getElementById('tools-always').dataset.on !== '1'`);
  report(
    'turning it off stops the tools being armed for every message',
    await evalJs(`document.getElementById('tools-always').getAttribute('aria-checked') === 'false'`),
  );
  report(
    'and the chips no longer claim to be always on',
    await evalJs(`!document.querySelector('.tools-always')`),
  );

  /* Narrow screens: the picker stays inside the viewport. */
  await cdp('Emulation.setDeviceMetricsOverride', {
    width: 390,
    height: 844,
    deviceScaleFactor: 2,
    mobile: true,
  });
  await evalJs(`(() => {
    const i = document.getElementById('input');
    i.value = '/tool';
    i.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
  await waitFor(`document.getElementById('tool-menu').hidden === false`);
  report(
    'at 390px the picker fits inside the viewport',
    await evalJs(`(() => {
      const r = document.getElementById('tool-menu').getBoundingClientRect();
      return r.left >= 0 && r.right <= window.innerWidth && r.width > 200;
    })()`),
  );
  await cdp('Emulation.clearDeviceMetricsOverride');

  /* ---- settings: the on-state track is a live gradient ---- */

  await cdp('Emulation.setEmulatedMedia', {
    features: [{ name: 'prefers-color-scheme', value: 'dark' }],
  });
  await cdp('Page.navigate', { url: new URL('settings#tools', APP_URL).href });
  const gotToggle = await waitFor(
    `document.readyState === 'complete' && document.querySelectorAll('.tool-toggle').length > 0`,
    15000,
  );
  report('the settings page renders tool switches', gotToggle, 'no .tool-toggle found');

  if (gotToggle) {
    const readToggle = () =>
      evalJs(`(() => {
        const t = document.querySelector('.tool-toggle');
        const track = getComputedStyle(t, '::before');
        const knob = getComputedStyle(t, '::after');
        return {
          on: t.dataset.on === '1',
          opacity: track.opacity,
          animation: track.animationName,
          play: track.animationPlayState,
          image: track.backgroundImage,
          knobLeft: knob.left,
        };
      })()`);

    const start = await readToggle();
    if (!start.on) {
      await evalJs(`document.querySelector('.tool-toggle').click()`);
      await waitFor(`document.querySelector('.tool-toggle').dataset.on === '1'`);
    }
    const on = await readToggle();
    report(
      'a switch that is on streams a gradient through its track',
      on.opacity === '1' &&
        on.animation === 'tool-toggle-stream' &&
        on.play === 'running' &&
        on.image.includes('linear-gradient'),
      JSON.stringify(on),
    );
    report('the knob sits at the right end when on', on.knobLeft === '20px', on.knobLeft);

    /* The accessibility path keeps the gradient but stops the movement. */
    await cdp('Emulation.setEmulatedMedia', {
      features: [
        { name: 'prefers-color-scheme', value: 'dark' },
        { name: 'prefers-reduced-motion', value: 'reduce' },
      ],
    });
    await sleep(150);
    report(
      'reduced motion freezes the stream instead of dropping the state',
      await evalJs(
        `getComputedStyle(document.querySelector('.tool-toggle'), '::before').animationName === 'none'`,
      ),
    );
    await cdp('Emulation.setEmulatedMedia', {
      features: [{ name: 'prefers-color-scheme', value: 'dark' }],
    });

    await evalJs(`document.querySelector('.tool-toggle').click()`);
    await waitFor(`document.querySelector('.tool-toggle').dataset.on === ''`);
    await waitFor(
      `getComputedStyle(document.querySelector('.tool-toggle'), '::before').opacity === '0'`,
    );
    const off = await readToggle();
    report(
      'a switch that is off fades the stream out and pauses it',
      off.opacity === '0' && off.play === 'paused',
      JSON.stringify(off),
    );
    report('the knob returns to the left when off', off.knobLeft === '2px', off.knobLeft);

    // Leave the setting exactly as it was found.
    if (start.on) {
      await evalJs(`document.querySelector('.tool-toggle').click()`);
      await waitFor(`document.querySelector('.tool-toggle').dataset.on === '1'`);
    }
  }

  ws.close();
} finally {
  await cleanup();
}

console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length ? 1 : 0);
