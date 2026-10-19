#!/usr/bin/env node
/**
 * What a downloaded model's row is allowed to claim about running it.
 *
 * The models page labelled every `.safetensors`/`.bin`/`.pth` row with the
 * runtime that *could* load that format — "Transformers" — regardless of
 * whether any job in the app actually claims the repo. A model nothing can run
 * therefore read:
 *
 *     prajjwal1/bert-tiny   SAFETENSORS   Transformers · no job uses this
 *
 * Two claims in one line, and the one a reader believes is the runtime. The Hub
 * says the model is text-to-speech or speech-to-text, the app offers a runtime
 * name, and running it does nothing — which is the report that started this.
 *
 * The fix is that a row no picker offers shows its state and nothing else; the
 * format chip already says what the file is. This checks that on the live page,
 * because the rule lives in the page's own rendering, not in an API.
 *
 * The jobless set is computed here from `/api/hf/files` and `/api/runnable`
 * rather than read off the page, so the test cannot agree with a broken page by
 * using the page's own idea of what is runnable. It skips when this machine has
 * no jobless weights to look at — every check is about a rendered row.
 *
 * Run with the app up (`npm run dev`):
 *   npm run test:models-ui
 */
import { spawn, spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000/';
const CDP_PORT = Number(process.env.CDP_PORT ?? 9381);

/** Formats the page maps to a runtime label, and the ones that matter here. */
const WEIGHTS = /\.(safetensors|bin|pth|pt)$/i;

const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
if (!browser) {
  console.log('SKIP — no chromium/chrome found on PATH');
  process.exit(0);
}

/** Every repo a job in this app will run, straight from the app. */
async function runnableRepos() {
  const repos = new Set();
  for (const task of ['stt', 'tts', 'image']) {
    const res = await fetch(new URL(`/api/runnable?task=${task}`, APP_URL), {
      signal: AbortSignal.timeout(15000),
    });
    for (const m of (await res.json())?.[task] ?? []) repos.add(m.repo);
  }
  return repos;
}

let files;
try {
  const res = await fetch(new URL('/api/hf/files', APP_URL), { signal: AbortSignal.timeout(15000) });
  files = (await res.json())?.files ?? [];
} catch {
  console.log(`SKIP — app not reachable at ${APP_URL} (start it with npm run dev)`);
  process.exit(0);
}

const runnable = await runnableRepos();

/**
 * The rows this test is about: weights on disk that no job claims.
 *
 * A file the record marks `llama` or `transformers` belongs to a chat provider
 * instead, and those rows have their own — honest — state.
 */
const byRepo = new Map();
for (const f of files) {
  if (!f.repo) continue;
  if (!byRepo.has(f.repo)) byRepo.set(f.repo, []);
  byRepo.get(f.repo).push(f);
}
// Chat GGUFs carry a VRAM estimate from the server; the same page renders
// both kinds of row, so one visit can check both claims.
const estimated = files.filter((f) => f.llama?.vram);
const jobless = [...byRepo.entries()]
  .filter(([repo, fs]) => {
    if (runnable.has(repo)) return false;
    if (fs.some((f) => f.llama || f.transformers)) return false;
    return fs.some((f) => WEIGHTS.test(f.file ?? ''));
  })
  .map(([repo]) => repo)
  .sort();

if (jobless.length === 0 && estimated.length === 0) {
  console.log('SKIP — this machine has no downloaded weights to look at');
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PROFILE = join(tmpdir(), `starr-models-ui-${process.pid}`);
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
  console.log('\n=== downloaded-model rows (headless chromium) ===');
  console.log(`  jobless weights on this machine: ${jobless.join(', ')}`);

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
  const waitFor = async (expr, timeoutMs = 20000) => {
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
  await cdp('Page.navigate', { url: new URL('/settings', APP_URL).href });
  await waitFor(`document.readyState === 'complete'`);
  if (jobless.length) {
    const target = JSON.stringify(jobless[jobless.length - 1]);
    const shown = await waitFor(
      `[...document.querySelectorAll('.local-download-row .local-meta__repo')]
         .some(el => el.textContent === ${target})`,
    );
    report('the jobless model is listed on the models page', shown, `waiting for ${jobless.join(', ')}`);

    if (shown) {
    // Every row for every jobless repo, since a repo can be rendered more than
    // once (one row per bundle member) and the rule has to hold on all of them.
    const rows = JSON.parse(
      (await evalJs(`JSON.stringify([...document.querySelectorAll('.local-download-row')].map(r => ({
        repo: r.querySelector('.local-meta__repo')?.textContent ?? '',
        runtime: r.querySelector('.local-meta__runtime')?.textContent ?? '',
        state: [...r.querySelectorAll('.local-meta__state')].map(s => s.textContent).join(' | '),
        meta: r.querySelector('.local-meta')?.textContent ?? '',
      })))`)) ?? '[]',
    );
    const wanted = rows.filter((r) => jobless.includes(r.repo));

    report(
      'every jobless weights row was rendered',
      jobless.every((repo) => wanted.some((r) => r.repo === repo)),
      wanted.map((r) => r.repo).join(', ') || '(none)',
    );
    report(
      'a jobless row does not name a runtime that would load its format',
      wanted.every((r) => !r.runtime),
      wanted.map((r) => `${r.repo}: "${r.runtime}"`).join('; '),
    );
    report(
      'a jobless row says outright that nothing here runs it',
      wanted.every((r) => /no local runner/i.test(r.state)),
      wanted.map((r) => `${r.repo}: "${r.state}"`).join('; '),
    );
    report(
      'the old "Transformers · no job uses this" pairing is gone',
      wanted.every((r) => !/Transformer/i.test(r.meta) || !/no job uses this/i.test(r.meta)),
      wanted.map((r) => `${r.repo}: "${r.meta}"`).join('; '),
    );
    }
  }

  if (estimated.length) {
    const pick = estimated[0];
    const target = JSON.stringify(pick.repo);
    const shown = await waitFor(
      `[...document.querySelectorAll('.local-download-row .local-meta__repo')]
         .some(el => el.textContent === ${target})`,
    );
    report(
      'a GGUF chat model is listed on the models page',
      shown,
      `waiting for ${pick.repo}`,
    );
    if (shown) {
      const tags = JSON.parse(
        (await evalJs(`JSON.stringify([...document.querySelectorAll('.local-download-row')]
          .filter(r => r.querySelector('.local-meta__repo')?.textContent === ${target})
          .map(r => ({
            text: r.querySelector('.local-meta__vram')?.textContent ?? '',
            title: r.querySelector('.local-meta__vram')?.title ?? '',
          })))`)) ?? '[]',
      );
      report(
        'the row states its VRAM estimate and the arithmetic behind it',
        tags.length > 0 && tags.every((t) => /VRAM/.test(t.text) && /Estimated VRAM/.test(t.title)),
        tags.map((t) => `"${t.text}" / "${t.title}"`).join('; ') || '(no tag)',
      );
      const v = pick.llama.vram;
      const suffix =
        v.fit === 'fits' ? 'fits' : v.fit === 'tight' ? 'tight' : v.fit === 'over' ? 'available' : 'at ';
      report(
        'the row carries the fit verdict the API computed',
        tags.every((t) => t.text.includes(suffix)),
        `fit=${v.fit}, texts: ${tags.map((t) => t.text).join(' | ')}`,
      );
    }
  }
} finally {
  await cleanup();
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);
