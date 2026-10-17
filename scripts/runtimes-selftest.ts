#!/usr/bin/env tsx
/**
 * Runtime install checks: the venv failure text, Ollama's per-OS install, the
 * on-demand password prompt, and the browser half of the password contract.
 *
 * Three things are being asserted here, and they are the same three that broke
 * on a fresh machine:
 *
 *  - a Python without the venv module says which package to install, instead of
 *    relaying "ensurepip is not available";
 *  - Ollama has an install command on every supported OS rather than a
 *    Linux-only dead end, and only the one that needs root says so;
 *  - the password box appears when an installer asks for one, and not before.
 *
 * The server needs to be up for the HTTP and browser sections
 * (`npm run dev`); those skip rather than fail without it. The unit checks
 * always run.
 */
import { spawn, spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describeVenvFailure, needsPrivilege, ollamaRecipe } from '../src/runtimes.ts';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000';
const PORT = Number(process.env.CDP_PORT ?? 9381);

let passed = 0;
const failures: string[] = [];
function report(name: string, ok: boolean, extra = '') {
  if (ok) {
    passed += 1;
    console.log(`  PASS  ${name}`);
  } else {
    failures.push(name);
    console.log(`  FAIL  ${name}${extra ? ` — ${extra}` : ''}`);
  }
}
function section(title: string) {
  console.log(`\n${title}`);
}

/* ---- 1. The venv failure names the package, not ensurepip ---- */
section('Python environment errors');
{
  // What a fresh Ubuntu actually prints.
  const ubuntu = describeVenvFailure(
    'The virtual environment was not created successfully because the ensurepip module is not available.',
  );
  report(
    'an ensurepip failure names the package to install',
    /python3-venv/.test(ubuntu) && !/ensurepip/.test(ubuntu),
    ubuntu,
  );
  report('the Ubuntu hint says apt', /sudo apt install python3-venv/.test(ubuntu), ubuntu);

  const noVenv = describeVenvFailure('No module named venv');
  report('a missing venv module also names the package', /python3-venv/.test(noVenv), noVenv);

  // Anything unrecognised keeps the interpreter's own words: a guess here
  // would be worse than the truth.
  const odd = describeVenvFailure('Error: [Errno 28] No space left on device');
  report(
    'an unrecognised failure keeps the raw message',
    odd.includes('No space left on device'),
    odd,
  );
  report('an empty failure does not become an empty error', describeVenvFailure('').length > 0);
}

/* ---- 2. Ollama has a command on every OS ---- */
section('Ollama installer per OS');
{
  const linux = ollamaRecipe('linux');
  report('Linux uses the vendor script', /install\.sh/.test(linux?.shown ?? ''), linux?.shown);
  report('Linux is the one that needs root', linux?.needsRoot === true);

  const mac = ollamaRecipe('darwin');
  report('macOS uses Homebrew', /brew install ollama/.test(mac?.shown ?? ''), mac?.shown);
  report(
    'macOS is not escalated (brew refuses to run as root)',
    mac?.needsRoot === false,
  );

  const win = ollamaRecipe('win32');
  report('Windows uses the vendor PowerShell script', /install\.ps1/.test(win?.shown ?? ''), win?.shown);
  report('Windows installs per-user, so no elevation', win?.needsRoot === false);
  report(
    'the Windows command is spawned, not piped through a shell',
    win?.cmd.includes('powershell.exe') && win?.cmd.includes('-NoProfile'),
    win?.cmd.join(' '),
  );

  report('an unsupported OS admits it rather than guessing', ollamaRecipe('aix' as NodeJS.Platform) === null);
}

/* ---- 3. "Needs a password" is read from the output, not the exit code ---- */
section('Privilege detection');
{
  report('sudo without a tty is recognised', needsPrivilege('sudo: a terminal is required to read the password'));
  report('the newer sudo wording is recognised', needsPrivilege('sudo: no tty present and no askpass program specified'));
  report('a permission denial is recognised', needsPrivilege('cp: /usr/local/bin/ollama: Permission denied'));
  report(
    'an ordinary install failure is not read as a password problem',
    !needsPrivilege('curl: (7) Failed to connect to ollama.com port 443: Connection timed out'),
  );
  report(
    'an empty log is not a privilege problem',
    !needsPrivilege(''),
  );
}

/* ---- 4. The server refuses a nonsense install and answers a real one ---- */
section('Server contract');
let serverUp = false;
try {
  const res = await fetch(`${APP_URL}/api/runtimes`, { signal: AbortSignal.timeout(3000) });
  serverUp = res.ok;
} catch {
  serverUp = false;
}
if (!serverUp) {
  console.log(`  SKIP  server not reachable at ${APP_URL} (start it with npm run dev)`);
} else {
  const status = await fetch(`${APP_URL}/api/runtimes`).then((r) => r.json());
  report(
    'runtimes status still lists every group',
    Boolean(status?.runtimes?.installed && Array.isArray(status?.groups)),
  );
  // Ollama is detected, not assumed: the old code answered "installed" on any
  // OS without looking, which is what made a missing install unrecoverable.
  // Read with one retry, because the server answers this from a probe that can
  // time out on a busy machine and a timeout is not what is under test here.
  let ollamaState: unknown;
  for (let i = 0; i < 2; i += 1) {
    ollamaState = (await fetch(`${APP_URL}/api/runtimes`).then((r) => r.json()))?.runtimes?.installed
      ?.ollama;
    if (ollamaState !== undefined) break;
    await new Promise((r) => setTimeout(r, 1500));
  }
  report(
    'ollama is reported from the machine, not assumed',
    typeof ollamaState === 'string' || ollamaState === null,
    String(ollamaState),
  );

  const bad = await fetch(`${APP_URL}/api/runtimes/install`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ groups: ['definitely-not-a-runtime'] }),
  });
  report('an unknown runtime id is refused', bad.status === 400, String(bad.status));

  const empty = await fetch(`${APP_URL}/api/runtimes/install`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ groups: [] }),
  });
  report('an empty selection is refused', empty.status === 400, String(empty.status));
}

/* ---- 5. The browser: no password box unless one is asked for ---- */
const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
if (!browser) {
  console.log('\n  SKIP  no chromium/chrome found on PATH');
} else if (!serverUp) {
  console.log('\n  SKIP  browser checks need the server up');
} else {
  console.log('\nBrowser: the password prompt is on demand');
  const PROFILE = join(tmpdir(), `starr-runtimes-${process.pid}`);
  rmSync(PROFILE, { recursive: true, force: true });
  const chrome = spawn(
    browser,
    [
      '--headless=new',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--window-size=1280,1400',
      `--remote-debugging-port=${PORT}`,
      `--user-data-dir=${PROFILE}`,
      'about:blank',
    ],
    { stdio: 'ignore' },
  );
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  const cleanup = () => {
    try {
      chrome.kill('SIGKILL');
    } catch {
      /* already gone */
    }
    // Chromium writes its profile out as it dies, so this can lose a race with
    // it. A leftover temp directory is not worth failing a check over.
    try {
      rmSync(PROFILE, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    } catch {
      /* leaving it behind */
    }
  };

  try {
    let wsUrl = '';
    for (let i = 0; i < 60; i += 1) {
      try {
        const list = (await (
          await fetch(`http://127.0.0.1:${PORT}/json/list`)
        ).json()) as Array<{ type: string; webSocketDebuggerUrl?: string }>;
        const page = list.find((t) => t.type === 'page');
        if (page?.webSocketDebuggerUrl) {
          wsUrl = page.webSocketDebuggerUrl;
          break;
        }
      } catch {
        /* not up yet */
      }
      await sleep(250);
    }
    if (!wsUrl) throw new Error('chromium never came up');

    const ws = new WebSocket(wsUrl);
    await new Promise<void>((resolve, reject) => {
      ws.addEventListener('open', () => resolve());
      ws.addEventListener('error', () => reject(new Error('cdp socket failed')));
    });
    let id = 0;
    const pending = new Map<number, (m: any) => void>();
    const pageErrors: string[] = [];
    ws.addEventListener('message', (ev) => {
      const m = JSON.parse(String((ev as MessageEvent).data));
      if (m.id && pending.has(m.id)) {
        pending.get(m.id)!(m);
        pending.delete(m.id);
      }
      if (m.method === 'Runtime.exceptionThrown') {
        pageErrors.push(
          String(m.params?.exceptionDetails?.exception?.description ?? 'unknown page error').split(
            '\n',
          )[0],
        );
      }
    });
    const cdp = (method: string, params: Record<string, unknown> = {}) =>
      new Promise<any>((resolve) => {
        const n = ++id;
        pending.set(n, resolve);
        ws.send(JSON.stringify({ id: n, method, params }));
      });
    const evalJs = async (expression: string) => {
      const r = await cdp('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      if (r.result?.exceptionDetails) {
        throw new Error(String(r.result.exceptionDetails.exception?.description ?? 'eval failed'));
      }
      return r.result?.result?.value;
    };

    await cdp('Page.enable');
    await cdp('Runtime.enable');
    await cdp('Page.navigate', { url: `${APP_URL}/settings` });
    await sleep(2500);
    await evalJs(`document.querySelector('[data-subview="runtimes"]')?.click(); true`);
    for (let i = 0; i < 40; i += 1) {
      await sleep(500);
      if (await evalJs(`document.querySelectorAll('.rt-row').length > 0`)) break;
    }

    // Stubs the install endpoint and records what the client sent. The real
    // install is not what is under test here; the prompt contract is.
    const stub = `(() => {
      window.__origFetch = window.fetch;
      window.__sent = [];
      window.__stubNext = null;
      const sse = (events) => {
        const enc = new TextEncoder();
        const body = events.map((e) => 'event: ' + e.event + '\\ndata: ' + JSON.stringify(e.data) + '\\n\\n').join('');
        return new Response(new ReadableStream({
          start(c) { c.enqueue(enc.encode(body)); c.close(); },
        }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
      };
      window.fetch = async (url, opts = {}) => {
        const u = String(url);
        if (u.includes('/api/runtimes/install')) {
          window.__sent.push(JSON.parse(opts.body ?? '{}'));
          return sse(window.__stubNext ?? [{ event: 'done', data: { runtimes: { installed: {} } } }]);
        }
        return window.__origFetch(url, opts);
      };
      // Ticking a group is the only precondition the install button has, and a
      // re-render un-ticks anything already installed — so this is a helper,
      // called again before each attempt rather than done once.
      window.__pick = () => {
        const boxes = [...document.querySelectorAll('.rt-row input[type=checkbox]')];
        const free = boxes.find((b) => !b.disabled) ?? boxes[0];
        if (!free) return false;
        free.disabled = false;
        if (!free.checked) { free.checked = true; free.dispatchEvent(new Event('change')); }
        document.getElementById('rt-install').disabled = false;
        return true;
      };
      return window.__pick();
    })()`;
    const staged = await evalJs(stub);
    report('a runtime group can be selected', staged === true);

    // A. An install that never asks for a password never shows the box.
    await evalJs(`window.__stubNext = [{ event: 'progress', data: 'Collecting kokoro\\n' }, { event: 'done', data: { runtimes: { installed: {} } } }]; true`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    await sleep(2500);
    report('the first install actually ran', (await evalJs(`window.__sent.length`)) === 1);
    report(
      'an install with no password request shows no password box',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === true,
    );
    report(
      'the first request carried no password field',
      (await evalJs(`typeof (window.__sent[0] ?? {}).password === 'undefined'`)) === true,
    );

    // B. An installer that asks opens the box, and the retry carries it.
    await evalJs(`window.__stubNext = [
      { event: 'progress', data: 'Installing Ollama on Linux\\n' },
      { event: 'privilege', data: { command: 'sudo curl -fsSL https://ollama.com/install.sh | sh' } },
      { event: 'error', data: { message: 'The Ollama installer needs root and could not elevate.' } },
    ]; true`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    await sleep(2500);
    report(
      'a privilege request opens the password box',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === false,
    );
    report(
      'the log says which command needs the password',
      /needs your password/i.test(
        String(await evalJs(`document.getElementById('rt-log-body').textContent`)),
      ),
    );

    await evalJs(`(() => {
      const i = document.getElementById('rt-pass-input');
      i.value = 'correct horse battery';
      document.getElementById('rt-pass-go').click();
      return true;
    })()`);
    await sleep(2500);
    report(
      'the retry sends the password in the request body',
      (await evalJs(`window.__sent[window.__sent.length - 1]?.password`)) === 'correct horse battery',
    );
    report('the password box closes once used', (await evalJs(`document.getElementById('rt-pass').hidden`)) === true);
    report(
      'the password is cleared from the box after use',
      (await evalJs(`document.getElementById('rt-pass-input').value`)) === '',
    );
    report(
      'the password never lands in the install log',
      !String(await evalJs(`document.getElementById('rt-log-body').textContent`)).includes(
        'correct horse battery',
      ),
    );

    // C. A second request with a password in hand does not re-open the box.
    await evalJs(`window.__stubNext = [
      { event: 'privilege', data: { command: 'sudo curl -fsSL https://ollama.com/install.sh | sh' } },
      { event: 'error', data: { message: 'That password was not accepted, so nothing was changed.' } },
    ]; true`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    await sleep(2000);
    await evalJs(`(() => {
      const i = document.getElementById('rt-pass-input');
      i.value = 'wrong one';
      document.getElementById('rt-pass-go').click();
      return true;
    })()`);
    await sleep(2500);
    report(
      'a refused password does not open the box again',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === true,
    );

    /* D. The GPU driver button, which used to prompt before it knew. */
    await evalJs(`(() => {
      window.__sent = [];
      window.fetch = async (url, opts = {}) => {
        const u = String(url);
        if (u.includes('/api/drivers/update')) {
          window.__sent.push(JSON.parse(opts.body ?? '{}'));
          const enc = new TextEncoder();
          const frames = [
            { event: 'log', data: { line: 'Reading what the machine has…' } },
            ...(window.__sent.length === 1
              ? [{ event: 'privilege', data: { command: 'sudo scripts/install-gpu-drivers.sh --yes   # needs your password' } },
                 { event: 'log', data: { line: 'The server could not elevate.' } }]
              : [{ event: 'done', data: { apply: true } }]),
          ];
          const body = frames.map((e) => 'event: ' + e.event + '\\ndata: ' + JSON.stringify(e.data) + '\\n\\n').join('');
          return new Response(new ReadableStream({
            start(c) { c.enqueue(enc.encode(body)); c.close(); },
          }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
        }
        return window.__origFetch(url, opts);
      };
      document.getElementById('rt-pass').hidden = true;
      return true;
    })()`);

    // First press arms, second presses starts it. Neither may prompt.
    await evalJs(`document.getElementById('rt-update-drivers').click(); true`);
    await sleep(600);
    report(
      'arming the driver update shows no password box',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === true,
    );
    await evalJs(`document.getElementById('rt-update-drivers').click(); true`);
    await sleep(1500);
    report(
      'the armed press starts the install rather than the prompt',
      (await evalJs(`window.__sent.length`)) === 1,
    );
    report(
      'that first request asked for no password',
      (await evalJs(`typeof (window.__sent[0] ?? {}).password === 'undefined'`)) === true,
    );
    report(
      'the password box opens only once the installer says it needs one',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === false,
    );

    await evalJs(`(() => {
      document.getElementById('rt-pass-input').value = 'driver secret';
      document.getElementById('rt-pass-go').click();
      return true;
    })()`);
    await sleep(2000);
    report(
      'the driver retry sends the password',
      (await evalJs(`window.__sent[window.__sent.length - 1]?.password`)) === 'driver secret',
    );
    report(
      'the driver password never lands in its log',
      !String(await evalJs(`document.getElementById('rt-driver-log').textContent`)).includes(
        'driver secret',
      ),
    );

    /* E. A retry that lands while the last attempt still holds the lock. */
    await evalJs(`(() => {
      window.__sent = [];
      let locked = false;
      const enc = new TextEncoder();
      const sse = (frames) => {
        const body = frames.map((e) => 'event: ' + e.event + '\\ndata: ' + JSON.stringify(e.data) + '\\n\\n').join('');
        return new Response(new ReadableStream({
          start(c) { c.enqueue(enc.encode(body)); c.close(); },
        }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
      };
      window.fetch = async (url, opts = {}) => {
        const u = String(url);
        if (u.includes('/api/runtimes/install')) {
          const sent = JSON.parse(opts.body ?? '{}');
          window.__sent.push(sent);
          if (!sent.password) {
            // The first attempt asks for a password, as the real Ollama
            // install does when it cannot elevate.
            return sse([
              { event: 'privilege', data: { command: 'sudo curl -fsSL https://ollama.com/install.sh | sh' } },
              { event: 'error', data: { message: 'The Ollama installer needs root and could not elevate.' } },
            ]);
          }
          if (window.__sent.filter((s) => s.password).length === 1) {
            // It answers the retry with the lock still held by the attempt
            // that just asked for the password, then frees it.
            locked = true;
            setTimeout(() => { locked = false; }, 900);
            return new Response(JSON.stringify({ error: 'An install is already running.' }), {
              status: 409, headers: { 'content-type': 'application/json' },
            });
          }
          return sse([{ event: 'done', data: { runtimes: { installed: {} } } }]);
        }
        if (u.endsWith('/api/runtimes')) {
          return new Response(JSON.stringify({ runtimes: { busy: locked, installed: {} } }), {
            status: 200, headers: { 'content-type': 'application/json' },
          });
        }
        return window.__origFetch(url, opts);
      };
      document.getElementById('rt-pass').hidden = true;
      return true;
    })()`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    await sleep(1500);
    await evalJs(`(() => {
      const i = document.getElementById('rt-pass-input');
      i.value = 'lock retry';
      document.getElementById('rt-pass-go').click();
      return true;
    })()`);
    await sleep(4000);
    report(
      'an install refused for the lock is retried once the lock clears',
      (await evalJs(`window.__sent.length`)) === 3,
      String(await evalJs(`window.__sent.length`)),
    );
    report(
      'the password is still on the retried request',
      (await evalJs(`window.__sent[window.__sent.length - 1]?.password`)) === 'lock retry',
    );

    await evalJs(`window.fetch = window.__origFetch; true`);
    report(
      'the settings page ran without a single uncaught error',
      pageErrors.length === 0,
      pageErrors.join(' | '),
    );
    ws.close();
  } catch (err) {
    failures.push('browser checks');
    console.log(`  FAIL  browser checks — ${(err as Error).message}`);
  } finally {
    cleanup();
  }
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);
