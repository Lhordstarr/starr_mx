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
import { chmodSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describeVenvFailure, needsPrivilege, ollamaCommand, ollamaRecipe } from '../src/runtimes.ts';

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
  report(
    'the Linux command keeps `sh -c` (dropping it installs nothing)',
    linux?.argv[0] === 'sh' && linux?.argv[1] === '-c' && linux?.argv.length === 3,
    linux?.argv.join(' '),
  );

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
    win?.argv.includes('powershell.exe') && win?.argv.includes('-NoProfile'),
    win?.argv.join(' '),
  );

  report('an unsupported OS admits it rather than guessing', ollamaRecipe('aix' as NodeJS.Platform) === null);

  // The escalated and bare forms are the same installer with a wrapper, so the
  // two argument vectors have to agree apart from that wrapper.
  const bare = ollamaCommand(linux!, false);
  const sudoed = ollamaCommand(linux!, true);
  report('bare form is the recipe itself', bare.cmd === 'sh' && bare.args.join(' ') === linux!.argv.slice(1).join(' '), `${bare.cmd} ${bare.args.join(' ')}`);
  report(
    'escalated form is sudo in front of the same recipe',
    sudoed.cmd === 'sudo' &&
      sudoed.args.join(' ') === ['-S', '-p', '', ...linux!.argv].join(' '),
    `${sudoed.cmd} ${sudoed.args.join(' ')}`,
  );

  /* And it actually runs. A fake `curl` earlier on PATH answers the real
     command line without touching the network, which is the only way to catch
     a mangled argv: `sh "curl … | sh"` fails with "cannot open …: No such
     file", which is exactly what the app used to do on Linux. */
  const runDir = join(tmpdir(), `starr-argv-${process.pid}`);
  try {
    mkdirSync(runDir, { recursive: true });
    const fakeCurl = join(runDir, 'curl');
    // The real thing is a shell script fetched and piped into `sh`, so the
    // fake has to output shell — not arbitrary text, which `sh` would rightly
    // try to run as a command.
    writeFileSync(fakeCurl, '#!/bin/sh\necho "echo ARGV_OK"\n', { mode: 0o755 });
    const { code, out } = await new Promise<{ code: number; out: string }>((resolve) => {
      const child = spawn(bare.cmd, bare.args, {
        env: { ...process.env, PATH: `${runDir}:${process.env.PATH ?? ''}` },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let seen = '';
      child.stdout.on('data', (d) => (seen += String(d)));
      child.stderr.on('data', (d) => (seen += String(d)));
      child.on('error', (err) => resolve({ code: 127, out: err.message }));
      child.on('close', (c) => resolve({ code: c ?? 0, out: seen }));
    });
    report(
      'the Linux command actually runs the script',
      code === 0 && out.includes('ARGV_OK'),
      `exit ${code}: ${out.trim().slice(0, 120)}`,
    );
  } finally {
    rmSync(runDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  }
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
// The root page is static and answers immediately. /api/runtimes is not: it
// probes Python, and a cold cache on a loaded machine takes longer than any
// sensible timeout — which used to read as "the server is down" and skip the
// checks that matter.
for (let i = 0; i < 3 && !serverUp; i += 1) {
  try {
    const res = await fetch(APP_URL, { signal: AbortSignal.timeout(5000) });
    serverUp = res.ok;
  } catch {
    await new Promise((r) => setTimeout(r, 1000));
  }
}
if (!serverUp) {
  console.log(`  SKIP  server not reachable at ${APP_URL} (start it with npm run dev)`);
} else {
  // A labelled block so the rate-limit case can bail out of the whole section
  // rather than fall through into checks that would all fail for the same
  // reason.
  serverSection: {
    const statusRes = await fetch(`${APP_URL}/api/runtimes`);
  const statusText = await statusRes.text();
  let status: any = null;
  try {
    status = JSON.parse(statusText);
  } catch {
    status = null;
  }
  // The server rate-limits /api to 30 a minute per route. A check suite run
  // several times in a row earns that, and every later check then fails for a
  // reason that has nothing to do with the code — so say what happened and stop
  // instead of reporting a wall of failures.
  if (statusRes.status === 429) {
    console.log('  SKIP  the server is rate-limiting this IP; run this on its own');
    serverUp = false;
    break serverSection;
  } else {
    report(
      'runtimes status still lists every group',
      Boolean(status?.runtimes?.installed && Array.isArray(status?.groups)),
      `HTTP ${statusRes.status}: ${statusText.slice(0, 160)}`,
    );
  }
  // Ollama is detected, not assumed: the old code answered "installed" on any
  // OS without looking, which is what made a missing install unrecoverable.
  // Read with one retry, because the server answers this from a probe that can
  // time out on a busy machine and a timeout is not what is under test here.
  let ollamaState: unknown;
  for (let i = 0; i < 2; i += 1) {
    try {
      ollamaState = (await fetch(`${APP_URL}/api/runtimes`, { signal: AbortSignal.timeout(40000) }).then(
        (r) => r.json(),
      ))?.runtimes?.installed?.ollama;
    } catch {
      /* the probe timed out; one more try */
    }
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

  /* The button label comes from a status read the client makes the moment it
     sees `done`, so the lock has to be free by then. It used to be released
     after, and that one ordering was enough to leave the button reading
     "Installing…" with nothing running.

     Checked against the real server with a real (already-installed, so it
     resolves offline) install. This asserts the invariant rather than the race:
     on localhost the release wins the photo finish, and what the browser
     actually lost to was its own follow-up work between `done` and the read.
     The check that pins the symptom is the browser one below — a status read
     answered "busy" a moment too late. */
  try {
    const res = await fetch(`${APP_URL}/api/runtimes/install`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ groups: ['onnxruntime'], device: 'cpu' }),
      signal: AbortSignal.timeout(180_000),
    });
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let sawDone = false;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      if (buf.includes('event: done')) {
        sawDone = true;
        break;
      }
    }
    await reader.cancel().catch(() => undefined);
    report('a real install reports completion', sawDone);
    if (sawDone) {
      const after = await fetch(`${APP_URL}/api/runtimes`).then((r) => r.json());
      report(
        'the lock is already free when completion is announced',
        after?.runtimes?.busy === false,
        `busy=${String(after?.runtimes?.busy)}`,
      );
    }
  } catch (err) {
    report('the lock-ordering check ran', false, (err as Error).message);
  }

  /* The stale-busy bug: `busy` is what disables the install button, and it
     used to be served from a 30-second cache. A click that landed while an
     install was already running was therefore allowed by the button, refused
     by the server, and answered with "Already running on the server" — for an
     install the user had only started once.

     Proved here on a throwaway server whose Python stalls, so an install is
     deterministically in flight with no network and nothing to install. */
  const stallDir = join(tmpdir(), `starr-stall-${process.pid}`);
  const stallPort = 3199;
  let stallServer: ReturnType<typeof spawn> | null = null;
  try {
    mkdirSync(stallDir, { recursive: true });
    const fakePython = join(stallDir, 'python3');
    // Answers the version probe like a real interpreter, then takes its time
    // about creating an environment — which is exactly the window in question.
    writeFileSync(
      fakePython,
      ['#!/bin/sh', 'case "$1 $2" in', '  "-c import"*) echo "3.12"; exit 0 ;;', 'esac', 'sleep 25', 'exit 1', ''].join(
        '\n',
      ),
    );
    chmodSync(fakePython, 0o755);
    stallServer = spawn(
      // The local binary, not `npx`: npx adds a wrapper process, and killing
      // the wrapper leaves the server listening — which then serves the *old*
      // build to the next run and turns this check into a coin toss.
      join(process.cwd(), 'node_modules', '.bin', 'tsx'),
      ['src/server.ts'],
      {
        cwd: process.cwd(),
        detached: true,
        env: {
          ...process.env,
          PORT: String(stallPort),
          RUNTIMES_PYTHON: fakePython,
          RUNTIMES_DIR: join(stallDir, 'runtimes'),
        },
        stdio: 'ignore',
      },
    );
    let ready = false;
    for (let i = 0; i < 60 && !ready; i += 1) {
      await new Promise((r) => setTimeout(r, 500));
      try {
        await fetch(`http://localhost:${stallPort}/api/runtimes`, { signal: AbortSignal.timeout(1500) });
        ready = true;
      } catch {
        /* still booting */
      }
    }
    if (!ready) throw new Error('the throwaway server never came up');

    // Warm the cache while nothing is installing.
    const idle = await fetch(`http://localhost:${stallPort}/api/runtimes`).then((r) => r.json());
    report('the throwaway server starts out not busy', idle?.runtimes?.busy === false);

    // Start an install and leave it hanging.
    void fetch(`http://localhost:${stallPort}/api/runtimes/install`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ groups: ['kokoro'], device: 'cpu' }),
    }).catch(() => undefined);

    let sawBusy = false;
    for (let i = 0; i < 20 && !sawBusy; i += 1) {
      await new Promise((r) => setTimeout(r, 500));
      const now = await fetch(`http://localhost:${stallPort}/api/runtimes`).then((r) => r.json());
      if (now?.runtimes?.busy === true) sawBusy = true;
    }
    report(
      'a cached read still reports an install that is running',
      sawBusy,
      'busy stayed false for 10s while an install held the lock',
    );
  } catch (err) {
    report('the stale-busy check ran', false, (err as Error).message);
  } finally {
    // The whole process group: tsx spawns the server it runs, and killing only
    // the parent leaves that child holding the port for the next run.
    if (stallServer?.pid) {
      try {
        process.kill(-stallServer.pid, 'SIGKILL');
      } catch {
        try {
          stallServer.kill('SIGKILL');
        } catch {
          /* already gone */
        }
      }
    }
    // Wait for the port to actually come back, so the next run cannot attach to
    // a server that is still shutting down.
    for (let i = 0; i < 20; i += 1) {
      try {
        await fetch(`http://localhost:${stallPort}/api/runtimes`, { signal: AbortSignal.timeout(500) });
      } catch {
        break;
      }
      await new Promise((r) => setTimeout(r, 250));
    }
    rmSync(stallDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  }
  }
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
    /**
     * Waits for a condition instead of sleeping a fixed time.
     *
     * These assertions are about what the page does after an async stream, and
     * a fixed wait makes them fail on a loaded machine while passing on an idle
     * one — which is how a check stops being a check.
     */
    const waitFor = async (expression: string, timeoutMs = 10000) => {
      const until = Date.now() + timeoutMs;
      for (;;) {
        if (await evalJs(`Boolean(${expression})`)) return true;
        if (Date.now() > until) return false;
        await sleep(150);
      }
    };

    await cdp('Page.enable');
    await cdp('Runtime.enable');
    await cdp('Page.navigate', { url: `${APP_URL}/settings` });
    // The subview buttons are in the static HTML, so they exist long before the
    // module finishes evaluating and wires their click handlers. Waiting on the
    // button's presence and clicking once is a race that a third of runs lost —
    // the click simply went to a button nobody was listening on yet. So: wait
    // for the page's own boot to show a panel (only the module does that), then
    // click, and click again if the panel has not switched.
    await waitFor(`document.querySelector('[data-subpanel]:not([hidden])')`, 30000);
    let opened = false;
    for (let i = 0; i < 10 && !opened; i += 1) {
      await evalJs(
        `document.querySelector('[data-subview="runtimes"]').click(); true`,
      );
      await sleep(250);
      opened = (await evalJs(`document.querySelector('[data-subpanel="runtimes"]').hidden`)) === false;
    }
    if (!opened) throw new Error('the Runtimes subview would not open');
    await waitFor(`document.querySelectorAll('.rt-row').length > 0`, 30000);

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
    await waitFor(`window.__sent.length === 1`);
    report('the first install actually ran', (await evalJs(`window.__sent.length`)) === 1);
    report(
      'an install with no password request shows no password box',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === true,
    );
    report(
      'the first request carried no password field',
      (await evalJs(`typeof (window.__sent[0] ?? {}).password === 'undefined'`)) === true,
    );
    // The label comes from a status read, so it has to end up right even when
    // that read is answered a moment late.
    const labelA = await waitFor(
      `document.getElementById('rt-install').textContent.trim() === 'Install selected'`,
      8000,
    );
    report('the button returns to its resting label after the install', labelA, String(await evalJs(`document.getElementById('rt-install').textContent.trim()`)));

    // B. An installer that asks opens the box, and the retry carries it.
    await evalJs(`window.__stubNext = [
      { event: 'progress', data: 'Installing Ollama on Linux\\n' },
      { event: 'privilege', data: { command: 'sudo curl -fsSL https://ollama.com/install.sh | sh' } },
      { event: 'error', data: { message: 'The Ollama installer needs root and could not elevate.' } },
    ]; true`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    await waitFor(`document.getElementById('rt-pass').hidden === false`);
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
    await waitFor(`window.__sent.length === 3`);
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
    await waitFor(`window.__sent.length === 5 && document.getElementById('rt-pass').hidden === true`);
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
    await waitFor(`document.getElementById('rt-driver-log').textContent.includes('Press "Update GPU drivers" again')`);
    report(
      'arming the driver update shows no password box',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === true,
    );
    await evalJs(`document.getElementById('rt-update-drivers').click(); true`);
    await waitFor(`window.__sent.length === 1`);
    report(
      'the armed press starts the install rather than the prompt',
      (await evalJs(`window.__sent.length`)) === 1,
    );
    report(
      'that first request asked for no password',
      (await evalJs(`typeof (window.__sent[0] ?? {}).password === 'undefined'`)) === true,
    );
    await waitFor(`document.getElementById('rt-pass').hidden === false`);
    report(
      'the password box opens only once the installer says it needs one',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === false,
    );

    await evalJs(`(() => {
      document.getElementById('rt-pass-input').value = 'driver secret';
      document.getElementById('rt-pass-go').click();
      return true;
    })()`);
    await waitFor(`window.__sent.length === 2`);
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
    await waitFor(`document.getElementById('rt-pass').hidden === false`);
    await evalJs(`(() => {
      const i = document.getElementById('rt-pass-input');
      i.value = 'lock retry';
      document.getElementById('rt-pass-go').click();
      return true;
    })()`);
    await waitFor(`window.__sent.length === 3`, 15000);
    report(
      'an install refused for the lock is retried once the lock clears',
      (await evalJs(`window.__sent.length`)) === 3,
      String(await evalJs(`window.__sent.length`)),
    );
    report(
      'the password is still on the retried request',
      (await evalJs(`window.__sent[window.__sent.length - 1]?.password`)) === 'lock retry',
    );

    /* F. A status read that is answered a moment too late.
       This is the shape that left the button stuck on "Installing…" with
       nothing running: the read that repaints the label is issued while the
       install still holds the lock, and lands after it has finished. The read
       is truthful about the moment it was made and wrong by the time it
       paints, so the only defence is to check it. */
    await evalJs(`(() => {
      window.__staleReads = 2;
      window.fetch = async (url, opts = {}) => {
        const u = String(url);
        if (u.includes('/api/runtimes/install')) {
          const enc = new TextEncoder();
          const body = 'event: done\\ndata: {"runtimes":{"installed":{}}}\\n\\n';
          return new Response(new ReadableStream({
            start(c) { c.enqueue(enc.encode(body)); c.close(); },
          }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
        }
        if (u.includes('/api/runtimes')) {
          // The first reads after the install answer "busy" — exactly what the
          // server said at the instant the client asked, and exactly what left
          // the label lying.
          const stale = window.__staleReads > 0;
          if (stale) window.__staleReads -= 1;
          return new Response(JSON.stringify({ runtimes: { busy: stale, installed: {} }, groups: [], blocked: [] }), {
            status: 200, headers: { 'content-type': 'application/json' },
          });
        }
        return window.__origFetch(url, opts);
      };
      document.getElementById('rt-pass').hidden = true;
      return true;
    })()`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    // The stale read paints "Installing…" — the check is that it is not the
    // last word.
    const recovered = await waitFor(
      `document.getElementById('rt-install').textContent.trim() === 'Install selected' && window.__staleReads === 0`,
      15000,
    );
    report(
      'a late "busy" read does not leave the button stuck on Installing…',
      recovered,
      String(await evalJs(`document.getElementById('rt-install').textContent.trim()`)),
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
