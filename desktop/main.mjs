/**
 * STARR Matrix in a desktop window.
 *
 * Dev-mode shell, deliberately un-packaged: it finds the app server (starting
 * one with tsx if nothing answers on the port) and points a plain Chromium
 * window at it. `npm run dev` in a browser keeps working exactly the same, and
 * this file is the whole desktop story until the app is ready to be packaged —
 * no builder config, nothing to keep in sync with the server.
 *
 * The server stays a separate process rather than running inside this one: it
 * is watched and reloaded by tsx, it spawns the Python jobs, and keeping it
 * outside is what lets the window attach to the dev server that is already up.
 */
import { app, BrowserWindow, Menu, nativeTheme, session, shell } from 'electron';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORT = Number(process.env.PORT || 3000);
const APP_URL = `http://127.0.0.1:${PORT}/`;

// The page picks its own theme (stored choice, else the system's), but the
// window paints before any of that JS runs. Following the system here matches
// the page for a fresh profile — which is what this profile is — and turns a
// black flash into the two tokens the app itself uses (#000000 / #f4f6fb).
const DARK = nativeTheme.shouldUseDarkColors;
const SURFACE = DARK ? '#000000' : '#f4f6fb';
const INK = DARK ? '#8f97a8' : '#525a6d';
const RING = DARK ? '#2a2f3a' : '#c9cedb';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** The child we started, if any; null when attached to a running dev server. */
let server = null;
let win = null;

/** True when something that looks like the app answers on the port. */
async function answering() {
  try {
    const res = await fetch(APP_URL, { signal: AbortSignal.timeout(1500) });
    return res.ok;
  } catch {
    return false;
  }
}

/** Attach to the dev server, or start one and wait for it to answer. */
async function ensureServer() {
  if (await answering()) {
    console.log(`[desktop] attaching to the server already on :${PORT}`);
    return;
  }
  const tsx = path.join(root, 'node_modules', '.bin', 'tsx');
  server = spawn(tsx, ['src/server.ts'], {
    cwd: root,
    env: { ...process.env, PORT: String(PORT) },
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  server.on('error', (err) => console.error('[desktop] could not start the server:', err.message));
  for (let i = 0; i < 240; i += 1) {
    if (server.exitCode !== null) {
      throw new Error(`the app server exited with code ${server.exitCode}`);
    }
    if (await answering()) return;
    await sleep(500);
  }
  throw new Error('the app server did not answer within two minutes.');
}

/** Stops the server we started; the server stops its Python jobs on SIGTERM. */
function stopServer() {
  const child = server;
  server = null;
  if (!child || child.exitCode !== null) return;
  child.kill('SIGTERM');
  const kill = setTimeout(() => child.kill('SIGKILL'), 5000);
  kill.unref?.();
}

const page = (html) => `data:text/html;charset=utf-8,${encodeURIComponent(html)}`;

const LOADING_PAGE = page(
  '<!doctype html><meta charset="utf-8"><style>' +
    'html,body{height:100%;margin:0;display:grid;place-items:center;' +
    `background:${SURFACE};color:${INK};font:600 14px system-ui,sans-serif}` +
    '.t{display:flex;align-items:center;gap:10px}' +
    `.s{width:14px;height:14px;border:2px solid ${RING};border-top-color:${INK};` +
    'border-radius:50%;animation:r .8s linear infinite}' +
    '@keyframes r{to{transform:rotate(1turn)}}</style>' +
    '<div class="t"><i class="s"></i>Starting STARR Matrix…</div>',
);

const errorPage = (message) =>
  page(
    '<!doctype html><meta charset="utf-8"><style>' +
      `html,body{height:100%;margin:0;display:grid;place-items:center;background:${SURFACE};` +
      `color:${INK};font:500 14px system-ui,sans-serif}main{max-width:520px;text-align:center}` +
      `h1{font-size:16px;margin:0 0 8px;color:${DARK ? '#e6e9f0' : '#23262f'}}p{margin:0;line-height:1.5}` +
      '</style><main><h1>STARR Matrix could not start</h1><p>' +
      message.replace(/&/g, '&amp;').replace(/</g, '&lt;') +
      '</p></main>',
  );

function createWindow() {
  const w = new BrowserWindow({
    width: 1200,
    height: 820,
    minWidth: 880,
    minHeight: 560,
    title: 'STARR Matrix',
    backgroundColor: SURFACE,
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  // The window is the product's name, not the page's: the tab title stays
  // whatever the document says, the window keeps the brand.
  w.on('page-title-updated', (event) => event.preventDefault());

  // The chat records with the microphone; everything else the page asks for is
  // denied. This origin is plain http on loopback, which a browser would treat
  // as insecure and treat nothing of — the window has to say it out loud.
  const allowed = new Set(['media', 'notifications', 'clipboard-sanitized-write']);
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(allowed.has(permission));
  });
  session.defaultSession.setPermissionCheckHandler((_wc, permission) => allowed.has(permission));

  // Links out (the Discord contact in Settings) belong in the system browser,
  // not in a second bare window with no chrome.
  w.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/i.test(url)) void shell.openExternal(url);
    return { action: 'deny' };
  });

  // No menu bar (the app draws its own chrome), so the dev shortcuts a menu
  // would have carried come from here instead: F12 / Ctrl+Shift+I devtools,
  // F5 / Ctrl+R reload. Ctrl+R mid-word in a text field is the one collision —
  // the same one a browser has, and it resolves the same way.
  w.webContents.on('before-input-event', (_event, input) => {
    if (input.type !== 'keyDown') return;
    const key = input.key.toLowerCase();
    if (key === 'f12' || (input.control && input.shift && key === 'i')) {
      w.webContents.toggleDevTools();
    } else if (key === 'f5' || (input.control && key === 'r')) {
      w.webContents.reload();
    }
  });

  w.on('closed', () => {
    win = null;
  });

  const goto = async (url) => {
    if (!w.isDestroyed()) await w.loadURL(url);
  };

  void w.loadURL(LOADING_PAGE);
  void (async () => {
    try {
      await ensureServer();
      await goto(APP_URL);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('[desktop]', message);
      await goto(errorPage(message));
    }
  })();

  return w;
}

// Wayland sessions pick their scaling automatically; on X11 this is a no-op.
app.commandLine.appendSwitch('ozone-platform-hint', 'auto');
// Opt-in remote debugging, so the window can be driven with CDP in tests.
if (process.env.STARR_DEBUG_PORT) {
  app.commandLine.appendSwitch('remote-debugging-port', process.env.STARR_DEBUG_PORT);
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!win) return;
    if (win.isMinimized()) win.restore();
    win.focus();
  });

  app.whenReady().then(() => {
    Menu.setApplicationMenu(null);
    win = createWindow();
  });

  app.on('window-all-closed', () => app.quit());
  app.on('before-quit', stopServer);
}
