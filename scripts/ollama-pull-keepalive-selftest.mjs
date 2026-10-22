/**
 * A pull must keep the Ollama daemon alive past its idle window.
 *
 * The daemon is stopped after `OLLAMA_IDLE_MS` of inactivity. A pull is a long
 * stream, not one request that finishes immediately, so the pull route has to
 * report activity for every progress line. Without that, the idle timer kills
 * the daemon halfway through the download and the browser sees "terminated".
 *
 * This starts a real app server against a fake Ollama daemon, sets the idle
 * window to 2 s, and makes the fake pull take 4 s. With the keepalive the pull
 * finishes and the daemon is then stopped by the same idle timer; without it
 * the stream dies before the pull is halfway done.
 *
 * Run with: npm run test:ollama-pull-keepalive
 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = dirname(dirname(fileURLToPath(import.meta.url)));
const appPort = Number(process.env.TEST_APP_PORT ?? 3307);
const ollamaPort = Number(process.env.TEST_OLLAMA_PORT ?? 11737);
const appBase = `http://127.0.0.1:${appPort}`;
const model = 'fake-vision:latest';
const idleMs = 2000;

const tmp = mkdtempSync(join(tmpdir(), 'starr-ollama-keepalive-'));
const binDir = join(tmp, 'bin');
mkdirSync(binDir);
const pidFile = join(tmp, 'daemon.pid');
const fakeOllama = join(binDir, 'ollama');

// A stand-in daemon: it answers the reachability probe and then streams a
// deliberately slow pull. SIGTERM is the app's idle/user stop; exiting at once
// makes the failure (if any) show up as a dead stream, like the real daemon.
writeFileSync(
  fakeOllama,
  `#!/usr/bin/env node
const http = require('http');
const fs = require('fs');
const port = ${ollamaPort};
const pidFile = ${JSON.stringify(pidFile)};

const server = http.createServer((req, res) => {
  if (req.url === '/api/tags') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end('{"models":[]}');
    return;
  }
  if (req.url === '/api/pull' && req.method === 'POST') {
    req.resume();
    req.on('end', () => {
      res.writeHead(200, { 'content-type': 'application/x-ndjson' });
      const send = (value) => res.write(JSON.stringify(value) + '\\n');
      send({ status: 'pulling manifest' });
      let sent = 0;
      const timer = setInterval(() => {
        sent += 1;
        send({ status: 'downloading', total: 40, completed: sent });
        if (sent >= 40) {
          clearInterval(timer);
          send({ status: 'success' });
          res.end();
        }
      }, 100);
    });
    return;
  }
  res.writeHead(404, { 'content-type': 'application/json' });
  res.end('{"error":"not found"}');
});

console.error('fake ollama starting on ' + port);
server.on('error', (err) => console.error('fake ollama error: ' + err.message));
server.listen(port, '127.0.0.1', () => {
  console.error('fake ollama ready');
  fs.writeFileSync(pidFile, String(process.pid));
});
process.on('SIGTERM', () => {
  try { fs.unlinkSync(pidFile); } catch {}
  process.exit(0);
});
`,
);
chmodSync(fakeOllama, 0o755);

const app = spawn(process.execPath, ['--import', 'tsx', 'src/server.ts'], {
  cwd: repo,
  env: {
    ...process.env,
    PATH: `${binDir}:${process.env.PATH}`,
    OLLAMA_BASE_URL: `http://127.0.0.1:${ollamaPort}`,
    OLLAMA_IDLE_MS: String(idleMs),
    PORT: String(appPort),
    HOST: '127.0.0.1',
  },
  stdio: ['ignore', 'pipe', 'pipe'],
});

let serverLog = '';
app.stdout.on('data', (chunk) => (serverLog += chunk));
app.stderr.on('data', (chunk) => (serverLog += chunk));

async function waitForServer() {
  const deadline = Date.now() + 30_000;
  for (;;) {
    try {
      const res = await fetch(`${appBase}/api/providers`);
      if (res.ok) return;
    } catch {
      /* not listening yet */
    }
    if (Date.now() > deadline) throw new Error(`server did not start\n${serverLog}`);
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
}

let failed = false;
try {
  await waitForServer();
  const res = await fetch(`${appBase}/api/ollama/pull`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: model }),
  });
  const stream = await res.text();
  if (res.status !== 200) throw new Error(`pull route answered ${res.status}: ${stream}`);
  const events = stream
    .split('\n')
    .filter((line) => line.startsWith('event: '))
    .map((line) => line.slice('event: '.length));
  console.log(`  events: ${events.join(', ') || '(none)'}`);
  if (!events.includes('done')) throw new Error(`pull did not finish\n${stream}`);

  assert.ok(existsSync(pidFile), 'the daemon should still be alive when the pull finishes');

  // The keepalive is not a permanent exemption: once the pull is over, the
  // ordinary idle timer must stop the daemon again.
  await new Promise((resolve) => setTimeout(resolve, idleMs + 400));
  assert.equal(existsSync(pidFile), false, 'the daemon should idle-stop after the pull');

  console.log('PASS  a pull keeps the daemon alive, then normal idle stop resumes');
} catch (err) {
  failed = true;
  console.error(`FAIL  ${(err).message}`);
  if (serverLog.trim()) console.error(`\nserver log:\n${serverLog}`);
} finally {
  app.kill('SIGTERM');
  await Promise.race([
    new Promise((resolve) => app.once('exit', resolve)),
    new Promise((resolve) => setTimeout(resolve, 6000)),
  ]);
  try {
    const pid = Number(readFileSync(pidFile, 'utf8'));
    if (pid) process.kill(pid, 'SIGKILL');
  } catch {
    /* already stopped */
  }
  rmSync(tmp, { recursive: true, force: true });
}

if (failed) process.exit(1);
