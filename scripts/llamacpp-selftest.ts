#!/usr/bin/env tsx
/**
 * llama.cpp chat checks: which GGUF files this machine has, which of them are
 * chat models, and the server that gets started to run the one you picked.
 *
 * The two failures this guards are the ones that made the runtime feel broken:
 *
 *  - a downloaded GGUF that no picker ever listed (the runtime was installed and
 *    the weights were on disk, and nothing joined them up);
 *  - a model that needs loading answering as a hang (llama-server serves one
 *    model, so a request for a different one has to start it again);
 *  - and the reverse of the second: a server left running after the app itself
 *    was killed, holding a model in RAM and a port the app believes is free.
 *
 * Everything runs against a temporary models directory and a stand-in server,
 * so the suite needs neither the real llama.cpp build nor a real GGUF: the
 * checks are about this app's wiring, and a fake binary makes them fast and
 * deterministic. The binary answers /health the way upstream does — 503 while
 * loading — so "the port is open" can never pass for "the model is ready".
 */
import { spawn, spawnSync } from 'node:child_process';
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const work = mkdtempSync(join(tmpdir(), 'llamacpp-selftest-'));
const models = join(work, 'models');

// Point the whole app's model discovery at the sandbox *before* anything reads
// it: `modelsDir()` and `hubCacheDir()` are functions, so this is enough — and
// an empty temp cache means the real one on this machine cannot leak in.
process.env.HF_MODELS_DIR = models;
process.env.HF_HUB_CACHE = join(work, 'hub');
process.env.LLAMACPP_SERVER = join(work, 'not-installed-yet');

import {
  cachedGgufModels,
  ggufChatModels,
  ggufModels,
  llamaHint,
  llamaServerState,
  prepareLlama,
  resolveGgufModel,
} from '../src/llamacpp.ts';

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

/** A weight file, in the layout the app's own downloads use. */
function put(repoDir: string, file: string, bytes = 64): string {
  const dir = join(models, repoDir);
  mkdirSync(dir, { recursive: true });
  const full = join(dir, file);
  writeFileSync(full, Buffer.alloc(bytes, 7));
  return full;
}

/* ---------------- The sandbox ---------------- */

const qwenPath = put('unsloth--Qwen3-4B-GGUF', 'Qwen3-4B-Q4_K_M.gguf');
put('ggml-org--gemma-3-4b-it-GGUF', 'gemma-3-4b-it-Q4_K_M.gguf');
put('ggml-org--gemma-3-4b-it-GGUF', 'mmproj-model-f16.gguf');
// Named so the file itself says nothing about transcription: only the catalog's
// task for the repo can classify it, which is the rule that exists for exactly
// this case (a quant whose name drifted from its purpose).
put('handy-computer--whisper-small-gguf', 'encore-Q4_K_M.gguf');
put('city96--stable-diffusion-3.5-medium-gguf', 'turbo-Q4_K_M.gguf');
// No catalog entry at all: the name is the only signal, and it says vocoder.
put('someone--Speech-Tools', 'vocoder-f16.gguf');
// Same file name in two repos: chat models both, and neither may shadow the
// other in a picker that keys on the id.
put('acme--Max-GGUF', 'Max-Q4_K_M.gguf');
put('cedar--Max-GGUF', 'Max-Q4_K_M.gguf');

/* ------- 1. What is on this machine, and what can chat ------- */
section('GGUF discovery and classification');
{
  const found = await ggufModels(true);
  const byFile = new Map(found.map((m) => [m.file, m]));

  const qwen = byFile.get('Qwen3-4B-Q4_K_M.gguf');
  report(
    'a catalog GGUF chat model is offered under a readable id',
    qwen?.chat === true && qwen.id === 'Qwen3-4B-Q4_K_M',
    JSON.stringify(qwen),
  );
  report(
    'its path is the file on disk',
    qwen?.path === qwenPath && existsSync(qwen?.path ?? ''),
    qwen?.path,
  );
  report('an uncatalogued chat GGUF is offered too', byFile.get('gemma-3-4b-it-Q4_K_M.gguf')?.chat === true);

  const mmproj = byFile.get('mmproj-model-f16.gguf');
  report(
    'an mmproj projector beside a chat model is not offered as a chat model',
    mmproj?.chat === false,
    JSON.stringify(mmproj),
  );

  const whisper = byFile.get('encore-Q4_K_M.gguf');
  report(
    'a whisper GGUF is classified by its catalog task, not its file name',
    whisper?.chat === false && /speech-to-text/.test(whisper?.why ?? ''),
    JSON.stringify(whisper),
  );
  const diffusion = byFile.get('turbo-Q4_K_M.gguf');
  report(
    'a diffusion GGUF is classified by its catalog task',
    diffusion?.chat === false && /image-generation/.test(diffusion?.why ?? ''),
    JSON.stringify(diffusion),
  );
  const vocoder = byFile.get('vocoder-f16.gguf');
  report(
    'an uncatalogued GGUF whose name is not a text model is left out',
    vocoder?.chat === false,
    JSON.stringify(vocoder),
  );

  const ids = found.map((m) => m.id);
  report(
    'two same-named files are told apart by their repo',
    ids.includes('acme/Max-GGUF/Max-Q4_K_M') && ids.includes('cedar/Max-GGUF/Max-Q4_K_M'),
    ids.join(', '),
  );
  report('no bare colliding id is left behind', !ids.includes('Max-Q4_K_M'), ids.join(', '));
  report(
    'the listing is sorted the way a picker shows it',
    ids.join('\n') === [...ids].sort((a, b) => a.localeCompare(b)).join('\n'),
    ids.join(', '),
  );
}

/* ------- 2. Resolving what a conversation stored ------- */
section('Resolving a model id');
{
  const byId = await resolveGgufModel('Qwen3-4B-Q4_K_M');
  report('an exact id resolves', byId?.file === 'Qwen3-4B-Q4_K_M.gguf', JSON.stringify(byId));

  const byName = await resolveGgufModel('Qwen3-4B-Q4_K_M.gguf');
  report('the file name also resolves', byName?.id === 'Qwen3-4B-Q4_K_M', JSON.stringify(byName));

  const byCase = await resolveGgufModel('qwen3-4b-q4_k_m');
  report('a lowercased id resolves', byCase?.id === 'Qwen3-4B-Q4_K_M', JSON.stringify(byCase));

  report('an unknown id resolves to nothing', (await resolveGgufModel('Llama-99B')) === null);

  const chat = await ggufChatModels();
  const chatIds = chat.map((m) => m.id);
  report(
    'the offered list is exactly the chat models',
    chatIds.includes('Qwen3-4B-Q4_K_M') &&
      !chatIds.some((id) => /mmproj|vocoder|encore|turbo/.test(id)),
    chatIds.join(', '),
  );
}

/* ------- 3. Refusals name the problem ------- */
section('Refusals before anything is started');
{
  const progress: string[] = [];
  const note = (line: string) => progress.push(line);

  let nonChat = '';
  await prepareLlama('encore-Q4_K_M', note).catch((err: Error) => {
    nonChat = err.message;
  });
  report('a non-chat GGUF is refused by name', /not a chat model/.test(nonChat), nonChat);

  let unknown = '';
  await prepareLlama('No-Such-Model', note).catch((err: Error) => {
    unknown = err.message;
  });
  report('an unknown model says so', /no GGUF model called/.test(unknown), unknown);

  // The binary path points at a file that does not exist, so this is a machine
  // with the runtime missing — the state the hint has to be honest about.
  report('the runtime reads as absent', llamaServerState().installed === false);
  report('the hint says the runtime is missing', /not installed yet/.test(llamaHint()), llamaHint());

  let missing = '';
  await prepareLlama('Qwen3-4B-Q4_K_M', note).catch((err: Error) => {
    missing = err.message;
  });
  report(
    'a chat request without the runtime says where to install it',
    /not installed on this machine/.test(missing) && /Runtimes/.test(missing),
    missing,
  );
  report('nothing was narrated for a request that never started', progress.length === 0, progress.join(' | '));
}

/* ------- 4. The stand-in server ------- */

const fake = join(work, 'fake-llama-server.mjs');
const fakeLog = join(work, 'fake-llama.log');
const delayFile = join(work, 'fake-delay.txt');
writeFileSync(delayFile, '0');

// Written rather than embedded from a fixture file so the contract is visible:
// /health answers 503 until the delay elapses, /v1/chat/completions streams two
// OpenAI-shaped deltas and [DONE], and every start and stop lands in the log
// with the model it was given.
writeFileSync(
  fake,
  `#!/usr/bin/env node
import { appendFileSync, readFileSync } from 'node:fs';
import { createServer } from 'node:http';

const argv = process.argv.slice(2);
const arg = (n) => {
  const i = argv.indexOf(n);
  return i >= 0 ? argv[i + 1] : null;
};
const model = arg('--model');
const alias = arg('--alias');
const port = Number(arg('--port'));
const log = (entry) => appendFileSync(process.env.FAKE_LOG, JSON.stringify(entry) + '\\n');
log({ event: 'start', pid: process.pid, model, alias, port });

let loadMs = 0;
try {
  loadMs = Number(readFileSync(process.env.FAKE_DELAY_FILE, 'utf8').trim()) || 0;
} catch {}
const readyAt = Date.now() + loadMs;

const server = createServer((req, res) => {
  if (req.url.startsWith('/health')) {
    if (Date.now() < readyAt) {
      res.writeHead(503, { 'content-type': 'application/json' });
      res.end('{"status":"loading model"}');
      return;
    }
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end('{"status":"ok"}');
    return;
  }
  if (req.url.startsWith('/v1/models')) {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ object: 'list', data: [{ id: alias, object: 'model' }] }));
    return;
  }
  if (req.url.startsWith('/v1/chat/completions')) {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write('data: ' + JSON.stringify({ choices: [{ delta: { content: 'Hello' } }] }) + '\\n\\n');
    res.write('data: ' + JSON.stringify({ choices: [{ delta: { content: ' from ' + alias } }] }) + '\\n\\n');
    res.write('data: [DONE]\\n\\n');
    res.end();
    return;
  }
  res.writeHead(404);
  res.end();
});
server.listen(port, '127.0.0.1');
process.on('SIGTERM', () => {
  log({ event: 'stop', pid: process.pid });
  server.close();
  process.exit(0);
});
`,
);
chmodSync(fake, 0o755);

process.env.LLAMACPP_SERVER = fake;
report('the runtime is found once the binary exists', llamaServerState().installed === true);
report(
  'the hint counts the chat models when the runtime is installed',
  /GGUF chat model/.test(llamaHint()),
  llamaHint(),
);
report(
  'the hint counts the same models the picker will show',
  cachedGgufModels().filter((m) => m.chat).length === 4,
  String(cachedGgufModels().filter((m) => m.chat).length),
);

/* ------- 5. A real chat request through the app ------- */

const appPort = Number(process.env.LLAMACPP_TEST_PORT ?? 3297);
const llamaPort = Number(process.env.LLAMACPP_TEST_LLAMA_PORT ?? 3298);
const appUrl = `http://127.0.0.1:${appPort}`;

const starts = () =>
  (existsSync(fakeLog) ? readFileSync(fakeLog, 'utf8').split('\n').filter(Boolean) : [])
    .map((line) => JSON.parse(line) as { event: string; pid: number; alias?: string; model?: string })
    .filter((e) => e.event === 'start');

const alive = (pid: number) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

async function portFree(port: number): Promise<boolean> {
  try {
    await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(500) });
    return false;
  } catch {
    return true;
  }
}

interface Ev {
  event: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any;
}

/** POSTs a chat and returns its SSE events, in order. */
async function chat(body: unknown): Promise<Ev[]> {
  const res = await fetch(`${appUrl}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(90000),
  });
  if (!res.ok) throw new Error(`chat HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const text = await res.text();
  const events: Ev[] = [];
  for (const block of text.split(/\n\n/)) {
    let event = 'message';
    let data = '';
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data += line.slice(5).trim();
    }
    if (!data) continue;
    try {
      events.push({ event, data: JSON.parse(data) });
    } catch {
      /* keep-alives and other non-JSON frames are not part of the contract */
    }
  }
  return events;
}

const deltas = (events: Ev[]) => events.filter((e) => e.event === 'delta').map((e) => e.data.text).join('');
const notices = (events: Ev[]) => events.filter((e) => e.event === 'notice').map((e) => e.data.message);
const errors = (events: Ev[]) => events.filter((e) => e.event === 'error').map((e) => e.data.message);

const ASK = (model: string) => ({
  provider: 'llamacpp',
  model,
  messages: [{ role: 'user', content: 'hi' }],
  tools: [],
});

let app: ReturnType<typeof spawn> | null = null;
try {
  if (!(await portFree(appPort)) || !(await portFree(llamaPort))) {
    report(`${appPort}/${llamaPort} are free for the throwaway server`, false, 'another process is listening');
    throw new Error('ports busy');
  }

  const appLog = join(work, 'app.log');
  app = spawn(join(process.cwd(), 'node_modules', '.bin', 'tsx'), ['src/server.ts'], {
    cwd: process.cwd(),
    detached: true,
    env: {
      ...process.env,
      PORT: String(appPort),
      LLAMACPP_PORT: String(llamaPort),
      LLAMACPP_SERVER: fake,
      LLAMACPP_CTX: '2048',
      // The idle window the real app gives ten minutes; six seconds proves the
      // same timer and keeps the suite quick — and it leaves a wide margin for
      // the hard-kill check at the end, which must not be able to mistake the
      // idle timer for its own result.
      LLAMACPP_IDLE_MS: '6000',
      FAKE_LOG: fakeLog,
      FAKE_DELAY_FILE: delayFile,
    },
    stdio: ['ignore', openSync(appLog, 'a'), openSync(appLog, 'a')],
  });

  let up = false;
  for (let i = 0; i < 60 && !up; i += 1) {
    await new Promise((r) => setTimeout(r, 500));
    // Poll the static page, not the API: the API is rate limited per route, and
    // a slow boot that legitimately needs twenty polls must not spend the quota
    // for the checks that follow.
    try {
      await fetch(`${appUrl}/`, { signal: AbortSignal.timeout(1500) });
      up = true;
    } catch {
      /* still booting */
    }
  }
  if (!up) {
    report('the throwaway server came up', false, readFileSync(appLog, 'utf8').slice(-400));
    throw new Error('server never came up');
  }

  section('Providers and the models page');
  {
    const providers = (await (await fetch(`${appUrl}/api/providers`)).json()) as {
      providers: { id: string; configured: boolean; local: boolean; models?: string[]; hint?: string }[];
    };
    const llama = providers.providers.find((p) => p.id === 'llamacpp');
    report(
      'llama.cpp is in the provider list as a local backend',
      llama?.local === true && llama.id === 'llamacpp',
      JSON.stringify(llama),
    );
    report(
      'it is configured on the first page load, before any provider list was asked for',
      llama?.configured === true,
      JSON.stringify(llama),
    );
    report(
      'the picker offers the chat models and nothing else',
      (llama?.models ?? []).includes('Qwen3-4B-Q4_K_M') &&
        (llama?.models ?? []).includes('gemma-3-4b-it-Q4_K_M') &&
        !(llama?.models ?? []).some((id) => /vocoder|encore|turbo|mmproj/.test(id)),
      JSON.stringify(llama?.models),
    );

    const files = (await (await fetch(`${appUrl}/api/hf/files`)).json()) as {
      files: { file: string; llama?: { provider: string; model: string } }[];
    };
    const gguf = files.files.find((f) => f.file.endsWith('Qwen3-4B-Q4_K_M.gguf'));
    report(
      'a downloaded chat GGUF carries the id the chat page must use',
      gguf?.llama?.provider === 'llamacpp' && gguf?.llama?.model === 'Qwen3-4B-Q4_K_M',
      JSON.stringify(gguf),
    );
    const whisperFile = files.files.find((f) => f.file.endsWith('encore-Q4_K_M.gguf'));
    report(
      'a non-chat GGUF gets no Chat link',
      whisperFile !== undefined && whisperFile.llama === undefined,
      JSON.stringify(whisperFile),
    );
  }

  section('Chatting through llama.cpp');
  {
    // A load slow enough to be narrated: the 503-while-loading window is what
    // turns a multi-gigabyte model into progress lines instead of a hang.
    writeFileSync(delayFile, '4500');
    const first = await chat(ASK('Qwen3-4B-Q4_K_M'));
    report(
      'the load is announced',
      notices(first).some((m) => /Starting llama\.cpp with Qwen3-4B-Q4_K_M/.test(m)),
      notices(first).join(' | '),
    );
    report(
      'a slow load keeps saying it is still working',
      notices(first).some((m) => /Still loading Qwen3-4B-Q4_K_M/.test(m)),
      notices(first).join(' | '),
    );
    report('the answer streams back', deltas(first) === 'Hello from Qwen3-4B-Q4_K_M', deltas(first));
    report('the stream ends with done', first[first.length - 1]?.event === 'done', JSON.stringify(first.at(-1)));

    let launch = starts();
    report(
      'exactly one server was started, with this model loaded and aliased',
      launch.length === 1 && launch[0].alias === 'Qwen3-4B-Q4_K_M' && /Qwen3-4B-Q4_K_M\.gguf$/.test(launch[0].model ?? ''),
      JSON.stringify(launch),
    );

    const running = (await (await fetch(`${appUrl}/api/providers`)).json()) as {
      providers: { id: string; hint?: string }[];
    };
    report(
      'the hint says what is loaded once it is up',
      /Running with Qwen3-4B-Q4_K_M loaded/.test(running.providers.find((p) => p.id === 'llamacpp')?.hint ?? ''),
      running.providers.find((p) => p.id === 'llamacpp')?.hint,
    );

    // Same model again: the server it already has must be reused, or every
    // message in a conversation would pay the load again.
    writeFileSync(delayFile, '0');
    const second = await chat(ASK('Qwen3-4B-Q4_K_M'));
    report('the second message answers with the same model', deltas(second) === 'Hello from Qwen3-4B-Q4_K_M', deltas(second));
    report(
      'no restart, and no start notice, for a model that is already loaded',
      starts().length === 1 && !notices(second).some((m) => /Starting llama\.cpp/.test(m)),
      `starts=${starts().length} notices=${notices(second).join(' | ')}`,
    );

    // A different model: llama-server serves one at a time, so this is a real
    // restart, and the old process must be gone rather than left holding RAM.
    const firstPid = starts()[0].pid;
    const third = await chat(ASK('gemma-3-4b-it-Q4_K_M'));
    report('switching models answers from the new one', deltas(third) === 'Hello from gemma-3-4b-it-Q4_K_M', deltas(third));
    launch = starts();
    report(
      'switching models restarts the server with the new alias',
      launch.length === 2 && launch[1].alias === 'gemma-3-4b-it-Q4_K_M',
      JSON.stringify(launch),
    );
    let reaped = false;
    for (let i = 0; i < 20 && !reaped; i += 1) {
      reaped = !alive(firstPid);
      if (!reaped) await new Promise((r) => setTimeout(r, 250));
    }
    report('the model it replaced is stopped, not left in memory', reaped);

    // A refusal must not spawn anything: the failure is knowable before the
    // expensive part starts.
    const before = starts().length;
    const bad = await chat(ASK('encore-Q4_K_M'));
    report('a non-chat model fails with an explanation', /not a chat model/.test(errors(bad).join(' ')), errors(bad).join(' '));
    const worse = await chat(ASK('No-Such-Model'));
    report('an unknown model fails with an explanation', /no GGUF model called/.test(errors(worse).join(' ')), errors(worse).join(' '));
    report('neither refusal started a server', starts().length === before, `starts=${starts().length}`);
  }

  section('Idle shutdown and coming back');
  {
    let stopped = false;
    for (let i = 0; i < 40 && !stopped; i += 1) {
      await new Promise((r) => setTimeout(r, 500));
      const last = starts().at(-1);
      stopped = Boolean(last) && !alive(last!.pid);
    }
    report('the server is shut down once it has sat unused', stopped);
    report(
      'and it says so in its own log rather than being killed outright',
      (existsSync(fakeLog) ? readFileSync(fakeLog, 'utf8') : '').includes('"event":"stop"'),
    );

    const again = await chat(ASK('gemma-3-4b-it-Q4_K_M'));
    report('the next message after that starts it again', deltas(again) === 'Hello from gemma-3-4b-it-Q4_K_M', deltas(again));
    report('which is a fresh process', starts().length === 3, `starts=${starts().length}`);
  }

  section('A hard kill leaves nothing behind');
  {
    // Killing the app the way a reload or a crash does — SIGKILL, no handler
    // runs — used to leave llama-server reparented to init, holding the model
    // in RAM and the port the app thinks is free. The supervisor's stdin pipe
    // is what changes that, and the check has to be able to tell the pipe apart
    // from the idle timer: the idle deadline is 6s, and the reap here is given
    // 4s, immediately after the last message.
    const last = starts().at(-1);
    const wrapper = last
      ? Number(spawnSync('ps', ['-o', 'ppid=', '-p', String(last.pid)]).stdout.toString().trim())
      : 0;
    const server = wrapper
      ? Number(spawnSync('ps', ['-o', 'ppid=', '-p', String(wrapper)]).stdout.toString().trim())
      : 0;
    if (!last || !server) {
      console.log('  SKIP  ps cannot walk the process tree on this machine');
    } else {
      process.kill(server, 'SIGKILL');
      let reaped = false;
      for (let i = 0; i < 20 && !reaped; i += 1) {
        reaped = !alive(last.pid);
        if (!reaped) await new Promise((r) => setTimeout(r, 200));
      }
      report('killing the app outright does not orphan the model server', reaped, `pid ${last.pid} survived`);
      report('and the supervisor goes with it', !alive(wrapper), `pid ${wrapper} survived`);
    }
  }
} catch (err) {
  if (!/ports busy|never came up/.test((err as Error).message)) {
    report('the llama.cpp chat section ran', false, (err as Error).message);
  }
} finally {
  // The whole process group: tsx spawns the server it runs, and killing only
  // the parent leaves that child holding the port for the next run.
  if (app?.pid) {
    try {
      process.kill(-app.pid, 'SIGKILL');
    } catch {
      try {
        app.kill('SIGKILL');
      } catch {
        /* already gone */
      }
    }
  }
  // Stand-ins are children of the app; a hard kill of the group orphans them,
  // so anything the log knows about is stopped by pid as well.
  for (const start of starts()) {
    if (alive(start.pid)) {
      try {
        process.kill(start.pid, 'SIGKILL');
      } catch {
        /* already gone */
      }
    }
  }
  for (let i = 0; i < 20; i += 1) {
    if (await portFree(appPort)) break;
    await new Promise((r) => setTimeout(r, 250));
  }
  rmSync(work, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}

/* ------- 6. The two screens the user sees ------- */

section('Client contract');
{
  const appJs = readFileSync(join(process.cwd(), 'public', 'app.js'), 'utf8');
  report(
    'the provider popover no longer tags rows "runs on this PC"',
    !/runs on this PC/.test(appJs),
  );
  report('the "On this machine" grouping stays', /On this machine \(/.test(appJs));

  const settingsJs = readFileSync(join(process.cwd(), 'public', 'settings.js'), 'utf8');
  report(
    'a chat model is offered the chat page it can run on',
    /llama/.test(settingsJs) && /provider=\$\{encodeURIComponent/.test(settingsJs),
  );
  report(
    'the GGUF format entry points at the runtime instead of declaring itself external',
    /id: 'llama-cpp'/.test(settingsJs) && /LLAMA_CHAT_RUNTIME/.test(settingsJs),
  );
}

/* ---------------- Summary ---------------- */

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
