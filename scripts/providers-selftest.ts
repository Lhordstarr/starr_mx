#!/usr/bin/env tsx
/**
 * The provider list is on the critical path of every page load, and it must not
 * freeze the server while it is built.
 *
 * It was built by asking each provider, in turn, "are you configured?" — and the
 * transformers provider answered that question with
 * `spawnSync(py, ['-c', 'import transformers'])`. That is a synchronous child
 * process, so it blocks the whole event loop for as long as the import takes:
 * 3.5 seconds on this machine, torch and all. The route asks three or four
 * times, so a single page load froze STARR Matrix for about fifteen seconds.
 * The provider list took fifteen seconds to appear, and every other request —
 * a keystroke's fetch, a panel that only wanted its settings — sat behind it
 * until it finished.
 *
 * Two things are checked here, and the second is the one that matters:
 *
 *   - the source, so the synchronous probe cannot come back quietly;
 *   - a running server, because the invariant is about the whole process: while
 *     the provider list is being built, another request must still be answered.
 *
 * Against the previous code both live checks fail — the list takes ~15 s and a
 * request made during it waits ~13 s. Skips (exit 0) when the app is not
 * reachable, since every live check needs a server to look at.
 */
import { readFileSync } from 'node:fs';
import * as path from 'node:path';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000/';

let passed = 0;
const failures: string[] = [];
function report(name: string, ok: boolean, detail = ''): void {
  if (ok) {
    passed += 1;
    console.log(`  PASS  ${name}`);
  } else {
    failures.push(name);
    console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

console.log('\n=== the provider list must not freeze the server ===');

/* ---- the source: no synchronous process on this path --------------------- */

const chat = readFileSync(path.resolve('src/transformers-chat.ts'), 'utf8');
const start = chat.indexOf('export function transformersAvailable');
const body = start === -1 ? '' : chat.slice(start, chat.indexOf('\n}\n', start) + 3);
report(
  'the transformers probe reads the filesystem instead of importing the package',
  body.length > 0 && !/\bspawnSync\(|\bexecSync\(|\bspawn\(/.test(body),
  'a synchronous import of transformers blocks the whole server for seconds',
);
report(
  'and it still answers the question it is asked',
  /sitePackages\(/.test(body) && /__init__\.py/.test(body),
  'the provider list has to know whether there is anything to offer',
);

/* ---- a running server: the list is quick and does not hold up the rest ---- */

// A dev server that is restarting (tsx watch reruns on every edit) is not an
// absent server, so this waits rather than skipping the moment it is unhappy.
async function appReachable(): Promise<boolean> {
  for (let i = 0; i < 30; i++) {
    const ok = await fetch(new URL('/api/tools', APP_URL), { signal: AbortSignal.timeout(5000) })
      .then((r) => r.ok)
      .catch(() => false);
    if (ok) return true;
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}

if (!(await appReachable())) {
  console.log(`\nSKIP — app not reachable at ${APP_URL} (start it with npm run dev)`);
  console.log(`\n${passed} passed, 0 failed`);
  process.exit(0);
}

async function timed(
  pathname: string,
): Promise<{ ms: number; status: number; body: unknown; error?: string }> {
  const t0 = Date.now();
  try {
    const res = await fetch(new URL(pathname, APP_URL), { signal: AbortSignal.timeout(60_000) });
    const body = await res.json().catch(() => null);
    return { ms: Date.now() - t0, status: res.status, body };
  } catch (e) {
    // A server frozen by a synchronous child process can reset the connection
    // instead of answering it. That is the failure this suite exists to catch,
    // so it is reported as one rather than crashing the run.
    return { ms: Date.now() - t0, status: 0, body: null, error: (e as Error).message };
  }
}

const first = await timed('/api/providers');
report(
  'the provider list answers in under two seconds',
  first.ms < 2000,
  first.error ? `no answer: ${first.error}` : `took ${first.ms} ms`,
);

// The list is built again here and left in flight while other requests are
// made. On a frozen event loop those requests wait for the list to finish; on a
// healthy one they are answered in milliseconds. Sampled several times because
// the freeze starts a moment after the request does.
const inFlight = fetch(new URL('/api/providers', APP_URL), { signal: AbortSignal.timeout(60_000) }).catch(
  () => null,
);
let worst = 0;
for (let i = 0; i < 8; i++) {
  await new Promise((r) => setTimeout(r, 100));
  const during = await timed('/api/tools');
  worst = Math.max(worst, during.ms);
}
report(
  'requests made while the list is still being built are answered promptly',
  worst < 1000,
  `worst wait ${worst} ms`,
);
await inFlight;

const providers = (first.body as { providers?: { id: string; hint?: string }[] } | null)?.providers ?? [];
report(
  'the list still names every provider',
  ['ollama', 'llamacpp', 'transformers-chat', 'openai'].every((id) => providers.some((p) => p.id === id)),
  providers.map((p) => p.id).join(', ') || '(none)',
);
report(
  'the transformers provider still reports a hint',
  typeof providers.find((p) => p.id === 'transformers-chat')?.hint === 'string',
  'the hint is a getter the probe feeds; a broken probe would take it out of the list',
);

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);
