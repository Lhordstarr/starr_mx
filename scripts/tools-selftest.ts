/**
 * Unit checks for agent tools and the agent loop. Covers what is hard to reach
 * through the browser: arg validation, the search parser (against a fixture,
 * no network), SSRF guards, persistence, the round loop with stub providers,
 * and both wire adapters with stubbed fetch.
 *
 * Live network checks (Wikipedia, one page fetch, DDG) run last and stay
 * narrow: Wikipedia and example.com are stable, while a DuckDuckGo bot
 * challenge is reported as a skip rather than a failure.
 *
 * Run with: npm run test:tools
 */
import '../src/net.js';
import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  assertPublicUrl,
  decodeEntities,
  describeTransportError,
  enabledTools,
  htmlToText,
  langChainTool,
  listTools,
  parseDuckDuckGo,
  resolveToolSelection,
  setToolEnabled,
  toolByName,
  toolsFilePath,
  validateArgs,
} from '../src/tools.js';
import type { Tool } from '../src/tools.js';
import {
  isToolFormatError,
  isToolsUnsupported,
  normalizeArgs,
  ollamaChatTools,
  openAIChatTools,
  runAgent,
  withToolGuidance,
} from '../src/agent.js';
import type { AgentTurn } from '../src/providers/types.js';
import type { ChatMessage } from '../src/types.js';

process.env.TOOLS_FILE = join(mkdtempSync(join(tmpdir(), 'tools-test-')), 'tools.json');

let passed = 0;
const failures: string[] = [];

async function check(name: string, fn: () => void | Promise<void>): Promise<void> {
  try {
    await fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures.push(name);
    console.log(`  FAIL  ${name}\n        ${(err as Error).message}`);
  }
}

/** A live check that degrades to SKIP on transport trouble, never a false fail.
 *  Only network-level failures skip (DNS, timeouts, resets, bot challenges);
 *  wrong content or HTTP errors still fail, so real bugs can't hide behind
 *  a SKIP. The vocabulary matches what describeTransportError emits
 *  ("Could not reach host: the connection timed out (ETIMEDOUT).") as well as
 *  the raw undici message, so a translated failure still skips. */
const SKIP_TRANSPORT = /fetch failed|timed out|ETIMEDOUT|Could not resolve|Timeout|timeout|abort|challenged|ENOTFOUND|EAI_AGAIN|ECONNRESET|ENETUNREACH|EHOSTUNREACH|ECONNREFUSED|socket hang up/i;

async function checkLive(name: string, fn: () => Promise<void>): Promise<void> {
  try {
    await fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    const msg = (err as Error).message;
    if (SKIP_TRANSPORT.test(msg)) {
      console.log(`  SKIP  ${name} (${msg.slice(0, 90)})`);
    } else {
      failures.push(name);
      console.log(`  FAIL  ${name}\n        ${msg}`);
    }
  }
}

const webSearch = toolByName('web_search')!;
const fetchPage = toolByName('fetch_page')!;
const wikipedia = toolByName('wikipedia')!;
const noSignal = AbortSignal.timeout(20000);

/* ---------------- arg validation ---------------- */

console.log('\n=== arg validation ===');
await check('valid args pass', () => {
  assert.equal(validateArgs(webSearch.definition, { query: 'x', count: 3 }), null);
  assert.equal(validateArgs(fetchPage.definition, { url: 'https://example.com' }), null);
});
await check('missing required arg fails', () => {
  assert.match(validateArgs(webSearch.definition, { count: 3 })!, /required.*query/);
});
await check('unknown arg fails', () => {
  assert.match(validateArgs(webSearch.definition, { query: 'x', bogus: 1 })!, /unknown argument/);
});
await check('wrong type fails', () => {
  assert.match(validateArgs(webSearch.definition, { query: 42 })!, /must be a string/);
});
await check('integer rejects floats', () => {
  assert.match(validateArgs(webSearch.definition, { query: 'x', count: 2.5 })!, /integer/);
});
await check('range and length caps hold', () => {
  assert.match(validateArgs(webSearch.definition, { query: 'x', count: 99 })!, /<= 10/);
  assert.match(validateArgs(webSearch.definition, { query: 'y'.repeat(301) })!, /too long/);
});
await check('non-object args fail', () => {
  assert.match(validateArgs(webSearch.definition, 'query')!, /must be an object/);
  assert.match(validateArgs(webSearch.definition, null)!, /must be an object/);
});

/* ---------------- LangChain tool definitions ---------------- */

console.log('\n=== LangChain tool definitions ===');
await check('wire schemas derive from the zod source', () => {
  assert.deepEqual(webSearch.definition.parameters, {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'The search query.', maxLength: 300 },
      count: { type: 'integer', description: 'How many results (1-10).', minimum: 1, maximum: 10 },
    },
    required: ['query'],
    additionalProperties: false,
  });
  assert.deepEqual(fetchPage.definition.parameters.required, ['url']);
  assert.deepEqual(toolByName('wikipedia')!.definition.parameters.required, ['query']);
});
await check('built-ins are real LangChain tools', () => {
  const t = langChainTool('web_search');
  assert.ok(t, 'web_search resolves');
  assert.equal(t!.name, 'web_search');
  assert.equal(typeof t!.invoke, 'function');
  assert.equal(langChainTool('ghost'), undefined);
});

/* ---------------- model-arg normalization ---------------- */

console.log('\n=== model-arg normalization ===');
const synthDef: Tool['definition'] = {
  name: 'synth',
  description: 'stub',
  parameters: {
    type: 'object',
    properties: {
      n: { type: 'integer' },
      flag: { type: 'boolean' },
      s: { type: 'string' },
    },
    required: [],
    additionalProperties: false,
  },
};
await check('stringified scalars coerce to their schema types', () => {
  assert.deepEqual(normalizeArgs(synthDef, { n: '5', flag: 'true', s: 'x' }), {
    n: 5,
    flag: true,
    s: 'x',
  });
});
await check('stray keys are dropped, and the cleaned call validates', () => {
  assert.deepEqual(normalizeArgs(webSearch.definition, { query: 'x', reason: 'why' }), {
    query: 'x',
  });
  const clean = normalizeArgs(webSearch.definition, { query: 'x', count: '1', confidence: 9 });
  assert.equal(validateArgs(webSearch.definition, clean), null);
});
await check('unrecoverable values still reach the validator unchanged', () => {
  const out = normalizeArgs(webSearch.definition, { query: 'x', count: 'lots' }) as {
    count: unknown;
  };
  assert.equal(out.count, 'lots');
  assert.match(validateArgs(webSearch.definition, out)!, /integer/);
});

/* ---------------- search parser fixture ---------------- */

const DDG_FIXTURE = `
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://example.com/llm">Large <b>Language</b> Models &amp; You</a>
  </h2>
  <div class="result__extras"><div class="result__extras__url">
    <span class="result__icon"></span>
    <a class="result__url" href="https://example.com/llm">example.com</a>
  </div></div>
  <div class="result__snippet">Everything about <b>LLMs</b> in one place.</div>
</div>
<div class="result">
  <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Ftwo&amp;rut=abc">Second Page</a></h2>
</div>`;

console.log('\n=== DuckDuckGo parser (fixture, no network) ===');
await check('direct and redirector links parse', () => {
  const r = parseDuckDuckGo(DDG_FIXTURE);
  assert.equal(r.length, 2);
  assert.equal(r[0].title, 'Large Language Models & You');
  assert.equal(r[0].url, 'https://example.com/llm');
  assert.equal(r[0].snippet, 'Everything about LLMs in one place.');
  assert.equal(r[1].url, 'https://example.org/two');
});
await check('empty and challenge pages yield nothing', () => {
  assert.deepEqual(parseDuckDuckGo('<html><body>no results</body></html>'), []);
  assert.deepEqual(parseDuckDuckGo('<html>anomaly challenge</html>'), []);
});

/* ---------------- html helpers ---------------- */

console.log('\n=== html helpers ===');
await check('entities decode', () => {
  assert.equal(decodeEntities('a &amp; b &lt;c&gt; &#65; &#x42; &quot;q&quot;'), 'a & b <c> A B "q"');
});
await check('scripts vanish, text survives with structure', () => {
  const t = htmlToText('<html><head><script>evil()</script><style>.x{}</style></head>' +
    '<body><h1>Hi</h1><p>One <b>two</b></p><!-- c --><p>Three</p></body></html>');
  assert.ok(!t.includes('evil'), 'script removed');
  assert.ok(t.includes('Hi') && t.includes('One two') && t.includes('Three'), `kept text: ${JSON.stringify(t)}`);
});

/* ---------------- SSRF guards (no network needed) ---------------- */

console.log('\n=== SSRF guards ===');
await check('loopback and private literals rejected', async () => {
  for (const u of [
    'http://127.0.0.1/', 'http://localhost/', 'http://[::1]/', 'http://0.0.0.0/',
    'http://10.0.0.5/', 'http://192.168.1.1/', 'http://172.16.0.9/',
    'http://169.254.169.254/latest', 'http://[::ffff:127.0.0.1]/',
  ]) {
    await assert.rejects(() => assertPublicUrl(u), /not fetchable|Could not resolve/);
  }
});
await check('non-http schemes and credentials rejected', async () => {
  await assert.rejects(() => assertPublicUrl('file:///etc/passwd'), /Only http/);
  await assert.rejects(() => assertPublicUrl('ftp://example.com/x'), /Only http/);
  await assert.rejects(() => assertPublicUrl('http://user:pass@example.com/'), /credentials/);
  await assert.rejects(() => assertPublicUrl('not a url'), /valid URL/);
});
await check('public hostnames pass (needs DNS)', async () => {
  const u = await assertPublicUrl('https://example.com/');
  assert.equal(u.hostname, 'example.com');
});

/* ---------------- transport diagnostics ---------------- */

console.log('\n=== transport diagnostics ===');
await check('a connection timeout names the host and the reason', () => {
  const raw = Object.assign(new TypeError('fetch failed'), {
    cause: Object.assign(new Error('connect ETIMEDOUT 185.15.59.224:443'), {
      code: 'ETIMEDOUT',
    }),
  });
  assert.equal(
    describeTransportError('https://en.wikipedia.org/w/api.php?x=1', raw).message,
    'Could not reach en.wikipedia.org: the connection timed out (ETIMEDOUT).',
  );
});
await check('a missing host is named, without leaking the query', () => {
  const raw = Object.assign(new TypeError('fetch failed'), {
    cause: Object.assign(new Error('getaddrinfo ENOTFOUND'), { code: 'ENOTFOUND' }),
  });
  const msg = describeTransportError('https://nope.example/api?token=secret', raw).message;
  assert.match(msg, /nope\.example/);
  assert.doesNotMatch(msg, /token=secret/);
});
await check('a timeout signal is reported as a timeout', () => {
  const err = new Error('The operation was aborted due to timeout');
  err.name = 'TimeoutError';
  assert.equal(
    describeTransportError('https://a.example/', err).message,
    'Timed out reaching a.example.',
  );
});
await check('a caller abort is passed through, not relabelled', () => {
  const err = new Error('The user stopped this');
  err.name = 'AbortError';
  assert.equal(describeTransportError('https://a.example/', err), err);
});
await check('the process does not race address families', async () => {
  const net = await import('node:net');
  assert.equal(net.getDefaultAutoSelectFamily(), false);
});

/* ---------------- persistence ---------------- */

console.log('\n=== enable/disable persistence ===');
await check('all tools enabled by default', async () => {
  const list = await listTools();
  assert.equal(list.length, 3);
  assert.ok(list.every((t) => t.enabled && t.keyless));
  assert.ok((await enabledTools()).length === 3);
  assert.ok(toolsFilePath().endsWith('tools.json'));
});
await check('disabling persists and re-enables', async () => {
  await setToolEnabled('web_search', false);
  assert.equal((await listTools()).find((t) => t.name === 'web_search')!.enabled, false);
  assert.equal((await enabledTools()).length, 2);
  await setToolEnabled('web_search', true);
  assert.equal((await enabledTools()).length, 3);
});
await check('unknown tool names throw', async () => {
  await assert.rejects(() => setToolEnabled('nope', false), /Unknown tool/);
});

await check('picker metadata rides along with the listing', async () => {
  const list = await listTools();
  const ws = list.find((t) => t.name === 'web_search')!;
  assert.equal(ws.label, 'Web Search');
  assert.equal(ws.icon, 'globe');
  assert.ok(
    list.every((t) => t.label && t.icon),
    'every built-in names its picker label and icon',
  );
});

/* ---------------- per-request tool selection ---------------- */

console.log('\n=== per-message tool selection ===');
await check('selection resolves to exactly the requested subset', async () => {
  const all = await enabledTools();
  assert.deepEqual(resolveToolSelection(undefined, all).tools, [], 'nothing asked for, nothing offered');
  assert.deepEqual(resolveToolSelection([], all).tools, [], 'an empty choice means no tools');
  assert.deepEqual(resolveToolSelection(false, all).tools, []);
  const one = resolveToolSelection(['wikipedia'], all);
  assert.equal(one.error, undefined);
  assert.deepEqual(one.tools.map((t) => t.definition.name), ['wikipedia']);
  const dup = resolveToolSelection(['wikipedia', 'web_search', 'wikipedia'], all);
  assert.deepEqual(dup.tools.map((t) => t.definition.name), ['wikipedia', 'web_search'], 'duplicates collapse');
  const legacy = resolveToolSelection(true, all);
  assert.equal(legacy.tools.length, all.length, 'true still means everything enabled');
});

await check('unknown and disabled ids are refused, never silently granted', async () => {
  const all = await enabledTools();
  assert.match(resolveToolSelection(['ghost'], all).error ?? '', /Unknown tool: ghost/);
  assert.match(resolveToolSelection(['web_search', 42], all).error ?? '', /array of tool names/);
  await setToolEnabled('fetch_page', false);
  try {
    const off = resolveToolSelection(['fetch_page'], await enabledTools());
    assert.match(off.error ?? '', /Tool not enabled: fetch_page/);
    assert.deepEqual(off.tools, [], 'a disabled tool resolves to nothing even if named');
    const legacy = resolveToolSelection(true, await enabledTools());
    assert.equal(legacy.tools.length, all.length - 1, 'the legacy flag follows the switch too');
  } finally {
    await setToolEnabled('fetch_page', true);
  }
  assert.ok(toolByName('fetch_page'), 'the refusal above was about the switch, not the registry');
});

/* ---------------- agent loop with stub providers ---------------- */

console.log('\n=== agent loop ===');

const stubTool: Tool = {
  definition: {
    name: 'stub_search',
    description: 'stub',
    parameters: {
      type: 'object',
      properties: { query: { type: 'string' } },
      required: ['query'],
      additionalProperties: false,
    },
  },
  keyless: true,
  execute: async (args) => `result for ${(args.query as string).toUpperCase()}`,
};

function stubChat(script: { text?: string; calls?: { name: string; args: unknown }[] }[]) {
  const seen: unknown[] = [];
  let n = 0;
  return {
    seen,
    chat: async function* ({ turns }: { turns: AgentTurn[] }) {
      seen.push(JSON.parse(JSON.stringify(turns)));
      const step = script[Math.min(n++, script.length - 1)];
      if (step.text) yield { type: 'text' as const, text: step.text };
      if (step.calls?.length) {
        yield {
          type: 'tool_calls' as const,
          calls: step.calls.map((c, i) => ({ id: c.name ? `id${i}` : '', ...c })),
        };
      }
    },
  };
}

const MSGS = [{ role: 'user' as const, content: 'hi' }];

await check('plain answer streams and ends', async () => {
  const s = stubChat([{ text: 'hello' }]);
  const events: string[] = [];
  for await (const ev of runAgent({
    chat: s.chat as never, model: 'm', messages: MSGS, tools: [stubTool], signal: noSignal,
  })) {
    events.push(`${ev.type}${ev.type === 'text' ? `:${ev.text}` : ''}`);
  }
  assert.deepEqual(events, ['text:hello']);
  assert.equal(s.seen.length, 1);
});

await check('tool round trip feeds results back', async () => {
  const s = stubChat([
    { text: 'Searching. ', calls: [{ name: 'stub_search', args: { query: 'cats' } }] },
    { text: 'Found it.' },
  ]);
  const events: string[] = [];
  for await (const ev of runAgent({
    chat: s.chat as never, model: 'm', messages: MSGS, tools: [stubTool], signal: noSignal,
  })) {
    events.push(ev.type === 'text' ? `text:${ev.text}` : `${ev.type}:${'name' in ev ? (ev as { name: string }).name : ''}`);
  }
  assert.deepEqual(events, ['text:Searching. ', 'tool_call:stub_search', 'tool_result:stub_search', 'text:Found it.']);
  assert.equal(s.seen.length, 2);
  const followUp = s.seen[1] as AgentTurn[];
  const asst = followUp.find((t) => t.role === 'assistant' && t.calls?.length);
  assert.ok(asst, 'assistant turn carries the calls');
  assert.equal(asst!.calls![0].name, 'stub_search');
  const tool = followUp.find((t) => t.role === 'tool');
  assert.equal(tool?.output, 'result for CATS');
});

await check('stringified tool args are executed coerced, not rejected', async () => {
  const numTool: Tool = {
    definition: {
      name: 'num_tool',
      description: 'stub',
      parameters: {
        type: 'object',
        properties: { n: { type: 'integer' } },
        required: ['n'],
        additionalProperties: false,
      },
    },
    keyless: true,
    execute: async (args) => `got:${typeof args.n}:${args.n}`,
  };
  const s = stubChat([{ calls: [{ name: 'num_tool', args: { n: '7' } }] }, { text: 'ok' }]);
  for await (const _ of runAgent({
    chat: s.chat as never, model: 'm', messages: MSGS, tools: [numTool], signal: noSignal,
  })) {
    // drain to the end
  }
  const followUp = s.seen[1] as AgentTurn[];
  assert.equal(followUp.find((t) => t.role === 'tool')?.output, 'got:number:7');
});

await check('tool-format rejections are recognized', () => {
  assert.equal(
    isToolFormatError(
      'Ollama error 500: {"error":"… does not match the expected peg-native format …"}',
    ),
    true,
  );
  assert.equal(
    isToolFormatError(
      'llama-server chat error: map[code:500 message:The model produced output that does not match the expected peg-native format type:server_error]',
    ),
    true,
  );
  assert.equal(isToolFormatError('Ollama error 500: disk full'), false);
});

await check('a tool-grammar rejection skips the round instead of failing', async () => {
  let rounds = 0;
  const chat = async function* ({ tools }: { tools: unknown[] }) {
    rounds += 1;
    if (rounds === 1) {
      throw new Error('Ollama error 500: … does not match the expected peg-native format …');
    }
    if (tools.length) {
      yield {
        type: 'tool_calls' as const,
        calls: [{ id: 'a', name: 'stub_search', args: { query: 'x' } }],
      };
    } else {
      yield { type: 'text' as const, text: 'plain answer' };
    }
  };
  const texts: string[] = [];
  for await (const ev of runAgent({
    chat: chat as never, model: 'm', messages: MSGS, tools: [stubTool],
    maxRounds: 2, signal: noSignal,
  })) {
    if (ev.type === 'text') texts.push(ev.text);
  }
  assert.equal(rounds, 3, 'the broken round is skipped, then tools + the plain round');
  assert.deepEqual(texts, ['plain answer']);
});

await check('unrelated provider errors still propagate', async () => {
  const chat = async function* (): AsyncGenerator<never> {
    throw new Error('Ollama error 500: disk full');
  };
  await assert.rejects(async () => {
    for await (const _ of runAgent({
      chat: chat as never, model: 'm', messages: MSGS, tools: [stubTool], signal: noSignal,
    })) {
      // drain
    }
  }, /disk full/);
});

await check('empty rounds are retried, not treated as an answer', async () => {
  let rounds = 0;
  const chat = async function* () {
    rounds += 1;
    if (rounds <= 2) return; // stall through both tool rounds: no text, no calls
    yield { type: 'text' as const, text: 'late plain' };
  };
  const texts: string[] = [];
  for await (const ev of runAgent({
    chat: chat as never, model: 'm', messages: MSGS, tools: [stubTool],
    maxRounds: 2, signal: noSignal,
  })) {
    if (ev.type === 'text') texts.push(ev.text);
  }
  assert.equal(rounds, 3, 'two stalls skipped, plain round answered');
  assert.deepEqual(texts, ['late plain']);
});

await check('unknown tools and bad args become error results, loop continues', async () => {
  const s = stubChat([
    { calls: [{ name: 'ghost', args: {} }] },
    { calls: [{ name: 'stub_search', args: {} }] },
    { text: 'done' },
  ]);
  const results: string[] = [];
  for await (const ev of runAgent({
    chat: s.chat as never, model: 'm', messages: MSGS, tools: [stubTool], signal: noSignal,
  })) {
    if (ev.type === 'tool_result') results.push(`${ev.name}:${ev.ok}:${ev.detail}`);
  }
  assert.equal(results.length, 2);
  assert.match(results[0], /ghost:false:Error: Unknown tool/);
  assert.match(results[1], /stub_search:false:.*required.*query/);
});

await check('a registered tool outside the request cannot execute', async () => {
  // web_search exists in the registry, but this request offered only the
  // stub — the loop must refuse it without ever hitting the network.
  const s = stubChat([
    { calls: [{ name: 'web_search', args: { query: 'x' } }] },
    { text: 'ok' },
  ]);
  const results: string[] = [];
  for await (const ev of runAgent({
    chat: s.chat as never, model: 'm', messages: MSGS, tools: [stubTool], signal: noSignal,
  })) {
    if (ev.type === 'tool_result') results.push(`${ev.name}:${ev.ok}:${ev.detail}`);
  }
  assert.equal(results.length, 1);
  assert.match(results[0], /web_search:false:Error: Tool not enabled for this request/);
  // The model was told why, in its own transcript, so it can answer instead.
  const followUp = s.seen[1] as AgentTurn[];
  assert.match(followUp.find((t) => t.role === 'tool')?.output ?? '', /not enabled/);
});

await check('a repeated identical call reuses the first result', async () => {
  let runs = 0;
  const counted: Tool = {
    ...stubTool,
    execute: async (args) => {
      runs += 1;
      return `result for ${args.query}`;
    },
  };
  const s = stubChat([
    { calls: [{ name: 'stub_search', args: { query: 'cats' } }] },
    { calls: [{ name: 'stub_search', args: { query: 'cats' } }] },
    { calls: [{ name: 'stub_search', args: { query: 'dogs' } }] },
    { text: 'done' },
  ]);
  const results: { ok: boolean; detail: string }[] = [];
  for await (const ev of runAgent({
    chat: s.chat as never, model: 'm', messages: MSGS, tools: [counted], signal: noSignal,
  })) {
    if (ev.type === 'tool_result') results.push({ ok: ev.ok, detail: ev.detail });
  }
  assert.equal(runs, 2, 'the duplicate did not run the tool again; the new query did');
  assert.equal(results.length, 3);
  assert.ok(results.every((r) => r.ok));
  assert.equal(results[0].detail, results[1].detail, 'the repeat sees the first result');
});

await check('round cap stops a stubborn model', async () => {
  const s = stubChat([{ calls: [{ name: 'stub_search', args: { query: 'x' } }] }]);
  let texts = 0;
  for await (const ev of runAgent({
    chat: s.chat as never, model: 'm', messages: MSGS, tools: [stubTool],
    maxRounds: 2, signal: noSignal,
  })) {
    if (ev.type === 'text') texts++;
  }
  // Rounds 0 and 1 carry tools; round 2 is the forced plain final round whose
  // calls (the stub always returns them) are dropped by the cap guard.
  assert.equal(s.seen.length, 3);
});

await check('abort ends the loop', async () => {
  const s = stubChat([{ text: 'partial ', calls: [{ name: 'stub_search', args: { query: 'x' } }] }]);
  const ctl = new AbortController();
  ctl.abort();
  let n = 0;
  for await (const _ of runAgent({
    chat: s.chat as never, model: 'm', messages: MSGS, tools: [stubTool], signal: ctl.signal,
  })) {
    n++;
  }
  assert.equal(n, 0);
});

await check('tool guidance notes differ by availability', () => {
  const msgs: ChatMessage[] = [{ role: 'user', content: 'hi' }];
  const on = withToolGuidance(msgs, true);
  const off = withToolGuidance(msgs, false);
  assert.equal(msgs.length, 1, 'input list untouched');
  assert.equal(on[0].role, 'system');
  assert.match(String(on[0].content), /never write out a pretend call/);
  assert.match(String(off[0].content), /not available/);
  assert.match(String(off[0].content), /never invent/);
});

await check('tool guidance names the tools enabled for this request', () => {
  const msgs: ChatMessage[] = [{ role: 'user', content: 'hi' }];
  const named = withToolGuidance(msgs, true, ['web_search', 'wikipedia']);
  assert.match(String(named[0].content), /only these: web_search, wikipedia/);
  assert.match(String(named[0].content), /never write out a pretend call/);
  const generic = withToolGuidance(msgs, true);
  assert.doesNotMatch(String(generic[0].content), /only these/);
});

await check('tool guidance merges into an existing system turn', () => {
  const msgs: ChatMessage[] = [
    { role: 'system', content: 'Be terse.' },
    { role: 'user', content: 'hi' },
  ];
  const out = withToolGuidance(msgs, false);
  assert.equal(out.length, 2);
  assert.match(String(out[0].content), /^Be terse\.\n\n/);
});

await check('tool-refusing models are recognized for the plain-answer fallback', () => {
  assert.equal(
    isToolsUnsupported(
      'Ollama error 400: {"error":"registry.ollama.ai/library/smollm2:135m does not support tools"}',
    ),
    true,
  );
  assert.equal(isToolsUnsupported("This model's tool calls are not supported"), true);
  assert.equal(isToolsUnsupported('Ollama error 500: model runner has crashed'), false);
  assert.equal(isToolsUnsupported('fetch failed'), false);
});

/* ---------------- wire adapters with stubbed fetch ---------------- */

console.log('\n=== OpenAI-compatible adapter ===');

function sseBody(frames: string[]): Response {
  return new Response(frames.map((f) => `data: ${f}\n\n`).join('') + 'data: [DONE]\n\n', {
    headers: { 'content-type': 'text/event-stream' },
  });
}

await check('text streams live, fragmented tool calls reassemble', async () => {
  const bodies: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: unknown, init: any) => {
    bodies.push(String(init?.body ?? ''));
    return sseBody([
      '{"choices":[{"delta":{"content":"Hi"}}]}',
      '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a1","function":{"name":"web_se"}}]}}]}',
      '{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "arch", "arguments": "{\\"query\\""}}]}}]}',
      '{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":\\"q\\"}"}}]}}]}',
    ]);
  }) as typeof fetch;
  try {
    const chat = openAIChatTools('http://x/v1', () => ({}), 'Stub');
    const events: unknown[] = [];
    for await (const ev of chat({
      model: 'm',
      turns: [{ role: 'user', content: 'hi' }],
      tools: [webSearch.definition],
      signal: noSignal,
    })) {
      events.push(ev);
    }
    const req = JSON.parse(bodies[0]);
    assert.ok(Array.isArray(req.tools) && req.tools[0].function.name === 'web_search');
    assert.equal(req.tool_choice, 'auto');
    assert.equal(req.messages[0].content, 'hi');
    assert.deepEqual(events[0], { type: 'text', text: 'Hi' });
    assert.equal(events.length, 2);
    const calls = (events[1] as { calls: { id: string; name: string; args: unknown }[] }).calls;
    assert.equal(calls.length, 1);
    assert.equal(calls[0].id, 'a1');
    assert.equal(calls[0].name, 'web_search');
    assert.deepEqual(calls[0].args, { query: 'q' });
  } finally {
    globalThis.fetch = realFetch;
  }
});

await check('assistant tool turns map back with ids', async () => {
  const bodies: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (_url: unknown, init: any) => {
    bodies.push(String(init?.body ?? ''));
    return sseBody(['{"choices":[{"delta":{"content":"ok"}}]}']);
  }) as typeof fetch;
  try {
    const chat = openAIChatTools('http://x/v1', () => ({}), 'Stub');
    const turns: AgentTurn[] = [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'Searching. ', calls: [{ id: 'a1', name: 'web_search', args: { query: 'q' } }] },
      { role: 'tool', content: '', toolCallId: 'a1', output: 'r' },
    ];
    for await (const _ of chat({ model: 'm', turns, tools: [], signal: noSignal })) {
      /* drain */
    }
    const req = JSON.parse(bodies[0]);
    assert.equal(req.messages[1].tool_calls[0].id, 'a1');
    assert.equal(req.messages[1].tool_calls[0].function.name, 'web_search');
    assert.equal(req.messages[2].role, 'tool');
    assert.equal(req.messages[2].tool_call_id, 'a1');
    assert.ok(!('tools' in req), 'no tools key on the plain final round');
  } finally {
    globalThis.fetch = realFetch;
  }
});

console.log('\n=== Ollama Hermes fallback (models that emit calls as text) ===');

async function ollamaRound(contents: string[], toolNames: string[]) {
  const realFetch = globalThis.fetch;
  const ndjson =
    contents.map((content) => JSON.stringify({ message: { content } })).join('\n') +
    '\n' +
    // Real Ollama always terminates the stream, and the client expects it.
    JSON.stringify({ message: { content: '' }, done: true });
  globalThis.fetch = (async () =>
    new Response(ndjson, { headers: { 'content-type': 'application/x-ndjson' } })) as typeof fetch;
  try {
    const chat = ollamaChatTools('http://localhost:11434');
    const tools = toolNames.map((name) => ({
      name,
      description: name,
      parameters: {
        type: 'object' as const,
        properties: { query: { type: 'string' as const } },
        required: ['query'],
        additionalProperties: false as const,
      },
    }));
    const events: unknown[] = [];
    for await (const ev of chat({
      model: 'm',
      turns: [{ role: 'user', content: 'hi' }],
      tools,
      signal: noSignal,
    })) {
      events.push(ev);
    }
    return events;
  } finally {
    globalThis.fetch = realFetch;
  }
}

await check('bare pseudo-call converts, no text leaks', async () => {
  const events = await ollamaRound(['{"name":"wikipedia","arguments":{"query":"X"}}'], ['wikipedia']);
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { name: string; args: unknown }[] }).calls;
  assert.equal(calls.length, 1);
  assert.equal(calls[0].name, 'wikipedia');
  assert.deepEqual(calls[0].args, { query: 'X' });
});

await check('a call-shaped closing round converts instead of leaking JSON', async () => {
  // No tool list on the wire (the capped final round): the JSON must still
  // convert — the loop drops these calls, and the server answers directly.
  const events = await ollamaRound(
    ['{"name":"wikipedia","arguments":{"query":"Today\'s Parent","sentences":2}}'],
    [],
  );
  assert.equal(events.length, 1);
  assert.equal((events[0] as { type: string }).type, 'tool_calls');
  const calls = (events[0] as { calls: { name: string; args: unknown }[] }).calls;
  assert.equal(calls[0].name, 'wikipedia');
  assert.deepEqual(calls[0].args, { query: "Today's Parent", sentences: 2 });
});

await check('raw call JSON alongside a native call never leaks as text', async () => {
  const realFetch = globalThis.fetch;
  const ndjson = JSON.stringify({
    message: {
      content: '{"name":"wikipedia","arguments":{"query":"X"}}',
      tool_calls: [{ function: { name: 'wikipedia', arguments: { query: 'X' } } }],
    },
    done: true,
  });
  globalThis.fetch = (async () =>
    new Response(ndjson, { headers: { 'content-type': 'application/x-ndjson' } })) as typeof fetch;
  try {
    const chat = ollamaChatTools('http://localhost:11434');
    const events: unknown[] = [];
    for await (const ev of chat({
      model: 'm',
      turns: [{ role: 'user', content: 'hi' }],
      tools: [wikipedia.definition],
      signal: noSignal,
    })) {
      events.push(ev);
    }
    assert.equal(events.length, 1, 'the call-shaped content is dropped, not streamed');
    assert.equal((events[0] as { type: string }).type, 'tool_calls');
  } finally {
    globalThis.fetch = realFetch;
  }
});

await check('a leading-newline chunk does not flip a call to live text', async () => {
  // Two stream chunks: whitespace first (a common model padding), the call
  // second. Flushing the whitespace live would leak the JSON as the answer.
  const events = await ollamaRound(['\n', '{"name":"wikipedia","arguments":{"query":"W"}}'], ['wikipedia']);
  assert.equal(events.length, 1);
  assert.equal((events[0] as { type: string }).type, 'tool_calls');
  assert.deepEqual((events[0] as { calls: { args: unknown }[] }).calls[0].args, { query: 'W' });
});

await check('fenced pseudo-call converts', async () => {
  const events = await ollamaRound(
    ['```json\n{"name":"wikipedia","arguments":{"query":"Y"}}\n```'], ['wikipedia']);
  assert.equal(events.length, 1);
  assert.equal((events[0] as { calls: { name: string }[] }).calls[0].name, 'wikipedia');
});

await check('function-spelled pseudo-call converts', async () => {
  const events = await ollamaRound(
    ['{"function":"wikipedia","arguments":{"query":"Z"}}'], ['wikipedia']);
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { name: string; args: unknown }[] }).calls;
  assert.equal(calls[0].name, 'wikipedia');
  assert.deepEqual(calls[0].args, { query: 'Z' });
});

await check('one stray closing brace still converts', async () => {
  const events = await ollamaRound(
    ['{"name":"wikipedia","arguments":{"query":"W"}}}'], ['wikipedia']);
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { name: string; args: unknown }[] }).calls;
  assert.equal(calls[0].name, 'wikipedia');
  assert.deepEqual(calls[0].args, { query: 'W' });
});

await check('prose around the JSON stays plain text', async () => {
  const events = await ollamaRound(
    ['Here you go:\n{"name":"wikipedia","arguments":{"query":"X"}}'], ['wikipedia']);
  assert.equal(events.length, 1);
  assert.deepEqual(events[0], { type: 'text', text: 'Here you go:\n{"name":"wikipedia","arguments":{"query":"X"}}' });
});

await check('unknown tool names stay plain text', async () => {
  const events = await ollamaRound(['{"name":"ghost","arguments":{}}'], ['wikipedia']);
  assert.equal(events.length, 1);
  assert.equal((events[0] as { type: string }).type, 'text');
});

await check('a registered tool that was not offered converts, so the loop can refuse it', async () => {
  const events = await ollamaRound(['{"name":"web_search","arguments":{"query":"X"}}'], ['wikipedia']);
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { name: string }[] }).calls;
  assert.equal(calls.length, 1);
  assert.equal(calls[0].name, 'web_search');
});

await check('normal answers still stream live with tools offered', async () => {
  const events = await ollamaRound(['Hello ', 'world'], ['wikipedia']);
  assert.deepEqual(events, [
    { type: 'text', text: 'Hello ' },
    { type: 'text', text: 'world' },
  ]);
});

await check('token-at-a-time pseudo-call still converts', async () => {
  const events = await ollamaRound(
    ['{"', 'name', '":', ' "wikipedia"', ', "arguments"', ': {"query"', ': "Paris"', '}}'],
    ['wikipedia'],
  );
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { name: string; args: unknown }[] }).calls;
  assert.equal(calls.length, 1);
  assert.equal(calls[0].name, 'wikipedia');
  assert.deepEqual(calls[0].args, { query: 'Paris' });
});

await check('brace-led answer that diverges streams live', async () => {
  const events = await ollamaRound(['{"a"', ': 1, "b": 2}'], ['wikipedia']);
  assert.deepEqual(events, [
    { type: 'text', text: '{"a"' },
    { type: 'text', text: ': 1, "b": 2}' },
  ]);
});

await check('function-spelled call buffers incrementally then converts', async () => {
  const events = await ollamaRound(
    ['{"fun', 'ction":"wikipedia"', ', "arguments"', ': {"query"', ': "Q"}}'],
    ['wikipedia'],
  );
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { name: string; args: unknown }[] }).calls;
  assert.equal(calls[0].name, 'wikipedia');
  assert.deepEqual(calls[0].args, { query: 'Q' });
});

await check('non-json fences stream live, never convert', async () => {
  const events = await ollamaRound(['```python\n', 'print(1)\n```'], ['wikipedia']);
  assert.deepEqual(events, [
    { type: 'text', text: '```python\n' },
    { type: 'text', text: 'print(1)\n```' },
  ]);
});

await check('two pseudo-calls in one message both convert', async () => {
  const events = await ollamaRound(
    ['{"name":"web_search","arguments":{"query":"A"}}\n' +
      '{"name":"fetch_page","arguments":{"url":"https://example.com"}}'],
    ['web_search', 'fetch_page'],
  );
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { name: string; args: unknown }[] }).calls;
  assert.equal(calls.length, 2);
  assert.equal(calls[0].name, 'web_search');
  assert.deepEqual(calls[0].args, { query: 'A' });
  assert.equal(calls[1].name, 'fetch_page');
  assert.deepEqual(calls[1].args, { url: 'https://example.com' });
});

await check('pretty-printed pair converts', async () => {
  const events = await ollamaRound(
    ['{\n  "name": "wikipedia",\n  "arguments": {\n    "query": "P"\n  }\n}\n' +
      '{"name":"wikipedia","arguments":{"query":"Q"}}'],
    ['wikipedia'],
  );
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { args: unknown }[] }).calls;
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[0].args, { query: 'P' });
  assert.deepEqual(calls[1].args, { query: 'Q' });
});

await check('fenced pair converts', async () => {
  const events = await ollamaRound(
    ['```json\n{"name":"wikipedia","arguments":{"query":"A"}}\n' +
      '{"name":"wikipedia","arguments":{"query":"B"}}\n```'],
    ['wikipedia'],
  );
  assert.equal(events.length, 1);
  assert.equal((events[0] as { calls: unknown[] }).calls.length, 2);
});

await check('streamed fenced pretty call converts', async () => {
  const events = await ollamaRound(
    ['```json\n{\n', '  "name": "web_search",\n', '  "arguments": { "query": "X" }\n', '}\n```'],
    ['web_search'],
  );
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { name: string; args: unknown }[] }).calls;
  assert.equal(calls.length, 1);
  assert.equal(calls[0].name, 'web_search');
  assert.deepEqual(calls[0].args, { query: 'X' });
});

await check('calls wrapped in <tool_call> tags convert', async () => {
  const events = await ollamaRound(
    ['<tool_call>\n{"name":"wikipedia","arguments":{"query":"T"}}\n</tool_call>\n' +
      '<tool_call>\n{"function":"wikipedia","arguments":{"query":"U"}}\n</tool_call>'],
    ['wikipedia'],
  );
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { args: unknown }[] }).calls;
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[0].args, { query: 'T' });
  assert.deepEqual(calls[1].args, { query: 'U' });
});

await check('token-at-a-time pair still converts', async () => {
  const events = await ollamaRound(
    ['{"name":"wikipedia"', ',"arguments":{"query":"Q"}}', '\n{"na',
      'me":"wikipedia","arguments":{"query":"R"}}'],
    ['wikipedia'],
  );
  assert.equal(events.length, 1);
  const calls = (events[0] as { calls: { args: unknown }[] }).calls;
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[0].args, { query: 'Q' });
  assert.deepEqual(calls[1].args, { query: 'R' });
});

await check('prose after the calls stays plain text', async () => {
  const events = await ollamaRound(
    ['{"name":"wikipedia","arguments":{"query":"X"}}\nHope that helps!'],
    ['wikipedia'],
  );
  assert.equal(events.length, 1);
  const ev = events[0] as { type: string; text: string };
  assert.equal(ev.type, 'text');
  assert.match(ev.text, /Hope that helps!/);
});

await check('one unknown name keeps the whole message text', async () => {
  const events = await ollamaRound(
    ['{"name":"wikipedia","arguments":{"query":"X"}}\n{"name":"ghost","arguments":{}}'],
    ['wikipedia'],
  );
  assert.equal(events.length, 1);
  assert.equal((events[0] as { type: string }).type, 'text');
});

/* ---------------- XML tool-call dialects ---------------- */

console.log('\n=== XML tool-call dialects ===');

// Tags are built from escapes on purpose: a fixture IS the bytes a model
// writes, and anything that mangles one tag into another would quietly turn a
// parser test into a test of prose that happens to parse.
const LT = '\x3c';
const GT = '\x3e';
const closeTag = (name: string) => `${LT}/${name}${GT}`;
const el = (name: string, body = '') => `${LT}${name}${GT}${body}${closeTag(name)}`;
const glmCall = (...pairs: [string, string][]) =>
  el('tool_call', 'wikipedia' + pairs.map(([k, v]) => el('arg_key', k) + el('arg_value', v)).join(''));
const qwenCall = (...pairs: [string, string][]) =>
  el(
    'tool_call',
    `${LT}function=wikipedia${GT}` +
      pairs.map(([k, v]) => `${LT}parameter=${k}${GT}${v}${closeTag('parameter')}`).join('') +
      closeTag('function'),
  );
const minimaxCall = (...pairs: [string, string][]) =>
  el(
    'minimax:tool_call',
    `${LT}invoke name="wikipedia"${GT}` +
      pairs.map(([k, v]) => `${LT}parameter name="${k}"${GT}${v}${closeTag('parameter')}`).join('') +
      closeTag('invoke'),
  );
const functionaryCall = (args: string) =>
  `${LT}function=wikipedia${GT}${args}${closeTag('function')}`;

function callsOf(events: unknown[]): { id: string; name: string; args: unknown }[] {
  const ev = events.find((e) => (e as { type?: string }).type === 'tool_calls');
  assert.ok(ev, 'expected a tool_calls event');
  return (ev as { calls: { id: string; name: string; args: unknown }[] }).calls;
}

await check('GLM-4.5 arg_key/arg_value calls convert', async () => {
  const events = await ollamaRound([glmCall(['query', 'Turing'])], ['wikipedia']);
  const calls = callsOf(events);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].name, 'wikipedia');
  assert.deepEqual(calls[0].args, { query: 'Turing' });
});

await check('Qwen3-Coder parameter calls convert, values coerced', async () => {
  const events = await ollamaRound(
    [qwenCall(['query', 'Turing'], ['limit', '2'])],
    ['wikipedia'],
  );
  const calls = callsOf(events);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].name, 'wikipedia');
  // `2` is a number after parsing: models put JSON scalars inside the tags.
  assert.deepEqual(calls[0].args, { query: 'Turing', limit: 2 });
});

await check('MiniMax M2 invoke/parameter calls convert', async () => {
  const events = await ollamaRound([minimaxCall(['query', 'Turing'])], ['wikipedia']);
  const calls = callsOf(events);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].name, 'wikipedia');
  assert.deepEqual(calls[0].args, { query: 'Turing' });
});

await check('Functionary raw-JSON calls convert', async () => {
  const events = await ollamaRound(
    [functionaryCall('{"query":"Turing"}')],
    ['wikipedia'],
  );
  const calls = callsOf(events);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].args, { query: 'Turing' });
});

await check('prose before a tagged call is kept and streams first', async () => {
  const events = await ollamaRound(
    ['Let me check that. ' + glmCall(['query', 'X'])],
    ['wikipedia'],
  );
  assert.equal(events.length, 2);
  assert.equal((events[0] as { type: string }).type, 'text');
  assert.equal((events[0] as { text: string }).text.trim(), 'Let me check that.');
  assert.equal((events[1] as { type: string }).type, 'tool_calls');
  assert.deepEqual(callsOf(events)[0].args, { query: 'X' });
});

await check('two calls in one Qwen scope both convert', async () => {
  const one = `${LT}function=wikipedia${GT}${LT}parameter=query${GT}A${closeTag('parameter')}${closeTag('function')}`;
  const two = `${LT}function=web_search${GT}${LT}parameter=query${GT}B${closeTag('parameter')}${closeTag('function')}`;
  const events = await ollamaRound([el('tool_call', one + two)], ['wikipedia', 'web_search']);
  const calls = callsOf(events);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].name, 'wikipedia');
  assert.equal(calls[1].name, 'web_search');
});

await check('a dialect block split mid-tag across chunks still converts', async () => {
  const raw = 'Sure. ' + glmCall(['query', 'Turing']);
  const events = await ollamaRound(
    [raw.slice(0, 24), raw.slice(24, 60), raw.slice(60)],
    ['wikipedia'],
  );
  const text = events
    .filter((e) => (e as { type: string }).type === 'text')
    .map((e) => (e as { text: string }).text)
    .join('');
  assert.equal(text, 'Sure. ');
  assert.deepEqual(callsOf(events)[0].args, { query: 'Turing' });
});

await check('an unknown tool name in a dialect block stays text', async () => {
  const events = await ollamaRound(
    [glmCall(['query', 'X']).replace('wikipedia', 'ghost')],
    ['wikipedia'],
  );
  assert.equal(events.length, 1);
  assert.equal((events[0] as { type: string }).type, 'text');
});

await check('malformed dialect markup leaves the message as text', async () => {
  // The value tag never closes: no call can be read, so nothing is invented.
  const broken = el(
    'tool_call',
    'wikipedia' + el('arg_key', 'query') + `${LT}arg_value${GT}Turing`,
  );
  const events = await ollamaRound([broken], ['wikipedia']);
  assert.equal(events.length, 1);
  assert.equal((events[0] as { type: string }).type, 'text');
});

await check('a closing-round dialect call still converts (loop drops it)', async () => {
  // No tool list on the wire: shape-check only, so the loop can drop the call
  // instead of surfacing raw markup as the answer.
  const events = await ollamaRound([qwenCall(['query', 'X'])], []);
  assert.equal(events.length, 1);
  assert.equal((events[0] as { type: string }).type, 'tool_calls');
});

await check('double-wrapped JSON calls still convert', async () => {
  const events = await ollamaRound(
    ['{{"name":"wikipedia","arguments":{"query":"X"}}}'],
    ['wikipedia'],
  );
  const calls = callsOf(events);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].args, { query: 'X' });
});

await check('openAIChatTools converts a dialect call in content', async () => {
  // The llama.cpp and transformers paths get no tool_calls wire: the call is
  // only ever text, and the same dialect parser has to catch it.
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    sseBody([
      JSON.stringify({ choices: [{ delta: { content: glmCall(['query', 'X']) } }] }),
    ])) as typeof fetch;
  try {
    const chat = openAIChatTools('http://x/v1', () => ({}), 'Stub');
    const events: unknown[] = [];
    for await (const ev of chat({
      model: 'm',
      turns: [{ role: 'user', content: 'hi' }],
      tools: [
        {
          name: 'wikipedia',
          description: 'd',
          parameters: {
            type: 'object' as const,
            properties: { query: { type: 'string' as const } },
            required: ['query'],
            additionalProperties: false as const,
          },
        },
      ],
      signal: noSignal,
    })) {
      events.push(ev);
    }
    assert.equal(events.length, 1);
    assert.equal((events[0] as { type: string }).type, 'tool_calls');
    assert.deepEqual(callsOf(events)[0].args, { query: 'X' });
  } finally {
    globalThis.fetch = realFetch;
  }
});

console.log('\n=== Ollama adapter (native wire) ===');

await check('NDJSON content and tool calls parse (object and string args)', async () => {
  const bodies: string[] = [];
  const realFetch = globalThis.fetch;
  const ndjson = [
    JSON.stringify({ message: { content: 'Yo' } }),
    JSON.stringify({
      message: {
        content: '',
        tool_calls: [
          { function: { name: 'wikipedia', arguments: { query: 'X' } } },
          { function: { name: 'fetch_page', arguments: '{"url":"https://example.com"}' } },
        ],
      },
      done: true,
    }),
  ].join('\n');
  globalThis.fetch = (async (_url: unknown, init: any) => {
    bodies.push(String(init?.body ?? ''));
    return new Response(ndjson, { headers: { 'content-type': 'application/x-ndjson' } });
  }) as typeof fetch;
  try {
    const chat = ollamaChatTools('http://localhost:11434');
    const events: unknown[] = [];
    for await (const ev of chat({
      model: 'm',
      turns: [{ role: 'user', content: 'hi' }],
      tools: [wikipedia.definition, fetchPage.definition],
      signal: noSignal,
    })) {
      events.push(ev);
    }
    const req = JSON.parse(bodies[0]);
    assert.equal(req.tools.length, 2);
    // Built-ins bind as LangChain tools: routed through the zod schema
    // conversion, still an OpenAI-shaped function record on the wire.
    assert.equal(req.tools[0].function.name, 'wikipedia');
    assert.equal(req.tools[0].function.parameters.properties.query.type, 'string');
    assert.equal(req.messages[0].content, 'hi');
    assert.deepEqual(events[0], { type: 'text', text: 'Yo' });
    const calls = (events[1] as { calls: { name: string; args: unknown }[] }).calls;
    assert.equal(calls.length, 2);
    assert.deepEqual(calls[0].args, { query: 'X' });
    assert.deepEqual(calls[1].args, { url: 'https://example.com' });
  } finally {
    globalThis.fetch = realFetch;
  }
});

/* ---------------- live network (narrow, honest skips) ---------------- */

console.log('\n=== live executors ===');

function stubServer(handler: (url: string) => { status: number; body: string; type: string }) {
  const server = createServer((req, res) => {
    const out = handler(req.url ?? '/');
    res.writeHead(out.status, { 'content-type': out.type });
    res.end(out.body);
  });
  return new Promise<{ base: string; close: () => Promise<void> }>((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as AddressInfo;
      resolve({ base: `http://127.0.0.1:${port}`, close: () => new Promise((r) => server.close(() => r())) });
    });
  });
}

await checkLive('DDG executor honors the SSRF guard end to end', async () => {
  const fixture = `<html><body><div class="result"><h2>
    <a class="result__a" href="https://example.com/a">Alpha &amp; Omega</a></h2>
    <div class="result__snippet">first <b>hit</b></div></div></body></html>`;
  const srv = await stubServer(() => ({ status: 200, body: fixture, type: 'text/html' }));
  const prev = process.env.DDG_BASE_URL;
  process.env.DDG_BASE_URL = srv.base;
  try {
    // The stub is loopback, which the SSRF guard blocks before any parsing —
    // so this documents the guard, while the parser is covered by fixture.
    await webSearch.execute({ query: 'x' }, { signal: noSignal });
    assert.fail('loopback should have been blocked');
  } catch (err) {
    assert.match((err as Error).message, /not fetchable/);
  } finally {
    if (prev === undefined) delete process.env.DDG_BASE_URL;
    else process.env.DDG_BASE_URL = prev;
    await srv.close();
  }
});

await checkLive('wikipedia lookup works end to end', async () => {
  const out = await wikipedia.execute({ query: 'Albert Einstein' }, { signal: noSignal });
  assert.match(out, /Wikipedia — /);
  assert.match(out, /en\.wikipedia\.org\/wiki\//);
});

await checkLive('fetch_page reads a small public page', async () => {
  const out = await fetchPage.execute({ url: 'https://example.com' }, { signal: noSignal });
  assert.match(out, /Example Domain/);
});

await checkLive('web_search degrades honestly when challenged', async () => {
  try {
    const out = await webSearch.execute({ query: 'large language model' }, { signal: noSignal });
    assert.match(out, /^\d+\. |No results for/);
  } catch (err) {
    assert.match((err as Error).message, /challenged/);
    throw err; // rethrow so checkLive reports SKIP, not PASS-on-throw
  }
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log(`\nFailing: ${failures.join(', ')}`);
  process.exit(1);
}
