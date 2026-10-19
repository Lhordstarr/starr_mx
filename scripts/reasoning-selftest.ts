/**
 * Unit checks for thinking separation: the streaming splitter, its wiring
 * through the agent loop, and the provider fields that carry scratchpad.
 *
 * The splitter is the load-bearing piece: thinking models write their
 * reasoning into the same byte stream as the answer, and a mistake either
 * dumps scratchpad into the transcript or swallows the real answer. The tag
 * table, the hold-back across chunk boundaries, and the no-guessing rule for
 * a lone closer all matter, so they are exercised directly and then again
 * through a stubbed model round.
 *
 * Run with: npm run test:reasoning
 */
import '../src/net.js';
import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ThinkingSplitter, reasoningText } from '../src/reasoning.js';
import { ollamaChatTools, openAIChatTools, runAgent } from '../src/agent.js';
import type { AgentEvent, ChatToolsEvent } from '../src/agent.js';
import type { AgentTurn } from '../src/providers/types.js';

process.env.TOOLS_FILE = join(mkdtempSync(join(tmpdir(), 'reasoning-test-')), 'tools.json');

let passed = 0;
const failures: string[] = [];

async function check(name: string, fn: () => void | Promise<void>): Promise<void> {
  try {
    await fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    fail(name, (err as Error).message);
  }
}

function fail(name: string, message: string): void {
  failures.push(name);
  console.log(`  FAIL  ${name}\n        ${message}`);
}

const noSignal = AbortSignal.timeout(20000);

/** Feeds pieces through one splitter and merges the two lanes. */
function feed(pieces: string[]): { reasoning: string; content: string } {
  const s = new ThinkingSplitter();
  let reasoning = '';
  let content = '';
  for (const p of pieces) {
    const d = s.push(p);
    reasoning += d.reasoning;
    content += d.content;
  }
  const tail = s.flush();
  return { reasoning: reasoning + tail.reasoning, content: content + tail.content };
}

/** The server's forwarding loop in miniature: provider events through the
 *  splitter, reasoning fields bypassing it. */
async function throughServer(events: AsyncIterable<ChatToolsEvent>) {
  const think = new ThinkingSplitter();
  let reasoning = '';
  let content = '';
  for await (const ev of events) {
    if (ev.type === 'reasoning') {
      reasoning += ev.text;
    } else if (ev.type === 'text') {
      const d = think.push(ev.text);
      reasoning += d.reasoning;
      content += d.content;
    }
  }
  const tail = think.flush();
  return { reasoning: reasoning + tail.reasoning, content: content + tail.content };
}

console.log('\n=== ThinkingSplitter ===');

await check('plain answers stream straight through', async () => {
  const out = feed(['Hello ', 'world']);
  assert.deepEqual(out, { reasoning: '', content: 'Hello world' });
});

await check('the first chunk of a non-tag answer is not held back', () => {
  const s = new ThinkingSplitter();
  const d = s.push('Hi there');
  assert.equal(d.content, 'Hi there');
  assert.equal(d.reasoning, '');
});

const DEEPSEEK_PAIRS: [string, string][] = [
  ['<|START_THINKING|>', '<|END_THINKING|>'],
  ['<|inner_prefix|>', '<|inner_suffix|>'],
  ['<seed:think>', '</seed:think>'],
  ['<think>', '</think>'],
  ['<|think|>', '<|end|><|begin|>assistant<|content|>'],
  ['<|channel>thought', '<channel|>'],
  ['<thinking>', '</thinking>'],
  ['[THINK]', '[/THINK]'],
];

await check('every tag pair in the table splits reasoning from answer', () => {
  for (const [start, end] of DEEPSEEK_PAIRS) {
    const out = feed([`${start}weighing options${end}The answer.`]);
    assert.equal(out.reasoning, 'weighing options', `reasoning for ${start}`);
    assert.equal(out.content, 'The answer.', `content for ${start}`);
  }
});

await check('tags never leak into either lane', () => {
  const out = feed(['<thinking>a</thinking>'],
  );
  assert.equal(out.reasoning, 'a');
  assert.equal(out.content, '');
});

await check('an opener split across chunks is held, not painted', () => {
  const out = feed(['<thin', 'king>Hmm', '</think', 'ing>Done']);
  assert.equal(out.reasoning, 'Hmm');
  assert.equal(out.content, 'Done');
});

await check('a closer split across chunks is held, not misread as reasoning', () => {
  const out = feed(['<thinking>deep', '</thi', 'nking>Answer']);
  assert.equal(out.reasoning, 'deep');
  assert.equal(out.content, 'Answer');
});

await check('an unclosed block: everything after the opener is reasoning', () => {
  const out = feed(['<thinking>still going']);
  assert.equal(out.reasoning, 'still going');
  assert.equal(out.content, '');
});

await check('a lone closer is never guessed as reasoning', () => {
  // Prefilled templates are covered by structured fields (Ollama `thinking`,
  // llama.cpp `reasoning_content`); guessing from a stray closer would eat
  // real answers, so the text stays answer text.
  const out = feed(['The tag </think> closes a thinking block.']);
  assert.equal(out.reasoning, '');
  assert.equal(out.content, 'The tag </think> closes a thinking block.');
});

await check('whitespace padding before a tag is dropped, not painted', () => {
  const out = feed(['\n\n<thinking>x</thinking>Answer']);
  assert.equal(out.reasoning, 'x');
  assert.equal(out.content, 'Answer');
});

await check('a second block at a line start is extracted too', () => {
  const out = feed(['<thinking>one</thinking>Middle\n<thinking>two</thinking>End']);
  assert.equal(out.reasoning, 'onetwo');
  assert.equal(out.content, 'Middle\nEnd');
});

await check('a tag quoted mid-sentence stays answer text', () => {
  const out = feed(['The literal <thinking> tag opens a block.']);
  assert.equal(out.reasoning, '');
  assert.equal(out.content, 'The literal <thinking> tag opens a block.');
});

await check('tool-call JSON at the head is answer text, not reasoning', () => {
  const out = feed(['{"name":"wikipedia","arguments":{"query":"X"}}']);
  assert.equal(out.reasoning, '');
  assert.equal(out.content, '{"name":"wikipedia","arguments":{"query":"X"}}');
});

await check('held text flushes to the lane it was heading for', () => {
  const s = new ThinkingSplitter();
  const a = s.push('<thinking>unfinished');
  const b = s.flush();
  // Whether it streams in push or waits for flush, the lane must be right.
  assert.deepEqual(
    { reasoning: a.reasoning + b.reasoning, content: a.content + b.content },
    { reasoning: 'unfinished', content: '' },
  );
  // And a held non-tag tail flushes as content.
  const t = new ThinkingSplitter();
  t.push('<thin');
  assert.deepEqual(t.flush(), { reasoning: '', content: '<thin' });
});

await check('incremental output equals whole-buffer output, char by char', () => {
  const text = '\n<thinking>Let me weigh: {\"a\":1}\nand </thinking>\nFinal.\n[THINK]more[/THINK] tail';
  const oneShot = feed([text]);
  const pieces: string[] = [];
  for (const ch of text) pieces.push(ch);
  const incremental = feed(pieces);
  assert.deepEqual(incremental, oneShot);
});

console.log('\n=== reasoningText ===');

await check('wire fields: string, parts, wrapper, junk', () => {
  assert.equal(reasoningText('thinking hard'), 'thinking hard');
  assert.equal(reasoningText([{ text: 'a' }, { text: 'b' }]), 'ab');
  assert.equal(reasoningText({ content: 'wrapped' }), 'wrapped');
  assert.equal(reasoningText(undefined), '');
  assert.equal(reasoningText(42), '');
});

console.log('\n=== provider fields ===');

await check('openAIChatTools reads reasoning_content and reasoning', async () => {
  const realFetch = globalThis.fetch;
  const sse = [
    '{"choices":[{"delta":{"reasoning_content":"hmm"}}]}',
    '{"choices":[{"delta":{"reasoning":"…more"}}]}',
    '{"choices":[{"delta":{"content":"Answer"}}]}',
  ]
    .map((f) => `data: ${f}\n\n`)
    .join('');
  globalThis.fetch = (async () =>
    new Response(sse + 'data: [DONE]\n\n', {
      headers: { 'content-type': 'text/event-stream' },
    })) as typeof fetch;
  try {
    const chat = openAIChatTools('http://x/v1', () => ({}), 'Stub');
    const events: ChatToolsEvent[] = [];
    for await (const ev of chat({
      model: 'm',
      turns: [{ role: 'user', content: 'hi' }],
      tools: [],
      signal: noSignal,
    })) {
      events.push(ev);
    }
    assert.deepEqual(events, [
      { type: 'reasoning', text: 'hmm' },
      { type: 'reasoning', text: '…more' },
      { type: 'text', text: 'Answer' },
    ]);
  } finally {
    globalThis.fetch = realFetch;
  }
});

await check('ollamaChatTools reads message.thinking', async () => {
  const realFetch = globalThis.fetch;
  const ndjson =
    [
      JSON.stringify({ message: { thinking: 'hmm' } }),
      JSON.stringify({ message: { content: 'Answer' } }),
      JSON.stringify({ message: { content: '' }, done: true }),
    ].join('\n') + '\n';
  globalThis.fetch = (async () =>
    new Response(ndjson, { headers: { 'content-type': 'application/x-ndjson' } })) as typeof fetch;
  try {
    const chat = ollamaChatTools('http://localhost:11434');
    const events: ChatToolsEvent[] = [];
    for await (const ev of chat({
      model: 'm',
      turns: [{ role: 'user', content: 'hi' }],
      tools: [],
      signal: noSignal,
    })) {
      events.push(ev);
    }
    assert.deepEqual(events, [
      { type: 'reasoning', text: 'hmm' },
      { type: 'text', text: 'Answer' },
    ]);
  } finally {
    globalThis.fetch = realFetch;
  }
});

await check('thinking tags written into content are split end to end', async () => {
  // The transformers server and raw gateways produce neither field: the tags
  // arrive inside content and the server-side splitter has to catch them.
  const realFetch = globalThis.fetch;
  const ndjson =
    [
      JSON.stringify({ message: { content: 'Let me think. <thi' } }),
      JSON.stringify({ message: { content: 'nking>First option\nSeco' } }),
      JSON.stringify({ message: { content: 'nd.</thinking>The answer is 4' } }),
      JSON.stringify({ message: { content: '2.' } }),
      JSON.stringify({ message: { content: '' }, done: true }),
    ].join('\n') + '\n';
  globalThis.fetch = (async () =>
    new Response(ndjson, { headers: { 'content-type': 'application/x-ndjson' } })) as typeof fetch;
  try {
    const chat = ollamaChatTools('http://localhost:11434');
    const { reasoning, content: answer } = await throughServer(
      chat({ model: 'm', turns: [{ role: 'user', content: 'hi' }], tools: [], signal: noSignal }),
    );
    assert.equal(reasoning, 'First option\nSecond.');
    assert.equal(answer, 'Let me think. The answer is 42.');
  } finally {
    globalThis.fetch = realFetch;
  }
});

console.log('\n=== the agent loop ===');

await check('runAgent forwards reasoning and keeps it out of history turns', async () => {
  const seenTurns: AgentTurn[][] = [];
  let round = 0;
  const chat = async function* (opts: { turns: AgentTurn[] }): AsyncGenerator<ChatToolsEvent> {
    seenTurns.push(opts.turns.map((t) => ({ ...t })));
    round += 1;
    if (round === 1) {
      yield { type: 'reasoning', text: 'pondering ' };
      yield { type: 'reasoning', text: 'the request' };
      yield { type: 'text', text: 'Looking.' };
      yield { type: 'tool_calls', calls: [{ id: '', name: 'ghost', args: {} }] };
    } else {
      yield { type: 'text', text: 'Final answer.' };
    }
  };
  const events: AgentEvent[] = [];
  for await (const ev of runAgent({
    chat,
    model: 'm',
    messages: [{ role: 'user', content: 'hi' }],
    tools: [],
    signal: noSignal,
  })) {
    events.push(ev);
  }
  const reasoning = events.filter((e) => e.type === 'reasoning').map((e) => e.text).join('');
  assert.equal(reasoning, 'pondering the request');
  // The assistant turn the loop hands back carries the answer text only.
  const assistant = seenTurns[1].find((t) => t.role === 'assistant')!;
  assert.equal(assistant.content, 'Looking.');
  assert.ok(
    !JSON.stringify(seenTurns[1]).includes('pondering'),
    'scratchpad must not be replayed to the provider',
  );
});

console.log('');
const total = passed + failures.length;
if (failures.length) {
  console.log(`${passed}/${total} checks failed:`);
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
console.log(`${passed}/${total} checks passed.`);
