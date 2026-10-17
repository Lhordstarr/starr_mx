/**
 * Unit checks for the attachment layer. Covers the parts that are hard to reach
 * through the browser: the size/count caps, and the per-provider wire formats.
 *
 * Run with: npm run test:attachments
 */
import assert from 'node:assert/strict';

import {
  checkAttachments,
  contentLength,
  parseContent,
  toAnthropicFormat,
  toGeminiParts,
  toOllamaMessage,
  toOpenAIFormat,
  IMAGE_MIME,
  isTextFile,
  resolveMime,
  LIMITS,
} from '../src/attachments.js';
import type { ImagePart, TextPart } from '../src/types.js';

let passed = 0;
const failures: string[] = [];

function check(name: string, fn: () => void) {
  try {
    fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures.push(name);
    console.log(`  FAIL  ${name}\n        ${err instanceof Error ? err.message : err}`);
  }
}

const image = (bytes: number, mimeType = 'image/png'): ImagePart => ({
  type: 'image',
  // base64 is 4/3 the byte length, so this yields `bytes` when decoded.
  data: 'A'.repeat(Math.ceil((bytes * 4) / 3)),
  mimeType,
});

const turn = (content: unknown) => [{ role: 'user' as const, content: content as never }];

console.log('\n=== caps ===');

check('a small image is allowed', () => {
  assert.deepEqual(checkAttachments(turn([image(1000)])), []);
});

check('an image over the per-image cap is rejected', () => {
  const problems = checkAttachments(turn([image(LIMITS.imageBytes + 1)]));
  assert.equal(problems.length, 1);
  assert.match(problems[0].reason, /under 5MB/);
});

check('an image exactly at the per-image cap is allowed', () => {
  assert.deepEqual(checkAttachments(turn([image(LIMITS.imageBytes)])), []);
});

check(`${LIMITS.imagesPerMessage} images are allowed`, () => {
  assert.deepEqual(checkAttachments(turn(Array(LIMITS.imagesPerMessage).fill(image(1000)))), []);
});

check('one image over the count cap is rejected', () => {
  const many = Array(LIMITS.imagesPerMessage + 1).fill(image(1000));
  const problems = checkAttachments(turn(many));
  assert.equal(problems.length, 1);
  assert.match(problems[0].reason, /At most 4 images/);
});

check('the count cap applies across messages, not per message', () => {
  // Two messages of three each is six images overall, which is over the cap.
  const messages = [
    { role: 'user' as const, content: [image(1000), image(1000), image(1000)] as never },
    { role: 'user' as const, content: [image(1000), image(1000), image(1000)] as never },
  ];
  assert.equal(checkAttachments(messages).length, 1);
});

check('many medium images exceeding the total cap are rejected', () => {
  // Four images, each under the 5MB per-image cap, together over 12MB.
  const each = LIMITS.imageBytes - 1024;
  assert.equal(checkAttachments(turn(Array(4).fill(image(each)))).length, 1);
});

check('a plain string message produces no problems', () => {
  assert.deepEqual(checkAttachments(turn('just text')), []);
});

console.log('\n=== content length ===');

check('a string counts its characters', () => {
  assert.equal(contentLength('hello'), 5);
});

check('an image counts as a flat allowance', () => {
  const one = contentLength([image(10)]);
  assert.equal(one, 4_000);
  // Independent of the payload: a bigger image is not "more tokens" to the guard.
  assert.equal(contentLength([image(1_000_000)]), 4_000);
});

check('mixed content adds up', () => {
  const text: TextPart = { type: 'text', text: 'hello' };
  assert.equal(contentLength([text, image(10)]), 5 + 4_000);
});

console.log('\n=== OpenAI wire format ===');

check('a plain string passes through untouched', () => {
  assert.equal(toOpenAIFormat('hello'), 'hello');
});

check('text parts become {type:text}', () => {
  const out = toOpenAIFormat([{ type: 'text', text: 'hi' }]) as { type: string }[];
  assert.deepEqual(out, [{ type: 'text', text: 'hi' }]);
});

check('an image becomes an image_url data URL', () => {
  const out = toOpenAIFormat([image(3)]) as { type: string; image_url: { url: string } }[];
  assert.equal(out[0].type, 'image_url');
  assert.match(out[0].image_url.url, /^data:image\/png;base64,/);
});

check('the base64 carries no data: prefix inside the field', () => {
  const out = toOpenAIFormat([image(3)]) as { image_url: { url: string } }[];
  assert.equal(out[0].image_url.url.split(',').length, 2);
});

check('a trailing image gets an empty text part appended', () => {
  const out = toOpenAIFormat([image(3)]) as { type: string }[];
  assert.equal(out.at(-1)?.type, 'text');
  // Gateways reject a turn whose final part is not text.
  assert.equal(out.length, 2);
});

check('a trailing image followed by text is left alone', () => {
  const out = toOpenAIFormat([image(3), { type: 'text', text: 'tail' }]) as { type: string }[];
  assert.equal(out.length, 2);
  assert.equal(out.at(-1)?.type, 'text');
});

console.log('\n=== Ollama wire format ===');

check('a plain string keeps the {content} shape', () => {
  assert.deepEqual(toOllamaMessage('hello'), { content: 'hello' });
});

check('images move to a separate array', () => {
  const out = toOllamaMessage([{ type: 'text', text: 'look' }, image(3)]);
  assert.equal(out.content, 'look');
  assert.equal(out.images?.length, 1);
});

check('no images means no images key', () => {
  const out = toOllamaMessage([{ type: 'text', text: 'look' }]);
  assert.equal(out.images, undefined);
});

check('an image-only turn keeps empty text', () => {
  const out = toOllamaMessage([image(3)]);
  assert.equal(out.content, '');
  assert.equal(out.images?.length, 1);
});

console.log('\n=== Anthropic wire format ===');

check('an image becomes a typed source block', () => {
  const out = toAnthropicFormat([image(3)]) as Record<string, unknown>[];
  assert.equal(out[0].type, 'image');
  const source = out[0].source as Record<string, unknown>;
  assert.equal(source.type, 'base64');
  assert.equal(source.media_type, 'image/png');
  assert.equal(typeof source.data, 'string');
});

check('text becomes a text block', () => {
  const out = toAnthropicFormat([{ type: 'text', text: 'hi' }]) as Record<string, unknown>[];
  assert.deepEqual(out, [{ type: 'text', text: 'hi' }]);
});

console.log('\n=== Gemini wire format ===');

check('an image becomes inline_data', () => {
  const out = toGeminiParts([image(3)]) as Record<string, unknown>[];
  const inline = out[0].inline_data as Record<string, unknown>;
  assert.equal(inline.mime_type, 'image/png');
  assert.equal(typeof inline.data, 'string');
});

check('a plain string becomes one text part', () => {
  assert.deepEqual(toGeminiParts('hi'), [{ text: 'hi' }]);
});

console.log('\n=== input validation ===');

check('a plain string parses', () => {
  assert.equal(parseContent('hi'), 'hi');
});

check('valid parts parse', () => {
  const parsed = parseContent([{ type: 'text', text: 'hi' }, image(3)]);
  assert.equal(Array.isArray(parsed), true);
  assert.equal((parsed as unknown[]).length, 2);
});

for (const [label, bad] of [
  ['an image with no data', [{ type: 'image', mimeType: 'image/png' }]],
  ['an image with no mimeType', [{ type: 'image', data: 'x' }]],
  ['an image with an unsupported type', [{ type: 'image', data: 'x', mimeType: 'image/svg+xml' }]],
  ['a text part with no text', [{ type: 'text' }]],
  ['an unknown part type', [{ type: 'video', data: 'x' }]],
  ['a non-object part', [42]],
  ['a numeric content', 42],
  ['a null content', null],
] as const) {
  check(`${label} is rejected`, () => {
    assert.equal(parseContent(bad), null);
  });
}

console.log('\n=== mime resolution ===');

check('a declared type is trusted', () => {
  assert.equal(resolveMime('a.png', 'image/png'), 'image/png');
});

check('a generic type falls back to the extension', () => {
  assert.equal(resolveMime('a.png', 'application/octet-stream'), 'image/png');
  assert.equal(resolveMime('a.JPEG', 'application/octet-stream'), 'image/jpeg');
});

check('a parameterised type is trimmed', () => {
  assert.equal(resolveMime('a.png', 'image/png; charset=binary'), 'image/png');
});

check('svg is not an accepted image type', () => {
  assert.equal(IMAGE_MIME.has('image/svg+xml'), false);
  // Scriptable, so it must never reach a provider or be rendered inline.
});

console.log('\n=== text file detection ===');

check('common code and data extensions are text', () => {
  for (const name of ['a.txt', 'a.md', 'a.json', 'a.py', 'a.ts', 'a.yaml', 'Makefile']) {
    if (!name.includes('.')) continue; // extensionless special names
    assert.equal(isTextFile(name), true, name);
  }
});

check('binary formats are not text', () => {
  for (const name of ['a.png', 'a.zip', 'a.pdf', 'a.exe']) {
    assert.equal(isTextFile(name), false, name);
  }
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log(`\nFailing: ${failures.join(', ')}`);
  process.exit(1);
}
