#!/usr/bin/env tsx
/**
 * A generated image must be labelled with the size it actually has.
 *
 * The pollinations adapter asks for 1024×1024 and the service answers with a
 * 768×768 JPEG. The adapter reported the request, so the gallery caption and
 * the saved session both said 1024×1024 for a file that was never that big.
 * The bytes are the authority: these checks build real headers for the four
 * formats the providers return and drive the same function `generateImage`
 * uses to settle the answer.
 *
 * The last check is the one that matters: a provider claiming 1024 while
 * sending a 768 JPEG must be reported as 768.
 */
import assert from 'node:assert/strict';

import { actualImageSize, imageDimensions } from '../src/image-size.ts';

let passed = 0;
const failures: string[] = [];
function check(name: string, fn: () => unknown): void {
  try {
    fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err: unknown) {
    failures.push(name);
    console.log(`  FAIL  ${name} — ${(err as Error).message}`);
  }
}

/** A PNG signature and the start of an IHDR chunk with the given size. */
function png(width: number, height: number): Buffer {
  const bytes = Buffer.alloc(33);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(bytes, 0);
  bytes.writeUInt32BE(13, 8);
  bytes.write('IHDR', 12, 'latin1');
  bytes.writeUInt32BE(width, 16);
  bytes.writeUInt32BE(height, 20);
  return bytes;
}

/** Two segments before a start-of-frame: APP0 and SOF0, like a camera JPEG. */
function jpeg(width: number, height: number): Buffer {
  return Buffer.concat([
    Buffer.from([0xff, 0xd8]),
    Buffer.from([
      0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00,
      0x01, 0x00, 0x00,
    ]),
    Buffer.from([
      0xff, 0xc0, 0x00, 0x11, 0x08, (height >> 8) & 0xff, height & 0xff, (width >> 8) & 0xff,
      width & 0xff, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01,
    ]),
    Buffer.from([0xff, 0xd9]),
  ]);
}

await check('a PNG header gives its dimensions', () => {
  assert.deepEqual(imageDimensions(png(512, 512)), { width: 512, height: 512 });
});

await check('a JPEG header is found after the segments that precede the frame', () => {
  assert.deepEqual(imageDimensions(jpeg(768, 768)), { width: 768, height: 768 });
});

await check('a WebP VP8X header gives its dimensions', () => {
  const bytes = Buffer.alloc(30);
  bytes.write('RIFF', 0, 'latin1');
  bytes.writeUInt32LE(22, 4);
  bytes.write('WEBP', 8, 'latin1');
  bytes.write('VP8X', 12, 'latin1');
  bytes.writeUInt32LE(10, 16);
  bytes.writeUIntLE(639, 24, 3);
  bytes.writeUIntLE(479, 27, 3);
  assert.deepEqual(imageDimensions(bytes), { width: 640, height: 480 });
});

await check('a GIF header gives its dimensions', () => {
  const bytes = Buffer.alloc(13);
  bytes.write('GIF89a', 0, 'latin1');
  bytes.writeUInt16LE(320, 6);
  bytes.writeUInt16LE(200, 8);
  assert.deepEqual(imageDimensions(bytes), { width: 320, height: 200 });
});

await check('a smaller answer overrides the size the provider claims', () => {
  // This is the real Pollinations case: request 1024, receive 768.
  const answer = actualImageSize({
    data: jpeg(768, 768).toString('base64'),
    width: 1024,
    height: 1024,
  });
  assert.deepEqual(answer, { width: 768, height: 768 });
});

await check('an unreadable payload keeps the provider claim', () => {
  assert.deepEqual(actualImageSize({ data: '', width: 1024, height: 1024 }), {
    width: 1024,
    height: 1024,
  });
  assert.deepEqual(
    actualImageSize({ data: Buffer.from('not an image').toString('base64'), width: 512, height: 512 }),
    { width: 512, height: 512 },
  );
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  for (const failure of failures) console.log(`  - ${failure}`);
  process.exit(1);
}
