/**
 * Exercises the STT path against a mock OpenAI-compatible provider.
 *
 * The multipart envelope is built by hand, so this checks the parts a real
 * provider would reject: boundary framing, the filename extension, the model
 * field, and that the audio bytes arrive unmodified.
 */
import { createServer } from 'node:http';
import { openAICompatible } from '../src/providers/openai-compat.js';

let lastUpload: { contentType: string; body: Buffer } | null = null;

const mock = createServer((req, res) => {
  if (req.url === '/v1/models') {
    // Mirrors a real provider's list, including models that must be filtered out.
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        data: [
          { id: 'whisper-large-v3' },
          { id: 'whisper-large-v3-turbo' },
          { id: 'distil-whisper-large-v3-en' },
          { id: 'gpt-4o-transcribe' },
          { id: 'gpt-4o-transcribe-diarize' },
          { id: 'tts-1-hd' },
          { id: 'text-embedding-3-large' },
          { id: 'gpt-4o-mini' },
        ],
      }),
    );
    return;
  }
  if (req.url === '/v1/audio/transcriptions') {
    const chunks: Buffer[] = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      lastUpload = {
        contentType: req.headers['content-type'] ?? '',
        body: Buffer.concat(chunks),
      };
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ text: '  Hello from the mock.  ' }));
    });
    return;
  }
  res.writeHead(404).end();
});

const ok = (cond: boolean, label: string) =>
  console.log((cond ? '  PASS ' : '  FAIL ') + label);

await new Promise<void>((r) => mock.listen(0, '127.0.0.1', r));
const port = (mock.address() as { port: number }).port;

process.env.GROQ_API_KEY = 'test-key-for-mock';
const provider = openAICompatible({
  id: 'groq',
  label: 'MockSTT',
  baseUrl: `http://127.0.0.1:${port}/v1`,
  envKey: 'GROQ_API_KEY',
  stt: true,
});

console.log('=== STT model discovery (no hardcoded ids) ===');
const sttModels = (await provider.listSttModels?.()) ?? [];
console.log('  discovered:', sttModels);
ok(sttModels.includes('whisper-large-v3'), 'finds whisper models');
ok(sttModels.includes('gpt-4o-transcribe'), 'finds transcribe models');
ok(!sttModels.some((m) => m.startsWith('tts')), 'excludes TTS models');
ok(!sttModels.some((m) => m.includes('diarize')), 'excludes diarization variants');
ok(!sttModels.some((m) => m.includes('embedding')), 'excludes embedding models');
ok(!sttModels.some((m) => m.startsWith('gpt-4o-mini')), 'excludes plain chat models');

console.log('\n=== multipart upload ===');
// Bytes that would be corrupted by an accidental string conversion.
const audio = Buffer.from(Array.from({ length: 512 }, (_, i) => i % 256));
const controller = new AbortController();
const text = await provider.transcribe!({
  model: 'whisper-large-v3',
  audio,
  mimeType: 'audio/webm;codecs=opus',
  signal: controller.signal,
});

ok(lastUpload !== null, 'provider received the upload');
const ct = lastUpload!.contentType;
ok(ct.startsWith('multipart/form-data; boundary='), `content-type carries a boundary (${ct.slice(0, 45)}…)`);
const boundary = /boundary=(.+)$/.exec(ct)?.[1];
ok(Boolean(boundary), 'boundary is extractable');

const body = lastUpload!.body.toString('latin1');
ok(body.includes(`--${boundary}\r\n`), 'starts with a boundary line');
ok(body.trimEnd().endsWith(`--${boundary}--`), 'closes with the final boundary');
ok(
  body.includes('Content-Disposition: form-data; name="file"; filename="recording.webm"'),
  'file part carries a webm filename derived from the MIME type',
);
ok(body.includes('Content-Type: audio/webm'), 'file part declares the base MIME type');
ok(body.includes('Content-Disposition: form-data; name="model"'), 'includes a model part');
ok(body.includes('whisper-large-v3\r\n--'), 'model value precedes the closing boundary');

const start = body.indexOf('audio/webm\r\n\r\n') + 'audio/webm\r\n\r\n'.length;
const end = body.indexOf('\r\n--', start);
const received = Buffer.from(body.slice(start, end), 'latin1');
ok(received.equals(audio), `audio bytes round-trip unmodified (${received.length} of ${audio.length})`);

ok(text === 'Hello from the mock.', `transcript is returned trimmed (got ${JSON.stringify(text)})`);

console.log('\n=== error handling ===');
let msg = '';
try {
  await provider.transcribe!({
    model: 'whisper-large-v3',
    audio,
    mimeType: 'audio/flac', // valid audio, but not a container this API takes
    signal: controller.signal,
  });
} catch (e) {
  msg = (e as Error).message;
}
ok(msg.includes('unsupported audio type'), `rejects an unsupported container (${msg})`);

mock.close();