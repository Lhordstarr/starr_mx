/**
 * Mock OpenAI-compatible endpoint for manual testing of custom providers.
 * Serves /models, /chat/completions (SSE) and /audio/transcriptions.
 *
 * Usage: CUSTOM_MOCK_PORT=4599 npx tsx scripts/mock-openai-endpoint.ts
 */
import { createServer } from 'node:http';

const port = Number(process.env.CUSTOM_MOCK_PORT ?? 4599);
const PREFIX = '/openai/v1';

const server = createServer((req, res) => {
  // Accept both with and without the /openai/v1 prefix.
  const path = req.url?.replace(PREFIX, '') ?? '';
  const auth = req.headers.authorization;

  if (path === '/models') {
    // MOCK_NO_STT=1 models an endpoint that serves chat but no speech models.
    const ids = ['local-model-7b', 'local-model-13b-instruct', 'tts-1'];
    if (!process.env.MOCK_NO_STT) ids.push('whisper-large-v3');
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        object: 'list',
        data: ids.map((id) => ({ id, object: 'model', owned_by: 'mock' })),
      }),
    );
    return;
  }

  if (path === '/chat/completions') {
    const chunks: Buffer[] = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const body = JSON.parse(Buffer.concat(chunks).toString() || '{}');
      const parts = describeContent(body.messages);
      console.log(`[mock] chat model=${body.model} auth=${auth ? 'yes' : 'no'} ${parts}`);
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });
      // MOCK_ECHO_MODEL replies with the model id verbatim, so a test can drive
      // arbitrary text through the real streaming + rendering path by putting
      // that text in the model field. Handy for checking markdown rendering.
      const words = process.env.MOCK_ECHO_MODEL ? String(body.model) : `mock reply using ${body.model}`;
      for (const w of words.split(' ')) {
        res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: `${w} ` } }] })}\n\n`);
      }
      res.write('data: [DONE]\n\n');
      res.end();
    });
    return;
  }

  if (path === '/audio/transcriptions') {
    const chunks: Buffer[] = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      console.log(`[mock] transcription auth=${auth ? 'yes' : 'none'} bytes=${Buffer.concat(chunks).length}`);
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ text: 'transcribed by the mock endpoint' }));
    });
    return;
  }

  res.writeHead(404, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: `no route for ${req.url}` }));
});

/**
 * Summarises the message shapes the client sent, so a test can assert on the
 * wire format rather than scraping logs for the whole base64 payload.
 */
function describeContent(messages: unknown): string {
  if (!Array.isArray(messages)) return 'messages=malformed';
  const out: string[] = [];
  for (const m of messages) {
    const msg = m as { role?: string; content?: unknown };
    if (typeof msg.content === 'string') {
      out.push(`${msg.role}:text(${msg.content.length})`);
    } else if (Array.isArray(msg.content)) {
      const kinds = msg.content.map((part) => {
        const p = part as { type?: string; mime_type?: string; text?: string };
        if (p.type === 'image_url') {
          const url = (p.image_url as { url?: string } | undefined)?.url ?? '';
          const [meta, b64] = url.split(',');
          return `image(${meta?.replace('data:', '')};b64:${b64?.length ?? 0})`;
        }
        if (p.type === 'inline_data') {
          return `image(${p.mime_type};b64:${((p as { data?: string }).data ?? '').length})`;
        }
        return `${p.type}(${(p.text ?? '').length})`;
      });
      out.push(`${msg.role}:[${kinds.join(' ')}]`);
    } else if (msg.content && typeof msg.content === 'object') {
      const o = msg.content as { images?: string[]; content?: string };
      out.push(`${msg.role}:ollama(text:${(o.content ?? '').length} images:${o.images?.length ?? 0})`);
    } else {
      out.push(`${msg.role}:empty`);
    }
  }
  return out.join(' | ');
}

server.listen(port, '127.0.0.1', () => console.log(`[mock] OpenAI-compatible endpoint on ${port}${PREFIX}`));