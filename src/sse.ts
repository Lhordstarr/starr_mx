/**
 * Streaming helpers shared by the provider adapters: a UTF-8 safe byte-stream
 * decoder, an SSE parser and an NDJSON parser.
 */

const encoder = new TextEncoder();

/**
 * Splits a byte stream into decoded text chunks.
 * Buffers partial UTF-8 sequences and partial SSE frames at the caller's
 * discretion; this only guarantees no multi-byte character is split.
 */
export async function* decodeStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<string, void, unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) yield decoder.decode(value, { stream: true });
    }
    const tail = decoder.decode();
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

/**
 * Parses an SSE byte stream into `data:` payloads.
 * Skips comments/keepalives, parses `[DONE]`, and tolerates multi-line data.
 */
export async function* parseSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<string, void, unknown> {
  let buffer = '';
  for await (const chunk of decodeStream(body)) {
    buffer += chunk;
    let sep: number;
    // Events are separated by a blank line; handle \n\n and \r\n\r\n.
    while ((sep = buffer.search(/\r?\n\r?\n/)) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + (buffer[sep] === '\r' ? 4 : 2));
      const data = raw
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())
        .join('\n');
      if (!data) continue;
      if (data === '[DONE]') return;
      yield data;
    }
  }
}

/** Parses Ollama's newline-delimited JSON stream into objects. */
export async function* parseNDJSON(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<any, void, unknown> {
  let buffer = '';
  for await (const chunk of decodeStream(body)) {
    buffer += chunk;
    let nl: number;
    while ((nl = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) yield JSON.parse(line);
    }
  }
  const rest = buffer.trim();
  if (rest) yield JSON.parse(rest);
}

/** Throws with the provider's own error text when a response is not 2xx. */
export async function ensureOk(res: Response, provider: string): Promise<Response> {
  if (res.ok) return res;
  let detail = '';
  try {
    detail = (await res.text()).slice(0, 500);
  } catch {
    /* body already consumed or unreadable */
  }
  throw new Error(`${provider} error ${res.status}: ${detail || res.statusText}`);
}

export { encoder };