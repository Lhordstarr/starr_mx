import { ensureOk, parseSSE } from '../sse.js';
import { getKey, keySource } from '../store.js';
import { toAnthropicFormat } from '../attachments.js';
import type { Provider, StreamOptions } from './types.js';

const apiKey = () => getKey('ANTHROPIC_API_KEY');
const API_URL = 'https://api.anthropic.com/v1/messages';
const API_VERSION = '2023-06-01';

/**
 * Anthropic does not implement the OpenAI shape, so it gets a dedicated
 * adapter. Note the system prompt is hoisted out of the message array into a
 * top-level field, which the other providers do not require.
 */
export const anthropic: Provider = {
  info: {
    id: 'anthropic',
    label: 'Anthropic',
    get configured() {
      return Boolean(apiKey());
    },
    local: false,
    envKey: 'ANTHROPIC_API_KEY',
    get hint() {
      const source = keySource('ANTHROPIC_API_KEY');
      if (source === 'settings') return 'Key from settings page';
      if (source === 'env') return 'Key from .env';
      return 'Missing ANTHROPIC_API_KEY';
    },
  },

  async listModels() {
    const key = apiKey();
    if (!key) return [];
    try {
      const res = await fetch('https://api.anthropic.com/v1/models', {
        headers: { 'x-api-key': key, 'anthropic-version': API_VERSION },
      });
      if (!res.ok) return [];
      const data = (await res.json()) as { data?: { id: string }[] };
      return (data.data ?? []).map((m) => m.id).sort();
    } catch {
      return [];
    }
  },

  async *stream({ model, messages, signal, temperature }: StreamOptions) {
    const key = apiKey();
    if (!key) {
      throw new Error(
        'Anthropic: no API key. Add one on the settings page or set ANTHROPIC_API_KEY in .env.',
      );
    }

    const system = messages
      .filter((m) => m.role === 'system')
      .map((m) => (typeof m.content === 'string' ? m.content : ''))
      .join('\n\n');

    const res = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'x-api-key': key,
        'anthropic-version': API_VERSION,
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model,
        max_tokens: 4096,
        temperature,
        system: system || undefined,
        messages: messages
          .filter((m) => m.role !== 'system')
          .map((m) => ({ role: m.role, content: toAnthropicFormat(m.content) })),
      }),
      signal,
    });
    await ensureOk(res, 'Anthropic');

    let inTextBlock = false;
    // Extended thinking blocks arrive as their own content block (a gateway
    // may stream them even though this adapter never asks for thinking).
    let inThinkingBlock = false;
    for await (const payload of parseSSE(res.body!)) {
      const event = JSON.parse(payload);
      if (event.type === 'content_block_start') {
        inTextBlock = event.content_block?.type === 'text';
        inThinkingBlock = event.content_block?.type === 'thinking';
      } else if (event.type === 'content_block_delta' && inThinkingBlock) {
        const piece = event.delta?.thinking;
        if (typeof piece === 'string' && piece) yield { type: 'reasoning', text: piece };
      } else if (event.type === 'content_block_delta' && inTextBlock) {
        const piece = event.delta?.text;
        if (piece) yield { type: 'text', text: piece as string };
      } else if (event.type === 'error') {
        throw new Error(`Anthropic: ${event.error?.message ?? 'stream error'}`);
      }
    }
  },
};