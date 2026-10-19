import { ensureOk, parseSSE } from '../sse.js';
import { getKey, keySource } from '../store.js';
import { toGeminiParts } from '../attachments.js';
import type { Provider, StreamOptions } from './types.js';

const apiKey = () => getKey('GEMINI_API_KEY');
const BASE = 'https://generativelanguage.googleapis.com/v1beta';

/**
 * Google's Generative Language API. Unlike the OpenAI-compatible providers it
 * takes `contents` with `parts`, uses "model" rather than "assistant" as the
 * assistant role, and carries the system prompt in its own field.
 *
 * Streaming is requested with `alt=sse`, which turns the otherwise
 * chunked-JSON response into ordinary Server-Sent Events.
 */
export const gemini: Provider = {
  info: {
    id: 'gemini',
    label: 'Google Gemini',
    get configured() {
      return Boolean(apiKey());
    },
    local: false,
    envKey: 'GEMINI_API_KEY',
    get hint() {
      const source = keySource('GEMINI_API_KEY');
      if (source === 'settings') return 'Key from settings page';
      if (source === 'env') return 'Key from .env';
      return 'Missing GEMINI_API_KEY';
    },
  },

  async listModels() {
    const key = apiKey();
    if (!key) return [];
    try {
      // Key goes in a header rather than the query string, so it can't end up
      // in proxy or server logs.
      const res = await fetch(`${BASE}/models?pageSize=200`, {
        headers: { 'x-goog-api-key': key },
      });
      if (!res.ok) return [];
      const data = (await res.json()) as {
        models?: { name: string; supportedGenerationMethods?: string[] }[];
      };
      return (data.models ?? [])
        // Drop embedding and tuning models; they can't serve a chat turn.
        .filter((m) => (m.supportedGenerationMethods ?? []).includes('generateContent'))
        .map((m) => m.name.replace(/^models\//, ''))
        .sort();
    } catch {
      return [];
    }
  },

  async *stream({ model, messages, signal, temperature }: StreamOptions) {
    const key = apiKey();
    if (!key) {
      throw new Error(
        'Google Gemini: no API key. Add one on the settings page or set GEMINI_API_KEY in .env.',
      );
    }

    const system = messages
      .filter((m) => m.role === 'system')
      .map((m) => (typeof m.content === 'string' ? m.content : ''))
      .join('\n\n');

    const res = await fetch(`${BASE}/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`, {
      method: 'POST',
      headers: { 'x-goog-api-key': key, 'content-type': 'application/json' },
      body: JSON.stringify({
        contents: messages
          .filter((m) => m.role !== 'system')
          .map((m) => ({
            role: m.role === 'assistant' ? 'model' : 'user',
            parts: toGeminiParts(m.content),
          })),
        systemInstruction: system ? { parts: [{ text: system }] } : undefined,
        generationConfig: {
          temperature,
          maxOutputTokens: 8192,
        },
      }),
      signal,
    });
    await ensureOk(res, 'Google Gemini');

    for await (const payload of parseSSE(res.body!)) {
      const event = JSON.parse(payload);
      // A candidate can carry several parts; take them in order, keeping
      // thought parts (Gemini 2.5's thinking) out of the answer text.
      for (const part of event.candidates?.[0]?.content?.parts ?? []) {
        if (!part.text) continue;
        if (part.thought) yield { type: 'reasoning', text: part.text as string };
        else yield { type: 'text', text: part.text as string };
      }
      if (event.error) {
        throw new Error(`Google Gemini: ${event.error.message ?? 'stream error'}`);
      }
    }
  },
};