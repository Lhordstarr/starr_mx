import { parseNDJSON } from '../sse.js';
import { toOllamaMessage } from '../attachments.js';
import { ollamaChatTools } from '../agent.js';
import { listLocalModels } from '../ollama-models.js';
import {
  ensureOllamaRunning,
  hookOllamaExit,
  noteOllamaUsed,
  ollamaReachable,
} from '../ollama-server.js';
import type { Provider, StreamOptions } from './types.js';

const baseUrl = process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434';

/**
 * Makes sure the daemon is up, then does the thing.
 *
 * Never used by `listModels`, which runs on every page load and every settings
 * refresh: starting a daemon because someone opened the settings page is the
 * surprise this app has spent its life avoiding. A chat request and a model pull
 * are the two moments a daemon is genuinely wanted.
 */
async function withDaemon<T>(what: () => Promise<T>): Promise<T> {
  hookOllamaExit();
  if (!(await ollamaReachable())) await ensureOllamaRunning();
  noteOllamaUsed();
  return what();
}

/**
 * Fully local provider. Ollama streams newline-delimited JSON from /api/chat,
 * which we flatten into plain text deltas.
 *
 * Images ride in a separate `images` array on each message rather than as
 * content parts, which is Ollama's own shape.
 */
export const ollama: Provider = {
  info: {
    id: 'ollama',
    label: 'Ollama (local)',
    configured: true,
    local: true,
    hint: `Running at ${baseUrl}`,
    // Native /api/chat tools, plus a Hermes-text fallback: small models often
    // emit calls as content instead of binding to tool_calls (verified live).
    supportsTools: true,
  },

  async listModels() {
    // No `withDaemon` here on purpose. This runs on every page load and every
    // settings refresh, and launching a daemon because someone opened a page is
    // the kind of surprise this app has spent its life avoiding. When the
    // daemon is down — it is killed when this app exits — the list comes from
    // Ollama's manifest store, so a restart still offers the installed models
    // (and one of them can start the daemon). Most recent first: that model is
    // the one most likely already resident in memory.
    try {
      return (await listLocalModels()).map((m) => m.name);
    } catch {
      return [];
    }
  },

  /**
   * Ollama is the one provider that actually reports vision support: /api/show
   * returns a `capabilities` array, so this is a real answer rather than a
   * guess from the model's name.
   */
  async listVisionModels() {
    const names = await ollama.listModels();
    const found: string[] = [];
    // Sequential rather than parallel: /api/show loads model metadata, and a
    // dozen of those at once makes a local daemon thrash.
    for (const name of names) {
      try {
        const res = await fetch(`${baseUrl}/api/show`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ name, model: name }),
        });
        if (!res.ok) continue;
        const data = (await res.json()) as { capabilities?: string[] };
        if (data.capabilities?.includes('vision')) found.push(name);
      } catch {
        // A model that cannot be described is simply not claimed as vision-capable.
      }
    }
    return found;
  },

  async *stream({ model, messages, signal, temperature }: StreamOptions) {
    await withDaemon(async () => undefined);
    const res = await fetch(`${baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: messages.map((m) => ({ role: m.role, ...toOllamaMessage(m.content) })),
        stream: true,
        options: temperature != null ? { temperature } : undefined,
      }),
      signal,
    });
    if (!res.ok) {
      // A non-ok response means Ollama answered — 404 is its "model not
      // found", not a dead daemon. Telling people to start a daemon that is
      // already serving is how an unpulled model looks like a broken install.
      const body = await res.text().catch(() => '');
      let detail = '';
      try {
        detail = (JSON.parse(body) as { error?: string }).error ?? '';
      } catch {
        /* not JSON — fall through to the status */
      }
      if (detail) throw new Error(`Ollama: ${detail}`);
      throw new Error(
        `Ollama answered HTTP ${res.status} at ${baseUrl}: ` +
          (body.slice(0, 200) || res.statusText),
      );
    }

    for await (const event of parseNDJSON(res.body!)) {
      if (event.error) throw new Error(`Ollama: ${event.error}`);
      // Thinking models put the scratchpad in `message.thinking`; it belongs
      // in the transcript's thinking pane, not in the answer.
      const thinking: string = event.message?.thinking ?? '';
      if (thinking) yield { type: 'reasoning', text: thinking };
      const piece: string = event.message?.content ?? '';
      if (piece) yield { type: 'text', text: piece };
      if (event.done) break;
    }
  },

  chatWithTools: ollamaChatTools(baseUrl),
};
