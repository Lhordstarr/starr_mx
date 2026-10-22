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
import type { DescribeImagesOptions, Provider, StreamOptions } from './types.js';

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

  /**
   * Describes one image for a text-only chat model.
   *
   * This is deliberately a separate model call, not a second `images` entry on
   * the chat request: the assigned vision model may be a different size or even
   * a different family, and the text it produces is what the active model
   * receives. Ollama's non-streaming response is small enough to collect here;
   * the image bytes never leave the machine.
   */
  async describeImages({ model, images, prompt, signal }: DescribeImagesOptions) {
    await withDaemon(async () => undefined);

    const askOnce = async (content: string): Promise<string> => {
      const res = await fetch(`${baseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: [
            {
              role: 'user',
              content,
              images: images.map((i) => i.data),
            },
          ],
          stream: false,
          options: { temperature: 0.1, num_predict: 768 },
          keep_alive: '5m',
        }),
        signal,
      });
      if (!res.ok) {
        const body = await res.text().catch(() => '');
        let detail = '';
        try {
          detail = (JSON.parse(body) as { error?: string }).error ?? '';
        } catch {
          /* not JSON */
        }
        throw new Error(
          `Vision model ${model} failed (HTTP ${res.status})` +
            (detail ? `: ${detail}` : '.'),
        );
      }
      const data = (await res.json()) as {
        message?: { content?: string };
        error?: string;
      };
      if (data.error) throw new Error(`Vision model ${model}: ${data.error}`);
      return data.message?.content?.trim() ?? '';
    };

    // Keep the instruction short. Moondream 2 answers this reliably, while a
    // long set of rules ending in "do not answer the question yourself" made
    // it return an empty message — the failure looked like the attachment was
    // broken. The context is useful but not required, so an empty result gets
    // one retry with the bare description instruction.
    const describe =
      'Describe this image in detail for someone who cannot see it. Include the ' +
      'layout, objects, people, colours, and transcribe all visible text exactly. ' +
      'Be factual.';
    const context = prompt.trim()
      ? `\n\nThe conversation is about: ${prompt.trim().slice(0, 2_000)}`
      : '';
    let text = await askOnce(`${describe}${context}`);
    if (!text) text = await askOnce('Describe this image in detail.');
    if (!text) throw new Error(`Vision model ${model} returned an empty description.`);
    return text;
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
      // A long generation is still use. Without this, a reply that runs past
      // the daemon's idle window gets its process stopped mid-stream.
      noteOllamaUsed();
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
