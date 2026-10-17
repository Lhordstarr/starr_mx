import { ensureOk, parseSSE } from '../sse.js';
import { getKey, keySource } from '../store.js';
import { toOpenAIFormat } from '../attachments.js';
import { openAIChatTools } from '../agent.js';
import type { Provider, ProviderId, StreamOptions, TranscribeOptions } from './types.js';

export interface OpenAICompatConfig {
  id: ProviderId;
  label: string;
  baseUrl: string;
  /**
   * Environment variable holding the API key. Required for built-ins, which
   * read it from .env or the settings page; omitted for custom providers, which
   * supply their key through getApiKey instead.
   */
  envKey?: string;
  hint?: string;
  /** Some gateways want app-identification headers (e.g. OpenRouter). */
  extraHeaders?: Record<string, string>;
  /** Set when the provider exposes an OpenAI-style /audio/transcriptions API. */
  stt?: boolean;
  /**
   * True when the server runs on this machine. The UI badges every such
   * provider identically; getting this wrong is what made Ollama appear
   * without the marker while a discovered LM Studio carried it, implying the
   * only self-hosted option was the one the app happened to start.
   */
  local?: boolean;
  /** Narration-only hint for the UI. */
  sttHint?: string;
  /**
   * Custom providers have no env var, so they supply the key directly. When
   * present it replaces getKey(cfg.envKey).
   */
  getApiKey?: () => string | undefined;
  /**
   * Built-in providers refuse to run without a key, because that is always a
   * misconfiguration. User-added endpoints are assumed reachable, so the
   * Authorization header is simply omitted when no key is saved.
   */
  keyOptional?: boolean;
  /** Marks a provider as user-added, so the UI can offer edit/delete. */
  custom?: boolean;
}

/** Audio extensions the OpenAI-compatible transcription API accepts. */
const STT_EXTENSIONS: Record<string, string> = {
  'audio/webm': 'webm',
  'audio/ogg': 'ogg',
  'audio/mp4': 'mp4',
  'audio/mpeg': 'mp3',
  'audio/mp3': 'mp3',
  'audio/wav': 'wav',
  'audio/x-wav': 'wav',
  'audio/webm;codecs=opus': 'webm',
  'audio/webm;codecs=vorbis': 'webm',
};

/**
 * Transcription models, identified by name shape. A hardcoded id would go stale
 * the moment a provider renames or retires a model, so ids come from the live
 * /models list and are filtered to the ones that look like STT models.
 */
const STT_MODEL = /(whisper|transcribe|parakeet|wav2vec|speech-to-text|^stt-|asr)/i;
const NON_STT = /(tts|embed|diariz|realtime|voice|\/v\d)/i;

/**
 * Builds a provider for any OpenAI-compatible /chat/completions endpoint.
 * OpenAI, OpenRouter and Groq all share this wire format.
 */
export function openAICompatible(cfg: OpenAICompatConfig): Provider {
  /**
   * Model ids the provider itself reported as image-capable, filled in from
   * whatever /models returns. Empty for the providers that publish no such
   * field, which the UI reads as "unknown" rather than "no".
   */
  const visionModels = new Set<string>();

  // Read at call time, never cached: the settings page can change keys while
  // the server is running.
  const apiKey = () => (cfg.getApiKey ? cfg.getApiKey() : cfg.envKey ? getKey(cfg.envKey) : undefined);

  /** Authorization header, omitted entirely when there's no key to send. */
  const authHeader = (): Record<string, string> => {
    const key = apiKey();
    return key ? { authorization: `Bearer ${key}` } : {};
  };

  const hint = () => {
    if (cfg.hint) return cfg.hint;
    if (cfg.custom) return cfg.stt ? 'Custom provider · supports dictation' : 'Custom provider';
    const source = cfg.envKey ? keySource(cfg.envKey) : 'none';
    if (source === 'settings') return 'Key from settings page';
    if (source === 'env') return 'Key from .env';
    return `Missing ${cfg.envKey ?? 'API key'}`;
  };

  const provider: Provider = {
    info: {
      id: cfg.id,
      label: cfg.label,
      get configured() {
        // A user-added provider counts as configured by definition: they added
        // it on purpose, and a 401 (if any) surfaces from the real request.
        if (cfg.keyOptional) return true;
        return Boolean(apiKey());
      },
      local: Boolean(cfg.local),
      envKey: cfg.envKey,
      custom: cfg.custom,
      get hint() {
        return hint();
      },
      // Same wire format everywhere here, so tool calls work on all of them.
      supportsTools: true,
    },

    async listModels() {
      try {
        const res = await fetch(`${cfg.baseUrl}/models`, {
          headers: {
            ...authHeader(),
            ...cfg.extraHeaders,
          },
        });
        if (!res.ok) return [];
        const data = (await res.json()) as {
          data?: {
            id: string;
            // OpenRouter and a few gateways report what a model accepts here.
            // Absent everywhere else, which the UI reads as "unknown".
            architecture?: { input_modalities?: string[] };
          }[];
        };
        for (const m of data.data ?? []) {
          if (m.architecture?.input_modalities?.includes('image')) {
            visionModels.add(m.id);
          }
        }
        return (data.data ?? []).map((m) => m.id).sort();
      } catch {
        return [];
      }
    },

    async *stream({ model, messages, signal, temperature }: StreamOptions) {
      const key = apiKey();
      if (!key && !cfg.keyOptional) {
        throw new Error(
          `${cfg.label}: no API key. Add one on the settings page or set ${cfg.envKey} in .env.`,
        );
      }
      const res = await fetch(`${cfg.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          ...authHeader(),
          'content-type': 'application/json',
          ...cfg.extraHeaders,
        },
        // Messages keep their content-parts shape: this wire format already
        // accepts text interleaved with images.
        body: JSON.stringify({
          model,
          messages: messages.map((m) => ({ role: m.role, content: toOpenAIFormat(m.content) })),
          stream: true,
          temperature,
        }),
        signal,
      });
      await ensureOk(res, cfg.label);

      for await (const payload of parseSSE(res.body!)) {
        const delta = JSON.parse(payload)?.choices?.[0]?.delta?.content;
        if (delta) yield delta as string;
      }
    },

    /**
     * Models the provider advertised as image-capable. Returning an empty list
     * is not the same as "none can": it means this provider publishes no
     * capability information, and the UI treats that as unknown.
     */
    listVisionModels: async () => Array.from(visionModels).sort(),

    chatWithTools: openAIChatTools(
      cfg.baseUrl,
      () => ({ ...authHeader(), ...cfg.extraHeaders }),
      cfg.label,
    ),
  };

  // Only providers flagged `stt` get these; the rest simply omit them, which
  // the UI reads as "no speech-to-text here".
  if (cfg.stt) {
    provider.listSttModels = async () => {
      const all = await provider.listModels();
      return all.filter((id) => STT_MODEL.test(id) && !NON_STT.test(id));
    };

    provider.transcribe = async ({ model, audio, mimeType, signal }: TranscribeOptions) => {
      const key = apiKey();
      if (!key && !cfg.keyOptional) {
        throw new Error(
          `${cfg.label}: no API key. Add one on the settings page or set ${cfg.envKey} in .env.`,
        );
      }

      // The provider sniffs the container from the filename extension, so the
      // browser's recording type has to be mapped to one it recognises.
      const base = mimeType.split(';')[0].trim().toLowerCase();
      const ext = STT_EXTENSIONS[base] ?? STT_EXTENSIONS[mimeType.trim().toLowerCase()];
      if (!ext) {
        throw new Error(`${cfg.label}: unsupported audio type "${mimeType}"`);
      }

      // Built by hand rather than with FormData so the Content-Type boundary is
      // set explicitly; providers are strict about the multipart envelope.
      const boundary = `----llmchat${crypto.randomUUID().replace(/-/g, '')}`;
      const head = Buffer.from(
        `--${boundary}\r\n` +
          `Content-Disposition: form-data; name="file"; filename="recording.${ext}"\r\n` +
          `Content-Type: ${base}\r\n\r\n`,
      );
      const tail = Buffer.from(
        `\r\n--${boundary}\r\n` +
          `Content-Disposition: form-data; name="model"\r\n\r\n${model}\r\n` +
          `--${boundary}--\r\n`,
      );
      const body = Buffer.concat([head, audio, tail]);

      const res = await fetch(`${cfg.baseUrl}/audio/transcriptions`, {
        method: 'POST',
        headers: {
          ...authHeader(),
          'content-type': `multipart/form-data; boundary=${boundary}`,
          ...cfg.extraHeaders,
        },
        body,
        signal,
      });
      await ensureOk(res, cfg.label);

      const data = (await res.json()) as { text?: string };
      return (data.text ?? '').trim();
    };
  }

  return provider;
}