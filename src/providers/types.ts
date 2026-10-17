import type { ChatMessage } from '../types.js';
import type { ToolDefinition } from '../tools.js';

/** Ids of the providers compiled into the app. */
export type BuiltinProviderId =
  | 'ollama'
  | 'llamacpp'
  | 'openai'
  | 'openrouter'
  | 'anthropic'
  | 'gemini'
  | 'groq';

/**
 * A provider id. Custom providers get generated slugs, so this is a string —
 * the `& {}` half keeps literal autocomplete for the built-ins while still
 * accepting an arbitrary id.
 */
export type ProviderId = BuiltinProviderId | (string & {});

export interface ProviderInfo {
  id: ProviderId;
  label: string;
  /** Whether this provider can be used as configured. */
  configured: boolean;
  /** Local providers need no key and are always available. */
  local: boolean;
  /**
   * Environment variable / settings field holding this provider's key.
   *
   * There is deliberately no default model: the set a key can reach comes from
   * the provider's own /models endpoint, and hardcoded ids go stale or 404.
   */
  envKey?: string;
  /** True for user-added providers, which the settings page can edit/delete. */
  custom?: boolean;
  /** Shown in the UI to explain configuration state. */
  hint?: string;
  /**
   * Whether this provider can run the agent loop (tool calls). The Tools page
   * reads this to say where tools actually work; the chat falls back to a
   * plain completion anywhere else.
   */
  supportsTools?: boolean;
}

export interface StreamOptions {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  signal: AbortSignal;
}

export interface TranscribeOptions {
  model: string;
  /** Encoded audio bytes, forwarded to the provider untouched. */
  audio: Buffer;
  /** MIME type of `audio`, e.g. `audio/webm`. */
  mimeType: string;
  signal: AbortSignal;
}

/**
 * Streams assistant text for a request, yielding raw deltas as they arrive.
 * The server turns each delta into an SSE event for the browser.
 */
export type StreamFn = (opts: StreamOptions) => AsyncGenerator<string, void, unknown>;

/**
 * One turn of the agent loop's working history. Tool turns are neutral here;
 * each provider adapter translates them to its own wire shape.
 */
export interface AgentTurn {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: ChatMessage['content'] | string;
  calls?: { id: string; name: string; args: unknown }[];
  toolCallId?: string;
  output?: string;
}

/** A single model-requested tool call, args already parsed to an object. */
export interface ToolCallRequest {
  id: string;
  name: string;
  args: unknown;
}

/** What a tool-capable round yields: live text, then any calls once complete. */
export type ChatToolsEvent =
  | { type: 'text'; text: string }
  | { type: 'tool_calls'; calls: ToolCallRequest[] };

export interface ChatToolsOptions {
  model: string;
  turns: AgentTurn[];
  tools: ToolDefinition[];
  temperature?: number;
  signal: AbortSignal;
}

/**
 * One provider round inside the agent loop. Present only on providers whose
 * wire format carries tool calls (OpenAI-compatible endpoints, Ollama);
 * absent everywhere else, which the chat reads as "plain completion only".
 */
export type ChatToolsFn = (opts: ChatToolsOptions) => AsyncGenerator<ChatToolsEvent, void, unknown>;

/** Fetches the provider's available models. Returns [] when unavailable. */
export type ListModelsFn = () => Promise<string[]>;

/**
 * Models the provider reported as accepting images.
 *
 * An empty result means the provider published no capability information, not
 * that none of its models can see — the UI distinguishes "known text-only"
 * from "unknown" rather than guessing from the model's name.
 */
export type ListVisionModelsFn = () => Promise<string[]>;

/**
 * Speech-to-text. Optional: only providers with a transcription API expose it.
 */
export type TranscribeFn = (opts: TranscribeOptions) => Promise<string>;

/**
 * Gets a local backend ready to answer a specific model, narrating what it is
 * doing.
 *
 * Only llama.cpp needs this today: it serves one model at a time, so a request
 * for a different one means starting a server with it loaded — seconds of work
 * that must not look like a hang. Providers that are always ready simply omit
 * it, and the chat calls it when it is there.
 */
export type PrepareFn = (model: string, onProgress: (line: string) => void) => Promise<void>;

/** Narration-only hint shown in the UI. */
export const NO_STT = 'This provider has no speech-to-text API';

export interface Provider {
  info: ProviderInfo;
  listModels: ListModelsFn;
  stream: StreamFn;
  /**
   * Present only when the provider can run tool calls. See ChatToolsFn for
   * the round protocol the agent loop speaks.
   */
  chatWithTools?: ChatToolsFn;
  /**
   * Present only when the provider can transcribe audio. The listed models are
   * the subset of `listModels()` that are transcription models.
   */
  transcribe?: TranscribeFn;
  /** Transcription-capable models, discovered from the provider's own list. */
  listSttModels?: ListModelsFn;
  /**
   * Present when a request may need something started first (llama.cpp has to
   * load the chosen model). Called before the stream begins.
   */
  prepare?: PrepareFn;
  /**
   * Models known to accept image attachments. Optional, and may legitimately
   * return [] to mean "this provider tells us nothing".
   */
  listVisionModels?: ListVisionModelsFn;
}