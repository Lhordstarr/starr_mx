/**
 * A piece of a user turn's body. Mirrors the OpenAI content-parts shape so the
 * server can pass it through to OpenAI-compatible endpoints untouched and only
 * translate for the providers that differ.
 */
export interface TextPart {
  type: 'text';
  text: string;
}

/**
 * An image attached to a user turn.
 *
 * Bytes travel as base64 without a `data:` prefix. Providers disagree about
 * which they want: OpenAI-compatible endpoints expect the whole thing in
 * `image_url.url`, while Anthropic and Gemini want the prefix stripped off and
 * the pieces in separate fields. See providers/attachments.ts.
 */
export interface ImagePart {
  type: 'image';
  /** Base64-encoded image bytes, no data: prefix. */
  data: string;
  mimeType: string;
}

/**
 * A short video attached to a user turn.
 *
 * No chat API this app speaks accepts a video part directly, and Ollama's
 * `images` array is images only. Video is therefore media for the *assigned
 * vision model*: the server samples frames with ffmpeg and describes them, so
 * the active chat model receives text. Keeping the part typed lets the server
 * reject video honestly when no vision model is assigned instead of silently
 * dropping it.
 */
export interface VideoPart {
  type: 'video';
  /** Base64-encoded video bytes, no data: prefix. */
  data: string;
  mimeType: string;
}

export type ContentPart = TextPart | ImagePart | VideoPart;

/**
 * What a message body may carry. Most turns are a plain string; turns with
 * media need the structured form.
 */
export type MessageContent = string | ContentPart[];

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: MessageContent;
  /** A thinking model's scratchpad, kept beside the answer for display and
   *  storage. Never sent back to a provider — the agent loop builds its turns
   *  from role and content only. */
  reasoning?: string;
}

export interface ChatRequestBody {
  provider: string;
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  /** Ask for the agent loop (tool calls). The server may still answer plainly. */
  tools?: boolean;
}
