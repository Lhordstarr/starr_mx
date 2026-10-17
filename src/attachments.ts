import type { ChatMessage, ContentPart, ImagePart, MessageContent } from './types.js';

/**
 * Caps on what a single turn may carry.
 *
 * Base64 inflates bytes by about a third, so a 5MB image becomes ~6.7MB of
 * JSON. Four images is roughly where a hosted provider starts rejecting the
 * request for being too large, which is a worse outcome than saying so here.
 */
export const LIMITS = {
  /** Bytes per image, before base64 inflation. */
  imageBytes: 5 * 1024 * 1024,
  /** Images per message. */
  imagesPerMessage: 4,
  /** Combined image bytes across one message. */
  totalImageBytes: 12 * 1024 * 1024,
} as const;

/**
 * Image types worth accepting. Deliberately a closed list: these bytes go
 * straight to a provider, and an allow-list is easier to reason about than
 * pattern-matching a hostile MIME type. SVG is excluded on purpose — it is
 * scriptable, and the browser would have to sanitise it to render inline.
 */
export const IMAGE_MIME = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif']);

/** Extensions the browser reports as an empty or generic type. */
const MIME_BY_EXT: Record<string, string> = {
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  webp: 'image/webp',
  gif: 'image/gif',
};

/**
 * Text files that can be folded into the message body as-is.
 *
 * All of these are already text on disk, so there is no parsing step and no
 * dependency. A PDF would need a real text extractor to be useful and is
 * deliberately absent.
 */
export const TEXT_EXT = new Set([
  'txt', 'md', 'markdown', 'csv', 'tsv', 'json', 'jsonl', 'ndjson', 'yaml', 'yml',
  'toml', 'xml', 'html', 'htm', 'log', 'ini', 'cfg', 'conf', 'env', 'properties',
  'sh', 'bash', 'zsh', 'fish', 'py', 'rb', 'pl', 'php', 'lua',
  'js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx', 'go', 'rs', 'java', 'kt', 'kts',
  'c', 'h', 'cc', 'cpp', 'hpp', 'cs', 'swift', 'scala', 'dart', 'sql',
  'dockerfile', 'makefile', 'gitignore', 'gradle', 'tf', 'tfvars',
]);

/** Resolves a usable MIME type for an upload, falling back to its extension. */
export function resolveMime(name: string, declared: string): string {
  const type = (declared ?? '').split(';')[0].trim().toLowerCase();
  if (type && type !== 'application/octet-stream') return type;
  const ext = name.split('.').pop()?.toLowerCase() ?? '';
  if (MIME_BY_EXT[ext]) return MIME_BY_EXT[ext];
  if (TEXT_EXT.has(ext)) return 'text/plain';
  return type;
}

/** True when the file is one we accept inline, as text rather than an image. */
export const isTextFile = (name: string) => TEXT_EXT.has(name.split('.').pop()?.toLowerCase() ?? '');

/** One problem found while validating an incoming attachment. */
export interface AttachmentProblem {
  reason: string;
}

/**
 * Checks a message's images against the caps above.
 *
 * Returns every problem rather than throwing, so the caller can report them all
 * at once instead of failing on the first bad file.
 */
export function checkAttachments(messages: ChatMessage[]): AttachmentProblem[] {
  const problems: AttachmentProblem[] = [];
  let images = 0;
  let totalBytes = 0;
  let largest = 0;

  for (const message of messages) {
    if (typeof message.content === 'string') continue;
    for (const part of message.content) {
      if (part.type !== 'image') continue;
      images += 1;
      // base64 length * 3/4 recovers roughly the original byte count.
      const bytes = Math.floor((part.data.length * 3) / 4);
      totalBytes += bytes;
      if (bytes > largest) largest = bytes;
    }
  }

  if (largest > LIMITS.imageBytes) {
    problems.push({
      reason: `Each image must be under ${Math.round(LIMITS.imageBytes / 1024 / 1024)}MB.`,
    });
  }
  if (images > LIMITS.imagesPerMessage) {
    problems.push({ reason: `At most ${LIMITS.imagesPerMessage} images per message.` });
  }
  if (totalBytes > LIMITS.totalImageBytes) {
    problems.push({
      reason: `Attachments total over ${Math.round(LIMITS.totalImageBytes / 1024 / 1024)}MB.`,
    });
  }
  return problems;
}

/**
 * Rough token cost of a message, for the conversation-length guard.
 *
 * Images have no exact character cost, so they get a flat allowance. The
 * constant is deliberately generous: better to over-count and refuse early
 * than to hand a provider a payload it will reject mid-stream.
 */
const IMAGE_TOKEN_ALLOWANCE = 4_000;

export function contentLength(content: MessageContent): number {
  if (typeof content === 'string') return content.length;
  let n = 0;
  for (const part of content) n += part.type === 'image' ? IMAGE_TOKEN_ALLOWANCE : part.text.length;
  return n;
}

/* ---------------- Per-provider wire formats ---------------- */

const asDataUrl = (p: ImagePart) => `data:${p.mimeType};base64,${p.data}`;

/**
 * OpenAI, OpenRouter, Groq and user-added endpoints share one content-parts
 * array, so their messages pass through with the shape unchanged.
 */
export function toOpenAIFormat(content: MessageContent): unknown {
  if (typeof content === 'string') return content;
  const parts: unknown[] = content.map((part) =>
    part.type === 'image'
      ? { type: 'image_url', image_url: { url: asDataUrl(part) } }
      : { type: 'text', text: part.text },
  );
  // A trailing image gets an empty text part after it: several gateways reject a
  // turn whose final part isn't text, and the browser legitimately sends
  // image-only turns. The empty part is harmless to providers that don't care.
  if (content.at(-1)?.type === 'image') parts.push({ type: 'text', text: '' });
  return parts;
}

/**
 * Ollama takes images as a separate base64 array on the message instead of as
 * content parts, and keeps the text a plain string.
 */
export function toOllamaMessage(content: MessageContent): { content: string; images?: string[] } {
  if (typeof content === 'string') return { content };
  const images: string[] = [];
  let text = '';
  for (const part of content) {
    if (part.type === 'image') images.push(part.data);
    else text += part.text;
  }
  return images.length ? { content: text, images } : { content: text };
}

/** Anthropic uses typed blocks with a base64 source object. */
export function toAnthropicFormat(content: MessageContent): unknown {
  if (typeof content === 'string') return content;
  return content.map((part) =>
    part.type === 'image'
      ? { type: 'image', source: { type: 'base64', media_type: part.mimeType, data: part.data } }
      : { type: 'text', text: part.text },
  );
}

/** Gemini carries images as inline_data parts. */
export function toGeminiParts(content: MessageContent): unknown[] {
  if (typeof content === 'string') return [{ text: content }];
  return content.map((part) =>
    part.type === 'image'
      ? { inline_data: { mime_type: part.mimeType, data: part.data } }
      : { text: part.text },
  );
}

/** Narrows an unknown to an image part, discarding anything malformed. */
export function isImagePart(value: unknown): value is ImagePart {
  const p = value as Partial<ImagePart> | null;
  return (
    !!p &&
    p.type === 'image' &&
    typeof p.data === 'string' &&
    p.data.length > 0 &&
    typeof p.mimeType === 'string' &&
    IMAGE_MIME.has(p.mimeType)
  );
}

/**
 * Validates an incoming content value from the browser.
 *
 * Returns null for anything malformed so a bad payload is rejected outright
 * rather than being forwarded to a provider that will 400 on it anyway.
 */
export function parseContent(value: unknown): MessageContent | null {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return null;
  const parts: ContentPart[] = [];
  for (const raw of value) {
    if (!raw || typeof raw !== 'object') return null;
    const p = raw as Partial<ContentPart>;
    if (p.type === 'text' && typeof p.text === 'string') {
      parts.push({ type: 'text', text: p.text });
    } else if (p.type === 'image') {
      if (!isImagePart(raw)) return null;
      parts.push(raw as ImagePart);
    } else {
      return null;
    }
  }
  return parts;
}
