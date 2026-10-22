import type { ChatMessage, ContentPart, ImagePart, MessageContent, VideoPart } from './types.js';

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
  /** Bytes per video, before base64 inflation. */
  videoBytes: 12 * 1024 * 1024,
  /** Videos per message. Four frames per video is already a lot of vision work. */
  videosPerMessage: 1,
  /** Combined image + video bytes across one message. */
  totalMediaBytes: 18 * 1024 * 1024,
} as const;

/**
 * Image types worth accepting. Deliberately a closed list: these bytes go
 * straight to a provider, and an allow-list is easier to reason about than
 * pattern-matching a hostile MIME type. SVG is excluded on purpose — it is
 * scriptable, and the browser would have to sanitise it to render inline.
 */
export const IMAGE_MIME = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif']);

/**
 * Video containers the browser can pick and ffmpeg can demux here.
 *
 * Deliberately narrow: these are the formats the bundled frame sampler has
 * been exercised on. An arbitrary container would fail in ffmpeg with a
 * message no one can act on, long after the attachment looked accepted.
 */
export const VIDEO_MIME = new Set([
  'video/mp4',
  'video/webm',
  'video/quicktime',
  'video/x-matroska',
]);

/** Extensions the browser reports as an empty or generic type. */
const MIME_BY_EXT: Record<string, string> = {
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  webp: 'image/webp',
  gif: 'image/gif',
  mp4: 'video/mp4',
  m4v: 'video/mp4',
  webm: 'video/webm',
  mov: 'video/quicktime',
  mkv: 'video/x-matroska',
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
  let videos = 0;
  let imageBytes = 0;
  let totalBytes = 0;
  let largestImage = 0;
  let largestVideo = 0;

  for (const message of messages) {
    if (typeof message.content === 'string') continue;
    for (const part of message.content) {
      if (part.type !== 'image' && part.type !== 'video') continue;
      // base64 length * 3/4 recovers roughly the original byte count.
      const bytes = Math.floor((part.data.length * 3) / 4);
      totalBytes += bytes;
      if (part.type === 'image') {
        images += 1;
        imageBytes += bytes;
        if (bytes > largestImage) largestImage = bytes;
      } else {
        videos += 1;
        if (bytes > largestVideo) largestVideo = bytes;
      }
    }
  }

  if (largestImage > LIMITS.imageBytes) {
    problems.push({
      reason: `Each image must be under ${Math.round(LIMITS.imageBytes / 1024 / 1024)}MB.`,
    });
  }
  if (largestVideo > LIMITS.videoBytes) {
    problems.push({
      reason: `Each video must be under ${Math.round(LIMITS.videoBytes / 1024 / 1024)}MB.`,
    });
  }
  if (images > LIMITS.imagesPerMessage) {
    problems.push({ reason: `At most ${LIMITS.imagesPerMessage} images per message.` });
  }
  if (videos > LIMITS.videosPerMessage) {
    problems.push({ reason: `At most ${LIMITS.videosPerMessage} video per message.` });
  }
  if (imageBytes > LIMITS.totalImageBytes) {
    problems.push({
      reason: `Images total over ${Math.round(LIMITS.totalImageBytes / 1024 / 1024)}MB.`,
    });
  }
  // The image total is the more specific message when both caps are crossed.
  // Saying both in one response reads as two unrelated failures.
  if (totalBytes > LIMITS.totalMediaBytes && imageBytes <= LIMITS.totalImageBytes) {
    problems.push({
      reason: `Attachments total over ${Math.round(LIMITS.totalMediaBytes / 1024 / 1024)}MB.`,
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
/** Four sampled frames, plus the block labels the preprocessor adds. */
const VIDEO_TOKEN_ALLOWANCE = 16_000;

export function contentLength(content: MessageContent): number {
  if (typeof content === 'string') return content.length;
  let n = 0;
  for (const part of content) {
    if (part.type === 'image') n += IMAGE_TOKEN_ALLOWANCE;
    else if (part.type === 'video') n += VIDEO_TOKEN_ALLOWANCE;
    else n += part.text.length;
  }
  return n;
}

/* ---------------- Per-provider wire formats ---------------- */

const asDataUrl = (p: ImagePart) => `data:${p.mimeType};base64,${p.data}`;

/**
 * No provider wire format in this app accepts a raw video part.
 *
 * Reaching this means the vision preprocessing did not run (no model assigned,
 * or a caller bypassed the chat route). Throwing here is deliberate: silently
 * dropping the video would let the model answer a question about footage it
 * never saw.
 */
function videoNeedsVisionModel(): never {
  throw new Error(
    'Video attachments need an assigned vision model. Set one under Settings → What each job uses.',
  );
}

/**
 * OpenAI, OpenRouter, Groq and user-added endpoints share one content-parts
 * array, so their messages pass through with the shape unchanged.
 */
export function toOpenAIFormat(content: MessageContent): unknown {
  if (typeof content === 'string') return content;
  const parts: unknown[] = content.map((part) => {
    if (part.type === 'image') {
      return { type: 'image_url', image_url: { url: asDataUrl(part) } };
    }
    if (part.type === 'video') videoNeedsVisionModel();
    return { type: 'text', text: part.text };
  });
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
    else if (part.type === 'video') videoNeedsVisionModel();
    else text += part.text;
  }
  return images.length ? { content: text, images } : { content: text };
}

/** Anthropic uses typed blocks with a base64 source object. */
export function toAnthropicFormat(content: MessageContent): unknown {
  if (typeof content === 'string') return content;
  return content.map((part) => {
    if (part.type === 'image') {
      return { type: 'image', source: { type: 'base64', media_type: part.mimeType, data: part.data } };
    }
    if (part.type === 'video') videoNeedsVisionModel();
    return { type: 'text', text: part.text };
  });
}

/** Gemini carries images as inline_data parts. */
export function toGeminiParts(content: MessageContent): unknown[] {
  if (typeof content === 'string') return [{ text: content }];
  return content.map((part) => {
    if (part.type === 'image') {
      return { inline_data: { mime_type: part.mimeType, data: part.data } };
    }
    if (part.type === 'video') videoNeedsVisionModel();
    return { text: part.text };
  });
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

/** Narrows an unknown to a video part, discarding anything malformed. */
export function isVideoPart(value: unknown): value is VideoPart {
  const p = value as Partial<VideoPart> | null;
  return (
    !!p &&
    p.type === 'video' &&
    typeof p.data === 'string' &&
    p.data.length > 0 &&
    typeof p.mimeType === 'string' &&
    VIDEO_MIME.has(p.mimeType)
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
    } else if (p.type === 'video') {
      if (!isVideoPart(raw)) return null;
      parts.push(raw as VideoPart);
    } else {
      return null;
    }
  }
  return parts;
}
