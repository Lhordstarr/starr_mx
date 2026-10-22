/**
 * Turns media into text for the model that is actually answering.
 *
 * A chat model may be text-only, and even a multimodal chat model should not
 * be handed four video frames as raw bytes when a small local model can do the
 * seeing once. The assignment in Settings decides which model performs that
 * step. This module is the one place the replacement happens, so every
 * provider receives a plain text turn and no adapter has to grow video support.
 *
 * Descriptions are cached by image bytes and model. A follow-up question about
 * the image the user sent two turns ago therefore reuses the earlier
 * description instead of running vision again — which matters most on a CPU,
 * where every extra vision call is seconds of work.
 */
import { createHash } from 'node:crypto';

import { extractVideoFrames, type VideoFrame } from './media.js';
import type { ChatMessage, ImagePart, VideoPart } from './types.js';
import type { Provider } from './providers/types.js';

export interface VisionTarget {
  providerId: string;
  model: string;
}

const CACHE_LIMIT = 64;
const cache = new Map<string, string>();

/** Test seam; production never needs to empty the cache. */
export function clearVisionCache(): void {
  cache.clear();
}

/**
 * Parses an assignment id such as `ollama:moondream:latest`.
 *
 * Only the first colon separates provider from model, because Ollama tags
 * themselves contain one. `local:` is reserved for a future bundled vision
 * runner and deliberately cannot be mistaken for an Ollama model.
 */
export function parseVisionAssignment(value: string | null | undefined): VisionTarget | null {
  if (!value) return null;
  const split = value.indexOf(':');
  if (split <= 0 || split === value.length - 1) return null;
  const providerId = value.slice(0, split);
  const model = value.slice(split + 1);
  if (providerId === 'local') return null;
  return { providerId, model };
}

export interface VisionPreprocessOptions {
  /** `ollama:<tag>` from Settings, or null when no model is assigned. */
  assignment: string | null;
  resolveProvider: (id: string) => Provider | undefined;
  signal: AbortSignal;
  /** Overridable so a unit test can exercise the replacement without ffmpeg. */
  extractFrames?: (video: VideoPart) => Promise<VideoFrame[]>;
}

export interface VisionPreprocessResult {
  messages: ChatMessage[];
  /** Media pieces that were actually described, for logs and tests. */
  described: number;
  cacheHits: number;
}

function cacheKey(model: string, part: ImagePart): string {
  const hash = createHash('sha256').update(part.data).digest('hex');
  return `${model}\0${part.mimeType}\0${hash}`;
}

function remember(key: string, description: string): void {
  if (cache.size >= CACHE_LIMIT) {
    const oldest = cache.keys().next().value;
    if (oldest) cache.delete(oldest);
  }
  cache.set(key, description);
}

function hasVideo(messages: ChatMessage[]): boolean {
  return messages.some(
    (m) => Array.isArray(m.content) && m.content.some((p) => p.type === 'video'),
  );
}

/**
 * Replaces every image and video part with text.
 *
 * With no vision assignment, images are left alone: a native multimodal chat
 * model is still allowed to handle them directly. Video is refused in that
 * state because none of the provider wire formats this app speaks accepts it.
 */
export async function preprocessVision(
  messages: ChatMessage[],
  opts: VisionPreprocessOptions,
): Promise<VisionPreprocessResult> {
  const target = parseVisionAssignment(opts.assignment);
  if (!target) {
    if (hasVideo(messages)) {
      throw new Error(
        'Video attachments need an assigned vision model. Set one under Settings → What each job uses.',
      );
    }
    if (opts.assignment) {
      throw new Error(
        `The assigned vision model "${opts.assignment}" cannot be used. Choose one under Settings → What each job uses.`,
      );
    }
    return { messages, described: 0, cacheHits: 0 };
  }

  const provider = opts.resolveProvider(target.providerId);
  if (!provider?.describeImages) {
    throw new Error(
      `The assigned vision model "${opts.assignment}" has no runner in this app. Choose another under Settings → What each job uses.`,
    );
  }

  const framesFor = opts.extractFrames ?? extractVideoFrames;
  let described = 0;
  let cacheHits = 0;

  const describe = async (image: ImagePart, prompt: string): Promise<string> => {
    const key = cacheKey(target.model, image);
    const cached = cache.get(key);
    if (cached) {
      cacheHits += 1;
      return cached;
    }
    const text = (
      await provider.describeImages!({
        model: target.model,
        images: [image],
        prompt,
        signal: opts.signal,
      })
    ).trim();
    if (!text) throw new Error(`Vision model ${target.model} returned an empty description.`);
    remember(key, text);
    described += 1;
    return text;
  };

  const out: ChatMessage[] = [];
  for (const message of messages) {
    if (message.role !== 'user' || !Array.isArray(message.content)) {
      out.push(message);
      continue;
    }

    const media = message.content.filter(
      (p): p is ImagePart | VideoPart => p.type === 'image' || p.type === 'video',
    );
    if (!media.length) {
      out.push(message);
      continue;
    }

    const text = message.content
      .filter((p) => p.type === 'text')
      .map((p) => p.text)
      .join('\n')
      .trim();
    const blocks: string[] = [];
    let imageIndex = 0;
    let videoIndex = 0;

    for (const part of media) {
      if (part.type === 'image') {
        imageIndex += 1;
        const description = await describe(part, text);
        blocks.push(
          `[Image ${imageIndex} — described by ${opts.assignment}]\n${description}`,
        );
        continue;
      }

      videoIndex += 1;
      const frames = await framesFor(part);
      if (!frames.length) {
        throw new Error('No video frames could be read from the attachment.');
      }
      const frameBlocks: string[] = [];
      for (let i = 0; i < frames.length; i += 1) {
        const frame = frames[i];
        const where =
          typeof frame.at === 'number' ? `, frame at ${frame.at.toFixed(1)}s` : '';
        const description = await describe(
          frame.image,
          `${text}\n\nThis is frame ${i + 1} of ${frames.length} from the attached video.`,
        );
        frameBlocks.push(`Frame ${i + 1}${where}:\n${description}`);
      }
      blocks.push(
        `[Video ${videoIndex} — sampled into ${frames.length} frames and described by ${opts.assignment}]\n${frameBlocks.join('\n\n')}`,
      );
    }

    out.push({
      ...message,
      content: [text, ...blocks].filter(Boolean).join('\n\n'),
    });
  }

  return { messages: out, described, cacheHits };
}
