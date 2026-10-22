/**
 * ffmpeg-backed helpers for media that no chat API accepts directly.
 *
 * Video is not passed to a provider. It is decoded into a handful of still
 * frames and described by the assigned vision model, which keeps the request
 * shape identical for Ollama, OpenAI, Anthropic and Gemini. Everything runs
 * through spawn with argument arrays and a private temp directory — no user
 * string ever becomes part of a shell command.
 */
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';

import type { ImagePart, VideoPart } from './types.js';

export interface VideoFrame {
  image: ImagePart;
  /** Approximate source timestamp, in seconds. */
  at: number;
}

const EXT_BY_MIME: Record<string, string> = {
  'video/mp4': 'mp4',
  'video/webm': 'webm',
  'video/quicktime': 'mov',
  'video/x-matroska': 'mkv',
};

const FRAME_COUNT = 4;

function run(
  command: string,
  args: string[],
  timeoutMs: number,
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs);
    child.stdout.on('data', (b) => (stdout += String(b)));
    child.stderr.on('data', (b) => (stderr += String(b)));
    child.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({ code: code ?? -1, stdout, stderr });
    });
  });
}

/** Turns ffmpeg's stderr into one line that fits in a chat error. */
function detail(stderr: string): string {
  return stderr
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(-2)
    .join(' ');
}

async function probeDuration(file: string): Promise<number> {
  try {
    const { code, stdout } = await run(
      'ffprobe',
      ['-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', file],
      20_000,
    );
    if (code !== 0) return 0;
    const seconds = Number(stdout.trim());
    return Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  } catch {
    return 0;
  }
}

/**
 * Samples up to four frames, spread across the clip.
 *
 * ffmpeg's input seek is used because it is fast on a long file; the exact
 * frame is less important than covering the beginning, middle and end. A
 * 640px box keeps the vision work bounded — a 4K frame costs the same tokens
 * as a 640px one but takes longer to encode and decode.
 */
export async function extractVideoFrames(video: VideoPart): Promise<VideoFrame[]> {
  const ext = EXT_BY_MIME[video.mimeType] ?? 'mp4';
  const dir = await mkdtemp(path.join(tmpdir(), 'starr-video-'));
  const input = path.join(dir, `input.${ext}`);
  const frames: VideoFrame[] = [];

  let lastError = '';
  try {
    await writeFile(input, Buffer.from(video.data, 'base64'));
    const duration = await probeDuration(input);
    // No duration means ffprobe could not read the container. Sampling the
    // same zero timestamp four times costs four vision calls and adds nothing.
    const count = duration <= 0 ? 1 : duration < 1 ? 2 : FRAME_COUNT;
    const span = duration > 0 ? duration : 0;

    for (let i = 0; i < count; i += 1) {
      const at = span ? Math.min(span - 0.05, (span * (i + 0.5)) / count) : 0;
      const out = path.join(dir, `frame-${i}.jpg`);
      let result;
      try {
        result = await run(
          'ffmpeg',
          [
            '-hide_banner',
            '-loglevel',
            'error',
            '-ss',
            at.toFixed(3),
            '-i',
            input,
            '-frames:v',
            '1',
            '-an',
            '-vf',
            'scale=640:640:force_original_aspect_ratio=decrease',
            '-q:v',
            '3',
            '-y',
            out,
          ],
          60_000,
        );
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
          throw new Error(
            'Video attachments need ffmpeg. Install it with your package manager, then send the video again.',
          );
        }
        throw err;
      }
      if (result.code !== 0) {
        // A seek past the last keyframe can fail on a very short clip; skip it
        // rather than losing the frames that did decode.
        lastError = detail(result.stderr);
        continue;
      }
      try {
        const info = await stat(out);
        if (info.size === 0) continue;
        frames.push({
          at,
          image: {
            type: 'image',
            mimeType: 'image/jpeg',
            data: (await readFile(out)).toString('base64'),
          },
        });
      } catch {
        /* this timestamp produced no frame */
      }
    }

    if (!frames.length) {
      throw new Error(
        `Could not read any frames from the video${lastError ? `: ${lastError}` : '.'}`,
      );
    }
    return frames;
  } finally {
    await rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}
