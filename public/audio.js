/**
 * Turning a recording into something the chosen runner can open.
 *
 * This exists for exactly one runner, and the reason is narrow. `whisper-cli`
 * reads wav, flac, mp3 and ogg, and refuses everything else — including both
 * formats a browser actually produces: MediaRecorder gives webm/opus on
 * Chromium and mp4/aac on Safari. So for a whisper.cpp model the audio has to
 * be decoded and re-encoded before it is uploaded, or it arrives as a container
 * the CLI rejects with a message about a file header.
 *
 * It is done here rather than on the server for three reasons: the browser
 * already has a decoder for every format it can record, no extra binary (ffmpeg
 * is a large install and needs root on many distributions), and whisper.cpp is
 * the only runtime that would have to wait for another one.
 *
 * The two Python runners are left alone. They bring their own audio libraries
 * and take the original bytes, and re-encoding for them would be a slower,
 * lossier round trip for nothing.
 */

/** MIME types whisper-cli opens without help. */
const READABLE = new Set([
  'audio/wav',
  'audio/x-wav',
  'audio/wave',
  'audio/vnd.wave',
  'audio/mpeg',
  'audio/mp3',
  'audio/flac',
  'audio/x-flac',
  'audio/ogg',
  'application/ogg',
]);

/** What whisper.cpp wants. Its resampler is fixed at this rate. */
const TARGET_RATE = 16_000;

function baseType(blob) {
  return String(blob?.type ?? '').split(';')[0].trim().toLowerCase();
}

/**
 * The audio to upload, in a form this model's runner can read.
 *
 * `driver` comes from the server's runnable payload, so the decision is made
 * from the runtime the model actually lists rather than from a guess here.
 * Anything unknown passes through untouched.
 */
export async function audioForUpload(blob, driver) {
  if (driver !== 'whisper-cpp') return blob;
  if (READABLE.has(baseType(blob))) return blob;
  return wav16k(blob);
}

/** True when this blob would have to be converted for whisper.cpp. */
export function needsDecoding(blob, driver) {
  return driver === 'whisper-cpp' && !READABLE.has(baseType(blob));
}

/**
 * Decodes any container the browser understands into 16 kHz mono PCM WAV.
 *
 * The context is asked for 16 kHz directly so the browser's own resampler does
 * the work — it band-limits properly, and a hand-rolled linear interpolation
 * aliases high frequencies down into the speech band, which is exactly the
 * error an ASR model cannot recover from. The fallback path exists only for a
 * browser that ignores the requested rate.
 */
async function wav16k(blob) {
  const Ctor = window.AudioContext ?? window.webkitAudioContext;
  if (!Ctor) throw new Error('This browser cannot decode the recording; save the audio as a .wav file instead.');
  const ctx = new Ctor({ sampleRate: TARGET_RATE });
  try {
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
    // `decoded` is already at the context's rate, which was asked to be
    // 16 kHz — so the target is just the target. It used to be a ternary that
    // passed the AudioBuffer itself on that path, making the frame count NaN
    // and producing a valid-looking WAV header with no samples in it, which is
    // the worst of both: it decodes to silence and reports success.
    const mono = toMono(decoded, TARGET_RATE);
    return new Blob([wavBytes(mono, TARGET_RATE)], { type: 'audio/wav' });
  } finally {
    // Left open, an AudioContext holds a realtime thread and, on some browsers,
    // will not let a second one be created. Dictation creates one per clip.
    await ctx.close().catch(() => undefined);
  }
}

/** Flattens every channel into one, at `rate`. */
function toMono(buffer, rate) {
  const seconds = Number(buffer?.duration);
  // A buffer with no stated duration still has samples; trusting the sample
  // count is what stops a silent or oddly-reported buffer becoming an empty
  // WAV that looks like a successful transcription of nothing.
  const frames = Number.isFinite(seconds) && seconds > 0
    ? Math.max(1, Math.round(seconds * rate))
    : Math.max(1, buffer?.length ?? 0);
  const out = new Float32Array(frames);
  const channels = buffer.numberOfChannels || 1;
  for (let ch = 0; ch < channels; ch += 1) {
    const data = buffer.getChannelData(ch);
    // Position in the source, walked once per output frame. `srcStep` is the
    // ratio between the two rates, so a context that already gave us 16 kHz
    // reads straight through at 1.0.
    const srcStep = buffer.sampleRate / rate;
    let src = 0;
    for (let i = 0; i < frames; i += 1) {
      const at = src;
      const i0 = Math.floor(at);
      const i1 = Math.min(buffer.length - 1, i0 + 1);
      const frac = at - i0;
      const a = data[i0] ?? 0;
      const b = data[i1] ?? a;
      out[i] = a + (b - a) * frac;
      src += srcStep;
    }
  }
  return out;
}

/** A 44-byte canonical RIFF/WAVE header followed by 16-bit mono samples. */
function wavBytes(samples, rate) {
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  const dataBytes = samples.length * 2;
  ascii(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // channels: mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, 'data');
  view.setUint32(40, dataBytes, true);
  let at = 44;
  for (let i = 0; i < samples.length; i += 1) {
    // Clamped rather than wrapped: one sample outside [-1, 1] would otherwise
    // come out as the loudest possible noise at the opposite polarity.
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(at, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    at += 2;
  }
  return bytes;
}