/**
 * How much VRAM a GGUF model wants, read from its own header.
 *
 * The resource bar says what the GPU is doing; this answers the question that
 * comes before it: will this model fit at all? A file's size is the floor, but
 * llama.cpp also allocates the KV cache (which grows with context) and some
 * scratch space, and those are exactly what turn a "4 GB model" into "does not
 * fit on a 4 GB card". The numbers here are an estimate, not a promise — the
 * build, the backend and the quantisation of the cache all move the line — so
 * the UI states it as one.
 *
 * Only the small metadata prefix of the file is read. The header walk stops as
 * soon as the architecture hyperparameters are in hand (they come before the
 * tokenizer tables in every writer that produces these files), with a hard cap
 * so a corrupt header cannot make the server read a whole checkpoint.
 */
import { open, stat, type FileHandle } from 'node:fs/promises';
import os from 'node:os';

/** Scalar/array type ids from the GGUF specification (v3). */
const enum GgufType {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
}

/** Fixed byte widths for scalar types; 0 means variable (string/array). */
const WIDTH: Record<number, number> = {
  [GgufType.Uint8]: 1,
  [GgufType.Int8]: 1,
  [GgufType.Uint16]: 2,
  [GgufType.Int16]: 2,
  [GgufType.Uint32]: 4,
  [GgufType.Int32]: 4,
  [GgufType.Float32]: 4,
  [GgufType.Bool]: 1,
  [GgufType.Uint64]: 8,
  [GgufType.Int64]: 8,
  [GgufType.Float64]: 8,
};

/** Header fields the estimate uses. */
export interface GgufMeta {
  architecture: string;
  /** Transformer layers. */
  blockCount: number;
  /** Hidden size (`n_embd`). */
  embeddingLength: number;
  /** Query heads (`n_head`). */
  headCount: number;
  /** Key/value heads (`n_head_kv`) — smaller than `headCount` on GQA models. */
  headCountKV: number;
  /** The model's trained context limit. */
  contextLength: number;
}

export interface VramEstimate {
  /** Bytes of weight files on disk. */
  weightsBytes: number;
  /** KV cache at `ctx`, f16 (llama.cpp's default cache type). */
  kvBytes: number;
  /** Runtime scratch: a small fraction of the weights plus a fixed floor. */
  overheadBytes: number;
  totalBytes: number;
  /** Context length the estimate was computed for. */
  ctx: number;
  /** Present only when the GGUF header could be parsed. */
  meta: GgufMeta | null;
}

const CHUNK = 1 << 20;
/** A header larger than this is not worth trusting (or waiting for). */
const MAX_HEADER_BYTES = 32 * 1024 * 1024;
/** KV cache element size: llama.cpp keeps f16 K and V unless told otherwise. */
const KV_BYTES_PER_ELEMENT = 2;
/** Fixed runtime allocation on top of the weights and cache. */
const OVERHEAD_FLOOR_BYTES = 150 * 1024 * 1024;
/** Fraction of the weights added for the compute buffers and allocator slack. */
const OVERHEAD_FRACTION = 0.02;

class HeaderReader {
  private buf: Buffer = Buffer.alloc(0);
  private off = 0;
  private bufBase = 0;
  private readSoFar = 0;

  constructor(
    private readonly fh: FileHandle,
    private readonly size: number,
  ) {}

  private async more(): Promise<boolean> {
    const at = this.bufBase + this.buf.length;
    if (at >= this.size) return false;
    const len = Math.min(CHUNK, this.size - at);
    const chunk = Buffer.allocUnsafe(len);
    const { bytesRead } = await this.fh.read(chunk, 0, len, at);
    if (!bytesRead) return false;
    const piece = chunk.subarray(0, bytesRead);
    this.buf = this.buf.length ? Buffer.concat([this.buf, piece]) : piece;
    this.readSoFar += bytesRead;
    if (this.readSoFar > MAX_HEADER_BYTES) {
      throw new Error('GGUF header is unreasonably large');
    }
    return true;
  }

  async take(n: number): Promise<Buffer> {
    if (n < 0 || n > this.size) throw new Error('GGUF read past end of file');
    const have = this.buf.length - this.off;
    if (n <= have) {
      const out = this.buf.subarray(this.off, this.off + n);
      this.off += n;
      return out;
    }
    // Drop the consumed prefix before piling more chunks on top of it.
    this.bufBase += this.off;
    this.buf = this.buf.subarray(this.off);
    this.off = 0;
    while (this.buf.length < n) {
      if (!(await this.more())) throw new Error('GGUF file ended before its header did');
    }
    const out = this.buf.subarray(0, n);
    this.off = n;
    return out;
  }

  async skip(n: number): Promise<void> {
    if (n < 0) throw new Error('negative GGUF skip');
    const abs = this.bufBase + this.off;
    if (abs + n > this.size) throw new Error('GGUF read past end of file');
    const inBuf = this.buf.length - this.off;
    if (n <= inBuf) {
      this.off += n;
      return;
    }
    // Everything buffered is being skipped too; release it and jump the file
    // cursor instead of reading the bytes only to throw them away.
    this.bufBase = abs + n;
    this.buf = Buffer.alloc(0);
    this.off = 0;
  }

  async u32(): Promise<number> {
    return (await this.take(4)).readUInt32LE(0);
  }

  async u64(): Promise<number> {
    const v = (await this.take(8)).readBigUInt64LE(0);
    // The header values that matter fit comfortably in a double; anything
    // larger is a corrupt file and reading it as Infinity would poison the
    // arithmetic downstream.
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('GGUF integer out of range');
    return Number(v);
  }

  async string(): Promise<string> {
    const len = await this.u64();
    if (len > this.size) throw new Error('GGUF string longer than the file');
    return (await this.take(len)).toString('utf8');
  }
}

/** Consume one metadata value, returning it only when it is a number. */
async function readValue(
  r: HeaderReader,
  type: number,
  want: 'skip' | 'number' | 'string',
): Promise<number | string | null> {
  if (type === GgufType.String) {
    const s = await r.string();
    return want === 'string' ? s : null;
  }
  if (type === GgufType.Array) {
    const elem = await r.u32();
    const count = await r.u64();
    // `head_count_kv` is written per layer by some converters; the largest
    // entry is the one that sizes the cache. Anything else is skipped whole.
    if (want !== 'number' || count > 1_000_000) {
      const width = WIDTH[elem];
      if (width) await r.skip(width * count);
      else for (let i = 0; i < count; i += 1) await readValue(r, elem, 'skip');
      return null;
    }
    let max: number | null = null;
    for (let i = 0; i < count; i += 1) {
      const v = await readValue(r, elem, 'number');
      if (typeof v === 'number') max = max === null ? v : Math.max(max, v);
    }
    return max;
  }
  const width = WIDTH[type];
  if (!width) throw new Error(`unknown GGUF value type ${type}`);
  const b = await r.take(width);
  if (want !== 'number') return null;
  switch (type) {
    case GgufType.Uint8:
      return b.readUInt8(0);
    case GgufType.Int8:
      return b.readInt8(0);
    case GgufType.Uint16:
      return b.readUInt16LE(0);
    case GgufType.Int16:
      return b.readInt16LE(0);
    case GgufType.Uint32:
      return b.readUInt32LE(0);
    case GgufType.Int32:
      return b.readInt32LE(0);
    case GgufType.Float32:
      return b.readFloatLE(0);
    case GgufType.Uint64:
      return Number(b.readBigUInt64LE(0));
    case GgufType.Int64:
      return Number(b.readBigInt64LE(0));
    case GgufType.Float64:
      return b.readDoubleLE(0);
    case GgufType.Bool:
      return b.readUInt8(0) ? 1 : 0;
    default:
      return null;
  }
}

/**
 * Read the architecture hyperparameters out of a GGUF file.
 *
 * Returns null for anything unreadable or not a GGUF — the caller falls back
 * to a weights-only estimate rather than showing a made-up cache figure.
 */
export async function readGgufMeta(file: string): Promise<GgufMeta | null> {
  let fh: FileHandle | null = null;
  try {
    fh = await open(file, 'r');
    const info = await fh.stat();
    if (!info.isFile() || info.size < 24) return null;
    const r = new HeaderReader(fh, info.size);
    const magic = (await r.take(4)).toString('latin1');
    if (magic !== 'GGUF') return null;
    const version = await r.u32();
    if (version < 2 || version > 3) return null;
    await r.u64(); // tensor count — only the metadata is needed
    const kvCount = await r.u64();
    if (kvCount > 1_000_000) return null;

    let architecture = '';
    let blockCount = 0;
    let embeddingLength = 0;
    let headCount = 0;
    let headCountKV = 0;
    let contextLength = 0;

    for (let i = 0; i < kvCount; i += 1) {
      const key = await r.string();
      const type = await r.u32();
      // The tokenizer tables are large and none of the fields below live after
      // them in files written by llama.cpp's converter. Once all the fields
      // are known, stop before walking megabytes of vocabulary.
      if (key.startsWith('tokenizer.') && architecture && blockCount && embeddingLength) break;
      if (key === 'general.architecture') {
        architecture = (await readValue(r, type, 'string')) as string;
        continue;
      }
      if (key === 'general.context_length') {
        contextLength = (await readValue(r, type, 'number')) as number;
        continue;
      }
      // Keys are `<architecture>.<name>`; accept the suffix even before the
      // architecture string is known, and resolve against it afterwards.
      const value = key.includes('.')
        ? (await readValue(r, type, 'number'))
        : ((await readValue(r, type, 'skip')), null);
      switch (key.slice(key.indexOf('.') + 1)) {
        case 'block_count':
          if (typeof value === 'number') blockCount = value;
          break;
        case 'embedding_length':
          if (typeof value === 'number') embeddingLength = value;
          break;
        case 'context_length':
          if (typeof value === 'number' && !contextLength) contextLength = value;
          break;
        case 'attention.head_count':
          if (typeof value === 'number') headCount = value;
          break;
        case 'attention.head_count_kv':
          if (typeof value === 'number') headCountKV = value;
          break;
        default:
          break;
      }
      // Architectures name their keys with their own prefix; a file with a
      // foreign prefix (rare, stitched models) simply keeps the first value.
    }

    if (!architecture && !blockCount && !embeddingLength) return null;
    return {
      architecture,
      blockCount,
      embeddingLength,
      headCount,
      headCountKV,
      contextLength,
    };
  } catch {
    return null;
  } finally {
    await fh?.close().catch(() => {});
  }
}

/**
 * Pure arithmetic: weights + KV at `ctx` + runtime overhead.
 *
 * KV is `2 (K and V) × layers × kv heads × head dim × ctx × 2 bytes`. The
 * layer count belongs in the formula even though LocalAI's estimator leaves it
 * out: without it a 32-layer model looks like it needs 1/32 of its real cache.
 */
export function estimateVram(
  weightsBytes: number,
  meta: GgufMeta | null,
  ctx: number,
): VramEstimate {
  const weights = Math.max(0, weightsBytes);
  const layers = meta?.blockCount && meta.blockCount > 0 ? meta.blockCount : 32;
  const embedding = meta?.embeddingLength && meta.embeddingLength > 0 ? meta.embeddingLength : 4096;
  const heads = meta?.headCount && meta.headCount > 0 ? meta.headCount : 32;
  const headDim = Math.max(1, Math.round(embedding / heads));
  const kvHeads = meta?.headCountKV && meta.headCountKV > 0 ? meta.headCountKV : heads;
  const tokens = Number.isFinite(ctx) && ctx > 0 ? Math.floor(ctx) : 0;
  const kvBytes =
    meta && tokens > 0
      ? 2 * layers * kvHeads * headDim * tokens * KV_BYTES_PER_ELEMENT
      : 0;
  const overhead = Math.round(weights * OVERHEAD_FRACTION + OVERHEAD_FLOOR_BYTES);
  return {
    weightsBytes: weights,
    kvBytes,
    overheadBytes: overhead,
    totalBytes: weights + kvBytes + overhead,
    ctx: tokens,
    meta,
  };
}

const metaCache = new Map<string, { key: string; meta: GgufMeta | null }>();
const CACHE_MAX = 64;

/**
 * Estimate for one file at one context length.
 *
 * The header read is cached by path + size + mtime, so the models page can ask
 * about every row on every visit without re-reading the same prefixes; the
 * estimate itself is cheap arithmetic and is recomputed when the context
 * length changes.
 */
export async function fileVramEstimate(file: string, ctx: number): Promise<VramEstimate> {
  let weights = 0;
  let key = `${file}\0?`;
  try {
    const info = await stat(file);
    weights = info.size;
    key = `${file}\0${info.size}\0${info.mtimeMs}`;
  } catch {
    // The caller may be racing a delete; the estimate still answers for zero
    // bytes rather than throwing out of an API response.
  }
  let hit = metaCache.get(file);
  if (!hit || hit.key !== key) {
    hit = { key, meta: weights > 0 ? await readGgufMeta(file) : null };
    if (metaCache.size >= CACHE_MAX) {
      const oldest = metaCache.keys().next().value;
      if (oldest !== undefined) metaCache.delete(oldest);
    }
    metaCache.set(file, hit);
  }
  return estimateVram(weights, hit.meta, ctx);
}

export type VramFit = 'fits' | 'tight' | 'over' | null;

/**
 * Does `est` fit in `vramTotalBytes`?
 *
 * `tight` is the honest band around the line: an estimate within 15% of the
 * card's total will fit or not depending on the driver, the display's own
 * allocation and llama.cpp's scratch buffers, so claiming a clean "fits"
 * would be a promise the app cannot keep.
 */
export function vramFit(est: VramEstimate, vramTotalBytes: number | null): VramFit {
  if (!vramTotalBytes || vramTotalBytes <= 0) return null;
  if (est.totalBytes > vramTotalBytes) return 'over';
  if (est.totalBytes > vramTotalBytes * 0.85) return 'tight';
  return 'fits';
}

/** Machine RAM as a fallback figure when there is no GPU to measure. */
export const systemRamBytes = () => os.totalmem();
