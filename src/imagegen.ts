/**
 * Free image generation providers.
 *
 * Deliberately separate from the chat provider registry: none of these speak
 * the chat wire protocol. The browser posts a prompt, the server calls the
 * provider and hands back the image bytes as base64, and the client renders it
 * as an assistant message like any attached image.
 *
 * Four adapters, in the order the UI presents them:
 *
 *   pollinations  — keyless. GET /prompt/{prompt}, returns bytes.
 *   stable-horde  — free GPU volunteers. Key optional (anonymous is
 *                   rate-limited), async job + polling.
 *   cloudflare    — free tier (no card), needs a token + account id, and the
 *                   flux-1-schnell model id.
 *   huggingface   — free tier behind a free token; serverless inference, can
 *                   take up to ~20s on first use.
 *
 * Rate and abuse limits are still load-bearing: every endpoint sits behind the
 * shared /api limiter, and a separate concurrency cap stops one browser from
 * pinning the CPU with parallel long generations.
 */
import { actualImageSize } from './image-size.js';
import { getKey } from './store.js';

export interface ImageGenSize {
  w: number;
  h: number;
}

export interface ImageGenView {
  id: string;
  label: string;
  keyless: boolean;
  configured: boolean;
  hint: string;
  models: string[];
  defaultModel: string;
  sizes: ImageGenSize[];
}

export interface GeneratedImage {
  /** Base64, no data: prefix — the same shape attachments already travel in. */
  data: string;
  mimeType: string;
  width: number;
  height: number;
  provider: string;
  model: string;
  prompt: string;
  label: string;
}

export interface GenerateCall {
  providerId: string;
  model: string;
  prompt: string;
  width: number;
  height: number;
  signal: AbortSignal;
}

/** Aspect ratios the UI offers. Providers that can't honour them keep defaults. */
export const IMAGE_SIZES: ImageGenSize[] = [
  { w: 1024, h: 1024 },
  { w: 1152, h: 896 },
  { w: 896, h: 1152 },
  { w: 1536, h: 640 },
];

export const MAX_PROMPT_LENGTH = 2000;

/** Caps long generations so one client can't pin the CPU. */
const MAX_CONCURRENT = 2;
let active = 0;
export function acquireImageGen(): boolean {
  if (active >= MAX_CONCURRENT) return false;
  active += 1;
  return true;
}
export function releaseImageGen(): void {
  active = Math.max(0, active - 1);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Runs an upstream fetch under both the caller's abort signal and a deadline.
 * Neither source can outlive the other: whichever fires first wins.
 */
async function race(
  timeoutMs: number,
  signal: AbortSignal,
  run: (s: AbortSignal) => Promise<Response>,
): Promise<Response> {
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal.reason);
  if (signal.aborted) controller.abort(signal.reason);
  else signal.addEventListener('abort', onAbort, { once: true });
  const timer = setTimeout(() => controller.abort(new Error('Image generation timed out')), timeoutMs);
  try {
    return await run(controller.signal);
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', onAbort);
  }
}

type GenerateFn = (opts: {
  model: string;
  prompt: string;
  width: number;
  height: number;
  signal: AbortSignal;
}) => Promise<Omit<GeneratedImage, 'provider' | 'model' | 'prompt' | 'label'>>;

interface ImageProvider {
  id: string;
  label: string;
  hint: string;
  /** The full set of env keys that must be present for this provider to run. */
  requires: string[];
  models: string[];
  defaultModel: string;
  generate: GenerateFn;
}

const keyless: ImageProvider = {
  id: 'pollinations',
  label: 'Pollinations',
  hint: 'Keyless — any prompt, hosted proxies over FLUX.',
  requires: [],
  models: ['flux'],
  defaultModel: 'flux',
  async generate({ model, prompt, width, height, signal }) {
    const url = new URL(`https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}`);
    url.searchParams.set('width', String(width));
    url.searchParams.set('height', String(height));
    url.searchParams.set('seed', String(Math.floor(Math.random() * 1_000_000_000)));
    url.searchParams.set('nologo', 'true');
    url.searchParams.set('model', model);

    const res = await race(50_000, signal, (s) => fetch(url, { signal: s }));
    if (!res.ok) {
      const detail = (await res.text().catch(() => '')).slice(0, 140);
      throw new Error(`Pollinations error (${res.status}) ${detail}`);
    }
    const bytes = Buffer.from(await res.arrayBuffer());
    if (!bytes.length) throw new Error('Pollinations returned an empty image');
    const header = res.headers.get('content-type') ?? '';
    return {
      data: bytes.toString('base64'),
      mimeType: header.startsWith('image/') ? header : 'image/png',
      width,
      height,
    };
  },
};

const stableHorde: ImageProvider = {
  id: 'stable-horde',
  label: 'Stable Horde',
  hint: 'Free volunteer GPUs. Anonymous works (rate-limited); a free kudos key lifts it.',
  requires: [],
  models: ['Deliberate'],
  defaultModel: 'Deliberate',
  async generate({ model, prompt, width, height, signal }) {
    const base = 'https://stablehorde.net/api/v2';
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const key = getKey('STABLE_HORDE_API_KEY');
    if (key) headers['apikey'] = key;

    const created = await race(30_000, signal, (s) =>
      fetch(`${base}/generate/async`, {
        method: 'POST',
        headers,
        signal: s,
        body: JSON.stringify({
          prompt,
          params: { width, height, nsfw: false },
          models: [model],
        }),
      }),
    );
    if (!created.ok) throw new Error(`Stable Horde rejected the request (${created.status})`);
    const job = (await created.json()) as { id?: string };
    if (!job.id) throw new Error('Stable Horde returned no job id');

    // Anonymous requests can wait behind a queue; give it a real minute.
    const deadline = Date.now() + 80_000;
    for (;;) {
      if (signal.aborted) throw new Error('Interrupted');
      if (Date.now() > deadline) throw new Error('Stable Horde timed out waiting for a volunteer');
      const check = await race(20_000, signal, (s) => fetch(`${base}/generate/check/${job.id}`, { signal: s }));
      const status = (await check.json()) as { done?: boolean };
      if (status.done) break;
      await sleep(2500);
    }

    const finished = await race(20_000, signal, (s) => fetch(`${base}/generate/status/${job.id}`, { signal: s }));
    const result = (await finished.json()) as { generations?: Array<{ img: string }>; message?: string };
    const raw = result.generations?.[0]?.img;
    if (!raw) throw new Error(result.message ?? 'Stable Horde produced no image');

    // The default response is base64; a cloudflare-r2 variant returns a URL.
    if (raw.startsWith('http')) {
      const got = await race(30_000, signal, (s) => fetch(raw, { signal: s }));
      const bytes = Buffer.from(await got.arrayBuffer());
      return {
        data: bytes.toString('base64'),
        mimeType: got.headers.get('content-type')?.startsWith('image/')
          ? (got.headers.get('content-type') as string)
          : 'image/png',
        width,
        height,
      };
    }
    return { data: raw, mimeType: 'image/jpeg', width, height };
  },
};

const cloudflare: ImageProvider = {
  id: 'cloudflare',
  label: 'Cloudflare Workers AI',
  hint: 'Free tier, no card. Needs a token and account id from the settings page.',
  requires: ['CLOUDFLARE_API_TOKEN', 'CLOUDFLARE_ACCOUNT_ID'],
  models: ['@cf/black-forest-labs/flux-1-schnell'],
  defaultModel: '@cf/black-forest-labs/flux-1-schnell',
  async generate({ model, prompt, width, height, signal }) {
    const token = getKey('CLOUDFLARE_API_TOKEN');
    const account = getKey('CLOUDFLARE_ACCOUNT_ID');
    if (!token || !account) throw new Error('Cloudflare needs an API token and account id (set them in Settings)');

    const res = await race(60_000, signal, (s) =>
      fetch(`https://api.cloudflare.com/client/v4/accounts/${encodeURIComponent(account)}/ai/run/${model}`, {
        method: 'POST',
        signal: s,
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ prompt, width, height }),
      }),
    );
    const body = (await res.json().catch(() => null)) as {
      success?: boolean;
      result?: { image?: string };
      errors?: Array<{ message?: string }>;
      message?: string;
    } | null;
    if (!res.ok || !body?.success) {
      const detail = body?.errors?.[0]?.message ?? body?.message ?? `HTTP ${res.status}`;
      throw new Error(`Cloudflare: ${detail}`);
    }
    if (!body.result?.image) throw new Error('Cloudflare returned no image');
    return { data: body.result.image, mimeType: 'image/png', width, height };
  },
};

const huggingface: ImageProvider = {
  id: 'huggingface',
  label: 'Hugging Face',
  hint: 'Free tier behind a free token. Serverless, so first use can take ~20s.',
  requires: ['HF_API_TOKEN'],
  models: ['black-forest-labs/FLUX.1-schnell'],
  defaultModel: 'black-forest-labs/FLUX.1-schnell',
  async generate({ model, prompt, signal }) {
    const token = getKey('HF_API_TOKEN');
    if (!token) throw new Error('Hugging Face needs an API token (set it in Settings)');

    let res = await race(95_000, signal, (s) =>
      fetch(`https://api-inference.huggingface.co/models/${model}`, {
        method: 'POST',
        signal: s,
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ inputs: prompt }),
      }),
    );
    // 503 means the model is still waking up; honour retry-after once.
    if (res.status === 503) {
      const retryMs = Math.min(Number(res.headers.get('retry-after') ?? '3') * 1000, 10_000);
      if (!signal.aborted) {
        await sleep(retryMs);
        res = await race(95_000, signal, (s) =>
          fetch(`https://api-inference.huggingface.co/models/${model}`, {
            method: 'POST',
            signal: s,
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
            body: JSON.stringify({ inputs: prompt }),
          }),
        );
      }
    }
    if (!res.ok) {
      const detail = (await res.text().catch(() => '')).slice(0, 140);
      throw new Error(`Hugging Face (${res.status}) ${detail}`);
    }
    const bytes = Buffer.from(await res.arrayBuffer());
    if (!bytes.length) throw new Error('Hugging Face returned an empty image');
    const header = res.headers.get('content-type') ?? '';
    return {
      data: bytes.toString('base64'),
      mimeType: header.startsWith('image/') ? header : 'image/png',
      width: 1024,
      height: 1024,
    };
  },
};

const imageProviders: ImageProvider[] = [keyless, stableHorde, cloudflare, huggingface];

const byId = new Map(imageProviders.map((p) => [p.id, p]));
export function imageProviderById(id: string): ImageProvider | undefined {
  return byId.get(id);
}

export function imageGenProvidersView(): ImageGenView[] {
  return imageProviders.map((p) => ({
    id: p.id,
    label: p.label,
    keyless: p.requires.length === 0,
    configured: p.requires.every((name) => Boolean(getKey(name))),
    hint: p.hint,
    models: p.models,
    defaultModel: p.defaultModel,
    sizes: IMAGE_SIZES,
  }));
}

/** Human labels for the settings page key rows that belong to image providers. */
export const IMAGE_FIELD_LABELS: Record<string, string> = {
  HF_API_TOKEN: 'Hugging Face',
  CLOUDFLARE_API_TOKEN: 'Cloudflare Workers AI',
  CLOUDFLARE_ACCOUNT_ID: 'Cloudflare account id',
  STABLE_HORDE_API_KEY: 'Stable Horde (optional)',
};

export async function generateImage(call: GenerateCall): Promise<GeneratedImage> {
  const provider = byId.get(call.providerId);
  if (!provider) throw new Error(`Unknown image provider: ${call.providerId}`);
  if (!provider.models.includes(call.model)) throw new Error(`Unknown model: ${call.model}`);
  const prompt = call.prompt.trim();
  if (!prompt) throw new Error('Prompt is required');
  if (prompt.length > MAX_PROMPT_LENGTH) throw new Error('Prompt is too long');
  if (!provider.requires.every((name) => Boolean(getKey(name)))) {
    throw new Error(`${provider.label} is not configured — set its key in Settings`);
  }

  const size = IMAGE_SIZES.find((s) => s.w === call.width && s.h === call.height) ?? IMAGE_SIZES[0];

  const result = await provider.generate({
    model: call.model,
    prompt,
    width: size.w,
    height: size.h,
    signal: call.signal,
  });
  return {
    ...result,
    // A provider may answer with a smaller picture than the one requested
    // (Pollinations returns 768×768 for a 1024 request), and its reported
    // width/height is the request, not the answer. The file says what it is.
    ...actualImageSize(result),
    provider: provider.id,
    model: call.model,
    prompt,
    label: provider.label,
  };
}