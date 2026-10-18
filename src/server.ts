import './net.js';
import 'dotenv/config';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import express from 'express';
import { readFileSync } from 'node:fs';
import { createServer as createTlsServer } from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { acquireStream, rateLimit, releaseStream } from './limits.js';
import { checkAttachments, contentLength, LIMITS, parseContent } from './attachments.js';
import { isToolFormatError, isToolsUnsupported, runAgent, withToolGuidance } from './agent.js';
import { enabledTools, listTools, resolveToolSelection, setToolEnabled } from './tools.js';
import type { Tool } from './tools.js';
import {
  acquirePull,
  deleteLocalModel,
  listLocalModels,
  listRunningModels,
  pullModel,
  releasePull,
  unloadModel,
  validateModelName,
} from './ollama-models.js';
import { ensureOllamaRunning, hookOllamaExit, ollamaReachable } from './ollama-server.js';
import { listProcesses, processMemoryBytes, stopAllProcesses, stopProcess } from './processes.js';
import {
  acquireDownload,
  avatarEtag,
  deleteCacheRepo,
  deleteDownload,
  downloadFile,
  modelsDir,
  fetchAvatar,
  imageMime,
  listDownloads,
  modelDetail,
  requiredFiles,
  pruneStalePartials,
  releaseDownload,
  searchModels,
  validateFilePath,
  validateRepoId,
} from './huggingface.js';
import { imageProviderById, imageGenProvidersView, generateImage, acquireImageGen, releaseImageGen, IMAGE_FIELD_LABELS, MAX_PROMPT_LENGTH } from './imagegen.js';
import { cacheImage, cacheNameError, imageCachePath } from './image-cache.js';
import { ggufModels, llamaServerState, stopLlamaServer } from './llamacpp.js';
import { llamaCppUpgradeKind } from './runtimes.js';
import { stopTransformersServer, transformersChatModels } from './transformers-chat.js';
import { allProviders, providerOrder } from './providers/index.js';
import { discoverLocalServers, discoveredProvider, type DiscoveredServer } from './discovery.js';
import { backendReports } from './backends.js';
import { runDriverInstaller } from './drivers.js';
import {
  assignments,
  loadAssignments,
  setAssignment,
  type TaskKey,
} from './assignments.js';
import { CATALOG } from './model-catalog.js';
import { runnableModels, type RunTask } from './runnable.js';
import {
  finishTransfer,
  isRunning,
  listTransfers,
  startTransfer,
  transferId,
  updateTransfer,
} from './transfers.js';
import { resourceReport } from './resources.js';
import {
  RUNTIME_GROUPS,
  detectGpu,
  deviceWarning,
  groupBlockedReason,
  groupById,
  installRuntimes,
  parseDevice,
  removeRuntimeGroup,
  removeRuntimes,
  runtimeStatus,
  venvPythonVersion,
  type RuntimeGroup,
} from './runtimes.js';
import { generateImageLocal, speak, transcribe } from './infer.js';
import {
  addCustomProvider,
  customApiKey,
  customProvidersFilePath,
  deleteCustomProvider,
  listCustomProviders,
  loadCustomProviders,
  updateCustomProvider,
} from './providers/custom.js';
import type { CustomProvider } from './providers/custom.js';
import type { Provider, ProviderId } from './providers/types.js';
import { ALLOWED_KEYS, clearAllKeys, clearKey, getKey, keysFilePath, loadKeys, maskKey, keySource, setKey } from './store.js';
import type { AllowedKey } from './store.js';
import type { ChatMessage } from './types.js';

const app = express();
app.disable('x-powered-by');

// Behind a reverse proxy, req.ip must come from X-Forwarded-For to be meaningful.
if (process.env.TRUST_PROXY) {
  app.set('trust proxy', process.env.TRUST_PROXY === 'true' ? 1 : process.env.TRUST_PROXY);
}

/**
 * Baseline security headers. The CSP is strict because the UI ships no inline
 * script or style; anything added later must be a real file to stay allowed.
 */
app.use((_req, res, next) => {
  res.setHeader(
    'Content-Security-Policy',
    [
      "default-src 'self'",
      "script-src 'self'",
      "style-src 'self'",
      "img-src 'self' data:",
      "connect-src 'self'",
      "form-action 'self'",
      "base-uri 'none'",
      "frame-ancestors 'none'",
      "object-src 'none'",
    ].join('; '),
  );
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
  if (process.env.TLS_CERT && process.env.TLS_KEY) {
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
  }
  next();
});

// Image attachments base64-inflate by about a third, so the chat route needs
// far more headroom than the rest of the API. Scoped to this one path so no
// other endpoint accepts a large body; checkUsage() still enforces the real
// per-message caps, and this is only the outer bound.
app.use('/api/chat', express.json({ limit: '20mb' }));
app.use(express.json({ limit: '1mb' }));

// A malformed JSON body would otherwise render Express's default error page,
// which leaks a stack trace and absolute paths. There's no auth on this server,
// so error detail stays server-side and the client gets a plain message.
app.use((err: any, _req: express.Request, res: express.Response, next: express.NextFunction) => {
  if (err instanceof SyntaxError && 'body' in err) {
    return res.status(400).json({ error: 'Malformed JSON body' });
  }
  if (err?.type === 'entity.too.large') {
    return res.status(413).json({ error: 'Request body too large' });
  }
  next(err);
});

// Audio uploads arrive as the raw request body, not JSON. Mounted here so the
// parser only sees the transcription route and never buffers chat bodies.
const audioUpload = express.raw({
  type: ['audio/*', 'application/octet-stream'],
  limit: '25mb',
});

const publicDir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'public');
// 'extensions' lets /settings resolve to settings.html instead of 404ing.
app.use(express.static(publicDir, { extensions: ['html'] }));

/**
 * The Studio used to be called the Image Studio and lived at /image. Both old
 * spellings still work: a bookmark or an old message should not 404 just
 * because the page learned to do more than images.
 */
app.get(['/image', '/imagestudio'], (_req, res) => res.redirect(302, '/studio'));

// The standalone Tools page merged into the settings hub; old links land on
// the Tools tab instead of 404ing.
app.get('/tools', (_req, res) => res.redirect('/settings#tools'));

/**
 * Blocks cross-site state changes. There's no session cookie to rely on, so this
 * Origin check is the only thing standing between a malicious page in your
 * browser and your API keys.
 */
function sameOriginOnly(req: express.Request, res: express.Response, next: express.NextFunction) {
  const origin = req.get('origin');
  if (!origin) return next();
  const host = req.get('host');
  if (host && new URL(origin).host === host) return next();
  res.status(403).json({ error: 'Cross-origin request blocked' });
}

const WRITE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);
app.use((req, res, next) =>
  WRITE_METHODS.has(req.method) ? sameOriginOnly(req, res, next) : next(),
);

/* ---------------- API ---------------- */

// Fixed-window rate limit. Not a login gate: it exists so a runaway client
// (or a loop feeding the chat endpoint) can't burn through your API quota.
app.use('/api', rateLimit);

/** Lists every provider — built-in and user-added — with config state and models. */
app.get('/api/providers', async (_req, res) => {
  // llama.cpp's provider answers "is anything runnable?" from a cached listing
  // of the GGUF files on this machine, and its info is read synchronously. One
  // warm-up here keeps a freshly started server (or a fresh page load) from
  // reporting "no models" for the first second.
  await ggufModels().catch(() => undefined);
  await transformersChatModels().catch(() => undefined);

  const registry = providerRegistry();

  // Self-hosted servers that happen to be running on this machine. Ollama is
  // always offered because the app manages it; the rest are found rather than
  // configured, so a llama.cpp or LM Studio the user already started shows up
  // without being typed in by hand.
  const discovered = await refreshDiscovered();
  const known = providerOrder();
  const order = [
    ...known,
    ...discovered.map((d) => d.id).filter((id) => !known.includes(id)),
  ];

  const entries = await Promise.all(
    order.map(async (id) => {
      const p = registry[id];
      if (!p) return null;
      const models = p.info.configured ? await p.listModels() : [];
      // Speech-to-text models are a subset of the same list; exposing them lets
      // the UI show a mic only where one would actually work.
      const sttModels = p.listSttModels && p.info.configured ? await p.listSttModels() : [];
      return {
        ...p.info,
        models,
        stt: Boolean(p.transcribe),
        sttModels,
        // Local vs cloud, so the picker can group rather than present one flat
        // list where "Ollama" sits among API keys with no indication it runs
        // on this machine. The provider's own flag is authoritative — Ollama
        // declares it, as does every discovered server — and the discovered set
        // is a fallback for anything that predates the flag.
        local: Boolean(p.info.local) || LOCAL_PROVIDER_IDS.has(id),
        discovered: discovered.some((d) => d.id === id),
      };
    }),
  );

  res.json({
    providers: entries.filter(Boolean),
    localServers: discovered,
  });
});

/**
 * Re-probe the well-known local ports.
 *
 * Separate from the provider list because discovery is a point-in-time answer:
 * starting LM Studio should make it appear without a full page reload.
 */
app.get('/api/providers/discover', async (_req, res) => {
  res.json({ servers: await refreshDiscovered() });
});

/** User-added providers. The API key is only ever returned masked. */
function customProviderView(p: CustomProvider) {
  return {
    id: p.id,
    label: p.label,
    baseUrl: p.baseUrl,
    stt: p.stt,
    createdAt: p.createdAt,
    // Masked only: the raw key never leaves the server.
    masked: p.apiKey ? maskKey(p.apiKey) : null,
    hasKey: Boolean(p.apiKey),
  };
}

app.get('/api/custom-providers', (_req, res) => {
  res.json({
    providers: listCustomProviders().map(customProviderView),
    storagePath: customProvidersFilePath(),
  });
});

app.post('/api/custom-providers', async (req, res) => {
  try {
    const created = await addCustomProvider(req.body ?? {});
    res.status(201).json({ provider: customProviderView(created) });
  } catch (err) {
    res.status(400).json({ error: (err as Error).message });
  }
});

app.put('/api/custom-providers/:id', async (req, res) => {
  try {
    const updated = await updateCustomProvider(req.params.id, req.body ?? {});
    res.json({ provider: customProviderView(updated) });
  } catch (err) {
    res.status(400).json({ error: (err as Error).message });
  }
});

app.delete('/api/custom-providers/:id', async (req, res) => {
  const removed = await deleteCustomProvider(req.params.id);
  if (!removed) return res.status(404).json({ error: 'No such provider' });
  res.json({ deleted: req.params.id });
});

/**
 * Asks a provider's own /models endpoint what it offers. Used by the settings
 * page to scan a freshly added endpoint, and to explain why a provider shows
 * no models. Never returns a key.
 */
async function scanModels(provider: Provider, hint: string) {
  const models = await provider.listModels();
  const sttModels = provider.listSttModels ? await provider.listSttModels() : [];
  // listModels swallows transport errors and returns [], so an empty list is
  // reported as "couldn't read any" rather than pretending the provider is empty.
  return {
    models,
    sttModels,
    ok: models.length > 0,
    note: models.length
      ? `${models.length} model${models.length > 1 ? 's' : ''} available`
      : `No models returned. Check the URL and key, and that ${hint} is reachable.`,
  };
}

app.get('/api/custom-providers/:id/models', async (req, res) => {
  const entry = listCustomProviders().find((p) => p.id === req.params.id);
  if (!entry) return res.status(404).json({ error: 'No such provider' });
  const provider = providerRegistry()[entry.id];
  if (!provider) return res.status(404).json({ error: 'No such provider' });
  res.json(await scanModels(provider, entry.baseUrl));
});

/**
 * Providers that run on this machine rather than someone else's.
 *
 * Ollama is here by construction. The discovered servers are added to the set
 * at request time; they are marked per-entry in the response as well, so this
 * stays the single source of truth for "is it local".
 */
const LOCAL_PROVIDER_IDS = new Set<string>(['ollama']);

/**
 * Self-hosted servers found on this machine, keyed by id.
 *
 * Held separately so it can be merged into every lookup. Discovery used to
 * happen only inside `/api/providers`, which made a discovered server appear in
 * the picker and then fail with "Unknown provider" the moment you tried to
 * chat with it — the list and the routes disagreed about what exists.
 */
let discoveredRegistry: Record<string, Provider> = {};

/** The provider registry every route must use. */
function providerRegistry(): Record<string, Provider> {
  return { ...allProviders(), ...discoveredRegistry };
}

/** Refreshes the discovered set. Called on boot and on demand. */
async function refreshDiscovered(): Promise<DiscoveredServer[]> {
  const servers = await discoverLocalServers(true);
  const next: Record<string, Provider> = {};
  for (const server of servers) {
    LOCAL_PROVIDER_IDS.add(server.id);
    next[server.id] = discoveredProvider(server);
  }
  discoveredRegistry = next;
  return servers;
}

/**
 * Every provider that can transcribe audio, so dictation availability is
 * visible in one place. Capability comes from the provider's own flags, so this
 * never makes a network call; model lists are fetched on demand.
 */
app.get('/api/dictation', async (_req, res) => {
  const registry = providerRegistry();
  const entries = providerOrder()
    .map((id) => registry[id])
    .filter((p) => p?.transcribe)
    .map((p) => ({
      id: p.info.id,
      label: p.info.label,
      custom: Boolean(p.info.custom),
      configured: p.info.configured,
      hasKey: p.info.custom
        ? Boolean(customApiKey(p.info.id))
        : Boolean(p.info.envKey && getKey(p.info.envKey)),
    }));

  // Dictation was cloud-only, so with no API keys configured the mic simply
  // vanished and there was no indication that a downloaded model on this
  // machine could transcribe just as well. Local speech-to-text now appears
  // alongside the cloud providers, with the same readiness rules.
  let local: Awaited<ReturnType<typeof runnableModels>> = [];
  try {
    local = await runnableModels('stt');
  } catch (err) {
    console.error('[dictation] could not list local models:', (err as Error).message);
  }

  const assigned = assignments();

  // Mark the assigned model so the UI can show the choice without a second
  // round trip, and so a stale assignment is visibly stale rather than silent.
  local = local.map((m) => ({
    ...m,
    assigned: assigned.stt === `local:${m.repo}`,
  }));

  res.json({ providers: entries, local, assigned });
});

/**
 * Every model this machine can run, by task, with the reason when it cannot.
 * The single source of truth behind the hub, the Run buttons and dictation.
 */
app.get('/api/runnable', async (req, res) => {
  const task = String(req.query.task ?? '');
  if (task && !['stt', 'tts', 'image'].includes(task)) {
    return res.status(400).json({ error: 'Unknown task' });
  }
  const tasks: RunTask[] = task ? [task as RunTask] : ['stt', 'tts', 'image'];
  const out: Record<string, unknown> = {};
  for (const t of tasks) out[t] = await runnableModels(t);
  res.json({ ...out, assigned: assignments() });
});

/**
 * Assigns a model to a job: speech-to-text, speech, or image generation.
 *
 * The single place a model gets tied to a task. The chat page and the Studio
 * both read the result rather than asking the user to choose again, which is
 * what put a model picker in the message bar in the first place.
 */
app.post('/api/assignments', async (req, res) => {
  const task = String(req.body?.task ?? '') as TaskKey;
  const model = typeof req.body?.model === 'string' ? req.body.model : null;
  if (!['stt', 'tts', 'image'].includes(task)) {
    return res.status(400).json({ error: 'Unknown task' });
  }
  try {
    res.json({ assigned: await setAssignment(task, model) });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

/**
 * Which models can accept image attachments.
 *
 * `known: true` means the provider actually reported the capability (Ollama's
 * /api/show `capabilities`, or a gateway's `input_modalities`). When
 * `known` is false the provider published nothing, so the browser treats
 * attachment as "unverified" rather than guessing from the model's name — a
 * name-based guess would go stale the moment a provider renames anything.
 *
 * Probed on demand, never on page load, because it costs a request per model.
 */
app.get('/api/vision', async (req, res) => {
  const registry = providerRegistry();
  const id = String(req.query.provider ?? '') as ProviderId;
  const provider = registry[id];
  if (!provider) return res.status(404).json({ error: `Unknown provider: ${id}` });

  // listModels() populates the capability set for providers that read it from
  // their /models response, so it has to run first.
  const all = await provider.listModels();
  const claimed = (await provider.listVisionModels?.()) ?? [];
  res.json({
    provider: provider.info.id,
    models: all,
    visionModels: claimed,
    known: provider.info.id === 'ollama' || claimed.length > 0,
    limits: LIMITS,
  });
});

/** Provider metadata for the settings form. Never returns a usable key. */
app.get('/api/settings', (_req, res) => {
  const registry = providerRegistry();
  const fields = ALLOWED_KEYS.map((envKey) => {
    const provider = Object.values(registry).find((p) => p.info.envKey === envKey);
    const value = getKey(envKey);
    return {
      envKey,
      providerId: provider?.info.id ?? null,
      // Image-provider fields live outside the chat registry, so their labels
      // come from a side table rather than a provider's info block.
      label: provider?.info.label ?? IMAGE_FIELD_LABELS[envKey] ?? envKey,
      configured: Boolean(value),
      source: keySource(envKey),
      // Masked only: the raw key never leaves the server.
      masked: value ? maskKey(value) : null,
      placeholder: placeholderFor(envKey),
    };
  });
  res.json({ fields, storagePath: keysFilePath() });
});

function placeholderFor(envKey: AllowedKey): string {
  switch (envKey) {
    case 'OPENAI_API_KEY':
      return 'sk-...';
    case 'OPENROUTER_API_KEY':
      return 'sk-or-v1-...';
    case 'ANTHROPIC_API_KEY':
      return 'sk-ant-...';
    case 'GEMINI_API_KEY':
      return 'AIza...';
    case 'GROQ_API_KEY':
      return 'gsk_...';
    case 'HF_API_TOKEN':
      return 'hf_...';
    case 'CLOUDFLARE_API_TOKEN':
      return 'Cloudflare API token';
    case 'CLOUDFLARE_ACCOUNT_ID':
      return '12-hex-digit account id';
    case 'STABLE_HORDE_API_KEY':
      return 'Optional — leave empty for anonymous';
  }
}

/** Saves one or more keys. An empty value removes the key. */
app.put('/api/settings', async (req, res) => {
  const body = req.body ?? {};
  const updates: AllowedKey[] = [];

  for (const [key, value] of Object.entries(body)) {
    if (!ALLOWED_KEYS.includes(key as AllowedKey)) {
      return res.status(400).json({ error: `Unknown key: ${key}` });
    }
    if (typeof value !== 'string') {
      return res.status(400).json({ error: `${key} must be a string` });
    }
    if (value.length > 500) {
      return res.status(400).json({ error: `${key} is too long` });
    }
    updates.push(key as AllowedKey);
    await setKey(key as AllowedKey, value);
  }

  if (updates.length === 0) {
    return res.status(400).json({ error: 'No keys supplied' });
  }
  res.json({ saved: updates });
});

app.delete('/api/settings/:envKey', async (req, res) => {
  const { envKey } = req.params;
  if (!ALLOWED_KEYS.includes(envKey as AllowedKey)) {
    return res.status(400).json({ error: `Unknown key: ${envKey}` });
  }
  await clearKey(envKey as AllowedKey);
  res.json({ cleared: envKey });
});

app.delete('/api/settings', async (_req, res) => {
  await clearAllKeys();
  res.json({ cleared: 'all' });
});

function validate(body: any): string | null {
  if (!body || typeof body !== 'object') return 'Invalid request body';
  if (!providerRegistry()[body.provider as ProviderId]) return `Unknown provider: ${body.provider}`;
  if (!Array.isArray(body.messages) || body.messages.length === 0) return 'messages is required';
  if (!body.model) return 'model is required';
  if (body.tools !== undefined && typeof body.tools !== 'boolean') {
    // Arrays are the per-message selection: names, bounded in count and
    // length so a hostile body can't turn validation into a workload.
    if (
      !Array.isArray(body.tools) ||
      body.tools.length > 32 ||
      body.tools.some((t: unknown) => typeof t !== 'string' || !t || t.length > 64)
    ) {
      return 'tools must be a boolean or an array of tool names';
    }
  }
  // Each message's content must be a string or a well-formed content-parts
  // array. Anything else is rejected here rather than being forwarded to a
  // provider that would 400 on it anyway.
  for (const message of body.messages) {
    if (!message || typeof message !== 'object') return 'Invalid message';
    if (!['system', 'user', 'assistant'].includes(message.role)) return 'Invalid message role';
    if (parseContent(message.content) === null) return 'Invalid message content';
  }
  return null;
}

/** Rejects absurd payloads before they reach a paid provider. */
function checkUsage(body: any): string | null {
  const chars = body.messages.reduce((n: number, m: ChatMessage) => n + contentLength(m.content), 0);
  if (chars > 200_000) return 'Conversation is too long';
  if (body.messages.length > 200) return 'Too many messages';
  const oversize = checkAttachments(body.messages as ChatMessage[]);
  if (oversize.length) return oversize.map((p) => p.reason).join(' ');
  const temperature = body.temperature;
  if (temperature !== undefined && (typeof temperature !== 'number' || temperature < 0 || temperature > 2)) {
    return 'temperature must be between 0 and 2';
  }
  return null;
}

/**
 * Speech-to-text. The browser posts raw recorded audio; the model is chosen
 * from what the provider actually offers, never hardcoded.
 */
app.post('/api/transcribe', audioUpload, async (req, res) => {
  const audio = req.body as Buffer;
  const providerId = String(req.query.provider ?? '') as ProviderId;
  const provider = providerRegistry()[providerId];

  if (!provider?.transcribe || !provider.listSttModels) {
    return res.status(400).json({ error: 'This provider has no speech-to-text API' });
  }
  if (!provider.info.configured) {
    return res.status(400).json({ error: `${provider.info.label} needs an API key` });
  }
  if (!audio?.length) {
    return res.status(400).json({ error: 'No audio received' });
  }

  // Honour a requested model only while the provider still offers it; a stale
  // id is a guaranteed 400. Otherwise take the best available one.
  const requested = typeof req.query.model === 'string' ? req.query.model : '';
  const available = await provider.listSttModels();
  const model = available.includes(requested) ? requested : available[0];
  if (!model) {
    return res
      .status(400)
      .json({ error: `${provider.info.label} listed no speech-to-text models` });
  }

  // Share the stream cap: transcription is a paid, slow upstream call too.
  if (!acquireStream()) {
    return res.status(503).json({ error: 'Server is busy, try again shortly' });
  }

  const controller = new AbortController();
  res.on('close', () => controller.abort());

  try {
    const text = await provider.transcribe({
      model,
      audio,
      mimeType: req.get('content-type') ?? 'application/octet-stream',
      signal: controller.signal,
    });
    res.json({ text, model });
  } catch (err) {
    if (!controller.signal.aborted) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('[stt]', message);
      res.status(502).json({ error: message });
    }
  } finally {
    releaseStream();
  }
});

/**
 * Lists image-generation providers with their config state. Rendering via the
 * shared /api limiter: a local script can't hammer the settings queries.
 */
app.get('/api/image/providers', (_req, res) => {
  res.json({ providers: imageGenProvidersView() });
});

/**
 * Serves one cached generation. The name is validated against the cache's own
 * naming scheme, so this cannot walk out of the directory; the file is only
 * ever named by `cacheImage`, never by the request.
 */
app.get('/api/image/file/:name', (req, res) => {
  const name = req.params.name;
  const problem = cacheNameError(name);
  if (problem) return res.status(400).json({ error: problem });
  res.sendFile(imageCachePath(name), (err) => {
    if (err) res.status(404).end();
  });
});

const downloadControllers = new Map<string, AbortController>();

/**
 * Cancels an in-flight model download.
 *
 * The work is server-owned, so leaving the page never stops it — which is the
 * feature that makes reloading safe. An explicit cancel is the one way to say
 * "I changed my mind": the controller aborts every fetch, the per-file catches
 * turn it into "Cancelled" and release their locks, and the transfers list
 * updates on the next poll.
 */
app.post('/api/hf/download/cancel', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  const problem = validateRepoId(repo);
  if (problem) return res.status(400).json({ error: problem });
  const controller = downloadControllers.get(repo);
  if (!controller) {
    return res.json({ cancelled: false, reason: 'not running' });
  }
  controller.abort();
  res.json({ cancelled: true });
});

/**
 * Discards a paused or failed download's partial file, so the next attempt
 * starts from scratch rather than resuming. Used by the explicit Cancel action.
 */
app.post('/api/hf/download/discard', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  const file = typeof req.body?.file === 'string' ? req.body.file : '';
  const problem = validateRepoId(repo) ?? (file ? validateFilePath(file) : null);
  if (problem) return res.status(400).json({ error: problem });
  const dir = path.join(modelsDir(), repo.replace(/\//g, '--'));
  const dest = file ? path.join(dir, ...file.split('/')) : null;
  const tmp = dest ? `${dest}.part` : null;
  // A repo-level discard drops every .part under that repo's directory.
  try {
    if (tmp) {
      await rm(tmp, { force: true });
    } else {
      const { readdir } = await import('node:fs/promises');
      const entries = await readdir(dir, { withFileTypes: true }).catch(() => []);
      const stack = [dir];
      while (stack.length) {
        const d = stack.pop()!;
        let es;
        try { es = await readdir(d, { withFileTypes: true }); } catch { continue; }
        for (const e of es) {
          const full = path.join(d, e.name);
          if (e.isDirectory()) stack.push(full);
          else if (e.isFile() && e.name.endsWith('.part')) await rm(full, { force: true });
        }
      }
    }
  } catch {
    /* best effort */
  }
  res.json({ discarded: true });
});

/**
 * Generates one image. The browser waits on this synchronously — nothing here
 * streams binary to the page, it comes back as a base64 object so the client
 * can drop it into an assistant message like any attached image.
 */
app.post('/api/image/generate', async (req, res) => {
  const body = req.body ?? {};
  const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : '';
  if (!prompt) return res.status(400).json({ error: 'Prompt is required' });
  if (prompt.length > MAX_PROMPT_LENGTH) {
    return res.status(400).json({ error: `Prompt is too long (max ${MAX_PROMPT_LENGTH})` });
  }
  if (!imageProviderById(body.provider)) {
    return res.status(400).json({ error: `Unknown image provider: ${body.provider}` });
  }
  const imageProvider = imageProviderById(body.provider)!;
  if (!imageProvider.models.includes(String(body.model ?? ''))) {
    return res.status(400).json({ error: `Unknown model for ${imageProvider.label}` });
  }

  if (!acquireImageGen()) {
    return res.status(503).json({ error: 'Already generating an image — wait a moment' });
  }

  const controller = new AbortController();
  res.on('close', () => controller.abort());
  try {
    const image = await generateImage({
      providerId: imageProvider.id,
      model: String(body.model),
      prompt,
      width: Number(body.width) || 1024,
      height: Number(body.height) || 1024,
      signal: controller.signal,
    });
    // The bytes go to disk before they go to the page. A data URL is megabytes
    // of localStorage the moment a session is saved — which is how one image
    // used to take every saved session with it, and why generated pictures came
    // back as "[image — not saved]". The page gets a URL; the file stays.
    let cached = null;
    if (image.data) {
      try {
        cached = await cacheImage(image.data, image.mimeType);
      } catch (err) {
        console.warn('[image] could not cache the file:', err);
      }
    }
    res.json(cached ? { ...image, data: '', url: `/api/image/file/${cached}` } : image);
  } catch (err) {
    if (!controller.signal.aborted) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('[image]', message);
      res.status(502).json({ error: message });
    }
  } finally {
    releaseImageGen();
  }
});

/**
 * Generated images travel as assistant content parts, but no upstream chat
 * API accepts images in assistant turns — every provider would reject the
 * turn. Drop those parts here so the wire messages stay clean regardless of
 * what the client sent.
 */
function stripAssistantImages(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((m) =>
    m.role === 'assistant' && Array.isArray(m.content)
      ? { ...m, content: m.content.filter((p) => p.type !== 'image') }
      : m,
  );
}

/**
 * Agent tools: definitions, keyless flags, and on/off state for the Tools
 * page. Everything runs server-side without API keys.
 */
app.get('/api/tools', async (_req, res) => {
  res.json({ tools: await listTools() });
});

app.put('/api/tools/:name', async (req, res) => {
  const enabled = req.body?.enabled;
  if (typeof enabled !== 'boolean') {
    return res.status(400).json({ error: 'enabled must be a boolean' });
  }
  try {
    res.json({ tool: await setToolEnabled(req.params.name, enabled) });
  } catch (err) {
    res.status(404).json({ error: (err as Error).message });
  }
});

/**
 * Local Ollama models for the settings page. Listing and deleting are quick
 * JSON; pulling streams progress as SSE because layers take minutes. Delete
 * takes the name in the body rather than the path: model ids can legally
 * contain slashes, which Express would not route reliably.
 */
app.get('/api/ollama/models', async (_req, res) => {
  try {
    res.json({ models: await listLocalModels() });
  } catch (err) {
    res.status(502).json({ error: (err as Error).message });
  }
});

app.post('/api/ollama/delete', async (req, res) => {
  const name = typeof req.body?.name === 'string' ? req.body.name.trim() : '';
  const problem = validateModelName(name);
  if (problem) return res.status(400).json({ error: problem });
  try {
    await deleteLocalModel(name);
    res.json({ deleted: name });
  } catch (err) {
    res.status(502).json({ error: (err as Error).message });
  }
});

app.post('/api/ollama/pull', async (req, res) => {
  const name = typeof req.body?.name === 'string' ? req.body.name.trim() : '';
  const problem = validateModelName(name);
  if (problem) return res.status(400).json({ error: problem });
  // One pull at a time: concurrent pulls thrash the disk and the progress UI.
  const pullIdEarly = transferId('pull', name);
  if (isRunning(pullIdEarly)) return res.json({ attached: pullIdEarly });
  if (!acquirePull(name)) {
    return res.status(503).json({ error: 'Already pulling a model — wait a moment' });
  }

  /* The daemon comes up first, and before the stream opens. A pull is the one
     operation that cannot happen without it — `/api/pull` is served by the
     daemon, so with nothing listening the request fails before a byte moves and
     the error names a port rather than anything the user can act on. Starting
     it after the headers would leave the browser watching a stream that was
     already doomed, and starting it as a separate step in the UI would be the
     terminal trip this is meant to remove. */
  try {
    hookOllamaExit();
    if (!(await ollamaReachable())) await ensureOllamaRunning((line) => console.log('[ollama]', line));
  } catch (err) {
    releasePull();
    return res.status(503).json({ error: (err as Error).message });
  }

  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache, no-transform',
    connection: 'keep-alive',
    'x-accel-buffering': 'no',
  });

  // An Ollama pull is a multi-gigabyte download, so it belongs to the server,
  // not to the socket that happened to start it. Reloading used to abort it and
  // throw away everything fetched so far; now ollama keeps writing into its own
  // blobs directory and a reconnecting client just reads the progress back.
  const controller = new AbortController();
  let clientGone = false;
  res.on('close', () => {
    clientGone = true;
  });

  const send = (event: string, data: unknown) => {
    if (clientGone) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  const pullId = transferId('pull', name);
  startTransfer({ id: pullId, kind: 'pull', label: name, repo: 'ollama', file: name });

  try {
    let latest = 0;
    for await (const progress of pullModel(name, controller.signal)) {
      if (clientGone) continue;
      // Ollama reports completed/total bytes per layer; the largest denominator
      // seen so far is the best estimate of the whole pull.
      if (typeof progress.total === 'number' && progress.total > 0) {
        latest = Math.max(latest, progress.total);
        updateTransfer(pullId, {
          received: progress.completed ?? 0,
          total: latest,
        });
      }
      send('progress', progress);
    }
    finishTransfer(pullId, 'done');
    send('done', { model: name });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (!clientGone) console.error('[ollama-pull]', message);
    finishTransfer(pullId, 'error', message);
    send('error', { message });
  } finally {
    if (!clientGone) res.end();
    releasePull();
  }
});
/**
 * Hugging Face browser + downloader for the Models settings tab. Search and
 * detail proxy the public Hub API (the saved HF token rides along for gated
 * repos); downloads stream to ./models with SSE progress. Delete takes
 * repo+file in the body, same slash-in-name reasoning as the Ollama delete.
 */

/* ---------------- Optional local runtimes ---------------- */

/**
 * Optional Python runtimes for the models Ollama can't serve. One install may
 * run at a time; the lock is a plain flag because every caller is on this
 * process and a second concurrent pip into the same venv would corrupt it.
 */
let runtimeBusy = false;

let runtimesCache: unknown = null;
let runtimesCacheAt = 0;
const RUNTIMES_CACHE_TTL = 30_000;

/**
 * Frees the one-install lock and drops the cached status with it.
 *
 * Both halves belong together: a cached payload that says "busy" outlives the
 * install that set it, and the button that reads it stays disabled.
 */
function releaseInstallLock() {
  runtimeBusy = false;
  runtimesCache = null;
}

app.get('/api/runtimes', async (_req, res) => {
  if (runtimesCache && Date.now() - runtimesCacheAt < RUNTIMES_CACHE_TTL) {
    // `busy` is the one field that must never be cached. It is what disables
    // the install button, and a stale "not busy" from before an install started
    // re-enables it mid-install — so a second click lands on the server while
    // the first install still holds the lock, and the answer is "already
    // running on the server" for an install the user never started twice.
    const cached = runtimesCache as { runtimes?: { busy?: boolean } };
    res.json({
      ...cached,
      runtimes: { ...cached.runtimes, busy: runtimeBusy },
    });
    return;
  }
  const status = await runtimeStatus(runtimeBusy);
  const device = parseDevice(_req.query.device);
  const python = await venvPythonVersion();
  // Which groups cannot be installed here, and why. A group in the shared venv
  // is judged by that venv's interpreter; one with its own venv (Kokoro) is
  // judged by whether a compatible interpreter exists to build it on. Offering
  // the button otherwise would only produce a pip resolver failure after the
  // user had already waited through torch.
  const blocked = [];
  for (const g of RUNTIME_GROUPS) {
    const reason = await groupBlockedReason(g);
    if (reason) blocked.push({ id: g.id, label: g.label, maxPython: g.maxPython, reason });
  }
  const payload = {
    runtimes: status,
    groups: RUNTIME_GROUPS,
    python,
    blocked,
    // Surfaced up front so the UI can warn before a multi-hundred-MB download.
    warning: await deviceWarning(device),
    // What the machine actually has, so Auto is not a guess.
    gpu: await detectGpu(),
    // Runtime upgrades available for the installed build (e.g. a CPU build on a
    // machine with a GPU). The UI offers a one-click reinstall for these.
    upgrades: (await llamaCppUpgradeKind()) ? { 'llama-cpp': 'vulkan' } : {},
  };
  runtimesCache = payload;
  runtimesCacheAt = Date.now();
  res.json(payload);
});

/**
 * What each backend is doing right now.
 *
 * Reported, not configured: every backend answers from its own live state, so
 * this cannot claim a GPU is unused because the page that asked knows less
 * about it than the backend does.
 */
app.get('/api/backends', async (_req, res) => {
  try {
    const [status, servers] = await Promise.all([
      runtimeStatus(runtimeBusy),
      discoverLocalServers(true).catch(() => []),
    ]);
    const backends = await backendReports({
      python: { venv: status.venv, torch: status.torch },
      groups: RUNTIME_GROUPS.map((g) => ({ id: g.id, label: g.label })),
      servers: servers.map((s) => ({ id: s.id, label: s.label, baseUrl: s.baseUrl })),
    });
    res.json({ backends, at: Date.now() });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

/**
 * Runs scripts/install-gpu-drivers.sh and streams what it prints.
 *
 * Two things this deliberately does not do. It does not accept arguments from
 * the request — the only choice is whether to actually install, so there is no
 * path from a URL to an arbitrary shell. And it does not pretend the install
 * will succeed unattended: the script needs root, so with no password supplied
 * it runs, fails to elevate, and says so plainly rather than emitting coloured
 * output that ends in "sudo: a terminal is required".
 *
 * A password *may* be supplied in the body, which is the only place it can
 * arrive from: a query string would land in access logs, and argv would be
 * visible to every process on the machine through `ps`. It is handed straight
 * to `sudo -S` over the child's stdin, never written to disk, never logged, and
 * never echoed back — drivers.ts scrubs it out of the output stream as a second
 * line of defence.
 *
 * The cross-origin check on writes is what makes this safe to expose: without
 * it, any page in the user's browser could POST here and drive the installer.
 */
app.post('/api/drivers/update', async (req, res) => {
  const apply = req.body?.apply === true;
  const raw = req.body?.password;
  // A password longer than this is not a password; refusing keeps an
  // accidentally pasted paragraph out of a subprocess pipe.
  const password = typeof raw === 'string' && raw.length > 0 && raw.length <= 200 ? raw : undefined;
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('X-Accel-Buffering', 'no');

  let clientGone = false;
  res.on('close', () => {
    clientGone = true;
  });
  const send = (event: string, data: unknown) => {
    if (clientGone) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  send('log', {
    message: apply
      ? password
        ? 'Running the driver installer with your password. This changes system packages.'
        : 'Running the driver installer. This changes system packages.'
      : 'Dry run — showing what would be installed.',
  });

  await runDriverInstaller({
    apply,
    password,
    onLog: (line) => send('log', { line }),
    // The client hanging up must not orphan a package manager mid-install.
    onNeedsPrivilege: (cmd) => send('privilege', { command: cmd }),
  });

  send('done', { apply });
});

app.post('/api/runtimes/install', async (req, res) => {
  const device = parseDevice(req.body?.device);
  const ids: unknown = req.body?.groups;
  // Only an installer that genuinely needs root ever reads this. It arrives in
  // the body rather than the query string (which lands in access logs) or argv
  // (visible to every process through `ps`), goes straight to `sudo -S` on the
  // child's stdin, and is never logged or echoed back.
  const rawPass = req.body?.password;
  const password =
    typeof rawPass === 'string' && rawPass.length > 0 && rawPass.length <= 200 ? rawPass : undefined;
  if (!Array.isArray(ids) || ids.length === 0) {
    return res.status(400).json({ error: 'Pick at least one runtime to install.' });
  }
  // Only ids from the fixed table; anything unknown is a hard error rather than
  // being passed along to pip.
  const groups = ids.map(groupById);
  if (groups.some((g) => g === null)) {
    return res.status(400).json({ error: 'Unknown runtime requested.' });
  }

  // Refuse up front when a requested group cannot be installed here — either
  // the interpreter it would run on is too new, or its own environment has no
  // compatible interpreter to be built on.
  for (const g of groups as RuntimeGroup[]) {
    const reason = await groupBlockedReason(g);
    if (reason) {
      return res.status(400).json({ error: reason, blocked: g.id });
    }
  }

  const known = transferId('install', (ids as string[]).slice().sort().join(','));
  if (runtimeBusy) {
    // Same install already running server-side: point the new client at it
    // rather than a 409 that would read as a failure after a mere reload.
    return isRunning(known)
      ? res.json({ attached: known })
      : res.status(409).json({ error: 'An install is already running.' });
  }
  runtimeBusy = true;
  const installId = transferId('install', (ids as string[]).slice().sort().join(','));
  startTransfer({
    id: installId,
    kind: 'install',
    label: (ids as string[]).join(' + '),
  });
  res.setHeader('content-type', 'text/event-stream');
  res.setHeader('cache-control', 'no-cache');
  res.setHeader('connection', 'keep-alive');
  // A reload mid-install leaves pip running and the socket dead. Writing to it
  // raises an unhandled 'error' on the response, which takes the whole server
  // down — so every write checks first, and the install is never cancelled.
  const clientGone = () => res.writableEnded || res.destroyed;
  const send = (event: string, data: unknown) => {
    if (clientGone()) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    const warning = await deviceWarning(device);
    if (warning) send('progress', warning);
    await installRuntimes({
      groups: groups as NonNullable<(typeof groups)[0]>[],
      device,
      password,
      // Report the need rather than failing on it: the browser only asks for a
      // password when an installer really is stuck on one, which is the same
      // contract the driver installer has.
      onNeedsPrivilege: (command) => send('privilege', { command }),
      onProgress: (line) => {
        // pip's log carries its own percentage; surface it on the downloads
        // page so an install looks the same whether or not you reloaded.
        const pct = /(\d{1,3})%/.exec(line);
        if (pct) updateTransfer(installId, { percent: Math.min(100, Number(pct[1])) });
        updateTransfer(installId, { label: line.trim().slice(0, 80) });
        send('progress', line);
      },
    });
    const status = await runtimeStatus(false);
    // Released before the completion is announced, not after. The browser
    // re-reads the runtime status the moment it sees `done` — that read is what
    // puts the Install button back — and while the lock was still held it read
    // "busy" and left the button saying "Installing…" with nothing left
    // running. The window was milliseconds wide, which is why it looked random.
    releaseInstallLock();
    finishTransfer(installId, 'done');
    send('done', { runtimes: status });
  } catch (err) {
    releaseInstallLock();
    finishTransfer(installId, 'error', (err as Error).message);
    send('error', { message: (err as Error).message });
  } finally {
    // Idempotent safety net: the lock must never outlive the install, whatever
    // path it left by.
    releaseInstallLock();
    if (!clientGone()) res.end();
  }
});

app.post('/api/runtimes/remove', async (_req, res) => {
  if (runtimeBusy) return res.status(409).json({ error: 'An install is already running.' });
  runtimeBusy = true;
  try {
    await removeRuntimes();
    res.json({ runtimes: await runtimeStatus(false) });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  } finally {
    runtimeBusy = false;
    runtimesCache = null;
  }
});

/**
 * Removes one runtime group rather than the whole environment.
 *
 * Shares the install lock: pip into the same venv from two processes is exactly
 * the corruption the lock exists to prevent, and removing torch while
 * transformers is being installed is a mess neither operation can report
 * cleanly. Streams pip's output the same way an install does, so the user sees
 * what happened instead of a spinner.
 */
app.post('/api/runtimes/remove-group', async (req, res) => {
  const id = typeof req.body?.id === 'string' ? req.body.id : '';
  // Only ids from the fixed table; anything else never reaches a command line.
  if (!groupById(id)) return res.status(400).json({ error: 'Unknown runtime requested.' });
  if (runtimeBusy) return res.status(409).json({ error: 'An install is already running.' });

  runtimeBusy = true;
  res.setHeader('content-type', 'text/event-stream; charset=utf-8');
  res.setHeader('cache-control', 'no-cache, no-transform');
  res.setHeader('connection', 'keep-alive');
  let clientGone = false;
  res.on('close', () => {
    clientGone = true;
  });
  const send = (event: string, data: unknown) => {
    if (!clientGone) res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    const result = await removeRuntimeGroup(id, (line) => send('log', { line }));
    send('done', { ...result, runtimes: await runtimeStatus(false) });
  } catch (err) {
    send('error', { message: (err as Error).message });
  } finally {
    runtimeBusy = false;
    runtimesCache = null;
    if (!clientGone) res.end();
  }
});

/**
 * Everything this app is currently running, and what it is using.
 *
 * Two kinds of work, which are genuinely different and are kept apart rather
 * than flattened into one list of "processes":
 *
 *  - Ollama models. Not this app's children at all — a separate daemon holding
 *    memory on a keep-alive timer. There is no pid to signal, so the action
 *    that actually frees the GPU is unloading the model.
 *  - Python jobs this app spawned. Real children of this process, so they can
 *    be signalled, which matters because image generation is allowed to run for
 *    an hour and there was previously no way to stop one from the UI.
 *
 * A model shows the runtimes its script imports rather than pretending each
 * runtime is its own service: PyTorch is a library inside a process, and
 * ONNX Runtime is not used by any bundled runner, so it never appears at all.
 */
app.get('/api/processes', async (_req, res) => {
  let ollama: { ok: boolean; models: unknown[]; detail: string } = {
    ok: false,
    models: [],
    detail: 'not checked',
  };
  try {
    // A short leash rather than the daemon client's usual 30s: this panel
    // refreshes every second, and while Ollama is loading a model /api/ps can
    // sit unanswered for far longer than that. Jobs must keep rendering and
    // the panel must answer promptly; "busy" is a state the UI can say out
    // loud, a hanging panel is not.
    ollama = {
      ok: true,
      models: await listRunningModels(AbortSignal.timeout(3000)),
      detail: 'daemon up',
    };
  } catch (err) {
    // "No daemon" is the normal state for someone who only uses cloud
    // providers, so it is reported as an answer rather than an error.
    const e = err as Error;
    const timedOut = e.name === 'TimeoutError' || e.name === 'AbortError';
    ollama = {
      ok: false,
      models: [],
      detail: timedOut
        ? 'Ollama did not answer within 3 s — it is likely loading or replying. Loaded models appear again once it answers.'
        : e.message,
    };
  }
  // Memory is read at report time, not stored on the entries: it is a reading
  // of the moment, and the manager polls often enough that a stale figure
  // would be worth less than none.
  const { active, recent } = listProcesses();
  res.json({
    active: await Promise.all(
      active.map(async (p) => ({ ...p, memoryBytes: await processMemoryBytes(p.pid) })),
    ),
    recent,
    ollama,
    // Total system RAM, so a per-job memory bar can be a fraction of something.
    ramTotalBytes: os.totalmem(),
  });
});

/**
 * Stops one job this app started, by the id the server minted for it.
 *
 * Ids only: a route that accepted a pid would let any page that can reach this
 * port kill any process on the machine, and there is no login to stop it. The
 * id comes from our own registry, so the blast radius is exactly the jobs we
 * spawned.
 */
app.post('/api/processes/:id/stop', async (req, res) => {
  const stopped = await stopProcess(req.params.id);
  if (!stopped) return res.status(404).json({ error: 'That job is no longer running' });
  res.json({ stopped });
});

app.post('/api/processes/stop-all', async (_req, res) => {
  res.json({ stopped: await stopAllProcesses() });
});

/**
 * Releases a loaded Ollama model.
 *
 * Deliberately unloads rather than kills the daemon: the model is what is
 * holding the GPU, and letting go of it is reversible. Stopping the daemon
 * itself would take down every model at once and break this app's own chat, so
 * that is not offered here.
 */
app.post('/api/ollama/unload', async (req, res) => {
  const name = typeof req.body?.name === 'string' ? req.body.name.trim() : '';
  const problem = validateModelName(name);
  if (problem) return res.status(400).json({ error: problem });
  try {
    await unloadModel(name);
    res.json({ unloaded: name });
  } catch (err) {
    res.status(502).json({ error: (err as Error).message });
  }
});

/* ---------------- Local inference ---------------- */

/**
 * Streams the Python log while the script runs, then returns its result.
 * Always closes the response: a finished SSE stream that is never ended leaves
 * the client waiting on a socket that will not deliver EOF.
 */
function withPythonLog(res: import('express').Response, onResult: (send: (e: string, d: unknown) => void) => Promise<unknown>) {
  res.setHeader('content-type', 'text/event-stream');
  res.setHeader('cache-control', 'no-cache');
  res.setHeader('connection', 'keep-alive');
  const send = (event: string, data: unknown) =>
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  onResult(send)
    .catch((err: Error) => send('error', { message: err.message }))
    .finally(() => res.end());
}

/**
 * Stages a browser-picked audio file for transcription. The runner deliberately
 * only accepts a path the downloader wrote, so an upload has to land in the same
 * tree to be usable. The name is ours, never the client's.
 */
/**
 * Identifies an uploaded recording by its magic bytes.
 *
 * Falls back to WebM, which is what `MediaRecorder` produces in every browser
 * that matters, so an unrecognised stream is still worth saving.
 */
function sniffAudioExt(buf: Buffer): string {
  if (buf.length >= 12) {
    const riff = buf.toString('ascii', 0, 4);
    const webm = buf.readUInt32BE(0) === 0x1a45dfa3;
    const ftyp = buf.toString('ascii', 4, 8);
    const ogg = buf.toString('ascii', 0, 4) === 'OggS';
    const id3 = buf.toString('ascii', 0, 3) === 'ID3';
    const flac = buf.toString('ascii', 0, 4) === 'fLaC';
    if (riff === 'RIFF' || riff === 'RIFX') return 'wav';
    if (webm) return 'webm';
    if (ftyp === 'ftyp') return 'mp4';
    if (ogg || id3) return 'mp3';
    if (flac) return 'flac';
  }
  return 'webm';
}

app.post('/api/infer/upload', audioUpload, async (req, res) => {
  const body = req.body as Buffer;
  if (!body?.length) return res.status(400).json({ error: 'No audio received.' });
  const dir = path.resolve(process.env.HF_MODELS_DIR ?? './models', '_uploads');
  // The extension follows the bytes, not the caller. A browser recording is
  // WebM/Opus, but a file picker can hand over WAV or MP4, and a `.webm` name on
  // a WAV file is a lie the decoders have to be taught to ignore.
  const sniffed = sniffAudioExt(body);
  const name = `clip-${Date.now()}.${sniffed}`;
  const dest = path.join(dir, name);
  // Belt and braces: the name is generated, but confirm it stays in the tree.
  if (!path.resolve(dest).startsWith(path.resolve(path.dirname(dir)) + path.sep)) {
    return res.status(400).json({ error: 'Bad upload name' });
  }
  try {
    await mkdir(dir, { recursive: true });
    await writeFile(dest, body);
    res.json({ file: `_uploads/${name}`, bytes: body.length });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

app.post('/api/infer/stt', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  // `file` is the model, `audio` is what to transcribe. Conflating them is what
  // made this endpoint hand the uploads folder to the model loader.
  const file = typeof req.body?.file === 'string' ? req.body.file : '';
  const audio = typeof req.body?.audio === 'string' ? req.body.audio : '';
  const device = parseDevice(req.body?.device);
  const problem = validateRepoId(repo) ?? validateFilePath(file) ?? validateFilePath(audio);
  if (problem) return res.status(400).json({ error: problem });
  withPythonLog(res, async (send) => {
    const r = await transcribe({ repo, file, audio, device, onLog: (l) => send('log', l.trim()) });
    if (!r.ok) send('error', { message: String(r.data?.error ?? 'Transcription failed.') });
    else send('done', r.data);
  });
});

app.post('/api/infer/tts', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  const file = typeof req.body?.file === 'string' ? req.body.file : '';
  const text = typeof req.body?.text === 'string' ? req.body.text : '';
  const device = parseDevice(req.body?.device);
  const problem = validateRepoId(repo) ?? validateFilePath(file);
  if (problem) return res.status(400).json({ error: problem });
  if (!text.trim()) return res.status(400).json({ error: 'Nothing to say.' });
  if (text.length > 20_000) return res.status(400).json({ error: 'Text is too long (20k max).' });
  const voice = typeof req.body?.voice === 'string' ? req.body.voice : 'af_heart';
  const speed = Math.min(Math.max(Number(req.body?.speed) || 1, 0.5), 2);
  withPythonLog(res, async (send) => {
    const { result } = await speak({
      repo, file, voice, speed, device, text,
      onLog: (l) => send('log', l.trim()),
    });
    if (!result.ok) send('error', { message: String(result.data?.error ?? 'Synthesis failed.') });
    else send('done', result.data);
  });
});

app.post('/api/infer/image', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  const file = typeof req.body?.file === 'string' ? req.body.file : '';
  const prompt = typeof req.body?.prompt === 'string' ? req.body.prompt : '';
  const device = parseDevice(req.body?.device);
  const problem = validateRepoId(repo) ?? validateFilePath(file);
  if (problem) return res.status(400).json({ error: problem });
  if (!prompt.trim()) return res.status(400).json({ error: 'Prompt is empty.' });
  if (prompt.length > MAX_PROMPT_LENGTH) {
    return res.status(400).json({ error: `Prompt is too long (${MAX_PROMPT_LENGTH} max).` });
  }
  const steps = Math.min(Math.max(Number(req.body?.steps) || 4, 1), 50);
  const width = Math.min(Math.max(Number(req.body?.width) || 512, 64), 1024);
  const height = Math.min(Math.max(Number(req.body?.height) || 512, 64), 1024);
  withPythonLog(res, async (send) => {
    const { result } = await generateImageLocal({
      repo, file, prompt, steps, width, height, device,
      negative: typeof req.body?.negative === 'string' ? req.body.negative : '',
      onLog: (l) => send('log', l.trim()),
    });
    if (!result.ok) send('error', { message: String(result.data?.error ?? 'Generation failed.') });
    else send('done', result.data);
  });
});

/** Serves a generated file. Only paths the runners themselves produced. */
app.get('/api/infer/out/:name', (req, res) => {
  const name = req.params.name;
  // A bare filename: no separators, so this cannot climb out of the out dir.
  if (!/^[\w.-]+\.(png|wav|jpg|jpeg)$/.test(name)) return res.status(400).json({ error: 'Bad name' });
  const base = path.resolve(process.env.RUNTIME_OUT_DIR ?? './.runtimes/out');
  const full = path.join(base, name);
  if (!path.resolve(full).startsWith(base + path.sep)) return res.status(400).json({ error: 'Bad path' });
  res.sendFile(full, (err) => {
    if (err) res.status(404).end();
  });
});

/**
 * Live CPU / RAM / GPU and backend reachability for the chat sidebar's resource
 * box. Polled, so it must stay cheap: no shell-outs unless nvidia-smi is
 * actually present, and the GPU counters are cached just under the poll
 * interval so a burst of tabs shares one reading without the meters freezing
 * between them.
 */
app.get('/api/resources', async (_req, res) => {
  try {
    res.json({ resources: await resourceReport() });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

/**
 * What is downloading right now, and what just finished. This is what makes a
 * reload safe: the page re-reads this instead of assuming nothing is running.
 */
app.get('/api/transfers', (_req, res) => {
  res.json(listTransfers());
});

/**
 * The curated "what can I actually run locally" catalog. Static data, so this
 * is a plain GET with no upstream call — it renders even when the Hub is
 * unreachable, which is exactly when you most want to know what to fetch.
 */
app.get('/api/catalog', (_req, res) => {
  res.json({ catalog: CATALOG });
});

app.get('/api/hf/search', async (req, res) => {
  const query = typeof req.query.q === 'string' ? req.query.q : '';
  if (query.length > 200) return res.status(400).json({ error: 'Search query is too long' });
  // An empty query lists the Hub's general listing; the browser asks for each
  // task separately as well, so no category depends on this one.
  const sort = typeof req.query.sort === 'string' ? req.query.sort : undefined;
  const pipeline = typeof req.query.pipeline === 'string' ? req.query.pipeline : undefined;
  const limit = req.query.limit !== undefined ? Number(req.query.limit) : undefined;
  // The Hub pages at 100. The old ceiling of 25 matched a three-column grid
  // and is what kept the browse page down to a dozen models.
  if (limit !== undefined && !(Number.isInteger(limit) && limit >= 1 && limit <= 100)) {
    return res.status(400).json({ error: 'limit must be 1-100' });
  }
  try {
    res.json({ models: await searchModels(query, { sort, limit, pipeline }) });
  } catch (err) {
    res.status(502).json({ error: (err as Error).message });
  }
});

app.get('/api/hf/model', async (req, res) => {
  const repo = typeof req.query.repo === 'string' ? req.query.repo : '';
  const problem = validateRepoId(repo);
  if (problem) return res.status(400).json({ error: problem });
  try {
    res.json({ model: await modelDetail(repo) });
  } catch (err) {
    res.status(502).json({ error: (err as Error).message });
  }
});

/**
 * The author's avatar, proxied. CSP pins img-src to 'self', so the browser
 * can't load cdn-avatars.huggingface.co itself; this pulls the image once per
 * author and caches it on disk. An author with no avatar answers 404, which
 * the card reads as "no image" and falls back to its generated tile.
 */
app.get('/api/hf/avatar/:author', async (req, res) => {
  try {
    const body = await fetchAvatar(req.params.author);
    const etag = avatarEtag(body);
    res.setHeader('cache-control', 'public, max-age=604800');
    res.setHeader('etag', etag);
    res.setHeader('content-type', imageMime(body) ?? 'application/octet-stream');
    if (req.headers['if-none-match'] === etag) return res.status(304).end();
    res.setHeader('content-length', String(body.length));
    res.end(body);
  } catch {
    // An <img> that 404s fires its own error handler, so the body is irrelevant;
    // only the status matters here.
    res.setHeader('cache-control', 'no-store');
    res.status(404).end();
  }
});

app.get('/api/hf/files', async (_req, res) => {
  // Opportunistic: clears out partials abandoned by cancelled downloads, and
  // the repo directories they leave behind.
  await pruneStalePartials().catch(() => {});
  const files = await listDownloads();
  // GGUF chat models are runnable from the chat page, and the id that names
  // them (`Qwen3-4B-Q4_K_M`) is derived here rather than re-derived in the
  // browser — the collision rule that qualifies two same-named files with their
  // repo has to have exactly one implementation, or a "Chat" button leads to a
  // model the server cannot resolve.
  const gguf = new Map((await ggufModels().catch(() => [])).map((m) => [`${m.repo}\n${m.file}`, m]));
  const tf = (await transformersChatModels().catch(() => [])).filter((m) => m.chat);
  const tfRoots = new Set(tf.map((m) => m.root));
  const tfByRepo = new Map(tf.map((m) => [m.repo, m]));
  res.json({
    files: files.map((f) => {
      const g = gguf.get(`${f.repo}\n${f.file}`);
      if (g?.chat) return { ...f, llama: { provider: 'llamacpp', model: g.id } };
      if (tfRoots.has(f.root) && tfByRepo.has(f.repo)) {
        return { ...f, transformers: { provider: 'transformers-chat', model: f.repo } };
      }
      return f;
    }),
  });
});

app.post('/api/hf/files/delete', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  const file = typeof req.body?.file === 'string' ? req.body.file : '';
  if (!repo) return res.status(400).json({ error: 'Model id is required' });
  const fileProblem = validateFilePath(file);
  if (fileProblem) return res.status(400).json({ error: fileProblem });
  try {
    await deleteDownload(repo, file);
    // If the file we just removed is the model a running llama-server holds,
    // that server is now pointing at nothing: stop it so the next request
    // starts a fresh one instead of failing on a missing file.
    const loaded = llamaServerState().model;
    const deletedName = path.basename(file).replace(/\.gguf$/i, '');
    if (loaded && loaded === deletedName) await stopLlamaServer();
    res.json({ deleted: { repo, file } });
  } catch (err) {
    const message = (err as Error).message;
    if (/no such file|ENOENT/i.test(message)) {
      return res.status(404).json({ error: 'No such downloaded file' });
    }
    res.status(500).json({ error: message });
  }
});

app.post('/api/hf/files/delete-many', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  const files = Array.isArray(req.body?.files)
    ? req.body.files.filter((f: unknown): f is string => typeof f === 'string')
    : [];
  const problem = validateRepoId(repo) ?? validateFilePath(files[0] ?? '');
  if (problem) return res.status(400).json({ error: problem });
  // One request for the whole set. The row used to fire one request per file,
  // so deleting a 58-file model was fifty-eight round trips and looked like a
  // hang for most of a minute before anything happened.
  let removed = 0;
  const failed = [];
  for (const file of files) {
    try {
      await deleteDownload(repo, file);
      removed += 1;
    } catch {
      failed.push(file);
    }
  }
  res.json({ removed, failed });
});

app.post('/api/hf/repo/delete', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  const problem = validateRepoId(repo);
  if (problem) return res.status(400).json({ error: problem });
  // Only files this app owns (the shared HF cache is never touched here).
  const rows = (await listDownloads()).filter((d) => d.repo === repo && d.location === 'models');
  if (!rows.length) {
    return res.status(404).json({ error: `${repo} has no downloaded files here.` });
  }
  let removed = 0;
  const failed = [];
  for (const r of rows) {
    try { await deleteDownload(repo, r.file); removed++; }
    catch { failed.push(r.file); }
  }
  res.json({ removed, failed, total: rows.length });
});

app.post('/api/hf/cache/delete', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  const problem = validateRepoId(repo);
  if (problem) return res.status(400).json({ error: problem });
  try {
    await deleteCacheRepo(repo);
    res.json({ removed: { repo } });
  } catch (err) {
    const message = (err as Error).message;
    if (/not in the Hugging Face cache/i.test(message)) {
      return res.status(404).json({ error: message });
    }
    res.status(500).json({ error: message });
  }
});

/**
 * The files a repository needs to be usable, and their total size.
 *
 * This is what makes "download the model" mean one action rather than a hunt
 * through a file list for the parts a loader will insist on.
 */
app.get('/api/hf/repo-files', async (req, res) => {
  const repo = typeof req.query.repo === 'string' ? req.query.repo : '';
  const problem = validateRepoId(repo);
  if (problem) return res.status(400).json({ error: problem });
  try {
    const files = await requiredFiles(repo);
    const total = files.reduce((a, f) => a + (f.size ?? 0), 0);
    // What is already here, so the button can say whether it has work to do.
    // Without this a fully-downloaded model offered an enabled "Download model"
    // that silently did nothing on click, which reads as a broken button.
    const have = new Set((await listDownloads()).map((d) => `${d.repo}::${d.file}`));
    const missing = files.filter((f) => !have.has(`${repo}::${f.path}`));
    res.json({
      files,
      total,
      count: files.length,
      present: files.length - missing.length,
      missing: missing.length,
      missingBytes: missing.reduce((a, f) => a + (f.size ?? 0), 0),
    });
  } catch (err) {
    res.status(502).json({ error: (err as Error).message });
  }
});

/**
 * Downloads every file a repository needs, in sequence, as one job.
 *
 * Each file is registered in the transfer registry exactly as a single download
 * is, so progress, cancellation and survival across a browser reload all work
 * unchanged. A file that fails is reported and the rest continue, because a
 * partial model with a clear list of what failed beats one silent abort.
 */
app.post('/api/hf/download-repo', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  const problem = validateRepoId(repo);
  if (problem) return res.status(400).json({ error: problem });

  const controller = new AbortController();
  downloadControllers.set(repo, controller);

  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache, no-transform',
    connection: 'keep-alive',
    'x-accel-buffering': 'no',
  });
  let clientGone = false;
  res.on('close', () => {
    clientGone = true;
    controller.abort();
  });
  const send = (event: string, data: unknown) => {
    if (clientGone) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    send('log', { message: `Reading the file list for ${repo}…` });
    const files = await requiredFiles(repo);
    if (!files.length) {
      send('error', { message: 'That repository published no downloadable files.' });
      return;
    }

    const total = files.reduce((a, f) => a + (f.size ?? 0), 0);
    let done = 0;
    const failed: string[] = [];

    // Already on disk? Leave it alone. Re-downloading a whole model because
    // one of its twenty files is missing means re-fetching the nineteen that
    // are already correct — gigabytes, every time.
    const already = new Set((await listDownloads()).map((d) => `${d.repo}::${d.file}`));
    const missing = files.filter((f) => !already.has(`${repo}::${f.path}`));
    const skipped = files.length - missing.length;
    if (skipped) send('skipped', { count: skipped, bytes: 0 });
    if (!missing.length) {
      send('done', { repo, files: 0, failed: [], total, alreadyPresent: skipped });
      return;
    }
    // `bytes` is the size of what is actually being fetched, which is not
    // `total`: `total` covers every required file including the ones already on
    // disk, so a percentage taken against it can never reach 100 once anything
    // has been skipped. The client needs the real denominator.
    const needBytes = missing.reduce((a, f) => a + (f.size ?? 0), 0);
    send('manifest', {
      repo,
      count: missing.length,
      total,
      bytes: needBytes,
      // Some Hub files publish no size at all. Say whether the denominator is
      // trustworthy rather than showing a percentage built on nothing.
      sized: missing.every((f) => typeof f.size === 'number' && f.size > 0),
      skipped,
    });

    // A small worker pool instead of a serial loop. The old code fetched one
    // file at a time, which turned a 12-file model into twelve sequential
    // downloads; the per-file lock above is what makes the pool safe, since
    // two workers never take the same key. Three at a time is the point where
    // Hugging Face's CDN stops getting faster and starts getting annoyed.
    const CONCURRENCY = 3;
    let cursor = 0;
    const workers = Array.from({ length: Math.min(CONCURRENCY, missing.length) }, async () => {
      while (true) {
        const i = cursor++;
        if (i >= missing.length) return;
        const f = missing[i];
      // A dead socket must not stop the work. Single-file downloads were
      // detached from their connection for exactly this reason, and the whole-model
      // route was missed: navigating away, or a reload, killed a multi-file
      // download partway and left a directory that looked complete but was not.
      // Events are simply not sent once the client is gone.
      if (!acquireDownload(`${repo}/${f.path}`)) {
        // Another transfer already holds this exact file. Skipping it means it
        // is NOT on disk when we finish, so it belongs in the failed list —
        // silently omitting it reported a complete model that was not one.
        failed.push(f.path);
        send('file-failed', {
          path: f.path,
          message: 'Already being downloaded elsewhere',
        });
        continue;
      }
      const id = transferId('download', `${repo}/${f.path}`);
      startTransfer({ id, kind: 'download', label: `${repo}/${f.path}`, repo, file: f.path });
      try {
        // One flaky socket should not cost the whole model. A multi-gigabyte
        // tree is a long sequence of transfers, and a single "terminated" left
        // a model that was three-quarters there and quietly unusable.
        let fileDone = null;
        let lastMessage = '';
        for (let attempt = 1; attempt <= 3; attempt++) {
          try {
            fileDone = await downloadFile(repo, f.path, {
              signal: controller.signal,
              onProgress: (p) => {
                updateTransfer(id, { received: p.received, total: p.total });
                send('progress', { path: f.path, index: done, of: missing.length, ...p });
              },
            });
            break;
          } catch (err) {
            lastMessage = err instanceof Error ? err.message : String(err);
            if (attempt < 3) {
              send('log', { message: `${f.path}: ${lastMessage} — retrying (${attempt}/2)` });
              await new Promise((r) => setTimeout(r, 400 * attempt));
            }
          }
        }
        if (!fileDone) throw new Error(lastMessage || 'download failed');
        finishTransfer(id, 'done');
        done += 1;
        send('file-done', { path: f.path, ...fileDone });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        finishTransfer(id, 'error', message);
        failed.push(f.path);
        send('file-failed', { path: f.path, message });
        if (!clientGone) console.error('[hf-download-repo]', message);
      } finally {
        releaseDownload(`${repo}/${f.path}`);
      }
      }
    });
    await Promise.all(workers);

    send('done', { repo, files: done, failed, total });
  } catch (err) {
    send('error', { message: (err as Error).message });
  } finally {
    downloadControllers.delete(repo);
    if (!clientGone) res.end();
  }
});

app.post('/api/hf/download', async (req, res) => {
  const repo = typeof req.body?.repo === 'string' ? req.body.repo : '';
  const file = typeof req.body?.file === 'string' ? req.body.file : '';
  const problem = validateRepoId(repo) ?? validateFilePath(file);
  if (problem) return res.status(400).json({ error: problem });
  const existingId = transferId('download', `${repo}/${file}`);
  if (isRunning(existingId)) {
    // Already going, server-side. Re-point this client at it rather than
    // refusing or, worse, starting a duplicate transfer.
    return res.json({ attached: existingId });
  }
  if (!acquireDownload(`${repo}/${file}`)) {
    return res.status(503).json({ error: 'Already downloading that file — wait a moment' });
  }

  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache, no-transform',
    connection: 'keep-alive',
    'x-accel-buffering': 'no',
  });

  // Deliberately NOT aborted when the client goes away. A reload used to kill
  // the transfer and strand its .part; now the download is server-owned and a
  // client that reconnects just re-reads the progress.
  const controller = new AbortController();
  downloadControllers.set(repo, controller);
  let clientGone = false;
  res.on('close', () => {
    clientGone = true;
  });

  const send = (event: string, data: unknown) => {
    if (clientGone) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  const id = transferId('download', `${repo}/${file}`);
  const transfer = startTransfer({ id, kind: 'download', label: file, repo, file });

  try {
    const done = await downloadFile(repo, file, {
      signal: controller.signal,
      onProgress: (p) => {
        updateTransfer(id, { received: p.received, total: p.total });
        send('progress', { file, ...p });
      },
    });
    finishTransfer(id, 'done');
    send('done', done);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    // A dead socket is not a failed download.
    if (!clientGone) console.error('[hf-download]', message);
    finishTransfer(id, 'error', message);
    send('error', { message });
  } finally {
    downloadControllers.delete(repo);
    if (!clientGone) res.end();
    releaseDownload(`${repo}/${file}`);
    void transfer;
  }
});

/**
 * Streams a chat completion over Server-Sent Events.
 * The client aborts the fetch to stop generation, which aborts the upstream
 * provider request via the shared AbortSignal.
 */
app.post('/api/chat', async (req, res) => {
  const problem = validate(req.body) ?? checkUsage(req.body);
  if (problem) return res.status(400).json({ error: problem });

  // Tools are per-message: the client names the exact subset it authorized
  // (true is the legacy "everything enabled" flag). Resolving before the
  // stream starts turns a stale or invented id into a clean 400 instead of a
  // stream that dies mid-way, and the resolved list is the request's whole
  // permission set — nothing outside it can execute.
  const selection = resolveToolSelection(req.body.tools, await enabledTools());
  if (selection.error) return res.status(400).json({ error: selection.error });
  const askedForTools =
    req.body.tools === true || (Array.isArray(req.body.tools) && req.body.tools.length > 0);

  // Bound simultaneous generations so one client can't pin the CPU or
  // exhaust a rate-limited provider quota.
  if (!acquireStream()) {
    return res.status(503).json({ error: 'Server is busy, try again shortly' });
  }

  const provider = providerRegistry()[req.body.provider as ProviderId];
  let messages: ChatMessage[] = stripAssistantImages(req.body.messages);

  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache, no-transform',
    connection: 'keep-alive',
    // Disable proxy buffering so tokens arrive immediately.
    'x-accel-buffering': 'no',
  });

  const controller = new AbortController();
  // Listen on the response, not the request: 'close' on req fires as soon as
  // the body is consumed, which would abort every stream immediately.
  res.on('close', () => controller.abort());

  const send = (event: string, data: unknown) =>
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  // A local backend may need bringing up before it can answer at all — llama.cpp
  // serves one model, so a request naming a different one starts a server with
  // it loaded. The progress lines become transcript notices, so a multi-gigabyte
  // load reads as work in progress, and a failure names what went wrong before
  // any "thinking" began.
  if (provider.prepare) {
    try {
      await provider.prepare(req.body.model, (line) => send('notice', { message: line }));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`[${provider.info.id}]`, message);
      if (!res.writableEnded) send('error', { message });
      res.end();
      releaseStream();
      return;
    }
  }

  // Agent mode: the model may call tools (web search, page fetch, wiki), and
  // each call streams as its own event so the transcript can show the work.
  // A request for tools degrades to plain chat — with a notice line, never an
  // error — when there is nothing to call with or the provider can't call, so
  // asking for tools never turns a working chat into a failure. Both paths
  // also state the tool situation up front (withToolGuidance): offered means
  // "call one when asked, never write pretend calls"; unavailable means
  // "never narrate a search that did not run".
  const streamPlain = async (turns: ChatMessage[]) => {
    const stream = provider.stream({
      model: req.body.model,
      messages: turns,
      temperature:
        typeof req.body.temperature === 'number' ? req.body.temperature : undefined,
      signal: controller.signal,
    });
    for await (const delta of stream) {
      if (controller.signal.aborted) break;
      send('delta', { text: delta });
    }
  };

  const runAgentStream = async (tools: Tool[]) => {
    const chatWithTools = provider.chatWithTools;
    const plainMessages = withToolGuidance(messages, false);
    if (!tools.length || !chatWithTools) {
      // Local models that can't do tools just answer directly; no need to
      // announce it — the reply itself is the feedback.
      if (!provider.info.local) {
        send('notice', {
          message: !tools.length
            ? 'No tools enabled — answering directly (see Settings → Tools).'
            : `${provider.info.label} doesn't support tool calls yet — answering directly.`,
        });
      }
      await streamPlain(plainMessages);
      send('done', {});
      return;
    }
    // Only answer text counts as replied: tool activity lines are not an
    // answer, so a loop that dies after showing its work can still degrade.
    let replied = false;
    try {
      for await (const ev of runAgent({
        chat: chatWithTools,
        model: req.body.model,
        messages: withToolGuidance(
          messages,
          true,
          tools.map((t) => t.definition.name),
        ),
        tools,
        temperature:
          typeof req.body.temperature === 'number' ? req.body.temperature : undefined,
        signal: controller.signal,
      })) {
        if (controller.signal.aborted) break;
        if (ev.type === 'text') {
          replied = true;
          send('delta', { text: ev.text });
        } else if (ev.type === 'tool_call') send('tool_call', { name: ev.name, args: ev.args });
        else send('tool_result', { name: ev.name, ok: ev.ok, detail: ev.detail });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      // Some models refuse tools outright (Ollama: "… does not support
      // tools"), and llama.cpp's tool grammar sometimes rejects its own
      // model's output. Nothing has been answered yet, so answer directly
      // with a notice — asking for tools must never fail a working chat.
      const refused = isToolsUnsupported(message);
      if (replied || controller.signal.aborted || !(refused || isToolFormatError(message))) {
        throw err;
      }
      // Local models that refuse tools just retry silently — the reply is
      // the feedback, and a notice line here reads as a failure.
      if (!provider.info.local) {
        send('notice', {
          message: refused
            ? `Model ${req.body.model} doesn't support tool calls — answering directly.`
            : 'Tool calls were interrupted — answering directly.',
        });
      }
      await streamPlain(plainMessages);
      replied = true;
    }
    // Tools ran but no answer text ever arrived — a stalled model, or one that
    // kept calling tools into the capped round. Answer directly rather than
    // leaving the reader with activity lines and silence.
    if (!replied && !controller.signal.aborted) {
      if (!provider.info.local) {
        send('notice', { message: 'The tool rounds ended without an answer — answering directly.' });
      }
      await streamPlain(plainMessages);
    }
    send('done', {});
  };

  try {
    if (askedForTools) {
      await runAgentStream(selection.tools);
    } else {
      const stream = provider.stream({
        model: req.body.model,
        // No tool was selected for this message, so the model is told so —
        // otherwise a request to "use the tools" comes back as a narrated
        // search that never happened.
        messages: withToolGuidance(messages, false),
        temperature:
          typeof req.body.temperature === 'number' ? req.body.temperature : undefined,
        signal: controller.signal,
      });

      for await (const delta of stream) {
        if (controller.signal.aborted) break;
        send('delta', { text: delta });
      }
      send('done', {});
    }
  } catch (err) {
    // An abort is a user-initiated stop, not a failure to report.
    if (!controller.signal.aborted) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`[${provider.info.id}]`, message);
      send('error', { message });
    }
  } finally {
    res.end();
    releaseStream();
  }
});

const port = Number(process.env.PORT) || 3000;
// Loopback by default: exposing on a network is an explicit, deliberate choice.
const host = process.env.HOST ?? '127.0.0.1';
const tlsCert = process.env.TLS_CERT;
const tlsKey = process.env.TLS_KEY;

// Saved keys must be in memory before the first request, but the port opens only
// once loading finishes, so a save can never race the initial read.
await loadKeys();
await loadAssignments();
await loadCustomProviders();
// The llama.cpp provider decides "configured" from a cached listing of the GGUF
// files on this machine, and the boot summary below asks synchronously. Waiting
// for one listing here keeps a machine that has the runtime and the models from
// being reported as missing them on the very page load that follows.
await ggufModels().catch(() => undefined);

const server =
  tlsCert && tlsKey
    ? createTlsServer({ cert: readFileSync(tlsCert), key: readFileSync(tlsKey) }, app)
    : app;

server.listen(port, host, () => {
  const scheme = tlsCert && tlsKey ? 'https' : 'http';
  const shown = host === '0.0.0.0' ? 'localhost' : host;

  console.log(`\n  STARR Matrix running at ${scheme}://${shown}:${port}`);
  console.log(`  Settings page:     ${scheme}://${shown}:${port}/settings`);
  console.log(`  Studio:            ${scheme}://${shown}:${port}/studio`);

  if (host === '0.0.0.0') {
    console.log('\n  ! Bound to all interfaces with NO password or login. Anyone');
    console.log('    who can reach this port can spend your API credits and read');
    console.log('    or change your keys at /settings.');
    if (!tlsCert || !tlsKey) {
      console.log('    ! No TLS configured, so traffic is plain text.');
    }
    console.log('    Keep it behind a reverse proxy with auth if that matters.');
  }

  // Probe the well-known local server ports before anyone asks. A chat request
  // that arrives first would otherwise 404 on a server that was running all
  // along, because discovery had not populated the registry yet.
  void refreshDiscovered()
    .then((found) => {
      if (found.length) {
        console.log(`  Local servers:     ${found.map((f) => `${f.label} :${new URL(f.baseUrl).port}`).join(', ')}`);
      }
    })
    .catch(() => {});

  // The runtime probe starts Python inside each venv and imports torch, which
  // is 20–40 s on a CPU-only machine; the models page waits for that answer
  // before it can name installed runtimes. Warm it at boot so the wait lands
  // here rather than on the first page the user opens.
  void runtimeStatus().catch(() => {});

  const ready = providerOrder().filter((id) => providerRegistry()[id].info.configured);
  console.log(`  Ready providers: ${ready.join(', ')}`);
  const missing = providerOrder().filter((id) => !providerRegistry()[id].info.configured);
  if (missing.length) {
    console.log(`  Not configured:  ${missing.join(', ')} (add keys at /settings or in .env)\n`);
  } else {
    console.log();
  }
});

// The desktop shell (desktop/main.mjs) stops this server with SIGTERM when its
// window closes, and Ctrl+C does the same in a terminal. Either way Node would
// exit without saying goodbye to the Python jobs this process started — a CPU
// transcription holds a core for minutes — so they are stopped first, the way
// the manager's Stop all button does it. The wait before exiting is for the
// SIGKILL fallback inside stopProcess, whose timer would otherwise die with
// this process before it ever fires.
//
// The managed llama-server goes the same way, for the same reason plus one more:
// it holds a model's weights in RAM, so leaving it behind would outlive the app
// that started it and there would be nothing left to ask it to stop.
for (const signal of ['SIGTERM', 'SIGINT'] as const) {
  process.on(signal, () => {
    void Promise.all([stopAllProcesses(), stopLlamaServer()]).then(([stopped]) => {
      if (stopped > 0) setTimeout(() => process.exit(0), 4200);
      else process.exit(0);
    });
  });
}