/**
 * Downloaded models that run under Hugging Face transformers, served for chat.
 *
 * The other half of the llama.cpp story. GGUF files are self-contained weights
 * that llama-server loads directly; a transformers model is a whole directory
 * (config, tokenizer, weights) and needs a Python process with torch and
 * transformers to hold a conversation. This module finds those directories and
 * starts that process on demand, the same way the GGUF path starts
 * llama-server: one model at a time, narrated loading, idle shutdown, and a
 * supervisor so nothing is left behind.
 *
 * Only causal language models are offered. A seq2seq or encoder model in the
 * same format is reported as "not a chat model" rather than being loaded into
 * a server that would answer nonsense.
 */
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import * as path from 'node:path';

import { listDownloads, type HFDownload } from './huggingface.js';
import { CATALOG } from './model-catalog.js';
import { venvPython } from './runtimes.js';
import {
  attachChild,
  finishProcess,
  registerProcess,
  registerStopHandler,
  unregisterStopHandler,
} from './processes.js';

const port = () => Number(process.env.TRANSFORMERS_PORT ?? 9933);
const idleMs = () => Number(process.env.TRANSFORMERS_IDLE_MS ?? 10 * 60_000);
const loadTimeoutMs = () => Number(process.env.TRANSFORMERS_LOAD_MS ?? 10 * 60_000);
const CACHE_MS = 5000;

const NON_TEXT_REPOS = new Map(
  CATALOG.flatMap((s) =>
    s.task === 'text' ? [] : s.entries.map((e) => [e.repo, s.task] as const),
  ),
);

const NOT_CHAT = /(whisper|wav2vec|parakeet|vocoder|tts|mmproj|\bclip\b|\bvae\b|unet|controlnet|diffusion|stable-diffusion|sdxl|sd3|flux|embed|rerank)/i;

const SUPERVISOR = `
const { spawn } = require('node:child_process');
const [bin, ...args] = process.argv.slice(1);
const child = spawn(bin, args, { stdio: ['ignore', 'inherit', 'inherit'] });
let stopping = false;
function stop() {
  if (stopping) return;
  stopping = true;
  try { child.kill('SIGTERM'); } catch {}
  const hard = setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, 4000);
  hard.unref();
}
child.on('exit', (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
process.on('SIGTERM', stop);
process.on('SIGINT', stop);
process.stdin.on('end', stop);
process.stdin.on('close', stop);
process.stdin.resume();
`;

export interface TransformersModel {
  id: string;
  repo: string;
  /** Directory the weights live in; passed to the Python server. */
  root: string;
  chat: boolean;
  why?: string;
}

let cached: { at: number; models: TransformersModel[] } | null = null;

/** Groups the downloaded files by their containing directory. */
async function groupedRoots() {
  const files = await listDownloads().catch(() => []);
  const byRoot = new Map<string, HFDownload[]>();
  for (const f of files) {
    if (!byRoot.has(f.root)) byRoot.set(f.root, []);
    byRoot.get(f.root)!.push(f);
  }
  return byRoot;
}

/**
 * Is this repo a quantized checkpoint the bundled server cannot load?
 *
 * `AutoModelForCausalLM` reads fp32/fp16/bf16 weights. GPTQ, AWQ and bitsandbytes
 * checkpoints store packed 4- and 8-bit weights behind a `quantize_config`, and
 * need `auto-gptq`, `gptqmodel` or `optimum` to unpack them — none of which this
 * app installs. Listing one as a chat model therefore produced a provider that
 * offered it, accepted the selection, and then failed at load time with a tensor
 * complaint that says nothing about a missing runtime.
 *
 * The config is the signal, not the name: `Qwen2.5-…-GPTQ-Int4` and a repo with
 * no hint in its name both carry `quantization_config` / `quantize_config`, and a
 * repo named "-4bit" can equally hold plain weights. The name is a fallback for a
 * tree whose config did not arrive.
 */
export function quantizedHow(repo: string, config: Record<string, unknown>): string | null {
  const cfg = config?.quantization_config ?? config?.quantize_config;
  if (cfg && typeof cfg === 'object') {
    const method = String((cfg as { quant_method?: unknown }).quant_method ?? '').toLowerCase();
    if (method === 'gptq') return 'GPTQ';
    if (method === 'awq') return 'AWQ';
    if (method) return method.toUpperCase();
    return 'quantized';
  }
  if (/\b(gptq|awq|bnb|bitsandbytes)\b/i.test(repo)) return 'GPTQ';
  if (/[-_](int4|int8|4bit|8bit|q4|q8)\b/i.test(repo)) return 'quantized';
  return null;
}

export async function transformersChatModels(force = false): Promise<TransformersModel[]> {
  if (!force && cached && Date.now() - cached.at < CACHE_MS) return cached.models;

  const byRoot = await groupedRoots();
  // One model per repo: when several snapshots exist, keep the newest.
  const byRepo = new Map<string, { root: string; files: HFDownload[]; mtime: number }>();
  for (const [root, files] of byRoot) {
    const repo = files[0]?.repo ?? '';
    if (!repo) continue;
    const task = NON_TEXT_REPOS.get(repo);
    const hasConfig = files.some((f) => f.file === 'config.json');
    const hasWeights = files.some((f) => /\.(safetensors|bin|pth|pt)$/i.test(f.file));
    if (!hasConfig || !hasWeights) continue;

    let architectures: string[] = [];
    try {
      const cfg = JSON.parse(readFileSync(path.join(root, 'config.json'), 'utf8'));
      architectures = Array.isArray(cfg?.architectures) ? cfg.architectures : [];
    } catch {
      architectures = [];
    }
    const isCausal = architectures.some((a) => /causal|lmhead/i.test(a)) && !architectures.some((a) => /conditionalgeneration|seq2seq|whisper|clip/i.test(a));
    const isMlx = /mlx/i.test(repo);
    const chat = !task && isCausal && !isMlx && !NOT_CHAT.test(repo);

    const mtime = files.reduce((m, f) => Math.max(m, Date.parse(f.modifiedAt) || 0), 0);
    const prev = byRepo.get(repo);
    if (!prev || mtime > prev.mtime) {
      byRepo.set(repo, { root, files, mtime });
    }
  }

  const out: TransformersModel[] = [];
  for (const [repo, { root, files }] of byRepo) {
    const task = NON_TEXT_REPOS.get(repo);
    let architectures: string[] = [];
    let config: Record<string, unknown> = {};
    try {
      config = JSON.parse(readFileSync(path.join(root, 'config.json'), 'utf8'));
      architectures = Array.isArray(config?.architectures) ? config.architectures : [];
    } catch {
      architectures = [];
    }
    const isCausal = architectures.some((a) => /causal|lmhead/i.test(a)) && !architectures.some((a) => /conditionalgeneration|seq2seq|whisper|clip/i.test(a));
    const isMlx = /mlx/i.test(repo);
    const quantized = quantizedHow(repo, config);
    const chat = !task && isCausal && !isMlx && !quantized && !NOT_CHAT.test(repo);
    out.push({
      id: repo,
      repo,
      root,
      chat,
      why: chat
        ? undefined
        : quantized
          ? `it is a ${quantized} checkpoint, which needs a runtime this app does not install`
          : isMlx
            ? 'it is an MLX (Apple) model, which needs a different runtime'
            : task
              ? `it is a ${task} model`
              : !isCausal
                ? `it is not a causal language model (${architectures.join(', ') || 'unknown architecture'})`
                : 'it is not a text model',
    });
  }
  out.sort((a, b) => a.id.localeCompare(b.id));
  cached = { at: Date.now(), models: out };
  return out;
}

export async function transformersChatModelsOnly(force = false): Promise<TransformersModel[]> {
  return (await transformersChatModels(force)).filter((m) => m.chat);
}

export function cachedTransformersModels(): TransformersModel[] {
  return cached?.models ?? [];
}

export async function resolveTransformersModel(id: string): Promise<TransformersModel | null> {
  const want = String(id ?? '').trim();
  if (!want) return null;
  const all = await transformersChatModels();
  return (
    all.find((m) => m.id === want) ??
    all.find((m) => m.id.toLowerCase() === want.toLowerCase()) ??
    null
  );
}

export function transformersAvailable(): boolean {
  const py = venvPython('venv');
  if (!py || !existsSync(py)) return false;
  try {
    return spawnSync(py, ['-c', 'import transformers'], { stdio: 'ignore' }).status === 0;
  } catch {
    return false;
  }
}

interface Serving {
  child: ChildProcess;
  model: string;
  jobId?: string;
  userStopped?: boolean;
  stopping?: boolean;
}

let serving: Serving | null = null;
let starting: Promise<void> | null = null;
let idleTimer: NodeJS.Timeout | null = null;
let tail: string[] = [];

function serverScript(): string {
  return path.join(process.cwd(), 'scripts', 'chat_transformers.py');
}

function serverBinary(): string | null {
  const py = venvPython('venv');
  if (!py || !existsSync(py)) return null;
  if (!existsSync(serverScript())) return null;
  if (!transformersAvailable()) return null;
  return py;
}

export interface TransformersServerState {
  installed: boolean;
  running: boolean;
  model: string | null;
  pid: number | null;
}

export function transformersServerState(): TransformersServerState {
  return {
    installed: Boolean(serverBinary()),
    running: Boolean(serving),
    model: serving?.model ?? null,
    pid: serving?.child.pid ?? null,
  };
}

function tailSuffix(): string {
  const last = tail.slice(-3).join(' · ');
  return last ? ` — server said: ${last.slice(0, 400)}` : '';
}

function remember(chunk: Buffer) {
  for (const line of String(chunk).split(/\r?\n/)) {
    if (!line.trim()) continue;
    tail.push(line.trim());
    while (tail.length > 40) tail.shift();
  }
}

function clearIdle() {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = null;
}

function touchIdle() {
  clearIdle();
  const ms = idleMs();
  if (!Number.isFinite(ms) || ms <= 0) return;
  idleTimer = setTimeout(() => void stopTransformersServer(), ms);
  idleTimer.unref?.();
}

export async function stopTransformersServer(byUser = false): Promise<void> {
  const current = serving;
  if (!current) return;
  current.userStopped = byUser;
  current.stopping = true;
  clearIdle();
  const { child } = current;
  if (child.exitCode !== null || child.signalCode) {
    serving = null;
    return;
  }
  await new Promise<void>((resolve) => {
    const hard = setTimeout(() => {
      try { child.kill('SIGKILL'); } catch {}
      resolve();
    }, 4000);
    child.once('exit', () => { clearTimeout(hard); resolve(); });
    try { child.kill('SIGTERM'); } catch { clearTimeout(hard); resolve(); }
  });
  if (serving === current) serving = null;
}

async function waitForReady(child: ChildProcess, model: TransformersModel) {
  const started = Date.now();
  for (;;) {
    if (child.exitCode !== null || child.signalCode) {
      throw new Error(`server stopped while loading ${model.id} (exit ${child.exitCode ?? child.signalCode})${tailSuffix()}`);
    }
    try {
      const res = await fetch(`http://127.0.0.1:${port()}/health`, { signal: AbortSignal.timeout(1500) });
      if (res.ok) return;
    } catch (e) {
      /* not listening yet */
    }
    if (Date.now() - started > loadTimeoutMs()) {
      throw new Error(`${model.id} did not finish loading within ${Math.round(loadTimeoutMs() / 1000)}s${tailSuffix()}`);
    }
    await new Promise((r) => setTimeout(r, 400));
  }
}

async function start(bin: string, model: TransformersModel) {
  await stopTransformersServer();

  const script = serverScript();
  const args = [script, '--model', model.root, '--host', '127.0.0.1', '--port', String(port()), '--device', 'auto'];

  tail = [];
  const child = spawn(process.execPath, ['-e', SUPERVISOR, bin, ...args], {
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  serving = { child, model: model.id };

  child.stdout?.on('data', remember);
  child.stderr?.on('data', remember);
  child.on('error', (err) => {
    if (serving?.child === child) serving = null;
    clearIdle();
    console.error('[transformers-chat] could not start:', err.message);
  });
  child.on('exit', (code, signal) => {
    const wasCurrent = serving?.child === child;
    const jobId = serving?.jobId;
    const userStopped = serving?.userStopped ?? false;
    if (wasCurrent) { serving = null; clearIdle(); }
    if (jobId) {
      unregisterStopHandler(jobId);
      finishProcess(jobId, code ?? null, signal, userStopped);
    }
    if (wasCurrent && code && !userStopped) {
      console.error(`[transformers-chat] server exited (${code ?? signal})${tailSuffix()}`);
    }
  });

  await waitForReady(child, model);

  // Find the real python server pid for the resource manager.
  let serverPid = child.pid ?? -1;
  for (let i = 0; i < 20 && serverPid === (child.pid ?? -1); i += 1) {
    await new Promise((r) => setTimeout(r, 100));
    try {
      const out = spawnSync('pgrep', ['-P', String(child.pid)], { encoding: 'utf8' });
      const found = out.stdout.trim().split(/\s+/).filter(Boolean)[0];
      if (found) serverPid = Number(found);
    } catch { /* pgrep missing */ }
  }
  const job = registerProcess({
    kind: 'llm',
    script: 'chat_transformers.py',
    pid: serverPid > 0 ? serverPid : (child.pid ?? -1),
    device: 'auto',
  });
  serving.jobId = job.id;
  attachChild(job.id, child);
  registerStopHandler(job.id, () => stopTransformersServer(true));

  touchIdle();
}

export async function prepareTransformers(modelId: string): Promise<void> {
  const model = await resolveTransformersModel(modelId);
  const bin = serverBinary();
  if (!model) throw new Error(`transformers: no model called "${String(modelId ?? '').slice(0, 80)}" on this machine.`);
  if (!model.chat) throw new Error(`transformers: ${model.repo} is not a chat model — ${model.why}.`);
  if (!bin) throw new Error('transformers is not available. Install torch and transformers under Settings → Runtimes, then try again.');

  if (serving?.model === model.id) { touchIdle(); return; }
  if (starting) {
    await starting.catch(() => undefined);
    if (serving?.model === model.id) { touchIdle(); return; }
  }
  const attempt = start(bin, model);
  starting = attempt;
  try { await attempt; } finally { if (starting === attempt) starting = null; }
}

export function transformersHint(): string {
  if (!transformersAvailable()) return 'transformers is not installed yet — install torch + transformers under Settings → Runtimes';
  const models = cachedTransformersModels().filter((m) => m.chat);
  if (!models.length) return 'No transformers chat models on this machine yet — download a causal-LM model from the Hub';
  const state = transformersServerState();
  if (state.running && state.model) return `Running with ${state.model} loaded`;
  return `${models.length} transformers chat model${models.length === 1 ? '' : 's'} on this machine — loaded when you send a message`;
}
