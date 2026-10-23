/**
 * Stable Diffusion 3.5 Medium through stable-diffusion.cpp.
 *
 * The diffusers route needs the whole pipeline in RAM: fp32 UNet, two text
 * encoders and a VAE, several gigabytes before activations. On a 7.5 GB laptop
 * that is not a tuning problem, it is a memory wall. stable-diffusion.cpp runs
 * the same model from GGUF-quantised weights, keeps the MMDiT in RAM, streams
 * the 2.75 GB T5 encoder from disk and runs the UNet on the card, so the model
 * genuinely fits without the full diffusers pipeline resident.
 *
 * The binary is installed by runtimes.ts. This module owns the layout on disk
 * and the CLI invocation; it never reaches a shell.
 */
import { spawn } from 'node:child_process';
import { mkdir, stat } from 'node:fs/promises';
import { readFileSync, readdirSync } from 'node:fs';
import * as path from 'node:path';

import { listDownloads, modelsDir } from './huggingface.js';
import {
  attachChild,
  finishProcess,
  registerProcess,
  setProcessDevice,
} from './processes.js';
import { runtimesDir } from './paths.js';
import type { Device } from './runtimes.js';

/** Pinned release; bump deliberately after checking the asset names. */
export const SDCPP_BUILD = 'master-987-a77d362';

export type SdCppKind = 'vulkan' | 'cpu' | 'metal';

export interface SdCppAsset {
  name: string;
  kind: SdCppKind;
}

/**
 * The build upstream publishes for this machine.
 *
 * Linux has no CUDA asset, so an NVIDIA machine uses the Vulkan build — it
 * drives the same card. macOS publishes Metal. Everything else that has a
 * render node gets Vulkan; the CPU archive always works.
 */
export function sdCppAsset(vulkan: boolean): SdCppAsset | null {
  if (process.platform === 'linux' && process.arch === 'x64') {
    return vulkan
      ? {
          name: `sd-master-a77d362-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip`,
          kind: 'vulkan',
        }
      : { name: `sd-master-a77d362-bin-Linux-Ubuntu-24.04-x86_64.zip`, kind: 'cpu' };
  }
  if (process.platform === 'win32' && process.arch === 'x64') {
    return vulkan
      ? { name: `sd-master-a77d362-bin-win-vulkan-x64.zip`, kind: 'vulkan' }
      : { name: `sd-master-a77d362-bin-win-cpu-x64.zip`, kind: 'cpu' };
  }
  if (process.platform === 'darwin' && process.arch === 'arm64') {
    return {
      name: `sd-master-a77d362-bin-Darwin-macOS-26.6.2-arm64.zip`,
      kind: 'metal',
    };
  }
  return null;
}

export const sdCppDir = (): string => path.join(runtimesDir(), 'stable-diffusion.cpp');
export const sdCppCliName = (): string =>
  process.platform === 'win32' ? 'sd-cli.exe' : 'sd-cli';

/** The CLI path recorded by the installer, or null when absent. */
export function sdCppCliPath(): string | null {
  try {
    const rel = readFileSync(path.join(sdCppDir(), 'build.txt'), 'utf8').trim();
    if (!rel) return null;
    return path.join(runtimesDir(), rel, sdCppCliName());
  } catch {
    return null;
  }
}

export async function sdCppInstalled(): Promise<boolean> {
  const cli = sdCppCliPath();
  if (!cli) return false;
  try {
    return (await stat(cli)).isFile();
  } catch {
    return false;
  }
}

/** Which backend the installed tree carries, read from the shared libraries. */
export function sdCppKind(): SdCppKind | null {
  const cli = sdCppCliPath();
  if (!cli) return null;
  try {
    const entries = readdirSync(path.dirname(cli));
    if (entries.some((n) => /vulkan/i.test(n))) return 'vulkan';
    if (process.platform === 'darwin') return 'metal';
    return 'cpu';
  } catch {
    return null;
  }
}

/**
 * Device names the CLI will accept, in its own order.
 *
 * Parsed from `--list-devices` rather than assumed: Vulkan0 is the Intel iGPU
 * on one machine and an NVIDIA card on another, and a cuda0 that does not
 * exist is a startup failure rather than a fallback.
 */
let deviceCache: Promise<{ name: string; label: string }[]> | null = null;
function listDevices(cli: string): Promise<{ name: string; label: string }[]> {
  if (deviceCache) return deviceCache;
  deviceCache = new Promise((resolve) => {
    const child = spawn(cli, ['--list-devices'], {
      cwd: path.dirname(cli),
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    child.stdout.on('data', (b) => (out += String(b)));
    child.stderr.on('data', (b) => (out += String(b)));
    child.on('error', () => resolve([]));
    child.on('close', () => {
      const devices: { name: string; label: string }[] = [];
      for (const line of out.split(/\r?\n/)) {
        const tab = line.indexOf('\t');
        if (tab <= 0) continue;
        const name = line.slice(0, tab).trim();
        const label = line.slice(tab + 1).trim();
        if (/^(vulkan|cuda|metal)\d+$/i.test(name)) devices.push({ name, label });
      }
      resolve(devices);
    });
  });
  return deviceCache;
}

/** Best device for the requested vendor, or null to stay on the CPU. */
async function pickBackend(
  cli: string,
  requested: Device,
): Promise<string | null> {
  if (requested === 'cpu' || sdCppKind() !== 'vulkan') return null;
  const devices = await listDevices(cli);
  const vulkan = devices.filter((d) => /^vulkan\d+$/i.test(d.name));
  if (!vulkan.length) return null;
  if (requested === 'nvidia') {
    return (
      vulkan.find((d) => /nvidia|geforce|quadro|rtx/i.test(d.label))?.name ??
      vulkan[0].name
    );
  }
  if (requested === 'amd') {
    return (
      vulkan.find((d) => /amd|radeon|ati/i.test(d.label))?.name ??
      vulkan[0].name
    );
  }
  // Auto: prefer a discrete card when the labels say so, otherwise the first
  // device Vulkan exposes. On a laptop this avoids choosing the iGPU over the
  // Radeon for no reason.
  return (
    vulkan.find((d) => /amd|radeon|nvidia|geforce|rtx/i.test(d.label))?.name ??
    vulkan[0].name
  );
}

export interface SdCppGenerateOptions {
  repo: string;
  prompt: string;
  negative?: string;
  steps: number;
  width: number;
  height: number;
  device: Device;
  onLog?: (line: string) => void;
}

/** The five paths a complete SD 3.5 Medium bundle resolves to. */
export interface SdCppFiles {
  model: string;
  t5: string;
  clipL: string;
  clipG: string;
  vae: string;
}

export interface SdCppResult {
  ok: boolean;
  data: Record<string, unknown>;
}

/** Only the catalog bundle this app knows how to drive. */
export function isSdCppRepo(repo: string): boolean {
  return repo === 'second-state/stable-diffusion-3.5-medium-GGUF';
}

/**
 * The canvas this card produces coherently. A 512×512 request sampled into
 * wood grain on the Radeon here; 384×384 at eight steps produced the intended
 * apple. Keeping the default honest matters more than matching a desktop GPU's
 * usual size.
 */
export const SDCPP_DEFAULT_WIDTH = 384;
export const SDCPP_DEFAULT_HEIGHT = 384;

function findFile(files: string[], pattern: RegExp): string | undefined {
  return files.filter((f) => pattern.test(f)).sort()[0];
}

/** All four GGUF components plus a VAE, resolved through download records. */
async function resolveBundle(repo: string): Promise<{
  model: string;
  t5: string;
  clipL: string;
  clipG: string;
  vae: string;
}> {
  const downloads = await listDownloads();
  const dirFor = (r: string) => path.join(modelsDir(), r.replace(/\//g, '--'));
  const inRepo = (r: string) =>
    downloads
      .filter((d) => d.repo === r)
      .map((d) => d.file);

  const main = inRepo(repo);
  const model = findFile(main, /^sd3\.5.*\.gguf$/i);
  const t5 = findFile(main, /^t5xxl.*\.gguf$/i);
  const clipL = findFile(main, /^clip_l.*\.gguf$/i);
  const clipG = findFile(main, /^clip_g.*\.gguf$/i);

  // The quantised repository omits the VAE. The catalog lists an ungated
  // mirror of the official VAE as an auxiliary file, so it lands in its own
  // repo folder. The official stabilityai repo is gated and would make a
  // token mandatory for a model that is otherwise public.
  const vaeRepo = 'adamo1139/stable-diffusion-3.5-medium-ungated';
  const vaeFiles = inRepo(vaeRepo).filter((f) => /^vae\/.*\.safetensors$/i.test(f));
  const vae =
    vaeFiles.find((f) => !/\.fp16\.safetensors$/i.test(f)) ?? vaeFiles[0];

  const missing: string[] = [];
  if (!model) missing.push('sd3.5_medium-Q4_0.gguf');
  if (!t5) missing.push('t5xxl-Q4_0.gguf');
  if (!clipL) missing.push('clip_l-Q4_0.gguf');
  if (!clipG) missing.push('clip_g-Q4_0.gguf');
  if (!vae) missing.push('vae/diffusion_pytorch_model.safetensors');
  if (missing.length) {
    throw new Error(
      `Stable Diffusion 3.5 Medium is incomplete — missing ${missing.join(', ')}. ` +
        'Download the whole card in Settings → Get models.',
    );
  }

  return {
    model: path.join(dirFor(repo), model!),
    t5: path.join(dirFor(repo), t5!),
    clipL: path.join(dirFor(repo), clipL!),
    clipG: path.join(dirFor(repo), clipG!),
    vae: path.join(dirFor(vaeRepo), vae!),
  };
}

/**
 * Builds the CLI argv for one SD 3.5 Medium generation.
 *
 * Exported rather than buried in the spawn so the memory flags can be tested
 * without launching a generation. `diffusion=cpu` is deliberate:
 * `diffusion=disk` is cheaper, but with this quantised MMDiT it produced pure
 * noise. The text encoders can stream safely, and that is where the 2.75 GB
 * T5 lives.
 */
export function sdCppArgv(
  opts: Pick<SdCppGenerateOptions, 'prompt' | 'negative' | 'steps' | 'width' | 'height'>,
  files: SdCppFiles,
  outPath: string,
  backend: string | null,
): string[] {
  const args = [
    '--diffusion-model',
    files.model,
    '--t5xxl',
    files.t5,
    '--clip_l',
    files.clipL,
    '--clip_g',
    files.clipG,
    '--vae',
    files.vae,
    '--prompt',
    opts.prompt,
    '--negative-prompt',
    opts.negative ?? '',
    '--steps',
    String(opts.steps),
    '--width',
    String(opts.width),
    '--height',
    String(opts.height),
    '--cfg-scale',
    '4.5',
    // SD 3.5 Medium's distilled guidance is 3.5 by default; SLG at 2.5 is the
    // value stable-diffusion.cpp documents for this specific model.
    '--guidance',
    '3.5',
    '--slg-scale',
    '2.5',
    '--sampling-method',
    'euler',
    '--scheduler',
    'simple',
    // Keep the quantised MMDiT in RAM. Streaming its weights from disk with
    // `diffusion=disk` produced pure noise with this SD 3.5 Medium build, even
    // though the same flags work for the text encoders. The full bundle is
    // 5.2 GiB, which does not fit beside a desktop session; the T5/CLIP
    // encoders are the pieces that can stream from disk instead.
    '--params-backend',
    'diffusion=cpu,te=disk,vae=cpu',
    '-o',
    outPath,
  ];
  // `--list-devices` prints `Vulkan0`, while `--backend` documents lowercase
  // names. Normalising here avoids depending on case-insensitive parsing in
  // whichever upstream build is installed.
  if (backend) args.push('--backend', `diffusion=${backend.toLowerCase()},te=cpu,vae=cpu`);
  else args.push('--backend', 'cpu');
  return args;
}

/**
 * Runs one image.
 *
 * Text encoders and VAE stay on the CPU when a GPU is used: the T5 encoder is
 * 2.75 GB and would not fit beside the UNet on a 2 GB laptop card. Its weights
 * stream from disk; the UNet stays in RAM and runs on the GPU. That split
 * produced a clean image here, while a disk-backed MMDiT did not.
 */
export async function generateWithSdCpp(
  opts: SdCppGenerateOptions,
): Promise<{ result: SdCppResult; outPath: string }> {
  const cli = sdCppCliPath();
  if (!cli) {
    throw new Error(
      'stable-diffusion.cpp is not installed. Install it on the Runtimes page.',
    );
  }
  const files = await resolveBundle(opts.repo);
  const outDir = path.resolve(process.env.RUNTIME_OUT_DIR ?? './.runtimes/out');
  await mkdir(outDir, { recursive: true });
  const outPath = path.join(outDir, `img-${Date.now()}.png`);
  const backend = await pickBackend(cli, opts.device);
  const args = sdCppArgv(opts, files, outPath, backend);

  opts.onLog?.(
    `stable-diffusion.cpp: ${opts.width}×${opts.height}, ${opts.steps} steps, ` +
      `${backend ?? 'cpu'} backend`,
  );

  const started = Date.now();
  const result = await new Promise<SdCppResult>((resolve) => {
    const child = spawn(cli, args, {
      cwd: path.dirname(cli),
      env: {
        ...process.env,
        LD_LIBRARY_PATH: [path.dirname(cli), process.env.LD_LIBRARY_PATH]
          .filter(Boolean)
          .join(path.delimiter),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    // stable-diffusion.cpp is a native child rather than one of infer.ts's
    // Python runners, so it does not pass through run() and would otherwise be
    // invisible in the resource manager. Registering it here makes this the
    // long job it is: visible from its first second, with a Stop button.
    const job = registerProcess({
      kind: 'image',
      script: 'sd-cli',
      pid: child.pid ?? -1,
      repo: opts.repo,
      device: opts.device,
    });
    attachChild(job.id, child);
    // The backend was already resolved from --list-devices, so the manager can
    // say "GPU (Vulkan)" immediately instead of showing the requested device.
    setProcessDevice(job.id, backend ?? 'cpu');

    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      resolve({ ok: false, data: { error: 'Image generation timed out after 90 minutes.' } });
    }, 90 * 60_000);

    const emit = (buf: Buffer) => {
      const text = String(buf);
      stderr += text;
      for (const line of text.split(/\r?\n/)) {
        const clean = line.trim();
        if (clean) opts.onLog?.(clean);
      }
    };
    child.stdout.on('data', (buf) => (stdout += String(buf)));
    child.stderr.on('data', emit);
    child.on('error', (err) => {
      clearTimeout(timer);
      finishProcess(job.id, null);
      resolve({ ok: false, data: { error: err.message } });
    });
    child.on('close', (code, signal) => {
      clearTimeout(timer);
      const finished = finishProcess(job.id, code, signal);
      // A job stopped from the resource manager is a decision, not a fault.
      // The close handler is the place that can still see the flag.
      if (finished?.stoppedByUser) {
        resolve({ ok: false, data: { error: 'Stopped from the resource manager.' } });
        return;
      }
      if (code !== 0) {
        const last = stderr
          .split(/\r?\n/)
          .map((s) => s.trim())
          .filter(Boolean)
          .slice(-4)
          .join(' ');
        resolve({
          ok: false,
          data: { error: `stable-diffusion.cpp exited ${code}${last ? `: ${last}` : '.'}` },
        });
        return;
      }
      resolve({
        ok: true,
        data: {
          path: outPath,
          width: opts.width,
          height: opts.height,
          steps: opts.steps,
          device: backend ?? 'cpu',
          seconds: Math.round((Date.now() - started) / 1000),
          log: stdout.trim().slice(-2_000),
        },
      });
    });
  });

  return { result, outPath };
}
