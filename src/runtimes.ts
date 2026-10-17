/**
 * Optional local runtimes for the models that Ollama can't serve.
 *
 * Ollama covers chat models and nothing else, so speech-to-text, text-to-speech
 * and image generation need a real Python environment. This manages the venvs
 * and installs packages into them with pip, entirely optional: nothing here
 * runs unless the user asks for it, and a failure here never affects the chat
 * page.
 *
 * Two constraints shaped the code:
 *
 *  - Nothing user-supplied reaches a shell. Package names come from a fixed
 *    table, the device selector is an enum, and pip is invoked through spawn
 *    with an argument array and shell:false. There is no string concatenation
 *    into a command line anywhere in this file.
 *  - The torch wheel depends on the device. PyPI's default linux torch drags in
 *    several GB of CUDA dependencies whether or not a GPU exists, so the index
 *    is chosen explicitly and "both" means "CUDA build, fall back to CPU".
 *
 * A group can also declare its own venv. Kokoro's dependency tree stops at
 * Python 3.12, and the shared venv follows the machine's interpreter — which on
 * a current distro is newer — so Kokoro gets a second environment built on the
 * newest compatible interpreter this machine has.
 */
import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import { needsPrivilege } from './drivers.js';
import { readFileSync, readdirSync, readlinkSync, realpathSync } from 'node:fs';
import { mkdir, readdir, rm, stat } from 'node:fs/promises';
import { homedir } from 'node:os';
import * as path from 'node:path';

export type Device = 'cpu' | 'nvidia' | 'amd' | 'auto';

export type GpuVendor = 'nvidia' | 'amd';

export interface GpuPresence {
  vendor: GpuVendor | null;
  detail: string;
  /** Present but the user may lack permission to open the device node. */
  needsGroup: boolean;
  /**
   * False when a card is present but cannot actually run CUDA or ROCm.
   * Absent means "no GPU", which is a different message entirely.
   */
  usable?: boolean;
  /** Real numbers read out of the driver, when it is willing to report them. */
  hardware?: GpuHardware;
  /**
   * A working Vulkan backend, which is a different question from ROCm.
   *
   * Worth separating because the two disagree exactly where it matters: ROCm
   * needs GCN3+ and refuses GCN1, while Mesa's RADV driver has run GCN1 fine
   * for years. A card can therefore be useless to a PyTorch ROCm build and
   * fully in use for inference at the same time — which is not a contradiction,
   * just two backends with different floors.
   */
  vulkan?: VulkanBackend;
  /** True when the GPU is genuinely being used right now, by whatever. */
  inUse?: boolean;
  /**
   * A card is plugged in, the driver is bound, and the card itself is not
   * answering. This is not the same as "too old for ROCm" and it is not the
   * same as "no card": the hardware is fine and a driver reload fixes it.
   */
  faulted?: boolean;
}

/**
 * A Vulkan device the loader can actually open.
 *
 * Enumerated by asking the loader rather than by looking for an ICD file: the
 * ICD manifest exists whenever the driver package is installed, including when
 * the card is faulted and the loader refuses to open it. What matters is
 * whether a device comes back.
 */
export interface VulkanBackend {
  available: boolean;
  /** The device's name as the driver reports it, e.g. "AMD Radeon R5 M330". */
  device: string | null;
  /** "radv", "intel", "nouveau" — which Mesa driver is serving it. */
  driver: string | null;
  /** True when the named device is discrete rather than integrated. */
  discrete: boolean;
}

const VULKAN_DISCRETE = /DISCRETE_GPU/i;

/** Asks the Vulkan loader what it can actually open. Cached briefly. */
let vulkanCache: { at: number; value: VulkanBackend } | null = null;

async function detectVulkan(): Promise<VulkanBackend> {
  if (vulkanCache && Date.now() - vulkanCache.at < 20_000) return vulkanCache.value;
  const missing: VulkanBackend = { available: false, device: null, driver: null, discrete: false };
  const out = await new Promise<string>((resolve) => {
    const child = spawn('vulkaninfo', ['--summary'], { stdio: ['ignore', 'pipe', 'ignore'] });
    let buf = '';
    child.stdout.on('data', (c) => {
      buf += String(c);
      if (buf.length > 200_000) child.kill('SIGTERM');
    });
    child.on('error', () => resolve(''));
    child.on('close', () => resolve(buf));
  }).catch(() => '');
  if (!out) {
    vulkanCache = { at: Date.now(), value: missing };
    return missing;
  }
  // vulkaninfo --summary prints one `GPU0:`/`GPU1:` block per device, each
  // carrying vendorID, deviceID, deviceType, deviceName and driverName. Split on
  // the block header rather than on a field: deviceType is printed *before*
  // deviceName, so splitting on deviceName attached each card's name to the
  // next card's type — which called the discrete Radeon integrated.
  //
  // The vendor id is what identifies the card. Names are marketing text and
  // vary by driver ("AMD Radeon R5 M330", "AMD Radeon HD 520"), but 0x1002 is
  // AMD on every one of them.
  const blocks = out.split(/^GPU\d+:\s*$/m).slice(1);
  const field = (b: string, k: string) =>
    b.match(new RegExp(`^\\s*${k}\\s*=\\s*(.+)$`, 'm'))?.[1]?.trim() ?? null;
  const devices = blocks.map((b) => ({
    vendor: field(b, 'vendorID')?.toLowerCase() ?? '',
    name: field(b, 'deviceName'),
    driver: field(b, 'driverName'),
    discrete: VULKAN_DISCRETE.test(b),
  }));
  const amd = devices.find((d) => d.vendor === '0x1002');
  const value: VulkanBackend = {
    available: devices.length > 0,
    device: amd?.name ?? null,
    driver: amd?.driver ?? null,
    discrete: amd?.discrete ?? false,
  };
  vulkanCache = { at: Date.now(), value };
  return value;
}

/**
 * What the kernel driver will admit about the card.
 *
 * `amdgpu` publishes its memory use through sysfs whether or not any compute
 * stack can open the device, so these numbers are available even while the card
 * is faulted — which is exactly when it is most useful to see that the hardware
 * is there and how big it is.
 */
export interface GpuHardware {
  /** Marketing name, from the PCI id table. */
  name: string;
  /** Bytes of on-board memory, when the driver reports any. */
  vramTotal?: number;
  vramUsed?: number;
  /** The GCN architecture generation, when it could be determined from the id. */
  arch?: string;
  /** The kernel driver bound to the card. */
  driver?: string;
  /** PCI revision, useful when identifying a card by hand. */
  revision?: string;
}

/** Where the venvs and their packages live. Gitignored alongside models/. */
export const runtimesDir = () => path.resolve(process.env.RUNTIMES_DIR ?? './.runtimes');

/**
 * Venv folder for a group. Most groups share the main `venv`; a group whose
 * tree cannot run on the machine's interpreter brings its own, named after it.
 */
const venvName = (group: RuntimeGroup): string => group.venv ?? 'venv';
const venvDir = (name = 'venv'): string => path.join(runtimesDir(), name);

/** Interpreter inside a venv, by platform layout. */
export function venvPython(name = 'venv'): string {
  return process.platform === 'win32'
    ? path.join(venvDir(name), 'Scripts', 'python.exe')
    : path.join(venvDir(name), 'bin', 'python');
}

/**
 * PyTorch wheel indexes, verified to publish cp314 manylinux x86_64 wheels.
 * Using PyPI's default would silently pull one vendor's runtime libraries onto a
 * CPU-only box, so the index is always explicit.
 *
 * AMD goes through ROCm. PyTorch's ROCm build deliberately aliases `torch.cuda`
 * to HIP, which is why the runner scripts do not need vendor branches — but the
 * wheel itself must match the vendor or nothing will load.
 */
const TORCH_INDEX: Record<Device, string> = {
  cpu: 'https://download.pytorch.org/whl/cpu',
  nvidia: 'https://download.pytorch.org/whl/cu128',
  amd: 'https://download.pytorch.org/whl/rocm6.4',
  // Whichever is actually present; resolved before use.
  auto: 'https://download.pytorch.org/whl/cpu',
};

export interface RuntimeGroup {
  id: string;
  /** Group installed by a vendor installer or a pinned binary rather than pip. */
  installer?: 'ollama' | 'llama-cpp';
  label: string;
  /** Why someone would want this, shown in the installer. */
  blurb: string;
  /** Tasks this group unlocks. */
  unlocks: string[];
  /** Fixed pip specs. Never user input. */
  pkgs: string[];
  /** Needs torch first; torch itself is installed via the device index. */
  needsTorch: boolean;
  /** Approximate download size, so the button can warn before a big fetch. */
  approxMB: number;
  /**
   * Import name to test for when detecting this group.
   *
   * Usually the pip name with dashes swapped for underscores, but stated
   * explicitly because a mismatch here means the installer and the detector
   * disagree — the app would offer an Install button for something it then
   * insists is not installed.
   */
  probe: string;
  /**
   * Upper bound on the Python that can run this group, e.g. '<3.13'.
   *
   * Kokoro and its whole dependency tree (misaki included) publish no wheels
   * for newer interpreters, so on a machine whose only Python is 3.14 the pip
   * install dies at the resolver with a wall of text. Declaring the bound lets
   * the installer say so plainly up front instead of downloading torch first
   * and failing afterwards.
   */
  maxPython?: string;
  /**
   * Venv folder this group installs into, when it is not the shared `venv`.
   *
   * A bound that the machine's interpreter cannot satisfy used to mean the
   * group was simply unavailable. A group with its own venv instead gets an
   * environment built on the newest installed interpreter its bound allows.
   */
  venv?: string;
}

export const RUNTIME_GROUPS: RuntimeGroup[] = [
  {
    id: 'ollama',
    probe: 'ollama',
    label: 'Ollama',
    blurb:
      'The model server itself, and the only way to run chat models here. ' +
      'Installs Ollama the way this OS does it — the vendor script on Linux ' +
      '(which asks for your password), Homebrew on macOS, the vendor ' +
      'PowerShell script on Windows.',
    unlocks: ['text'],
    // Not a pip package: Ollama ships its own installer. Kept here so
    // everything the app needs appears in one list, but `pkgs` is empty and
    // installRuntimes routes it through the vendor installer instead.
    pkgs: [],
    needsTorch: false,
    approxMB: 1500,
    installer: 'ollama',
  },
  {
    id: 'llama-cpp',
    probe: 'llama-server',
    label: 'llama.cpp',
    blurb:
      'The GGUF inference engine, installed as a pinned prebuilt binary into ' +
      'this app’s own folder — no root, no Python, no compiler. Use it to run ' +
      'the quantized chat models in Get models (Qwen3, Llama 3.2, Gemma), ' +
      'which Ollama cannot load.',
    unlocks: ['text'],
    // Not a pip package either: the upstream project ships prebuilt CPU
    // tarballs per build. Kept beside Ollama so "the two chat runtimes" read
    // as a pair, but this one installs into the app and can be removed again.
    pkgs: [],
    needsTorch: false,
    approxMB: 17,
    installer: 'llama-cpp',
  },
  {
    id: 'onnxruntime',
    probe: 'onnxruntime',
    label: 'ONNX Runtime',
    // Deliberately empty `unlocks`: none of the bundled runners use it. The
    // earlier claim that it unlocked TTS was wrong — speak.py imports the
    // kokoro package and torch. Kept as a genuinely small optional extra for
    // running ONNX exports by hand, and labelled as such rather than implying
    // it powers something it does not.
    blurb: 'Optional extra for running ONNX exports yourself. The bundled TTS, STT and image runners do not use it.',
    unlocks: [],
    pkgs: ['onnxruntime'],
    needsTorch: false,
    approxMB: 18,
  },
  {
    id: 'torch',
    probe: 'torch',
    label: 'PyTorch',
    blurb: 'The base tensor library. Everything below needs it, and it is by far the largest download.',
    unlocks: ['text-to-speech', 'speech-to-text', 'image-generation'],
    pkgs: [],
    needsTorch: false,
    approxMB: 529,
  },
  {
    id: 'transformers',
    probe: 'transformers',
    label: 'Transformers',
    blurb: 'Loads Whisper for transcription. Without it there is no local STT.',
    unlocks: ['speech-to-text'],
    pkgs: ['transformers', 'accelerate', 'sentencepiece'],
    needsTorch: true,
    approxMB: 45,
  },
  {
    id: 'diffusers',
    probe: 'diffusers',
    label: 'Diffusers',
    blurb: 'Runs Stable Diffusion and friends for image generation. Slow on CPU.',
    unlocks: ['image-generation'],
    pkgs: ['diffusers', 'safetensors', 'accelerate'],
    needsTorch: true,
    approxMB: 25,
  },
  {
    id: 'faster-whisper',
    probe: 'faster_whisper',
    label: 'Faster Whisper',
    blurb:
      'Speech-to-text via CTranslate2. Same Whisper models as transformers, ' +
      'about 40 MB of runtime instead of 529 MB of PyTorch, and CPU-only ' +
      'transcription is fast enough to be usable.',
    unlocks: ['speech-to-text'],
    pkgs: ['faster-whisper'],
    needsTorch: false,
    approxMB: 55,
  },
  {
    id: 'kokoro',
    probe: 'kokoro',
    label: 'Kokoro',
    blurb:
      'The TTS package for the Kokoro 82M voice model. It builds its own ' +
      'environment on Python 3.12 or older — nothing in its dependency tree ' +
      'has wheels for newer interpreters, so it cannot share the main venv.',
    unlocks: ['text-to-speech'],
    pkgs: ['kokoro', 'soundfile'],
    // Kokoro's wheel requires torch, so torch has to be in its venv first —
    // otherwise pip resolves the default index and drags several GB of CUDA
    // libraries onto the machine.
    needsTorch: true,
    approxMB: 12,
    maxPython: '<3.13',
    venv: 'venv-kokoro',
  },
];

/** Asks one interpreter its version. '' when it cannot run at all. */
async function probeVersion(bin: string): Promise<string> {
  return new Promise<string>((resolve) => {
    const child = spawn(bin, ['-c', 'import sys;print("%d.%d"%sys.version_info[:2])'], {
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    let out = '';
    child.stdout.on('data', (c) => {
      out += String(c);
    });
    child.on('close', (code) => resolve(code === 0 ? out.trim() : ''));
    child.on('error', () => resolve(''));
  });
}

/**
 * The interpreter a venv runs on, e.g. '3.14'. '' when the venv is missing.
 *
 * Read once and cached: it is a property of the machine, not of the request.
 * Creating or removing a venv clears its entry.
 */
const venvVersions = new Map<string, string>();
export async function venvPythonVersion(name = 'venv'): Promise<string | null> {
  const known = venvVersions.get(name);
  if (known !== undefined) return known;
  const version = await probeVersion(venvPython(name));
  venvVersions.set(name, version);
  return version;
}

/**
 * The newest interpreter on this machine that satisfies a bound, as an
 * executable `python -m venv` can be built with, or null when there is none.
 *
 * Without a bound this is the machine's `python3`, which is what the shared
 * venv has always used. With one — Kokoro's `<3.13` on a distro whose default
 * is newer — versioned binaries are tried newest first, then the interpreters
 * uv and pyenv keep out of PATH. RUNTIMES_PYTHON overrides the search.
 */
async function findInterpreter(bound?: string): Promise<string | null> {
  const explicit = process.env.RUNTIMES_PYTHON;
  if (explicit) {
    const version = await probeVersion(explicit);
    return version && pythonSatisfies(version, bound) ? explicit : null;
  }

  if (!bound) {
    for (const name of ['python3', 'python']) {
      if (await probeVersion(name)) return name;
    }
    return null;
  }

  for (const minor of ['3.12', '3.11', '3.10', '3.9', '3.8']) {
    const name = `python${minor}`;
    const version = await probeVersion(name);
    if (version && pythonSatisfies(version, bound)) return name;
  }

  // Managers keep interpreters outside PATH; look where they put them. The
  // directory name carries the version, so it doubles as the binary name.
  const roots = [
    path.join(homedir(), '.local', 'share', 'uv', 'python'),
    path.join(homedir(), '.pyenv', 'versions'),
  ];
  for (const root of roots) {
    let entries: string[];
    try {
      entries = await readdir(root);
    } catch {
      continue;
    }
    // Newest first: '3.9' < '3.12' as strings, so compare the parsed numbers.
    const candidates = entries.flatMap((entry) => {
      const m = /(\d+)\.(\d+)/.exec(entry);
      return m ? [{ name: entry, major: Number(m[1]), minor: Number(m[2]) }] : [];
    });
    candidates.sort((a, b) => b.major - a.major || b.minor - a.minor);
    for (const c of candidates) {
      const dir = path.join(root, c.name, 'bin');
      let bin = path.join(dir, `python${c.major}.${c.minor}`);
      let version = await probeVersion(bin);
      if (!version) {
        bin = path.join(dir, 'python3');
        version = await probeVersion(bin);
      }
      if (version && pythonSatisfies(version, bound)) return bin;
    }
  }

  // The default interpreter is still worth a try — a bound can be loose.
  for (const name of ['python3', 'python']) {
    const version = await probeVersion(name);
    if (version && pythonSatisfies(version, bound)) return name;
  }
  return null;
}

/** True when the venv's interpreter satisfies the group's upper bound. */
export function pythonSatisfies(version: string | null, bound: string | undefined): boolean {
  if (!bound) return true;
  if (!version) return true; // Unknown interpreter: let pip be the judge.
  const m = /^\s*(<|<=|==|>=|>)?\s*(\d+)(?:\.(\d+))?/.exec(bound);
  if (!m) return true;
  const op = m[1] ?? '==';
  const major = Number(m[2]);
  const minor = m[3] === undefined ? 0 : Number(m[3]);
  const parts = version.split('.').map((n) => Number(n) || 0);
  const have = [parts[0] ?? 0, parts[1] ?? 0];
  const cmp = have[0] !== major ? Math.sign(have[0] - major) : Math.sign(have[1] - minor);
  if (op === '<') return cmp < 0;
  if (op === '<=') return cmp <= 0;
  if (op === '>') return cmp > 0;
  if (op === '>=') return cmp >= 0;
  return cmp === 0;
}

export function groupById(id: unknown): RuntimeGroup | null {
  if (typeof id !== 'string') return null;
  return RUNTIME_GROUPS.find((g) => g.id === id) ?? null;
}

export function parseDevice(d: unknown): Device {
  return d === 'nvidia' || d === 'amd' || d === 'auto' ? d : 'cpu';
}

/**
 * Which GPU, if any, this machine actually has.
 *
 * NVIDIA is easy — nvidia-smi answers. AMD has no such universal tool, so it is
 * detected from the kernel's ROCm device node instead, and a permission problem
 * is reported as one rather than as "no GPU": /dev/kfd exists but unreadable
 * means the card is there and the user just needs to be in the render group.
 */
/** One display-class PCI device, identified by its real vendor ID. */
interface PciGpu {
  vendorId: string;
  deviceId: string;
  /** sysfs directory for this device, e.g. /sys/bus/pci/devices/0000:01:00.0 */
  sysfs: string;
}

const PCI_DISPLAY_CLASS = /^0x03/;

/**
 * Enumerates display-class PCI devices by their vendor and device IDs.
 *
 * This is the only detection worth trusting. The previous version inferred the
 * vendor from `/dev/kfd`, which is a kernel node that outlives the hardware:
 * on an Intel laptop it reported "AMD GPU detected" purely because a stale
 * `amdgpu` node was still on disk, and Auto then selected a ROCm install for a
 * card that could never support one.
 */
function listPciGpus(): PciGpu[] {
  const out: PciGpu[] = [];
  let dirs: string[] = [];
  try {
    dirs = readdirSync('/sys/bus/pci/devices');
  } catch {
    return out;
  }
  for (const name of dirs) {
    const base = `/sys/bus/pci/devices/${name}`;
    try {
      const cls = readFileSync(`${base}/class`, 'utf8').trim();
      if (!PCI_DISPLAY_CLASS.test(cls)) continue;
      out.push({
        vendorId: readFileSync(`${base}/vendor`, 'utf8').trim().toLowerCase(),
        deviceId: readFileSync(`${base}/device`, 'utf8').trim().toLowerCase(),
        sysfs: base,
      });
    } catch {
      // Unreadable device node: skip rather than guess.
    }
  }
  return out;
}

/**
 * A GPU's marketing name.
 *
 * Ordered best-source-first, because a lookup table was the only thing here and
 * a lookup table is the worst way to name a card. Five PCI ids out of thousands
 * are recognised; every other card fell through to the literal string
 * `GPU 0x1002:0x1234`, which is not a name and is what made this feel
 * hard-coded. On this machine, for instance, it reported an R5 M330 as an
 * "HD 520 (mobile)" because both ids land in that range.
 *
 * The real names are published by the stack that is actually driving the card:
 *
 *  1. Vulkan. `vulkaninfo` prints the name the driver reports for the device it
 *     can open, and on an AMD box that is a real product name
 *     ("AMD Radeon R5 M330"). Already detected here for the ROCm-vs-Vulkan
 *     question, so it costs nothing extra.
 *  2. The kernel's `product_name`, where the driver exposes it. Often absent on
 *     older amdgpu, but correct when present and free to read.
 *  3. NVIDIA, via `nvidia-smi --query-gpu=name`, which is the vendor's own
 *     answer and has never needed a table.
 *  4. The old table, as a last resort. Kept because it is better than an id
 *     string when it happens to be right, and labelled as a guess when used.
 *  5. The PCI id, which is always true even when it is not a name.
 */
async function resolveGpuName(g: PciGpu, vulkan: VulkanBackend | null): Promise<string> {
  // The driver already told us; nothing to ask.
  if (vulkan?.available && vulkan.device) return vulkan.device;

  if (g.vendorId === '0x10de') {
    // NVIDIA publishes the name itself. Anything else here would be guessing at
    // a card whose vendor answers the question directly.
    const named = await new Promise<string>((resolve) => {
      const child = spawn('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], {
        stdio: ['ignore', 'pipe', 'ignore'],
      });
      let out = '';
      child.stdout?.setEncoding('utf8');
      child.stdout?.on('data', (c: string) => (out += c));
      child.on('error', () => resolve(''));
      child.on('close', () => resolve(out.trim().split('\n')[0] ?? ''));
    });
    if (named) return named;
  }

  const sysfsName = readAttr(`${g.sysfs}/product_name`);
  if (sysfsName) return sysfsName;

  const guess = KNOWN_GPU_NAMES[`${g.vendorId}:${g.deviceId}`];
  if (guess) return guess;
  // Last resort, and honestly labelled: the ids are always right, even when
  // they are not a name anyone would recognise.
  return `${g.vendorId}:${g.deviceId}`;
}

/**
 * A short table of cards recognised well enough to name without asking.
 *
 * Only consulted after every real source has failed, so it cannot override the
 * driver. Five entries covers nothing in general — it exists so an old card with
 * no Vulkan and no sysfs name still gets a word rather than a hex pair.
 */
const KNOWN_GPU_NAMES: Record<string, string> = {
  '0x1002:0x6660': 'AMD Radeon HD 520 (mobile)',
  '0x1002:0x6640': 'AMD Radeon HD 6650 (mobile)',
  '0x8086:0x1916': 'Intel HD Graphics 520 (integrated)',
  '0x8086:0x191b': 'Intel HD Graphics 530 (integrated)',
  '0x8086:0x3e92': 'Intel UHD Graphics 620 (integrated)',
};

/**
 * The best name available without asking anything: the table, else the ids.
 * Synchronous, for the paths that never get as far as a driver query.
 */
function gpuIdName(g: PciGpu): string {
  return KNOWN_GPU_NAMES[`${g.vendorId}:${g.deviceId}`] ?? `${g.vendorId}:${g.deviceId}`;
}

/** Renders a byte count the way a person would say it. */function fmtBytes(n: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

/** GCN generation from a PCI device id, for the architectures we can name. */
function gcnArch(deviceId: string): string | undefined {
  const n = Number.parseInt(deviceId.replace(/^0x/i, ''), 16);
  if (!Number.isFinite(n)) return undefined;
  if (n >= 0x6600 && n <= 0x67df) return 'GCN1';
  if (n >= 0x1300 && n <= 0x16ff) return 'GCN2';
  if (n >= 0x6810 && n <= 0x6fff) return 'GCN3';
  if (n >= 0x7000 && n <= 0x7fff) return 'GCN4';
  if (n >= 0x8000 && n <= 0x8fff) return 'GCN5';
  if (n >= 0x9000 && n <= 0x9fff) return 'RDNA1';
  if (n >= 0xa000 && n <= 0xafff) return 'RDNA2';
  return undefined;
}

/** Reads a sysfs attribute, returning undefined when it is absent or unreadable. */
function readAttr(file: string): string | undefined {
  try {
    const v = readFileSync(file, 'utf8').trim();
    return v === '' ? undefined : v;
  } catch {
    return undefined;
  }
}

/** Reads a numeric sysfs attribute. */
function readNum(file: string): number | undefined {
  const v = readAttr(file);
  if (v === undefined) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

/**
 * Everything the kernel will tell us about an AMD card.
 *
 * `amdgpu` exports its memory accounting under the DRM node rather than under
 * /dev/kfd, so this keeps working when the compute interface is dead. That
 * matters: on a faulted card these are the only facts still available, and they
 * are what distinguishes "the hardware is there and healthy" from "there is
 * nothing".
 */
function radeonInfo(g: PciGpu): GpuHardware {
  // Provisional. detectGpu replaces this with the driver's own answer once
  // Vulkan has been asked, which is the only source that is right on a card
  // nobody has written a table entry for.
  const out: GpuHardware = { name: gpuIdName(g) };
  // The DRM node and the PCI node expose the same attributes; the DRM path is
  // the documented one and the only one present on every kernel that has amdgpu.
  const roots = [`/sys/class/drm/card${cardIndexFor(g)}/device`, g.sysfs];
  for (const root of roots) {
    if (out.vramTotal === undefined) {
      const t = readNum(`${root}/mem_info_vram_total`);
      if (t !== undefined && t > 0) out.vramTotal = t;
    }
    if (out.vramUsed === undefined) {
      const u = readNum(`${root}/mem_info_vram_used`);
      if (u !== undefined) out.vramUsed = u;
    }
    if (out.driver === undefined) {
      try {
        out.driver = readlinkSync(`${root}/driver`).split('/').pop();
      } catch {
        /* no driver bound */
      }
    }
    if (out.revision === undefined) out.revision = readAttr(`${root}/revision`);
    if (out.vramTotal !== undefined && out.driver !== undefined) break;
  }
  out.arch = gcnArch(g.deviceId);
  return out;
}

/** Finds the cardN whose device symlink resolves to this PCI device. */
function cardIndexFor(g: PciGpu): string {
  try {
    // Resolved, not raw: the two symlinks point at the same directory from
    // different depths, so readlink output differs
    // ("../../../devices/pci0000:00/0000:00:1c.0/0000:01:00.0" versus
    // "../../../0000:01:00.0"). Comparing the strings never matched, which left
    // every probe looking at card0 and reporting a healthy card as dead.
    const real = realpathSync(g.sysfs);
    for (const name of readdirSync('/sys/class/drm')) {
      if (!/^card\d+$/.test(name)) continue;
      try {
        if (realpathSync(`/sys/class/drm/${name}/device`) === real) return name.slice(4);
      } catch {
        /* not this one */
      }
    }
  } catch {
    /* fall through */
  }
  return '0';
}

/**
 * Whether the card is powered and answering.
 *
 * A driver can be bound, the card can be listed by lspci, and the GPU can still
 * be completely unreachable: `runtime_status` goes to `error` when the runtime
 * power-management core failed to resume the device, after which every open of
 * the render node returns EINVAL and ROCm reports it cannot open /dev/kfd. That
 * state is recoverable — it is a driver fault, not a hardware fault — and telling
 * the user their card is "too old for ROCm" when it is actually faulted sends
 * them off to install a CPU build that fixes nothing.
 */
async function radeonFaulted(g: PciGpu): Promise<boolean> {
  const status = readAttr(`${g.sysfs}/power/runtime_status`);
  if (status === 'error') return true;
  // No runtime PM directory at all on an AMD dGPU is itself odd but not proof.
  // Fall back to actually trying to open the render node, which is the real test.
  return !(await renderNodeOpens(g));
}

/** True when the card's render node can actually be opened. */
async function renderNodeOpens(g: PciGpu): Promise<boolean> {
  const idx = cardIndexFor(g);
  const candidates = [
    `/sys/class/drm/card${idx}/device/drm/renderD*`,
  ];
  for (const pattern of candidates) {
    const dir = pattern.slice(0, pattern.indexOf('/renderD'));
    let names: string[] = [];
    try {
      names = readdirSync(dir).filter((n) => /^renderD/.test(n));
    } catch {
      continue;
    }
    for (const n of names) {
      const ok = await new Promise<boolean>((resolve) => {
        const child = spawn('sh', ['-c', `test -r /dev/dri/${n}`], { stdio: ['ignore'] });
        child.on('error', () => resolve(false));
        child.on('close', (code) => resolve(code === 0));
      });
      if (ok) return true;
    }
  }
  return false;
}

/** True when the kernel's AMD compute module is genuinely loaded. */
async function kfdLoaded(): Promise<boolean> {
  if (!(await exists('/sys/module/amdgpu'))) return false;
  return exists('/dev/kfd');
}

/**
 * Whether ROCm can drive this AMD part at all.
 *
 * ROCm supports GCN3 and newer. GCN1 (Southern Islands, 2011-2013) and GCN2
 * (Sea Islands, 2013-2015) are far below that — a Radeon HD 520 is GCN1, and
 * pointing the installer at the ROCm wheel index for it downloads gigabytes
 * and produces a PyTorch build that cannot open the device.
 *
 * Those two generations occupy contiguous, well-known PCI id ranges, so a
 * range test separates them from anything ROCm could use. Anything outside the
 * ranges is reported as `null`: genuinely modern, but not something to claim
 * certainty about from an id alone.
 */
function rocmSupport(deviceId: string): 'yes' | 'no' | null {
  const n = Number.parseInt(deviceId.replace(/^0x/i, ''), 16);
  if (!Number.isFinite(n)) return null;
  // GCN1 "Southern Islands" 0x6600-0x67DF (Oland 0x66xx, Kaveri 0x6660,
  // Hawaii 0x67Bx, Tahiti 0x6780) and GCN2 "Sea Islands" 0x1300-0x16FF.
  // GCN3 begins at 0x6810, just above the first range, so the two do not
  // overlap and nothing ROCm can drive is caught by this test.
  if ((n >= 0x6600 && n <= 0x67df) || (n >= 0x1300 && n <= 0x16ff)) return 'no';
  return null;
}

export async function detectGpu(): Promise<GpuPresence> {
  const inRenderGroup = await new Promise<boolean>((resolve) => {
    const child = spawn('id', ['-G'], { stdio: ['ignore', 'pipe', 'ignore'] });
    let out = '';
    child.stdout.on('data', (c) => {
      out += String(c);
    });
    child.on('error', () => resolve(false));
    child.on('close', (code) => resolve(code === 0 && /\brender\b/.test(out)));
  });

  const gpus = listPciGpus();

  // NVIDIA is easy and unambiguous: the tool answers if the driver is loaded.
  const nvidia = await new Promise<boolean>((resolve) => {
    const child = spawn('nvidia-smi', ['-L'], { stdio: ['ignore'] });
    child.on('error', () => resolve(false));
    child.on('close', (code) => resolve(code === 0));
  });
  const nvidiaGpu = gpus.find((g) => g.vendorId === '0x10de');
  if (nvidia) {
    // nvidia-smi answered, so ask it for the name too rather than reaching for
    // the id table — the vendor publishes this itself.
    const named = nvidiaGpu ? await resolveGpuName(nvidiaGpu, null) : '';
    return {
      vendor: 'nvidia',
      detail: nvidiaGpu ? `NVIDIA GPU: ${named}` : 'NVIDIA GPU detected',
      needsGroup: false,
    };
  }

  // An AMD card is only usable when the amdgpu module is actually loaded. The
  // device node alone can survive the hardware being gone.
  const amdGpu = gpus.find((g) => g.vendorId === '0x1002');
  if (amdGpu) {
    const hw = radeonInfo(amdGpu);
    // Ask the driver what it is before anything else reads hw.name. A card with
    // no Vulkan and no sysfs product_name keeps the table's answer.
    hw.name = await resolveGpuName(amdGpu, await detectVulkan());
    // Checked before the ROCm verdict on purpose. A faulted card is the more
    // urgent fact and the more misleading one to omit: the card is present and
    // the driver is bound, so the arch test still answers, but telling someone
    // to install a CPU build when their GPU is simply switched off is wrong.
    if (await radeonFaulted(amdGpu)) {
      // Two different faults that look identical from the outside, and the fix
      // differs. Without a driver bound, the card failed to initialise at all —
      // the amdgpu probe gives up when the GPU's own firmware hangs during POST,
      // so no device node is ever created and reloading cannot help. With the
      // driver bound but the power state errored, the card came up once and
      // then failed to resume, and a reload is worth trying.
      const mem = hw.vramTotal ? ` with ${fmtBytes(hw.vramTotal)} of VRAM` : '';
      if (!hw.driver) {
        return {
          vendor: 'amd',
          detail:
            `${hw.name}${mem} is installed but its firmware is failing to start — ` +
            'the amdgpu driver could not bring it up, so it has no device node at ' +
            'all and no backend can reach it. A driver reload will not fix this; ' +
            'it needs a cold power cycle (shut down, unplug the charger, and hold ' +
            'the power button for 30 seconds).',
          needsGroup: false,
          usable: false,
          faulted: true,
          hardware: hw,
        };
      }
      return {
        vendor: 'amd',
        detail:
          `${hw.name}${mem} is installed and the ${hw.driver} driver is bound to ` +
          'it, but the card is not responding — its runtime power state has ' +
          'errored, so no backend can open it. Reloading the module or rebooting ' +
          'usually clears this one.',
        needsGroup: false,
        usable: false,
        faulted: true,
        hardware: hw,
      };
    }
    const rocm = rocmSupport(amdGpu.deviceId);
    const vk = await detectVulkan();
    if (rocm === 'no') {
      // The verdict that matters most on this kind of card. Saying only "ROCm
      // cannot use it" reads as "your GPU is dead", and it is routinely wrong:
      // Mesa's RADV driver has run GCN1 for years and Ollama ships a Vulkan
      // llama.cpp that is usually already using the card while this says no.
      const vkLine = vk.available && vk.device
        ? ` Vulkan does work on it — ${vk.driver ?? 'a Vulkan driver'} reports ` +
          `"${vk.device}"${vk.discrete ? ' as a discrete GPU' : ''}, and Ollama runs ` +
          'models on it through Vulkan. '
        : '';
      return {
        vendor: 'amd',
        detail:
          `${hw.name} is a ${hw.arch ?? 'GCN1/GCN2'} card from before ROCm existed` +
          `${hw.vramTotal ? ` (${fmtBytes(hw.vramTotal)})` : ''}, so ROCm — and ` +
          'therefore GPU PyTorch wheels — cannot run on it. ' +
          vkLine +
          'Install the CPU build for local Python models; chat still runs on the ' +
          'GPU through Ollama and Vulkan.',
        needsGroup: false,
        usable: false,
        vulkan: vk,
        hardware: hw,
      };
    }
    if (!(await kfdLoaded())) {
      return {
        vendor: 'amd',
        detail:
          `${hw.name} found, but the amdgpu kernel module is not ` +
          'loaded, so ROCm has no device to talk to. It will install but cannot run.',
        needsGroup: false,
        usable: false,
      };
    }
    const readable = await new Promise<boolean>((resolve) => {
      const child = spawn('sh', ['-c', 'test -r /dev/kfd'], { stdio: ['ignore'] });
      child.on('error', () => resolve(false));
      child.on('close', (code) => resolve(code === 0));
    });
    return {
      vendor: 'amd',
      detail: readable
        ? `AMD GPU: ${hw.name}${hw.vramTotal ? ` — ${fmtBytes(hw.vramTotal)} VRAM` : ''}`
        : `${hw.name} found, but /dev/kfd is unreadable — add your user to the render group`,
      needsGroup: !readable,
      usable: readable,
      vulkan: vk,
      hardware: hw,
    };
  }

  const intelGpu = gpus.find((g) => g.vendorId === '0x8086');
  if (intelGpu) {
    return {
      vendor: null,
      detail:
        `${await resolveGpuName(intelGpu, null)}. Intel integrated graphics run neither CUDA ` +
        'nor ROCm — install the CPU builds.',
      needsGroup: false,
      usable: false,
    };
  }

  if (!inRenderGroup) {
    return {
      vendor: null,
      detail: 'No GPU detected. If it is AMD, add your user to the render group and re-check.',
      needsGroup: false,
      usable: false,
    };
  }
  return { vendor: null, detail: 'No GPU detected', needsGroup: false, usable: false };
}

/** Resolves 'auto' to whatever is installed, or the CPU index if nothing is. */
function resolveIndex(device: Device, presence: GpuPresence): string {
  if (device !== 'auto') return TORCH_INDEX[device];
  if (presence.vendor === 'nvidia') return TORCH_INDEX.nvidia;
  if (presence.vendor === 'amd') return TORCH_INDEX.amd;
  return TORCH_INDEX.cpu;
}

export interface RuntimeStatus {
  venv: boolean;
  python: string | null;
  /** Group id -> installed version, or null. */
  installed: Record<string, string | null>;
  /** What torch reports about usable devices, when torch is present. */
  torch: {
    version: string | null;
    cudaAvailable: boolean;
    deviceCount: number;
    deviceName: string | null;
  };
  busy: boolean;
}

/**
 * Asks the venv what it actually has, in one interpreter launch rather than
 * one per package. Probes are importlib lookups, not real imports, except torch
 * — which genuinely has to load to report CUDA, so that one is slower.
 */
/**
 * The detector, built from RUNTIME_GROUPS rather than written out by hand.
 *
 * This used to be a literal list of six module names. Adding a runtime group
 * then left it invisible: the installer offered it, `runtimeStatus` never saw
 * it, and every surface that depends on "is this installed?" concluded the
 * model could not run. The list and the table drifted apart silently.
 *
 * Generating the probe from the same table the installer uses removes the
 * possibility: a group cannot exist without also being detected.
 */
function buildProbe(): string {
  const names = [...new Set(RUNTIME_GROUPS.map((g) => g.probe))];
  return `
import json, importlib.util
out = {"packages": {}, "torch": None}
for name in ${JSON.stringify(names)}:
    try:
        out["packages"][name] = True if importlib.util.find_spec(name) else False
    except Exception:
        out["packages"][name] = False
try:
    import torch
    out["torch"] = {
        "version": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception:
    out["torch"] = None
print(json.dumps(out))
`;
}

async function exists(p: string): Promise<boolean> {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

/**
 * Is the ollama binary present? The daemon may run without it, but without
 * the CLI there is nothing to install from and nothing to point at.
 *
 * Every OS is checked honestly. This used to answer "yes" on macOS and Windows
 * so the runtime would not offer an install it could not perform — which meant
 * the row read as installed on a machine that had never heard of Ollama, and
 * there was no way to fix it from inside the app.
 */
async function ollamaInstalled(): Promise<boolean> {
  if (process.platform === 'win32') {
    // The Windows installer drops the CLI in the per-user programs folder and
    // nowhere else, so PATH is not a reliable answer here.
    const exe = path.join(
      process.env.LOCALAPPDATA ?? path.join(homedir(), 'AppData', 'Local'),
      'Programs',
      'Ollama',
      'ollama.exe',
    );
    if (await exists(exe)) return true;
    return onPath('ollama.exe');
  }
  // Homebrew's prefix is not always on PATH for a process the user never
  // logged in with, and that is exactly where `brew install ollama` puts it.
  for (const dir of ['/usr/local/bin/ollama', '/opt/homebrew/bin/ollama', '/usr/bin/ollama']) {
    if (await exists(dir)) return true;
  }
  return onPath('ollama');
}

/** True when `name` resolves on this machine's PATH, by spawning `which`. */
function onPath(name: string): Promise<boolean> {
  return new Promise<boolean>((resolve) => {
    const done = (ok: boolean) => resolve(ok);
    const child: ChildProcess =
      process.platform === 'win32'
        ? spawn('where', [name], { stdio: 'ignore' })
        : spawn('sh', ['-c', `command -v ${name}`], { stdio: 'ignore' });
    child.on('error', () => done(false));
    child.on('close', (code) => done(code === 0));
  });
}

/* ---- llama.cpp (pinned prebuilt binary, installed into the app) ---- */

/**
 * The llama.cpp build this app installs.
 *
 * A build number, not the vX.Y.Z tag: the tag releases are source-only, and
 * every platform binary is published on the rolling `bNNNNN` releases. Pinned
 * rather than "latest" so an install is reproducible and a bad upstream build
 * cannot change what lands on someone's disk unasked. Bump it deliberately.
 */
const LLAMA_BUILD = 'b11706';

/** Where the pinned engine lives. Upstream's tarball unpacks a flat
 *  `llama-<build>/` directory (the binaries sit at its top level, beside the
 *  shared libraries, not under a build/ tree), so the layout is discovered on
 *  install and remembered here. */
const llamaDir = (): string => path.join(runtimesDir(), 'llama.cpp');
const llamaServerName = (): string =>
  process.platform === 'win32' ? 'llama-server.exe' : 'llama-server';
/** The HTTP server binary; this is what the chat provider talks to. */
function llamaServerPath(): string | null {
  // Recorded at install time: the directory that actually holds the binary,
  // so removal deletes exactly that tree rather than guessing at a layout.
  const marker = path.join(llamaDir(), 'build.txt');
  try {
    const dir = readFileSync(marker, 'utf8').trim();
    if (dir) return path.join(runtimesDir(), dir, llamaServerName());
  } catch {
    /* not installed, or the marker is gone */
  }
  return null;
}

/** True when the pinned server binary exists — llama.cpp's own "installed". */
async function llamaCppInstalled(): Promise<boolean> {
  const p = llamaServerPath();
  if (!p) return false;
  try {
    const info = await stat(p);
    return info.isFile();
  } catch {
    return false;
  }
}

/** CPU asset for this machine, or null when upstream has no build for it. */
function llamaCpuAsset(): string | null {
  if (process.platform !== 'linux') return null;
  const arch = process.arch === 'arm64' ? 'arm64' : 'x64';
  return `llama-${LLAMA_BUILD}-bin-ubuntu-${arch}.tar.gz`;
}

/** One venv's answer to the probe, or null when that venv has no interpreter.
 *  A parse failure counts as "nothing installed there", which is what an empty
 *  environment looks like anyway. */
type VenvProbe = {
  packages: Record<string, boolean>;
  torch: { version: string; cuda: boolean; count: number; name: string | null } | null;
};

/**
 * Probe answers, cached because the probe is expensive: a Python start plus a
 * torch import is 20–40 s on a CPU-only box, and the answer only changes when
 * this app installs or removes something. Installs, removals and venv creation
 * drop the cache, so no button is ever offered from an answer the app itself
 * has just made wrong; the TTL only bounds how long a change made outside the
 * app (a manual pip) stays invisible.
 */
const probeCache = new Map<string, { at: number; value: VenvProbe | null }>();
const probesInFlight = new Map<string, Promise<VenvProbe | null>>();
const PROBE_TTL_MS = 15 * 60_000;
// Keys carry the epoch so a probe that started before a pip run can never
// resolve afterwards and write its stale answer into the fresh cache.
let probeEpoch = 0;

/** Drops cached probe answers so the next status call re-reads the venvs. */
export function invalidateRuntimeProbes(): void {
  probeEpoch += 1;
  probeCache.clear();
}

function runVenvProbe(py: string): Promise<VenvProbe> {
  return new Promise((resolve, reject) => {
    const child = spawn(py, ['-c', buildProbe()], { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), 60_000);
    child.stdout.on('data', (d) => (out += d));
    child.on('error', reject);
    child.on('close', () => {
      clearTimeout(timer);
      try {
        resolve(JSON.parse(out.trim().split('\n').pop() ?? '{}'));
      } catch {
        resolve({ packages: {}, torch: null });
      }
    });
  });
}

async function probeVenv(name: string): Promise<VenvProbe | null> {
  const py = venvPython(name);
  // The interpreter check stays in front of the cache: a venv deleted from
  // under us must read as missing, not as its last answer.
  if (!(await exists(py))) return null;
  const key = `${probeEpoch}:${name}`;
  const cached = probeCache.get(key);
  if (cached && Date.now() - cached.at < PROBE_TTL_MS) return cached.value;
  // Concurrent callers — the models page asks for runtimes twice at once —
  // share one probe rather than starting a second torch import beside it.
  const running = probesInFlight.get(key);
  if (running) return running;
  const probe = runVenvProbe(py);
  probesInFlight.set(key, probe);
  try {
    const value = await probe;
    probeCache.set(key, { at: Date.now(), value });
    return value;
  } finally {
    probesInFlight.delete(key);
  }
}

/** Runs the probe in every venv and reports per group. Returns a blank status
 *  if the shared venv has no interpreter yet. */
export async function runtimeStatus(busy = false): Promise<RuntimeStatus> {
  const blank: RuntimeStatus = {
    venv: false,
    python: null,
    installed: {},
    torch: { version: null, cudaAvailable: false, deviceCount: 0, deviceName: null },
    busy,
  };
  const main = await probeVenv('venv');
  if (!main) return blank;

  // A group with its own venv is answered by that venv's probe, so "is Kokoro
  // installed?" asks the interpreter Kokoro would actually run on, not the one
  // that cannot import it.
  const side = new Map<string, Awaited<ReturnType<typeof probeVenv>>>();
  for (const g of RUNTIME_GROUPS) {
    const name = venvName(g);
    if (name !== 'venv' && !side.has(name)) side.set(name, await probeVenv(name));
  }

  // Every group is detected the same way, from the same table the installer
  // writes to. A new group is installable and detectable in the same commit.
  const installed: Record<string, string | null> = {};
  for (const g of RUNTIME_GROUPS) {
    if (g.installer === 'ollama') {
      installed[g.id] = (await ollamaInstalled()) ? 'installed' : null;
      continue;
    }
    if (g.installer === 'llama-cpp') {
      installed[g.id] = (await llamaCppInstalled()) ? 'installed' : null;
      continue;
    }
    if (g.id === 'torch') {
      installed[g.id] = main.torch?.version ?? null;
      continue;
    }
    const probe = venvName(g) === 'venv' ? main : (side.get(venvName(g)) ?? null);
    installed[g.id] = probe?.packages?.[g.probe] ? 'installed' : null;
  }

  return {
    venv: true,
    python: venvPython(),
    installed,
    torch: {
      version: main.torch?.version ?? null,
      cudaAvailable: main.torch?.cuda ?? false,
      deviceCount: main.torch?.count ?? 0,
      deviceName: main.torch?.name ?? null,
    },
    busy,
  };
}

/**
 * Creates a venv if it is missing, on the newest interpreter this machine has
 * that satisfies the bound its groups declare, and returns its interpreter.
 * ensurepip provides pip inside it, which is why a system pip is not required.
 */
export async function ensureVenv(
  name: string,
  bound: string | undefined,
  onProgress: (line: string) => void,
): Promise<void> {
  if (await exists(venvPython(name))) {
    onProgress('Reusing the existing environment.');
    return;
  }
  const interpreter = await findInterpreter(bound);
  if (!interpreter) {
    throw new Error(
      bound
        ? `This needs Python ${bound}, and no installed interpreter satisfies it. ` +
            'Install one (for example python3.12), or set RUNTIMES_PYTHON to point at one.'
        : 'No python3 interpreter was found on this machine.',
    );
  }
  await mkdir(runtimesDir(), { recursive: true });
  onProgress(
    name === 'venv'
      ? 'Creating the Python environment…'
      : `Creating ${name} with ${interpreter}${bound ? ` (needs Python ${bound})` : ''}…`,
  );
  await new Promise<void>((resolve, reject) => {
    // -m venv, not the venv console script: no activation state, no shell.
    const child = spawn(interpreter, ['-m', 'venv', venvDir(name)], { stdio: ['ignore', 'pipe', 'pipe'] });
    let err = '';
    child.stderr.on('data', (d) => (err += d));
    child.on('error', reject);
    child.on('close', (code) =>
      code === 0 ? resolve() : reject(new Error(describeVenvFailure(err))),
    );
  });
  if (!(await exists(venvPython(name)))) throw new Error('The environment was created but has no interpreter.');
  venvVersions.delete(name);
  invalidateRuntimeProbes();
}

/**
 * Turns a failed `python -m venv` into something the user can act on.
 *
 * The stock failure on a fresh Ubuntu names ensurepip and nothing else, which
 * sends the user to a Python bug rather than the package they are missing:
 * Ubuntu and Debian split the venv module and ensurepip into `python3-venv`,
 * which a minimal install does not pull in. So the known causes get the exact
 * package, and anything else keeps the interpreter's own message, which is more
 * use than a guess would be.
 */
export function describeVenvFailure(stderr: string): string {
  const text = stderr.trim();
  if (/\bensurepip\b/i.test(text)) {
    return (
      'This Python cannot create environments — the venv module is missing. ' +
      'On Ubuntu/Debian: sudo apt install python3-venv (or python3.<version>-venv, ' +
      'to match this interpreter)   ' +
      '(Fedora: sudo dnf install python3-libs; Arch: sudo pacman -S python). ' +
      'Then install the runtime again.'
    );
  }
  if (/No module named venv/i.test(text)) {
    return (
      'This Python was built without the venv module. ' +
      'On Ubuntu/Debian: sudo apt install python3-venv, or point RUNTIMES_PYTHON at ' +
      'another interpreter that has it.'
    );
  }
  return `python -m venv failed: ${text.slice(0, 300)}`;
}

/**
 * Why a group cannot be installed here, or null when it can.
 *
 * A group in the shared venv is judged by the interpreter that venv runs on. A
 * group with its own venv is judged by the best interpreter available for its
 * bound — before the venv exists there is nothing to ask, and after it exists
 * its own version answers. This is what the Runtimes page and the install route
 * both show, so the button and the refusal always tell the same story.
 */
export async function groupBlockedReason(group: RuntimeGroup): Promise<string | null> {
  const name = venvName(group);
  const version = await venvPythonVersion(name);

  // The venv exists and runs an interpreter: that is the final answer.
  if (version) {
    if (pythonSatisfies(version, group.maxPython)) return null;
    return name === 'venv'
      ? `${group.label} needs Python ${group.maxPython}; the environment runs ${version}.`
      : `${group.label} needs Python ${group.maxPython}, but its own environment runs ${version}.`;
  }

  // No venv yet. The shared one will be built on the machine's python3, which
  // every current bound tolerates; a group with its own venv needs a
  // compatible interpreter to exist before it can be built at all.
  if (name === 'venv') return null;
  if (await findInterpreter(group.maxPython)) return null;
  return `${group.label} needs Python ${group.maxPython} — none is installed. Install one (for example python3.12) and its own environment will be built on it.`;
}

export interface InstallOpts {
  groups: RuntimeGroup[];
  device: Device;
  onProgress: (line: string) => void;
  /**
   * The user's password, used once to satisfy `sudo` and never kept.
   *
   * Reaches the child on stdin only, never argv — argv is world-readable
   * through `ps`. Optional, because almost nothing a runtime install does
   * needs root: it is read only when an installer actually asks.
   */
  password?: string;
  /** Called when an installer needs root it could not get on its own. */
  onNeedsPrivilege?: (command: string) => void;
}

/** The one part of InstallOpts a single installer needs to see. */
type PrivilegedInstall = Pick<InstallOpts, 'onProgress' | 'password' | 'onNeedsPrivilege'>;

/**
 * Did an install stop because it wanted a password it had no way to get?
 *
 * Re-exported from the driver installer rather than re-listed here: both paths
 * hit the same wall (no terminal to type into), and two copies of the pattern
 * list would drift into disagreeing about what counts.
 */
export { needsPrivilege } from './drivers.js';


/**
 * Installs the given groups into their venvs, torch first when anything needs
 * it. One pip invocation per group keeps the progress log readable and means a
 * failure names the group that caused it.
 *
 * Groups are bucketed by venv: the shared one always exists — the probe, the
 * device readout and most runners live there — and a group with its own venv
 * gets one built on an interpreter its bound allows before anything is
 * installed into it.
 */
export async function installRuntimes({
  groups,
  device,
  onProgress,
  password,
  onNeedsPrivilege,
}: InstallOpts): Promise<void> {
  // The shared venv is the baseline, so it exists from the first install on.
  await ensureVenv('venv', undefined, onProgress);

  // pip is about to change these venvs: drop the cached answers up front and
  // again when it is finished, so a failed run cannot leave a stale half in
  // place either.
  invalidateRuntimeProbes();

  const members = new Map<string, RuntimeGroup[]>();
  for (const g of groups) {
    const name = venvName(g);
    const list = members.get(name) ?? [];
    list.push(g);
    members.set(name, list);
  }

  const presence = await detectGpu();
  const common = ['-m', 'pip', 'install', '--disable-pip-version-check', '--no-input'];
  const listed = new Set(groups.map((g) => g.id));

  const runPip = (py: string, args: string[], label: string) =>
    new Promise<void>((resolve, reject) => {
      const child = spawn(py, args, { stdio: ['ignore', 'pipe', 'pipe'] });
      const bump = (buf: Buffer) => {
        for (const line of buf.toString().split(/\r?\n/)) {
          if (line.trim()) onProgress(line.trim());
        }
      };
      child.stdout.on('data', bump);
      child.stderr.on('data', bump);
      child.on('error', reject);
      child.on('close', (code) =>
        code === 0
          ? resolve()
          : reject(new Error(`${label} failed (pip exited ${code}) — see the log above.`)),
      );
    });

  for (const [name, list] of members) {
    const bound = list.map((g) => g.maxPython).find((b) => b) ?? undefined;
    await ensureVenv(name, bound, onProgress);
    const py = venvPython(name);
    const where = name === 'venv' ? 'the environment' : `${list[0].label}'s own environment`;

    if (list.some((g) => g.id === 'torch' || g.needsTorch)) {
      const resolved = resolveIndex(device, presence);
      let deviceNote: string;
      if (device === 'cpu') deviceNote = 'CPU wheels';
      else if (device === 'nvidia') deviceNote = 'CUDA wheels (NVIDIA)';
      else if (device === 'amd') deviceNote = 'ROCm wheels (AMD)';
      else deviceNote = presence.vendor ? `${presence.vendor} wheels (auto-detected)` : 'CPU wheels (no GPU found)';
      onProgress(`Installing PyTorch into ${where} — ${deviceNote}. This is the big one…`);
      await runPip(py, [...common, '--index-url', resolved, 'torch'], 'PyTorch');
      onProgress('PyTorch installed.');
    }

    for (const g of list) {
      if (g.id === 'torch') continue;
      if (g.installer === 'ollama') {
        await installOllama({ onProgress, password, onNeedsPrivilege });
        continue;
      }
      if (g.installer === 'llama-cpp') {
        await installLlamaCpp(onProgress);
        continue;
      }
      if (!g.pkgs.length) continue;
      onProgress(`Installing ${g.label} into ${where}…`);
      await runPip(py, [...common, ...g.pkgs], g.label);
      onProgress(`${g.label} installed.`);
    }
  }

  if (listed.size === 0) onProgress('Nothing selected.');
  invalidateRuntimeProbes();
}

/**
 * Ollama's own installer, run the way this OS does it.
 *
 * Deliberately not a pkgs entry: Ollama is not a Python package, it ships an
 * installer, and on Linux that installer wants root. It is the only runtime
 * that can escalate, so it is also the only one that can ask for a password —
 * and it does that only when the step actually needs one.
 */
async function installOllama(opts: PrivilegedInstall): Promise<void> {
  const { onProgress, password, onNeedsPrivilege } = opts;
  if (await ollamaInstalled()) {
    onProgress('Ollama is already installed.');
    return;
  }

  const recipe = ollamaRecipe();
  if (!recipe) {
    // Nothing to run on this OS. Say so with the vendor's own instructions
    // rather than a dead end.
    throw new Error(
      'There is no scripted Ollama installer for this OS. Download it from ' +
        'https://ollama.com/download and re-check this page.',
    );
  }

  // A password is only ever offered for a step that genuinely needs root. On
  // macOS and Windows the installer runs as the user, so there is nothing to
  // ask for and the prompt never opens.
  const elevates = recipe.needsRoot;
  onProgress(`Installing Ollama on ${platformLabel()} — ${recipe.shown}…`);

  const scrub = (text: string) =>
    password && password.length >= 3 ? text.split(password).join('[redacted]') : text;
  // Whether sudo is on the other end of the pipe decides stdin: a password has
  // to be written, anything else is told to ignore input.
  const escalating = Boolean(password && elevates);
  let tail = '';
  let code: number;
  try {
    code = await new Promise<number>((resolve) => {
      const [cmd, cmdArgs] = escalating
        ? ['sudo', ['-S', '-p', '', ...recipe.cmd, ...recipe.args]]
        : [recipe.cmd[0], recipe.args];
      const child: ReturnType<typeof spawn> = spawn(cmd, cmdArgs, {
        stdio: escalating ? ['pipe', 'pipe', 'pipe'] : ['ignore', 'pipe', 'pipe'],
      });
      if (escalating) {
        child.stdin?.on('error', () => {
          /* sudo exited early — a wrong password does exactly that */
        });
        child.stdin?.end(`${password}\n`);
      }
      const bufs: Record<'out' | 'err', string> = { out: '', err: '' };
      const pump = (stream: NodeJS.ReadableStream | null, key: 'out' | 'err') => {
        stream?.setEncoding('utf8');
        stream?.on('data', (c: string) => {
          bufs[key] += c;
          const lines = bufs[key].split('\n');
          bufs[key] = lines.pop() ?? '';
          for (const line of lines) emit(line);
        });
      };
      const emit = (raw: string) => {
        const text = scrub(raw.replace(/\s+$/, ''));
        if (!text.trim()) return;
        onProgress(text.trim());
        tail = `${tail}\n${text}`.slice(-4000);
      };
      pump(child.stdout, 'out');
      pump(child.stderr, 'err');
      child.on('error', (err) => {
        const missing = (err as NodeJS.ErrnoException).code === 'ENOENT';
        emit(
          missing && recipe.cmd[0] === 'brew'
            ? 'Homebrew is not installed on this machine.'
            : `Could not start the installer: ${err.message}`,
        );
        resolve(127);
      });
      child.on('close', (c) => {
        for (const key of ['out', 'err'] as const) {
          emit(bufs[key]);
          bufs[key] = '';
        }
        resolve(c ?? 0);
      });
    });
  } catch (err) {
    throw new Error(`Could not start the Ollama installer: ${(err as Error).message}`);
  }

  if (password && /incorrect password|Sorry, try again|authentication failure/i.test(tail)) {
    throw new Error('That password was not accepted, so nothing was changed.');
  }

  // It stopped because it wanted a password it had no way to get, not because
  // the installer is broken. That distinction is the whole reason this checks
  // the output rather than the exit code.
  if (code !== 0 && needsPrivilege(tail)) {
    const shown = `sudo ${recipe.shown}`;
    onNeedsPrivilege?.(shown);
    throw new Error(
      'The Ollama installer needs root and could not elevate. ' +
        'Supply your password to let it run, or run this in a terminal yourself: ' +
        shown,
    );
  }

  if (code !== 0) {
    // A missing Homebrew is the difference between "this will never work" and
    // "install this one thing first", and the log line above only half says it.
    if (recipe.cmd[0] === 'brew' && /Homebrew is not installed/.test(tail)) {
      throw new Error(
        'Homebrew is not installed, and macOS installs Ollama through it. ' +
          'Install Homebrew from https://brew.sh, then install this again.',
      );
    }
    throw new Error(`The Ollama installer exited ${code} — see the log above.`);
  }
  onProgress('Ollama installed. Start it with `ollama serve` if the daemon is not already running.');
}

/** How to install Ollama on this OS, as the vendor documents it. */
export function ollamaRecipe(
  platform: NodeJS.Platform = process.platform,
):
  | { shown: string; cmd: string[]; args: string[]; needsRoot: boolean }
  | null {
  if (platform === 'win32') {
    // The vendor's own PowerShell one-liner. It installs per-user, so it does
    // not need an elevated prompt.
    return {
      shown: 'irm https://ollama.com/install.ps1 | iex',
      cmd: ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command'],
      args: ['irm https://ollama.com/install.ps1 | iex'],
      needsRoot: false,
    };
  }
  if (platform === 'darwin') {
    return {
      shown: 'brew install ollama',
      cmd: ['brew'],
      args: ['install', 'ollama'],
      // Homebrew refuses to run as root, so escalating here would break it
      // rather than help it. If brew still needs help the user is told the
      // command to run in a terminal.
      needsRoot: false,
    };
  }
  if (platform === 'linux') {
    return {
      shown: 'curl -fsSL https://ollama.com/install.sh | sh',
      cmd: ['sh', '-c'],
      args: ['curl -fsSL https://ollama.com/install.sh | sh'],
      needsRoot: true,
    };
  }
  return null;
}

/** A human name for this OS, for the install line in the log. */
function platformLabel(): string {
  if (process.platform === 'darwin') return 'macOS';
  if (process.platform === 'win32') return 'Windows';
  if (process.platform === 'linux') return 'Linux';
  return process.platform;
}

/**
 * Downloads and unpacks the pinned llama.cpp CPU build.
 *
 * No root and no Python, which is the whole point on this kind of box: it
 * installs into the app's own runtimes folder and can be removed the same way
 * Ollama's can. The tarball is fetched with Node's fetch and unpacked with
 * the system `tar`, so nothing is piped into a shell.
 */
async function installLlamaCpp(onProgress: (line: string) => void): Promise<void> {
  if (await llamaCppInstalled()) {
    onProgress('llama.cpp is already installed.');
    return;
  }
  const asset = llamaCpuAsset();
  if (!asset) {
    throw new Error(
      'The bundled llama.cpp installer is Linux-only (the project publishes ' +
        'per-platform CPU builds; see its releases for macOS and Windows).',
    );
  }

  onProgress(`Fetching llama.cpp ${LLAMA_BUILD} (CPU build)…`);
  const url = `https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/${asset}`;
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`Could not download ${asset} (HTTP ${res.status}).`);
  const bytes = Buffer.from(await res.arrayBuffer());

  await mkdir(llamaDir(), { recursive: true });
  const tarball = path.join(llamaDir(), asset);
  const { writeFile, chmod } = await import('node:fs/promises');
  await writeFile(tarball, bytes);
  onProgress(`Unpacking ${(bytes.length / 1024 / 1024).toFixed(0)} MB…`);
  await new Promise<void>((resolve, reject) => {
    // The archive unpacks into llama-<build>/…; extract into the runtimes dir
    // and remember which directory it was, since upstream's exact layout is
    // not something this app should hard-code.
    const child = spawn('tar', ['-xzf', tarball, '-C', runtimesDir()], {
      stdio: ['ignore', 'ignore', 'pipe'],
    });
    let err = '';
    child.stderr.on('data', (b) => (err += String(b)));
    child.on('error', reject);
    child.on('close', (code) =>
      code === 0
        ? resolve()
        : reject(new Error(`tar exited ${code}${err.trim() ? ` — ${err.trim()}` : ''}`)),
    );
  });
  await rm(tarball, { force: true });

  // Find the directory that actually holds llama-server, however it nested it.
  const unpacked = (await readdir(runtimesDir(), { withFileTypes: true }))
    .filter((e) => e.isDirectory() && e.name.startsWith('llama-'))
    .map((e) => e.name);
  let found: string | null = null;
  for (const name of unpacked) {
    const base = path.join(runtimesDir(), name);
    const stack = [base];
    while (stack.length) {
      const dir = stack.pop()!;
      let entries;
      try {
        entries = await readdir(dir, { withFileTypes: true });
      } catch {
        continue;
      }
      for (const e of entries) {
        const full = path.join(dir, e.name);
        if (e.isFile() && e.name === llamaServerName()) {
          found = path.relative(runtimesDir(), path.dirname(full));
          await chmod(full, 0o755);
          break;
        } else if (e.isDirectory()) {
          stack.push(full);
        }
      }
      if (found) break;
    }
    if (found) break;
  }
  if (!found) {
    throw new Error(
      'The build unpacked, but no llama-server binary was found. The archive ' +
        'layout may have changed; fetch it by hand from the llama.cpp releases.',
    );
  }
  // Store the directory that holds the binary itself (not its grandparent),
  // so removal deletes exactly the unpacked tree and nothing beside it.
  const serverDir = found ? path.dirname(path.join(runtimesDir(), found, llamaServerName())) : null;
  await writeFile(
    path.join(llamaDir(), 'build.txt'),
    serverDir ? path.relative(runtimesDir(), serverDir) : '',
  );
  onProgress('llama.cpp installed. Its GGUF chat models can be run with llama-server.');
}

/** Removes the whole environment. Everything in it goes; there is no partial
 *  uninstall, because pip cannot reliably drop torch's shared CUDA libraries. */
export async function removeRuntimes(): Promise<void> {
  await rm(runtimesDir(), { recursive: true, force: true });
  venvVersions.clear();
  invalidateRuntimeProbes();
}

/**
 * Removes one runtime group, leaving the rest of the environment alone.
 *
 * "Remove environment" is a blank cheque — it takes torch, transformers,
 * diffusers, faster-whisper and kokoro with it, and on this kind of box that is
 * hundreds of megabytes that may have been there for months. So a group can also
 * be removed on its own, which is `pip uninstall` against that group's fixed
 * package list.
 *
 * What it cannot do is remove Ollama. That is not a pip package in the venv: it
 * is a system-wide install put there by the vendor's own script, with shared
 * libraries the daemon needs. Saying so is better than a button that quietly
 * removed something else instead.
 */
export async function removeRuntimeGroup(
  id: string,
  onProgress: (line: string) => void,
): Promise<{ removed: string; label: string; reason: string | null }> {
  const group = groupById(id);
  if (!group) throw new Error(`Unknown runtime: ${id}`);

  if (group.installer === 'ollama') {
    return {
      removed: id,
      label: group.label,
      reason:
        'Ollama is installed system-wide by its own installer, not into this ' +
        'environment, so it cannot be removed from here. Use its own uninstall ' +
        'instructions if you want it gone.',
    };
  }
  if (group.installer === 'llama-cpp') {
    onProgress('Removing llama.cpp…');
    // The marker records the directory holding the binary; drop that whole
    // tree so a remove actually reclaims the space, not just the marker.
    const server = llamaServerPath();
    if (server) await rm(path.dirname(server), { recursive: true, force: true });
    await rm(llamaDir(), { recursive: true, force: true });
    onProgress('llama.cpp removed.');
    return { removed: id, label: group.label, reason: null };
  }
  if (!group.pkgs.length) {
    return {
      removed: id,
      label: group.label,
      reason: `${group.label} installs nothing removable in this environment.`,
    };
  }

  const name = venvName(group);
  const py = venvPython(name);
  if (!(await exists(py))) {
    throw new Error(
      name === 'venv'
        ? 'There is no Python environment to remove anything from.'
        : `There is no ${group.label} environment to remove.`,
    );
  }

  // A side venv exists for exactly this group, so removing the group means
  // removing the environment — torch and the rest of its tree go with it
  // rather than being left behind in a venv nothing else uses.
  if (name !== 'venv') {
    onProgress(`Removing the ${group.label} environment (${group.pkgs.join(', ')})…`);
    await rm(venvDir(name), { recursive: true, force: true });
    venvVersions.delete(name);
    invalidateRuntimeProbes();
    onProgress(`${group.label} removed.`);
    return { removed: id, label: group.label, reason: null };
  }

  onProgress(`Removing ${group.label} (${group.pkgs.join(', ')})…`);
  await new Promise<void>((resolve, reject) => {
    const child = spawn(
      py,
      ['-m', 'pip', 'uninstall', '-y', '--disable-pip-version-check', '--no-input', ...group.pkgs],
      { stdio: ['ignore', 'pipe', 'pipe'] },
    );
    const bump = (buf: Buffer) => {
      for (const line of buf.toString().split(/\r?\n/)) {
        if (line.trim()) onProgress(line.trim());
      }
    };
    child.stdout.on('data', bump);
    child.stderr.on('data', bump);
    child.on('error', reject);
    child.on('close', (code) =>
      code === 0
        ? resolve()
        : reject(
            new Error(`Removing ${group.label} failed (pip exited ${code}) — see the log above.`),
          ),
    );
  });

  onProgress(`${group.label} removed.`);
  invalidateRuntimeProbes();
  return { removed: id, label: group.label, reason: null };
}

/** Explains a device choice that can't work, before the user commits to a
 *  multi-hundred-megabyte download. Returns null when the choice is fine. */
export async function deviceWarning(device: Device): Promise<string | null> {
  if (device === 'cpu') return null;
  const presence = await detectGpu();
  if (presence.needsGroup) return presence.detail;

  // A card that exists but cannot run the requested backend is worth warning
  // about for Auto too, not only for an explicit choice: Auto is what gets
  // selected by default, so it is exactly where a silent mistake is worst.
  if (presence.vendor === null) return presence.detail;

  if (device === 'auto') {
    if (presence.usable === false) return presence.detail;
    return null;
  }
  if (device === 'nvidia') {
    return presence.vendor === 'nvidia'
      ? null
      : presence.vendor === 'amd'
        ? 'This looks like an AMD machine. Choose the AMD (ROCm) option — CUDA wheels will not load.'
        : 'No NVIDIA GPU detected. CUDA wheels will install but cannot run.';
  }
  if (device === 'amd') {
    return presence.vendor === 'amd'
      ? presence.usable === false
        ? presence.detail
        : null
      : presence.vendor === 'nvidia'
        ? 'This looks like an NVIDIA machine. Choose the NVIDIA (CUDA) option.'
        : 'No AMD GPU detected. ROCm wheels will install but cannot run.';
  }
  return null;
}