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
import { spawn, spawnSync } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import { needsPrivilege } from './drivers.js';
import {
  accessSync,
  constants,
  readFileSync,
  readdirSync,
  readlinkSync,
  realpathSync,
  statSync,
} from 'node:fs';
import { mkdir, mkdtemp, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
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
 * The first existing folder at or above `dir` that this process cannot write to,
 * with the uid that owns it. Null when somewhere along the way it can write.
 *
 * This exists to answer "needs sudo" with the truth rather than a guess. The
 * answer is almost never yes: nothing about building a Python environment in
 * your own folder requires root, and the one time it fails is that the folder
 * was *already* created by root — by an install run under `sudo` — so the fix is
 * to hand the folder back to you, not to build the next thing as root.
 *
 * Walking up matters because `.runtimes` is often created by this same code, and
 * a missing leaf is not the problem; the unwritable parent is.
 */
export function unwritableAncestor(dir: string): { path: string; uid: number } | null {
  let cur = path.resolve(dir);
  for (;;) {
    let uid: number | null = null;
    try {
      uid = statSync(cur).uid;
    } catch {
      // Does not exist yet — the answer, if there is one, is above it.
    }
    if (uid !== null) {
      try {
        accessSync(cur, constants.W_OK);
        return null;
      } catch {
        return { path: cur, uid };
      }
    }
    const up = path.dirname(cur);
    if (up === cur) return null;
    cur = up;
  }
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
      'this app’s own folder — no root, no Python, no compiler. The build ' +
      'matches the card: CUDA on NVIDIA, Vulkan where a render node and ' +
      'vulkaninfo are both present, CPU otherwise. Download the quantized ' +
      'chat models in Get models (Qwen3, Llama 3.2, Gemma) and they show up ' +
      'in the chat provider picker, run on this machine. A GPU build is a much ' +
      'larger download than the CPU one — roughly 173 MB, and up to 766 MB ' +
      'more if this machine has no CUDA runtime of its own.',
    unlocks: ['text'],
    // Not a pip package either: the upstream project ships prebuilt CPU
    // tarballs per build. Kept beside Ollama so "the two chat runtimes" read
    // as a pair, but this one installs into the app and can be removed again.
    pkgs: [],
    needsTorch: false,
    // The CPU and Vulkan builds are ~17 MB. The CUDA build is ~173 MB, plus a
// ~594 MB CUDA runtime archive when the machine has no runtime of its own —
// and "≈ 17 MB" on a card that will download 766 is worse than no figure at
// all. The real figure is only known once the machine has been inspected, so
// the Runtimes page is told it in the blurb rather than here, where a static
// number cannot be right for more than one of the three builds.
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
  /**
   * Whether Ollama is pinned to its Vulkan backend by this app's drop-in.
   * Separate from `installed`, because it is how Ollama will run rather than
   * whether it is present. False on every platform but Linux, and false when
   * the drop-in is absent — which includes machines where Ollama happens to
   * have Vulkan on by default, so this means "configured", not "using".
   */
  ollamaVulkan: boolean;
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
        # Which CUDA runtime this build links against. The llama.cpp CUDA build
        # has to match it: upstream ships 12.8 and 13.4 of the same pin, and
        # picking wrong gives a build that loads and then fails on the first device.
        "cuda_version": getattr(torch.version, "cuda", None),
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
export function llamaServerPath(): string | null {
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
/** Which backend the installed build actually has, read off the shared
 *  libraries beside the binary rather than off its name. */
/** What kind of llama.cpp build is installed here. */
type LlamaKind = 'vulkan' | 'cuda' | 'cpu';

function installedLlamaKind(): LlamaKind | null {
  const p = llamaServerPath();
  if (!p) return null;
  try {
    const dir = path.dirname(p);
    const entries = readdirSync(dir);
    // CUDA first: a CUDA build bundles its own copy of the Vulkan loader's
    // neighbour libraries in some builds, and the name that proves the build is
    // the CUDA one is the ggml backend it was compiled against.
    if (entries.some((n) => /^libggml-cuda/i.test(n))) return 'cuda';
    if (entries.some((n) => /^libggml-vulkan/i.test(n))) return 'vulkan';
    return 'cpu';
  } catch {
    return null;
  }
}

/** Whether a Vulkan-capable GPU is actually usable on this machine. */
export function hasVulkan(): boolean {
  // A render node is the real signal that a GPU is present; vulkaninfo being
  // on PATH means the ICD (driver) is installed, which is what llama.cpp's
  // Vulkan backend needs. Either missing → the CPU build, which always works.
  try {
    if (!readdirSync('/dev/dri').some((n) => /^renderD\d+$/.test(n))) return false;
  } catch {
    return false;
  }
  try {
    return spawnSync('vulkaninfo', ['--summary'], { stdio: 'ignore' }).status === 0;
  } catch {
    return false;
  }
}

/**
 * Which CUDA build this machine can actually run.
 *
 * Upstream publishes CUDA 12.8 and 13.4 of the same pin and they are not
 * interchangeable: 13.4 needs a newer driver, and asking for it on a machine
 * that has an older one produces a server that unpacks fine and then dies on the
 * first device with "CUDA driver version is insufficient for CUDA runtime
 * version".
 *
 * torch is asked first, because its version is the one already known to run on
 * this machine — matching it answers the question rather than guessing at it.
 *
 * The fallback is the driver, mapped through NVIDIA's own support matrix rather
 * than compared against a CUDA number: CUDA 12.x needs 525 or newer, CUDA 13.x
 * needs 580 or newer. An earlier version of this read the driver's major as if
 * it were a CUDA major, so 550 — a perfectly normal 12.x driver — came out as
 * 13.4 and the wrong build was fetched. When the driver is older than either,
 * the answer is null and the CPU build is the honest one.
 */
export function cudaRuntimeVersion(): string | null {
  const cached = probeCache.get('venv')?.value ?? null;
  const t = cached?.torch?.cudaVersion ? String(cached.torch.cudaVersion) : null;
  if (t && /^\d+\.\d+$/.test(t)) return t;
  const smi = spawnSync('nvidia-smi', ['--query-gpu=driver_version', '--format=csv,noheader'], {
    encoding: 'utf8',
  });
  if (smi.status !== 0 || !smi.stdout) return null;
  // "550.54.14" — the major is the driver series, which is what the minimums
  // below are stated in.
  const major = Number(/^(\d+)\./.exec(String(smi.stdout).trim())?.[1]);
  if (!Number.isFinite(major) || major <= 0) return null;
  if (major >= 580) return '13.4';
  if (major >= 525) return '12.8';
  return null;
}

/** Whether an NVIDIA card this app can build for is present. */
export function hasNvidiaGpu(): boolean {
  return spawnSync('nvidia-smi', ['-L'], { stdio: 'ignore' }).status === 0;
}

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

/** Non-null when a better (GPU) build is available than the one installed. */
export async function llamaCppUpgradeKind(): Promise<LlamaKind | null> {
  const kind = installedLlamaKind();
  if (kind === null) return null; // nothing installed yet — install handles it
  const best = bestLlamaKind();
  if (!best || best === 'cpu') return null;
  return kind === 'cpu' ? best : null;
}

/** The GPU build this machine should be running, or 'cpu'. */
function bestLlamaKind(): LlamaKind {
  // CUDA before Vulkan. An NVIDIA card has no usable Vulkan story here (no
  // render node in a container, and the CPU build is what the Vulkan check then
  // falls back to), so an NVIDIA machine that reached the CPU build is exactly
  // the case this fixes. On a card that answers both, CUDA is the native path.
  if (hasNvidiaGpu()) return 'cuda';
  if (hasVulkan()) return 'vulkan';
  return 'cpu';
}

/** Which CUDA tag upstream published for this machine, e.g. '12.8'.
 *
 * 12.8 is the default rather than the newest: it runs on every driver from 525
 * up, so a machine too new to have been classified still gets a build that
 * works. Falling forward to 13.4 on an unrecognised driver is what installed a
 * build Colab could not load. */
function cudaTag(): string {
  const v = cudaRuntimeVersion();
  return v === '13.4' ? '13.4' : '12.8';
}

/** The build to fetch for this machine. Null when upstream has no build for this
 *  platform (macOS and Windows are not scripted yet).
 *
 *  The CUDA build is published as a pair: the server itself, and a separate
 *  archive of the CUDA runtime libraries it links against. Both are named, so
 *  the installer can take the runtime only when the machine has none of its
 *  own. */
function llamaAsset(): { name: string; kind: LlamaKind; cudart?: string } | null {
  if (process.platform !== 'linux') return null;
  const arch = process.arch === 'arm64' ? 'arm64' : 'x64';
  const kind = bestLlamaKind();
  if (kind === 'cuda') {
    // Upstream ships CUDA for x64 only; there is no arm64 CUDA build at this pin
    // for 12.8, so an arm64 NVIDIA box keeps the build it can actually run.
    if (arch !== 'x64') return { name: `llama-${LLAMA_BUILD}-bin-ubuntu-${arch}.tar.gz`, kind: 'cpu' };
    const tag = cudaTag();
    return {
      name: `llama-${LLAMA_BUILD}-bin-ubuntu-cuda-${tag}-${arch}.tar.gz`,
      cudart: `cudart-llama-${LLAMA_BUILD}-bin-ubuntu-cuda-${tag}-${arch}.tar.gz`,
      kind: 'cuda',
    };
  }
  if (kind === 'vulkan') return { name: `llama-${LLAMA_BUILD}-bin-ubuntu-vulkan-${arch}.tar.gz`, kind: 'vulkan' };
  return { name: `llama-${LLAMA_BUILD}-bin-ubuntu-${arch}.tar.gz`, kind: 'cpu' };
}

/**
 * Whether this machine already has the CUDA runtime libraries llama.cpp links.
 *
 * The matching cudart archive is 594 MB, against 173 MB for the server itself,
 * so downloading it on a machine that already has them — every Colab runtime,
 * most CUDA installs — is 594 MB of nothing. Looked for where they actually
 * live: the CUDA toolkit, the loader's default paths, and the `nvidia-*` wheels
 * torch installs, which is how a pip-installed torch finds them without any
 * system CUDA at all.
 */
export function hasCudaRuntimeLibs(): boolean {
  const roots = [
    '/usr/local/cuda/lib64',
    '/usr/local/cuda/targets/x86_64-linux/lib',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/lib',
  ];
  for (const root of roots) {
    try {
      if (readdirSync(root).some((n) => /^libcudart\.so/i.test(n))) return true;
    } catch {
      /* not this root */
    }
  }
  // The `nvidia-*` wheels torch pulls in, which is how a pip-installed torch
  // finds libcudart without any system CUDA at all. The directory is named for
  // whatever torch resolved, so it is searched for rather than constructed.
  const libRoot = path.resolve(path.dirname(venvPython()), '..', 'lib');
  return findCudart(libRoot, 0);
}

function findCudart(dir: string, depth: number): boolean {
  if (depth > 6) return false;
  let entries: Array<{ name: string; isDirectory(): boolean }>;
  try {
    entries = readdirSync(dir, { withFileTypes: true }) as unknown as Array<{
      name: string;
      isDirectory(): boolean;
    }>;
  } catch {
    return false;
  }
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const full = path.join(dir, e.name);
    try {
      for (const inner of readdirSync(full, { withFileTypes: true }) as unknown as Array<{
        name: string;
        isDirectory(): boolean;
      }>) {
        const libDir = path.join(full, inner.name, 'lib');
        try {
          if (readdirSync(libDir).some((n) => /^libcudart\.so/i.test(n))) return true;
        } catch {
          /* not a lib dir */
        }
      }
    } catch {
      /* keep looking */
    }
    if (findCudart(full, depth + 1)) return true;
  }
  return false;
}

/** One venv's answer to the probe, or null when that venv has no interpreter.
 *  A parse failure counts as "nothing installed there", which is what an empty
 *  environment looks like anyway. */
type VenvProbe = {
  packages: Record<string, boolean>;
  torch: { version: string; cuda: boolean; count: number; name: string | null; cudaVersion?: string | null } | null;
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

/** Runs the probe in every venv and reports per group.
 *
 *  A machine with no Python environment is not a machine with nothing installed.
 *  This used to return a blank status when the shared venv had no interpreter,
 *  which reported Ollama and llama.cpp as absent too — on a fresh Linux image
 *  with no python3-venv, installing llama.cpp succeeded and the page went on
 *  saying it was not there and offering to install it again. Neither of those
 *  two uses Python at all, so they are answered directly, and only the groups
 *  that really run through pip depend on the venv probe. */
export async function runtimeStatus(busy = false): Promise<RuntimeStatus> {
  const main = await probeVenv('venv');

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
      installed[g.id] = main?.torch?.version ?? null;
      continue;
    }
    const probe = venvName(g) === 'venv' ? main : (side.get(venvName(g)) ?? null);
    installed[g.id] = probe?.packages?.[g.probe] ? 'installed' : null;
  }

  return {
    venv: main !== null,
    python: main ? venvPython() : null,
    installed,
    // Reported separately from `installed` because it is a property of the
    // install's configuration, not of whether Ollama is present. The Runtimes
    // page shows it so "is it on the GPU?" is answerable without a terminal.
    ollamaVulkan: await ollamaVulkanEnabled(),
    torch: {
      version: main?.torch?.version ?? null,
      cudaAvailable: main?.torch?.cuda ?? false,
      deviceCount: main?.torch?.count ?? 0,
      deviceName: main?.torch?.name ?? null,
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
    // An interpreter is not an environment. A venv built without pip — a Python
    // whose ensurepip is missing, an install that died part-way, a directory
    // copied in from elsewhere — passes every check this function used to make,
    // and then every single install fails with "No module named pip". So the
    // question asked here is the one that decides whether pip can run at all.
    if (await venvHasPip(name)) {
      onProgress('Reusing the existing environment.');
      return;
    }
    onProgress('This environment has no pip in it. Repairing it…');
    if (await repairVenv(name, bound, onProgress, false)) {
      venvVersions.delete(name);
      invalidateRuntimeProbes();
      return;
    }
    throw new Error(MISSING_PIP_MESSAGE);
  }
  await createVenv(name, bound, onProgress);
  // A freshly built environment is not automatically a working one: the very
  // machine this came from builds a venv with no pip in it, because its Python
  // has the venv module without the pip wheels. The build reports success; the
  // check below is what notices.
  if (!(await venvHasPip(name))) {
    onProgress('The new environment has no pip in it. Repairing it…');
    if (!(await repairVenv(name, bound, onProgress, true))) throw new Error(MISSING_PIP_MESSAGE);
  }
  venvVersions.delete(name);
  invalidateRuntimeProbes();
}

/** Builds the environment from scratch. */
async function createVenv(
  name: string,
  bound: string | undefined,
  onProgress: (line: string) => void,
): Promise<void> {
  const interpreter = await findInterpreter(bound);
  if (!interpreter) {
    throw new Error(
      bound
        ? `This needs Python ${bound}, and no installed interpreter satisfies it. ` +
            'Install one (for example python3.12), or set RUNTIMES_PYTHON to point at one.'
        : 'No python3 interpreter was found on this machine.',
    );
  }
  // Asked before anything is created, because this is the one failure whose
  // answer is a single command rather than an investigation — and because
  // running the build as root to "fix" it is what caused it.
  const blocked = unwritableAncestor(runtimesDir());
  if (blocked) {
    throw new Error(permissionMessage(blocked.path, blocked.uid));
  }
  await mkdir(runtimesDir(), { recursive: true });
  onProgress(
    name === 'venv'
      ? 'Creating the Python environment…'
      : `Creating ${name} with ${interpreter}${bound ? ` (needs Python ${bound})` : ''}…`,
  );
  try {
    // Probed before the build rather than after the failure: the failure is the
    // only place it is wanted, and asking for it there would mean a second wait
    // with the user already staring at a failure message.
    const version = await probeVersion(interpreter);
    await new Promise<void>((resolve, reject) => {
      // -m venv, not the venv console script: no activation state, no shell.
      const child = spawn(interpreter, ['-m', 'venv', venvDir(name)], { stdio: ['ignore', 'pipe', 'pipe'] });
      let err = '';
      // Both streams, because the interpreter picks and we do not know which one
      // it used. `python3 -m venv` reports on stderr, but the pip/ensurepip
      // machinery behind it writes to stdout, and a distribution's wrapper can
      // use either. Reading only stderr is how this ended up reporting
      // "python -m venv failed: " with nothing after it.
      child.stdout.on('data', (d) => (err += d));
      child.stderr.on('data', (d) => (err += d));
      child.on('error', reject);
      child.on('close', (code, signal) => {
        if (code === 0) return resolve();
        // A process killed by a signal has `code === null`, so this used to be
        // indistinguishable from a clean non-zero exit and produced an empty
        // message. SIGKILL here is the out-of-memory killer, which is a real
        // answer and not a venv problem at all.
        reject(
          new Error(
            describeVenvFailure(err, {
              killed: signal ? `The interpreter was killed by ${signal} before it could report anything.` : undefined,
              // One extra subprocess, and only on the failure path: it is what
              // turns "python3.<version>-venv" into the package name to type.
              version: version || undefined,
            }),
            // Carried structurally, not re-derived from the text above. This is
            // the one failure with a scripted fix, and recognising it by parsing
            // our own English would make the recovery depend on the wording.
            { cause: isMissingVenvModule(err) && version ? new MissingVenvPackage(version) : undefined },
          ),
        );
      });
    });
    if (!(await exists(venvPython(name)))) throw new Error('The environment was created but has no interpreter.');
  } catch (err) {
    /* A half-built environment is worse than none.
     *
     * `python -m venv` writes the directory and the interpreter before it runs
     * ensurepip, so a Python without python3-venv fails *after* leaving
     * something behind that passes every check this code used to make. Every
     * later attempt then reported "Reusing the existing environment" and died
     * with "No module named pip" — which is the exact sequence in the field log
     * this came from, and why the original error was never seen again.
     *
     * Removing it means the next attempt fails the way the first one did, and
     * says why, instead of failing somewhere unrelated much later. */
    await rm(venvDir(name), { recursive: true, force: true });
    throw err;
  }
}

/**
 * Puts pip back into an environment that has none.
 *
 * Two attempts, cheapest first. `ensurepip` in place keeps whatever is already
 * there; rebuilding from scratch is the fallback for a directory that is broken
 * in a way ensurepip will not complete inside. Rebuilding a pip-less
 * environment cannot lose anything — pip is the only thing that ever put a
 * package in there, so there is nothing to lose.
 *
 * `justBuilt` skips the rebuild: the directory was made by this code moments
 * ago, so building it again would produce the same pip-less result. When that
 * is true, a failure to find pip is a fact about the interpreter, not the
 * directory, and the caller should say so instead of churning.
 */
async function repairVenv(
  name: string,
  bound: string | undefined,
  onProgress: (line: string) => void,
  justBuilt: boolean,
): Promise<boolean> {
  onProgress('Running ensurepip…');
  const attempt = await runPython(venvPython(name), ['-m', 'ensurepip', '--upgrade'], 180_000);
  if (attempt.code === 0 && (await venvHasPip(name))) {
    onProgress('pip is back.');
    return true;
  }
  if (justBuilt) {
    // ensurepip is the thing that can be missing, and this interpreter has
    // just demonstrated that it is: the same build would repeat it.
    onProgress(
      /No module named ensurepip/i.test(attempt.err)
        ? 'This Python has no ensurepip, so it cannot put pip into an environment.'
        : 'ensurepip could not finish.',
    );
    return false;
  }
  // ensurepip is itself the thing that can be missing, so say which one it was
  // before throwing the directory away.
  if (/No module named ensurepip/i.test(attempt.err)) {
    onProgress(
      'This Python has no ensurepip either, so the environment is being rebuilt — ' +
        'anything that was installed into it is lost.',
    );
  } else {
    onProgress('ensurepip could not finish. Rebuilding the environment — anything already in it is lost.');
  }
  await rm(venvDir(name), { recursive: true, force: true });
  try {
    await createVenv(name, bound, onProgress);
  } catch (err) {
    onProgress(`Rebuilding failed: ${(err as Error).message}`);
    return false;
  }
  return venvHasPip(name);
}

/** Runs this environment's interpreter and collects everything it said. */
function runPython(
  python: string,
  args: string[],
  timeoutMs: number,
): Promise<{ code: number; out: string; err: string }> {
  return new Promise((resolve) => {
    const child = spawn(python, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs);
    child.stdout.on('data', (d) => (out += String(d)));
    child.stderr.on('data', (d) => (err += String(d)));
    child.on('error', (e) => {
      clearTimeout(timer);
      resolve({ code: 127, out, err: err + e.message });
    });
    child.on('close', (code, signal) => {
      clearTimeout(timer);
      // A killed process reports `code === null`, and `?? 0` turned that into a
      // *success* — so an interpreter killed by the out-of-memory killer, or by
      // this timeout, was reported as having pip and the install carried on
      // assuming the environment was fine. A death is never a success.
      resolve({ code: code ?? 128, out, err: err + (signal ? `\n[killed by ${signal}]` : '') });
    });
  });
}

/** Can this environment install packages at all? */
export async function venvHasPip(name: string): Promise<boolean> {
  const r = await runPython(venvPython(name), ['-m', 'pip', '--version'], 60_000);
  return r.code === 0 && /pip \d/.test(r.out);
}

/**
 * What to tell someone whose Python cannot build a working environment.
 *
 * One message, used whether the failure was at creation or at repair, because
 * it is the same missing package either way.
 */
const VENV_MODULE_HINT =
  'On Ubuntu/Debian: sudo apt install python3-venv (or python3.<version>-venv, to match this interpreter)   ' +
  '(Fedora: sudo dnf install python3-libs; Arch: sudo pacman -S python).';

/**
 * The same hint, with the package named exactly.
 *
 * `python3-venv` is a metapackage that only ever installs the module for the
 * distro's *default* Python. When the interpreter this code picked is not that
 * one — a group with an upper bound, on a machine whose default is newer — the
 * metapackage installs cleanly and the very next attempt fails identically,
 * which reads as the advice being wrong. It is: the versioned package is the
 * one that matches the interpreter, and the interpreter is already known here.
 */
function venvModuleHint(version?: string): string {
  if (!version) return VENV_MODULE_HINT;
  return (
    `On Ubuntu/Debian: sudo apt install python${version}-venv   ` +
    `(sudo apt install python3-venv on its own does not cover this interpreter — ` +
    `that package follows the distro's default Python, and this one is ${version})   ` +
    '(Fedora: sudo dnf install python3-libs; Arch: sudo pacman -S python).'
  );
}

/**
 * What to try when the interpreter could not create an environment at all, and
 * the reason was not the missing venv module.
 *
 * This used to be the bare `python -m venv failed: <stderr>` with nothing else,
 * which is the message that reached Ubuntu. Being unable to build the
 * environment is not a diagnosis — the interpreter's own words rarely are — so
 * the three things that account for nearly all of it are named instead:
 * PEP 668, a filesystem that cannot do what a venv needs, and no room to write.
 */
const VENV_BUILD_ADVICE =
  'A Python environment is a directory of symlinks, so it needs a filesystem that supports them and room to write.   ' +
  'Point RUNTIMES_DIR somewhere on your home filesystem if the app sits on a network or Windows-formatted drive, ' +
  'and check there is free space. Or set RUNTIMES_PYTHON to a different interpreter.';

/**
 * What to say when a folder this process has to write to is not writable.
 *
 * The answer to "does this need sudo" is almost always "no, and running it with
 * sudo is what broke it". A Python environment built by root cannot be repaired,
 * updated or removed by the user who owns the app afterwards, so the instinct to
 * reach for `sudo` here is what turns a one-line failure into a folder nobody can
 * use. So the command given is the one that gives the folder *back*, and the one
 * that creates the environment is never wrapped in sudo.
 */
export function permissionMessage(dir: string, uid: number): string {
  const owned = uid === 0 ? 'owned by root' : `owned by uid ${uid}`;
  return (
    `${dir} is ${owned}, so this process cannot write to it — and it will not work until that changes. ` +
    (uid === 0
      ? 'This normally means a previous install was run under sudo. The environment itself does not need root: ' +
        `hand the folder back to yourself with   sudo chown -R "$(id -un)":"$(id -un)" "${dir}"   ` +
        'then install the runtime again. Running the app as root instead will break it again the same way.'
      : `Grant yourself write access to "${dir}", or point RUNTIMES_DIR at a folder you own, then install the runtime again.`)
  );
}

/** Said when there is an environment but no way to install into it. */
export const MISSING_PIP_MESSAGE =
  'This Python cannot install packages — the environment has no pip, and it could not be repaired. ' +
  `${VENV_MODULE_HINT} Then install the runtime again.`;

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
export interface VenvFailureDetail {
  /** Set when the child was killed by a signal rather than exiting on its own. */
  killed?: string;
  /** The interpreter that was used, e.g. '3.12'. Names the exact package. */
  version?: string;
}

/**
 * The distro package that carries the venv module is not installed.
 *
 * Carried as a marker so the one failure with a scripted fix can be recognised
 * without matching on the message text this file also writes for humans.
 */
export class MissingVenvPackage extends Error {
  constructor(
    /** Interpreter version the package must match, e.g. '3.12'. */
    readonly version: string,
  ) {
    super(`python${version}-venv is not installed`);
    this.name = 'MissingVenvPackage';
  }
  /** The exact command, which is also what the password prompt is shown. */
  get command(): string {
    return `sudo apt-get install -y python${this.version}-venv`;
  }
}

/** Did the interpreter fail because its venv module is missing? */
function isMissingVenvModule(stderr: string): boolean {
  return /\bensurepip\b/i.test(stderr) || /No module named ['"]?venv/i.test(stderr);
}

export function describeVenvFailure(stderr: string, detail: VenvFailureDetail = {}): string {
  const text = stderr.trim();
  const hint = venvModuleHint(detail.version);
  // Killed before it could speak. SIGKILL is the out-of-memory killer; SIGTERM
  // is something outside this app asking it to stop. Neither is a venv fault,
  // so neither gets venv advice.
  if (detail.killed) {
    return (
      `${detail.killed} That is not a fault in the Python environment itself. ` +
      (detail.killed.includes('SIGKILL')
        ? 'SIGKILL here is almost always the out-of-memory killer: free memory, add swap, or close other applications, then install the runtime again.'
        : 'Something outside the app stopped it — check whether a system update or a package manager was running, then try again.')
    );
  }
  // Ubuntu and Debian print the same cause several ways depending on the
  // release and the Python build: the prose "ensurepip module is not available",
  // and the bare traceback "No module named ensurepip". Both mean one package.
  if (/\bensurepip\b/i.test(text)) {
    return (
      'This Python cannot create environments — the venv module is missing. ' +
      `${hint} Then install the runtime again.`
    );
  }
  if (/No module named ['"]?venv/i.test(text)) {
    return (
      'This Python was built without the venv module. ' +
      `${hint} Or set RUNTIMES_PYTHON to point at another interpreter.`
    );
  }
  // PEP 668. Ubuntu 23.04+ and every current Debian mark the system interpreter
  // externally managed, so pip inside a directory that is not a real venv refuses
  // to run at all. Reaching this means the environment was not one.
  if (/externally-managed-environment/i.test(text)) {
    return (
      "This Python's package manager refuses to install anything outside a real virtual environment (PEP 668). " +
      `${hint} Then install the runtime again.`
    );
  }
  // The environment is a set of symlinks. exFAT, some network mounts and a
  // Windows drive mounted into Linux refuse to make them, and venv reports the
  // refusal as a bare OSError with no mention of what it wanted to do.
  if (/Invalid cross-device link|Operation not supported|os\.symlink|Too many levels of symbolic links|Errno 18\b|Errno 95\b/i.test(text)) {
    return (
      `The filesystem this app is on cannot create the symbolic links a Python environment is made of — Python said: ` +
      `${text.slice(0, 200)}   ${VENV_BUILD_ADVICE}`
    );
  }
  if (/No space left on device|Errno 28\b/i.test(text)) {
    return (
      'There is not enough free disk space to build the Python environment. ' +
      'Free some space — or point RUNTIMES_DIR at a drive with room — then install the runtime again.'
    );
  }
  if (/Permission denied|Errno 13\b/i.test(text)) {
    // The pre-flight check normally catches this before Python is even run, so
    // arriving here means something deeper changed underneath. Ask the folder
    // anyway: if the cause is known, say so, rather than listing what else it
    // might have been.
    const blocked = unwritableAncestor(runtimesDir());
    return blocked
      ? permissionMessage(blocked.path, blocked.uid)
      : `The Python environment could not be written to (${text.slice(0, 120)}). ` +
        `${VENV_BUILD_ADVICE} If it is a permissions problem on this app's folder, fixing that and installing again will do it.`;
  }
  // Exited non-zero having said nothing. This is the case that produced a bare
  // "python -m venv failed: " with no reason at all: there is no text to relay,
  // so the sentence has to be built from what is actually known.
  if (!text) {
    return (
      'The Python interpreter exited without creating an environment and without saying why. ' +
      'Running `python3 -m venv .venv` in a terminal shows the error it is swallowing here — ' +
      'that output is the missing half. ' +
      VENV_BUILD_ADVICE
    );
  }
  // Anything else keeps the interpreter's own words, because a guess would be
  // worse than the truth — but it is no longer the end of the sentence.
  return `python -m venv failed: ${text.slice(0, 300)}   ${VENV_BUILD_ADVICE}`;
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
 * Creates the environment, installing the distro package that carries the venv
 * module if that is what is missing.
 *
 * This is the one place the app escalates for Python itself. It has to: the
 * module is a *system* package, so no amount of retrying, repairing or pointing
 * RUNTIMES_PYTHON at a different interpreter gets past it — and the alternative
 * is a user copying a command out of the log and back into a terminal, which is
 * where the version gets typed wrong and the failure repeats.
 *
 * Two constraints keep it narrow. The package name is derived from the
 * interpreter this code itself chose, never from anything the user sent, and only
 * one package is ever installed — so the worst case is the wrong venv module for
 * an interpreter already on the machine, not arbitrary package installation.
 */
async function ensureVenvWithRecovery(
  name: string,
  bound: string | undefined,
  onProgress: (line: string) => void,
  opts: Pick<InstallOpts, 'password' | 'onNeedsPrivilege'>,
): Promise<void> {
  try {
    await ensureVenv(name, bound, onProgress);
    return;
  } catch (err) {
    const missing = (err as Error).cause;
    if (!(missing instanceof MissingVenvPackage)) throw err;
    await installVenvPackage(missing, onProgress, opts);
  }
  // The package is in place now, so the failure that brought us here should not
  // repeat. If it does, this second failure is a different problem and says so.
  await ensureVenv(name, bound, onProgress);
}

/**
 * Runs one command behind `sudo -S`, streaming its output. Resolves the exit code.
 *
 * The password arrives on stdin and is scrubbed from everything that comes back,
 * because a tool that echoes its own input must not put it in the browser. The
 * argument vector is passed through untouched — nothing here builds a shell
 * string, which is the constraint the rest of this file is written to.
 */
async function sudoExec(
  argv: string[],
  password: string,
  onProgress: (line: string) => void,
): Promise<number> {
  const scrub = (text: string) =>
    password.length >= 3 ? text.split(password).join('[redacted]') : text;
  let tail = '';
  const code = await new Promise<number>((resolve) => {
    // `-p ''` stops sudo printing a second prompt of its own. `-k` is
    // deliberately not passed, so a recent sudo authentication in a terminal is
    // reused instead of asking again.
    const child = spawn('sudo', ['-S', '-p', '', ...argv], { stdio: ['pipe', 'pipe', 'pipe'] });
    child.stdin?.on('error', () => {
      /* sudo exited early — a wrong password does exactly that */
    });
    child.stdin?.end(`${password}\n`);
    const bufs: Record<'out' | 'err', string> = { out: '', err: '' };
    const pump = (stream: NodeJS.ReadableStream | null, key: 'out' | 'err') => {
      stream?.setEncoding('utf8');
      stream?.on('data', (c: string) => {
        bufs[key] += c;
        const lines = bufs[key].split('\n');
        bufs[key] = lines.pop() ?? '';
        for (const line of lines) {
          const text = scrub(line.replace(/\s+$/, '')).trim();
          if (text) onProgress(text);
          tail = `${tail}\n${text}`.slice(-4000);
        }
      });
    };
    pump(child.stdout, 'out');
    pump(child.stderr, 'err');
    child.on('error', (err) => {
      const missing = (err as NodeJS.ErrnoException).code === 'ENOENT';
      onProgress(
        missing ? 'sudo is not on this machine.' : `Could not start sudo: ${scrub(err.message)}`,
      );
      resolve(127);
    });
    child.on('close', (c) => {
      for (const key of ['out', 'err'] as const) {
        const rest = scrub(bufs[key]).trim();
        if (rest) {
          onProgress(rest);
          tail = `${tail}\n${rest}`.slice(-4000);
        }
      }
      resolve(c ?? 0);
    });
  });
  if (/incorrect password|Sorry, try again|authentication failure/i.test(tail)) {
    throw new Error('That password was not accepted, so nothing was changed.');
  }
  return code;
}

/**
 * The systemd drop-in that turns Ollama's Vulkan backend on.
 *
 * Current Ollama enables Vulkan by default, so this is belt-and-braces for the
 * versions that did not — but it is also the only place the setting can be
 * pinned per machine. Writing `Environment=` into the unit itself would be
 * overwritten by the next Ollama upgrade; a drop-in in `ollama.service.d` is
 * merged on top and survives, which is why the vendor says `systemctl edit`
 * rather than edit the unit.
 */
const OLLAMA_VULKAN_DROPIN = '[Service]\nEnvironment="OLLAMA_VULKAN=1"\n';
const OLLAMA_DROPIN_PATH = '/etc/systemd/system/ollama.service.d/override.conf';

/** True when this machine already has Ollama pinned to its Vulkan backend. */
export async function ollamaVulkanEnabled(): Promise<boolean> {
  if (process.platform !== 'linux') return false;
  try {
    return /OLLAMA_VULKAN=1/.test(await readFile(OLLAMA_DROPIN_PATH, 'utf8'));
  } catch {
    return false;
  }
}

/**
 * Points Ollama at Vulkan on a machine that has it.
 *
 * Two things, and the second is the one that actually decides whether it works:
 *
 *  - `OLLAMA_VULKAN=1`, pinned in a systemd drop-in so it survives upgrades.
 *  - the `ollama` user in the `render` group. Ollama runs as its own system
 *    user, and without that membership it cannot open `/dev/dri/renderD*` at
 *    all — so the driver is present, the setting is on, and Ollama silently
 *    falls back to the CPU. The vendor calls this out specifically for AMD.
 *
 * Best-effort by design: Ollama is installed and working by the time this runs,
 * so a step that cannot finish is reported and the install carries on rather
 * than throwing away a completed install over a GPU preference.
 */
async function enableOllamaVulkan(opts: PrivilegedInstall): Promise<void> {
  const { onProgress, password, onNeedsPrivilege } = opts;
  if (process.platform !== 'linux') return;

  const vk = await detectVulkan();
  if (!(vk.available || hasVulkan())) {
    onProgress(
      'No working Vulkan driver here, so Ollama is left on its default backend. ' +
        'Install vulkan-tools and your GPU driver, then install Ollama again.',
    );
    return;
  }
  const card = vk.device ? ` (${vk.device})` : '';

  if (await ollamaVulkanEnabled()) {
    onProgress(`Ollama is already set to use Vulkan${card}.`);
    return;
  }

  if (!password) {
    // Report the need rather than failing on it: the browser prompts, and this
    // is the same contract every other elevating step uses.
    onNeedsPrivilege?.(
      'sudo systemctl edit ollama   # sets OLLAMA_VULKAN=1, then sudo usermod -aG render ollama',
    );
    onProgress(
      `This machine has Vulkan${card}, but turning it on for Ollama needs your password. ` +
        'Supply it to finish, or run the command above in a terminal.',
    );
    return;
  }

  onProgress(`Setting Ollama to use Vulkan${card}…`);

  // Written to a temporary file and installed from there rather than piped into
  // a shell, so no part of this is ever interpreted by one.
  const tmp = await mkdtemp(path.join(tmpdir(), 'ollama-vk-'));
  const dropin = path.join(tmp, 'override.conf');
  try {
    await writeFile(dropin, OLLAMA_VULKAN_DROPIN, 'utf8');
    // `install -D` creates the drop-in directory systemd looks in, which does
    // not exist on a fresh install.
    const put = await sudoExec(
      ['install', '-D', '-m', '0644', dropin, OLLAMA_DROPIN_PATH],
      password,
      onProgress,
    );
    if (put !== 0) {
      onProgress('Could not write the Ollama service setting — Ollama is left as it was.');
      return;
    }
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }

  // The group membership is what makes the setting mean anything. Asked rather
  // than assumed, so a machine that already has it does not get a pointless
  // sudo round trip.
  const groups = spawnSync('id', ['-nG', 'ollama'], { encoding: 'utf8' });
  if (groups.status === 0 && /(^|\s)render(\s|$)/.test(groups.stdout ?? '')) {
    onProgress('The ollama user can already open the GPU.');
  } else if ((await sudoExec(['usermod', '-aG', 'render', 'ollama'], password, onProgress)) !== 0) {
    onProgress(
      'Could not add the ollama user to the render group, so Ollama stays on the CPU until you run: ' +
        'sudo usermod -aG render ollama',
    );
  } else {
    onProgress('Added the ollama user to the render group so it can open the GPU.');
  }

  // The service has to reread the unit and restart for either change to land.
  await sudoExec(['systemctl', 'daemon-reload'], password, onProgress);
  const restart = await sudoExec(['systemctl', 'restart', 'ollama'], password, onProgress);
  if (restart !== 0) {
    onProgress(
      'Could not restart Ollama, so the setting applies next time it starts. ' +
        'Start it with `ollama serve`.',
    );
  } else {
    onProgress('Ollama restarted with Vulkan enabled.');
  }
  invalidateRuntimeProbes();
}

/** Installs the one distro package that unblocks `python -m venv`. */
async function installVenvPackage(
  pkg: MissingVenvPackage,
  onProgress: (line: string) => void,
  opts: Pick<InstallOpts, 'password' | 'onNeedsPrivilege'>,
): Promise<void> {
  await aptInstall(
    `python${pkg.version}-venv`,
    'This Python is missing the venv module',
    onProgress,
    opts,
  );
}

/**
 * Installs one apt package, escalating only when a password was supplied.
 *
 * Every caller passes a package name this file chose for itself — never
 * anything a request carried in — so the blast radius is one package, chosen
 * from a fixed set, on a machine where the user is about to type their password
 * anyway. That is deliberately the same contract the Ollama and driver
 * installers already had; this just means the venv module and curl stop being
 * the two things a user has to leave the app to install.
 */
async function aptInstall(
  pkg: string,
  why: string,
  onProgress: (line: string) => void,
  { password, onNeedsPrivilege }: Pick<InstallOpts, 'password' | 'onNeedsPrivilege'>,
): Promise<void> {
  // Only apt-based systems have a scripted answer here. On anything else the
  // message already names the right package, and guessing a dnf or pacman
  // command here would be worse than printing what the interpreter said.
  if (process.platform !== 'linux') {
    throw new Error(
      `${why}, and this app cannot install it on ${platformLabel()}. ` +
        'Install it with your package manager, then install the runtime again.',
    );
  }

  const command = `sudo apt-get install -y ${pkg}`;

  if (!password) {
    // No password yet: report the need so the browser prompts, and say exactly
    // what will run. This is the same contract every other elevating step uses.
    onNeedsPrivilege?.(command);
    throw new Error(
      `${why}. Installing it needs your password:   ${command}   ` +
        'Supply your password to let it run, or run that command in a terminal and install the runtime again.',
    );
  }

  onProgress(`Installing ${command} — this needs your password…`);
  const scrub = (text: string) =>
    password.length >= 3 ? text.split(password).join('[redacted]') : text;
  const run = async (argv: string[]): Promise<{ code: number; tail: string }> => {
    let tail = '';
    const code = await new Promise<number>((resolve) => {
      // `sudo -S` reads the password from stdin; `-p ''` stops sudo printing a
      // second prompt of its own. `-k` is deliberately not passed, so a recent
      // sudo authentication in a terminal is reused instead of asking again.
      const child = spawn('sudo', ['-S', '-p', '', ...argv], { stdio: ['pipe', 'pipe', 'pipe'] });
      child.stdin?.on('error', () => {
        /* sudo exited early — a wrong password does exactly that */
      });
      child.stdin?.end(`${password}\n`);
      const bufs: Record<'out' | 'err', string> = { out: '', err: '' };
      const pump = (stream: NodeJS.ReadableStream | null, key: 'out' | 'err') => {
        stream?.setEncoding('utf8');
        stream?.on('data', (c: string) => {
          bufs[key] += c;
          const lines = bufs[key].split('\n');
          bufs[key] = lines.pop() ?? '';
          for (const line of lines) {
            const text = scrub(line.replace(/\s+$/, '')).trim();
            if (text) onProgress(text);
            tail = `${tail}\n${text}`.slice(-4000);
          }
        });
      };
      pump(child.stdout, 'out');
      pump(child.stderr, 'err');
      child.on('error', (err) => {
        const missing = (err as NodeJS.ErrnoException).code === 'ENOENT';
        onProgress(
          missing
            ? 'apt-get is not on this machine, so the package cannot be installed automatically.'
            : `Could not start apt-get: ${scrub(err.message)}`,
        );
        resolve(127);
      });
      child.on('close', (c) => {
        for (const key of ['out', 'err'] as const) {
          const rest = scrub(bufs[key]).trim();
          if (rest) {
            onProgress(rest);
            tail = `${tail}\n${rest}`.slice(-4000);
          }
        }
        resolve(c ?? 0);
      });
    });
    return { code, tail };
  };

  let { code, tail } = await run(['apt-get', 'install', '-y', pkg]);

  /* "Unable to locate package" means apt has never been told where the packages
     are, not that the package is missing. A container image that has not run an
     update since it was built fails exactly this way, so the update is run and
     the install retried — the pair is what a person types at a terminal, and
     doing only the second half is what made this need a manual fix. */
  if (code !== 0 && /unable to locate package|no such package|not found|404\s+Not Found/i.test(tail)) {
    onProgress('Refreshing the package list (apt-get update)…');
    const update = await run(['apt-get', 'update']);
    if (update.code !== 0) {
      throw new Error(
        'apt-get update failed, so the package list is unknown. Run it in a terminal ' +
          'to see why, then install the runtime again.',
      );
    }
    ({ code, tail } = await run(['apt-get', 'install', '-y', pkg]));
  }

  if (/incorrect password|Sorry, try again|authentication failure/i.test(tail)) {
    throw new Error('That password was not accepted, so nothing was changed.');
  }
  if (code !== 0) {
    throw new Error(
      needsPrivilege(tail)
        ? `The package could not be installed because it needs root:   ${command}`
        : `Installing ${command} failed (apt-get exited ${code}) — see the log above.`,
    );
  }
  onProgress(`${pkg} installed.`);
  invalidateRuntimeProbes();
}


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
/** Does this group's install go through pip, and so need a Python environment? */
function usesPython(group: RuntimeGroup): boolean {
  // Ollama and llama.cpp ship their own installers. Neither imports a package,
  // neither reads an interpreter, and neither is blocked by a Python that
  // cannot build a venv.
  return group.installer !== 'ollama' && group.installer !== 'llama-cpp';
}

export async function installRuntimes({
  groups,
  device,
  onProgress,
  password,
  onNeedsPrivilege,
}: InstallOpts): Promise<void> {
  // Only the groups that actually go through pip. Ollama and llama.cpp have
  // their own installers, so they carry no venv — which matters, because
  // building one is what used to fail: the environment was created before the
  // selection was looked at, so installing Ollama on a machine with no
  // python3-venv died in a Python error with no Ollama involved at all.
  const pipGroups = groups.filter(usesPython);

  // pip is about to change these venvs: drop the cached answers up front and
  // again when it is finished, so a failed run cannot leave a stale half in
  // place either.
  invalidateRuntimeProbes();

  const members = new Map<string, RuntimeGroup[]>();
  for (const g of pipGroups) {
    const name = venvName(g);
    const list = members.get(name) ?? [];
    list.push(g);
    members.set(name, list);
  }
  // The shared venv is the baseline for pip groups, built here rather than in
  // the loop so a selection of nothing but vendor installers creates nothing.
  if (members.has('venv')) {
    await ensureVenvWithRecovery('venv', undefined, onProgress, { password, onNeedsPrivilege });
  }

  const presence = await detectGpu();
  const common = ['-m', 'pip', 'install', '--disable-pip-version-check', '--no-input'];
  const listed = new Set(groups.map((g) => g.id));

  const runPip = (py: string, args: string[], label: string) =>
    new Promise<void>((resolve, reject) => {
      const child = spawn(py, args, { stdio: ['ignore', 'pipe', 'pipe'] });
      // Kept, not just printed. "pip exited 1" says nothing about why, and the
      // one reason worth naming — an environment with no pip in it — is
      // otherwise invisible to everything reading the transcript.
      let err = '';
      const bump = (buf: Buffer) => {
        for (const line of buf.toString().split(/\r?\n/)) {
          if (line.trim()) onProgress(line.trim());
        }
      };
      child.stdout.on('data', bump);
      child.stderr.on('data', (d: Buffer) => {
        err += d.toString();
        bump(d);
      });
      child.on('error', reject);
      child.on('close', (code) => {
        if (code === 0) {
          resolve();
          return;
        }
        if (/No module named pip/i.test(err)) {
          reject(new Error(MISSING_PIP_MESSAGE));
          return;
        }
        reject(new Error(`${label} failed (pip exited ${code}) — see the log above.`));
      });
    });

  for (const [name, list] of members) {
    const bound = list.map((g) => g.maxPython).find((b) => b) ?? undefined;
    // The shared environment was ensured at the top of this function. Asking
    // again printed "Reusing the existing environment" twice for every install
    // and, now that this check costs a subprocess, asked the same question
    // twice as well.
    if (name !== 'venv') {
      await ensureVenvWithRecovery(name, bound, onProgress, { password, onNeedsPrivilege });
    }
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
      if (!g.pkgs.length) continue;
      onProgress(`Installing ${g.label} into ${where}…`);
      await runPip(py, [...common, ...g.pkgs], g.label);
      onProgress(`${g.label} installed.`);
    }
  }

  /* The two that ship their own installers, outside the pip loop because they
     carry no environment. Run after it so a selection of both does the Python
     work first and leaves a vendored binary to a second. */
  for (const g of groups) {
    if (g.installer === 'ollama') {
      await installOllama({ onProgress, password, onNeedsPrivilege });
      continue;
    }
    if (g.installer === 'llama-cpp') {
      await installLlamaCpp(onProgress);
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

  /* The vendor's Linux installer is `curl … | sh`, which fails on a minimal or
   * server Ubuntu that has no curl — and "curl: command not found" reads as the
   * app being broken rather than the machine being minimal. wget does the same
   * job. With neither, curl is installed through the same password prompt the
   * Ollama install itself uses, so this is one extra step rather than a dead end
   * and a manual trip to a terminal. */
  let fetcher: 'curl' | 'wget' | null = null;
  if (process.platform === 'linux') {
    fetcher = await downloadTool();
    if (!fetcher) {
      await aptInstall('curl', 'Ollama is installed with curl, and this machine has neither curl nor wget', onProgress, opts);
      fetcher = (await downloadTool()) ?? 'curl';
    }
    onProgress(`Fetching the Ollama installer with ${fetcher}.`);
  }

  const recipe = ollamaRecipe(process.platform, fetcher ?? 'curl');
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

  const scrub = (text: string) =>
    password && password.length >= 3 ? text.split(password).join('[redacted]') : text;

  /** Runs the vendor installer once; returns its exit code and last words. */
  const runInstaller = async (): Promise<{ code: number; tail: string }> => {
    onProgress(`Installing Ollama on ${platformLabel()} — ${recipe.shown}…`);
    // Whether sudo is on the other end of the pipe decides stdin: a password has
    // to be written, anything else is told to ignore input.
    const escalating = Boolean(password && elevates);
    let tail = '';
    let code: number;
    try {
      code = await new Promise<number>((resolve) => {
        const { cmd, args: cmdArgs } = ollamaCommand(recipe, escalating);
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
            missing && recipe.argv[0] === 'brew'
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
    return { code, tail };
  };

  /**
   * Turns one run into: done, throw, or "install this first and try again".
   *
   * The vendor tarball is zstd-compressed, so a machine without zstd stops with
   * "This version requires zstd for extraction" — the same dead end as a missing
   * curl, and on a lean image just as common. Reading the package out of that
   * message is what makes it self-healing instead of a manual trip to a
   * terminal.
   *
   * Checked against a fixed list rather than installed on trust: this text comes
   * from a script fetched over the network, and an installer that could name any
   * package would be a way to run apt with a name it picked.
   */
  const classify = (run: { code: number; tail: string }): string | null => {
    if (run.code === 0) return null;
    if (password && /incorrect password|Sorry, try again|authentication failure/i.test(run.tail)) {
      throw new Error('That password was not accepted, so nothing was changed.');
    }
    // It stopped because it wanted a password it had no way to get, not because
    // the installer is broken. That distinction is the whole reason this checks
    // the output rather than the exit code.
    if (needsPrivilege(run.tail)) {
      const shown = `sudo ${recipe.shown}`;
      onNeedsPrivilege?.(shown);
      throw new Error(
        'The Ollama installer needs root and could not elevate. ' +
          'Supply your password to let it run, or run this in a terminal yourself: ' +
          shown,
      );
    }
    if (recipe.argv[0] === 'brew' && /Homebrew is not installed/.test(run.tail)) {
      throw new Error(
        'Homebrew is not installed, and macOS installs Ollama through it. ' +
          'Install Homebrew from https://brew.sh, then install this again.',
      );
    }
    return missingInstallerPrerequisite(run.tail);
  };

  let pending = classify(await runInstaller());

  // One retry, after installing what the installer said it needed. Bounded on
  // purpose: asked for the same package twice, it is a machine in a state this
  // cannot fix, and retrying would just repeat the same message.
  if (pending) {
    await aptInstall(pending, `The Ollama installer needs ${pending}`, onProgress, opts);
    const again = classify(await runInstaller());
    if (again === pending) {
      throw new Error(
        `The Ollama installer still needs ${pending} after it was installed — see the log above.`,
      );
    }
    pending = again;
  }

  if (pending) {
    throw new Error(`The Ollama installer says it needs ${pending}. See the log above.`);
  }

  onProgress('Ollama installed. Start it with `ollama serve` if the daemon is not already running.');
  // After the install, so the daemon exists to be restarted. Best-effort by
  // design — this reports and returns rather than throwing, because Ollama is
  // already installed and working at this point and a GPU preference is not a
  // reason to call the whole install a failure.
  await enableOllamaVulkan(opts);
}

/**
 * The tool that will fetch a URL on this machine, or null when there is none.
 *
 * The vendor install script is `curl … | sh`, and a minimal or server Ubuntu
 * has no curl — so the documented command fails with "curl: command not found"
 * and nothing is installed, which reads as the app being broken rather than the
 * machine being minimal. wget is nearly always present instead and writes the
 * same bytes to stdout, so it drops straight into a pipe. When neither exists,
 * the answer is to install curl rather than leave the user with a dead command.
 */
export async function downloadTool(): Promise<'curl' | 'wget' | null> {
  for (const t of ['curl', 'wget'] as const) {
    if (await onPath(t)) return t;
  }
  return null;
}

/**
 * The only packages this app will install because a downloaded script said it
 * wanted them.
 *
 * A fixed list, not whatever the message names. The text comes from an
 * installer fetched over the network; letting it choose an apt argument would
 * make "install this" an instruction rather than a report.
 */
const INSTALLER_PREREQUISITES = ['zstd', 'curl', 'ca-certificates', 'xz-utils'] as const;

/**
 * The package a failed vendor installer said it needs, or null.
 *
 * Ollama's tarball is zstd-compressed, so a machine without zstd stops with
 * "This version requires zstd for extraction. Please install zstd and try
 * again". The name is read from that sentence rather than from a list of its
 * own phrasings, so a wording change upstream keeps working — and anything
 * outside the allowlist returns null, which leaves the user with the
 * installer's own message rather than an app acting on it.
 */
export function missingInstallerPrerequisite(output: string): string | null {
  const named =
    /\brequires\s+([a-z0-9][a-z0-9.+-]*)\b/i.exec(output)?.[1] ??
    /\bplease install\s+([a-z0-9][a-z0-9.+-]*)\b/i.exec(output)?.[1] ??
    null;
  if (!named) return null;
  const pkg = named.toLowerCase();
  return (INSTALLER_PREREQUISITES as readonly string[]).includes(pkg) ? pkg : null;
}

/**
 * How to install Ollama on this OS, as the vendor documents it.
 *
 * `argv` is the whole argument vector, not a command with its own trailing
 * arguments: the exact same vector is used both bare and behind `sudo`, and
 * splitting it in two is how `sh -c` ended up being dropped — which made the
 * installer run `sh "curl … | sh"` and fail with "cannot open …: No such
 * file" instead of installing anything.
 *
 * `fetcher` is a parameter rather than a lookup in here so this stays a pure
 * function of its arguments, which is what its per-OS checks rely on.
 */
export function ollamaRecipe(
  platform: NodeJS.Platform = process.platform,
  fetcher: 'curl' | 'wget' = 'curl',
): { shown: string; argv: string[]; needsRoot: boolean } | null {
  if (platform === 'win32') {
    // The vendor's own PowerShell one-liner. It installs per-user, so it does
    // not need an elevated prompt.
    return {
      shown: 'irm https://ollama.com/install.ps1 | iex',
      argv: [
        'powershell.exe',
        '-NoProfile',
        '-ExecutionPolicy',
        'Bypass',
        '-Command',
        'irm https://ollama.com/install.ps1 | iex',
      ],
      needsRoot: false,
    };
  }
  if (platform === 'darwin') {
    return {
      shown: 'brew install ollama',
      argv: ['brew', 'install', 'ollama'],
      // Homebrew refuses to run as root, so escalating here would break it
      // rather than help it. If brew still needs help the user is told the
      // command to run in a terminal.
      needsRoot: false,
    };
  }
  if (platform === 'linux') {
    // wget writes the response to stdout with -qO-, which is the same shape
    // curl's default stdout has, so the pipe is identical either way.
    const pipe = fetcher === 'wget' ? 'wget -qO-' : 'curl -fsSL';
    return {
      shown: `${pipe} https://ollama.com/install.sh | sh`,
      argv: ['sh', '-c', `${pipe} https://ollama.com/install.sh | sh`],
      needsRoot: true,
    };
  }
  return null;
}

/**
 * The exact process to run for a recipe, with or without sudo in front.
 *
 * One place builds it, so the escalated and un-escalated forms cannot drift —
 * they are the same installer, and only the wrapper differs.
 */
export function ollamaCommand(
  recipe: { argv: string[] },
  escalating: boolean,
): { cmd: string; args: string[] } {
  return escalating
    ? { cmd: 'sudo', args: ['-S', '-p', '', ...recipe.argv] }
    : { cmd: recipe.argv[0], args: recipe.argv.slice(1) };
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
/** Unpacks one archive into the runtimes dir.
 *
 * The archives unpack into llama-<build>/…; extract into the runtimes dir and
 * remember which directory it was, since upstream's exact layout is not
 * something this app should hard-code. */
async function unpack(tarball: string): Promise<void> {
  await new Promise<void>((resolve, reject) => {
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
}

async function installLlamaCpp(onProgress: (line: string) => void): Promise<void> {
  if (await llamaCppInstalled() && !(await llamaCppUpgradeKind())) {
    onProgress('llama.cpp is already installed.');
    return;
  }
  if (await llamaCppInstalled()) {
    onProgress('Upgrading llama.cpp to the GPU build…');
  }
  const picked = llamaAsset();
  if (!picked) {
    throw new Error(
      'The bundled llama.cpp installer is Linux-only (the project publishes ' +
        'per-platform builds; see its releases for macOS and Windows).',
    );
  }
  const asset = picked.name;

  // A previous build (CPU when a GPU is now available, or the other way round)
  // would otherwise shadow the new one: the discovery below takes the first
  // llama-* directory that holds the binary, and the stale one would win.
  try {
    for (const e of await readdir(runtimesDir(), { withFileTypes: true })) {
      if (e.isDirectory() && /^llama-/.test(e.name)) {
        await rm(path.join(runtimesDir(), e.name), { recursive: true, force: true });
      }
    }
  } catch {
    /* runtimes dir may not exist yet */
  }

  const KIND_LABEL: Record<string, string> = {
    vulkan: 'Vulkan',
    cuda: `CUDA ${cudaTag()}`,
    cpu: 'CPU',
  };
  onProgress(`Fetching llama.cpp ${LLAMA_BUILD} (${KIND_LABEL[picked.kind]} build)…`);
  const url = `https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/${asset}`;

  // The CUDA build links against libcudart and friends, which upstream ships as
  // a separate 594 MB archive. Fetched only when this machine has none: a
  // Colab runtime, a CUDA toolkit install, or the nvidia-* wheels torch already
  // pulled in all provide them, and there is no reason to download a third
  // copy of the runtime onto those.
  const wantsCudart = picked.kind === 'cuda' && picked.cudart && !hasCudaRuntimeLibs();
  if (picked.kind === 'cuda' && !wantsCudart) {
    onProgress('This machine already has the CUDA runtime libraries.');
  }

  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`Could not download ${asset} (HTTP ${res.status}).`);
  const bytes = Buffer.from(await res.arrayBuffer());

  await mkdir(llamaDir(), { recursive: true });
  const tarball = path.join(llamaDir(), asset);
  const { writeFile, chmod } = await import('node:fs/promises');
  await writeFile(tarball, bytes);
  onProgress(`Unpacking ${(bytes.length / 1024 / 1024).toFixed(0)} MB…`);
  await unpack(tarball);

  if (wantsCudart && picked.cudart) {
    const rtUrl = `https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/${picked.cudart}`;
    onProgress('Fetching the CUDA runtime libraries (~594 MB, one time)…');
    const rt = await fetch(rtUrl, { redirect: 'follow' });
    if (!rt.ok) throw new Error(`Could not download ${picked.cudart} (HTTP ${rt.status}).`);
    const rtTarball = path.join(llamaDir(), picked.cudart);
    await writeFile(rtTarball, Buffer.from(await rt.arrayBuffer()));
    await unpack(rtTarball);
    await rm(rtTarball, { force: true });
  }
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