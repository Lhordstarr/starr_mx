/**
 * Live resource usage for the machine this app runs on.
 *
 * Everything here is best-effort and degrades rather than failing: a machine
 * without nvidia-smi simply reports no GPU instead of erroring, because the
 * box is just as likely to be CPU-only by design as misconfigured.
 *
 * CPU percent needs two samples to be meaningful (one delta can't know what the
 * previous idle looked like), so the first call after a cold start reports the
 * load average instead and the client polls again a beat later.
 */
import { execFile, spawn, type ChildProcess } from 'node:child_process';
import { readdir, readFile, stat } from 'node:fs/promises';
import os from 'node:os';

export interface ResourceReport {
  cpu: {
    /** Percent across all cores, or null before the second sample. */
    percent: number | null;
    cores: number;
    loadAvg1: number;
    /** Per-core percent, same null caveat. */
    perCore: (number | null)[];
    model: string;
  };
  ram: {
    totalMB: number;
    usedMB: number;
    freeMB: number;
    percent: number;
  };
  gpu: {
    available: boolean;
    vendor: 'nvidia' | 'amd' | null;
    reason: string | null;
    name: string | null;
    count: number;
    /** Percent across all GPUs, null when no reader is available. */
    percent: number | null;
    memUsedMB: number | null;
    memTotalMB: number | null;
  };
  at: number;
}

type CoreTimes = { idle: number; total: number }[];

/** Previous cumulative tick counters, kept module-scoped between polls. */
let lastCores: CoreTimes | null = null;
let lastSampleAt = 0;

function readCores(): CoreTimes {
  return os.cpus().map((c) => ({ idle: c.times.idle, total: c.times.user + c.times.nice + c.times.sys + c.times.irq + c.times.idle }));
}

/** Percent busy per core between two cumulative snapshots. */
function diffCores(prev: CoreTimes, next: CoreTimes): (number | null)[] {
  return next.map((n, i) => {
    const p = prev[i];
    if (!p) return null;
    const dTotal = n.total - p.total;
    if (dTotal <= 0) return 0;
    return Math.min(100, Math.max(0, ((dTotal - (n.idle - p.idle)) / dTotal) * 100));
  });
}

function cpuSample(): { percent: number | null; perCore: (number | null)[] } {
  const now = readCores();
  // Guard the denominator against a boot or a very short interval producing
  // nonsense. Under ~200ms the numbers are mostly sampling noise.
  const usable = lastCores !== null && Date.now() - lastSampleAt > 200;
  const perCore = usable ? diffCores(lastCores as CoreTimes, now) : now.map(() => null);
  lastCores = now;
  lastSampleAt = Date.now();
  const flat = perCore.filter((n): n is number => n !== null);
  const percent = flat.length ? flat.reduce((a, b) => a + b, 0) / flat.length : null;
  return { percent, perCore };
}

/**
 * GPU counters.
 *
 * NVIDIA is read with nvidia-smi. AMD is read with radeontop and nothing else.
 *
 * rocm-smi and amdsmi were the alternatives and both are gone on purpose: each
 * needs a full ROCm install and a readable /dev/kfd to read four numbers, and
 * neither works at all on the older cards radeontop handles. radeontop is ~80
 * KB, is in every distribution's repositories, and reads the same amdgpu
 * counters. The vendor is still detected first, so a card that is present but
 * unreadable is reported as "detected, unreachable" rather than as no GPU —
 * which is genuinely different, and points at a different fix.
 */
type Vendor = 'nvidia' | 'amd' | null;

let gpuProbe: Promise<{ vendor: Vendor; name: string | null; count: number; reason: string | null }> | null = null;
/**
 * Last sample, so a burst of polls does not spawn a reader per request.
 *
 * Kept under the page's poll interval on purpose. At 2 s the page repeated
 * every reading twice — the meters froze for two seconds and then stepped,
 * which is exactly what "live" is not supposed to look like. Below the
 * interval, every request gets a fresh number while a second tab arriving
 * inside the window still shares the same reader.
 *
 * The reason has to be cached too. It was not, so a caller arriving inside the
 * window got null counters with no explanation — which is exactly the state
 * that reads as "the app has no idea what is going on".
 */
const GPU_CACHE_MS = 900;
let gpuCache: {
  at: number;
  util: number | null;
  memUsed: number | null;
  memTotal: number | null;
  reason: string | null;
} | null = null;

function run(cmd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { timeout: 4000, maxBuffer: 1024 * 1024 }, (err, stdout) => {
      if (err) reject(err);
      else resolve(stdout);
    });
  });
}

/* ---------------- AMD: one long-lived sampler ---------------- */

/**
 * `radeontop -d -` prints one line per interval and keeps going. Run once per
 * request it costs a full second every time, because its dump interval is
 * measured in whole seconds and cannot be lowered below one — so a 3-second poll
 * missed the old 2-second cache every single time and every response took
 * ~1050ms to arrive. Measured, not assumed: five consecutive polls came back
 * at 2.26s, 1.05s, 1.05s, 1.06s, 1.07s.
 *
 * So it is started once and left running, and a request reads whatever the most
 * recent line was. That moves the second off the request path entirely: the
 * meters then update as fast as the page polls, which is the point of polling
 * them at all.
 */
interface AmdSample {
  util: number | null;
  memUsedMB: number | null;
  at: number;
  /** Set from radeontop's stderr the first time it says anything. */
  warn: string | null;
}

interface AmdSampler {
  node: string;
  child: ChildProcess;
  sample: AmdSample;
  /** Fires if nobody reads a sample for AMD_SAMPLE_IDLE_MS. */
  idleTimer: ReturnType<typeof setTimeout>;
}

let amdSampler: AmdSampler | null = null;

/** How the sampler is torn down when it is no longer wanted. */
function stopAmdSampler() {
  if (!amdSampler) return;
  clearTimeout(amdSampler.idleTimer);
  try {
    amdSampler.child.kill('SIGTERM');
  } catch {
    /* already gone */
  }
  amdSampler = null;
}

/**
 * Pulls the two numbers we want out of one radeontop dump line.
 *
 * The unanimity check is load-bearing: radeontop's fallback path when it cannot
 * reach the registers pins every counter to exactly 100.00%, which is a
 * plausible-looking lie that reads as a GPU at full load. A real GPU never does
 * that, so a line where the named counters all agree is not a measurement.
 */
function parseRadeontopLine(line: string): { util: number | null; memUsedMB: number | null; bogus: boolean } {
  const utilMatch = line.match(/gpu\s+([0-9.]+)%/);
  let util: number | null = utilMatch ? Number(utilMatch[1]) : null;
  // radeontop prints VRAM in MB already, so no conversion: an earlier *1024
  // here turned a reported 0.64mb into 655MB.
  const vram = line.match(/vram\s+[0-9.]+%\s+([0-9.]+)mb/i);
  const memUsedMB = vram ? Number(vram[1]) : null;

  const counters = [...line.matchAll(/([a-z]{2,3})\s+([0-9.]+)%/g)];
  const unanimous = counters.length >= 8 && new Set(counters.map((m) => m[2])).size === 1;
  if (unanimous) util = null;
  return { util, memUsedMB, bogus: unanimous };
}

/** Starts the sampler if it is not already running against this node. */
function ensureAmdSampler(node: string): AmdSampler | null {
  if (amdSampler && amdSampler.node === node && amdSampler.child.exitCode === null) {
    return amdSampler;
  }
  stopAmdSampler();

  let child: ChildProcess;
  try {
    child = spawn('radeontop', ['-d', '-', '-p', node], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } catch {
    return null;
  }

  const sampler: AmdSampler = {
    node,
    child,
    sample: { util: null, memUsedMB: null, at: 0, warn: null },
    idleTimer: setTimeout(() => stopAmdSampler(), AMD_SAMPLE_IDLE_MS),
  };
  sampler.idleTimer.unref?.();
  amdSampler = sampler;

  // Line-buffered: the dump is newline-delimited, so a partial tail is held
  // until the rest arrives rather than parsed as a short sample.
  let buf = '';
  child.stdout?.setEncoding('utf8');
  child.stdout?.on('data', (chunk: string) => {
    buf += chunk;
    let nl: number;
    while ((nl = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (!line.trim()) continue;
      const parsed = parseRadeontopLine(line);
      sampler.sample = {
        util: parsed.util,
        memUsedMB: parsed.bogus ? null : parsed.memUsedMB,
        at: Date.now(),
        // Sticky: once radeontop has said why it cannot read the card, that
        // answer holds until the process is restarted.
        warn: sampler.sample.warn ?? (parsed.bogus ? 'radeontop reported identical 100% readings for every counter, which a working GPU does not do.' : null),
      };
    }
  });

  // It says so on stderr while still dumping samples, so this has to be read
  // rather than inferred from the numbers.
  let err = '';
  child.stderr?.setEncoding('utf8');
  child.stderr?.on('data', (chunk: string) => {
    err += chunk;
    if (err.length > 2000) err = err.slice(-2000);
    const warn = /cannot access gpu registers|failed to open drm node|no vram support/i.exec(err);
    if (warn) sampler.sample.warn = warn[0];
  });

  child.on('error', () => {
    if (amdSampler === sampler) amdSampler = null;
  });
  child.on('close', () => {
    if (amdSampler === sampler) amdSampler = null;
  });
  return sampler;
}

/**
 * How long the sampler may sit with nobody reading it.
 *
 * With a page open it is read every few seconds and this never fires. Its real
 * job is the abandoned case: close every tab and the sidebar stops polling, so
 * without this the reader would run forever for nobody. It cannot help a server
 * that was SIGKILLed — nothing in the dying process gets to run, including this
 * timer — but it does mean a quiet server holds no reader.
 */
const AMD_SAMPLE_IDLE_MS = 120_000;

function touchAmdSampler() {
  if (!amdSampler) return;
  clearTimeout(amdSampler.idleTimer);
  amdSampler.idleTimer = setTimeout(() => stopAmdSampler(), AMD_SAMPLE_IDLE_MS);
  amdSampler.idleTimer.unref?.();
}

/** A sample newer than this is treated as stale, however healthy the process. */
const AMD_SAMPLE_MAX_AGE_MS = 10_000;

/**
 * The current AMD sample, starting the sampler if needed.
 *
 * Falls back to a single one-shot dump when the sampler has not produced a line
 * yet, so the very first request still answers with real numbers rather than an
 * empty bar — which is what a "warming up" placeholder would have cost.
 */
async function amdSample(node: string): Promise<{ util: number | null; memUsedMB: number | null; reason: string | null }> {
  const sampler = ensureAmdSampler(node);
  if (sampler) touchAmdSampler();
  const fresh = sampler && sampler.sample.at && Date.now() - sampler.sample.at < AMD_SAMPLE_MAX_AGE_MS;
  if (fresh) {
    return {
      util: sampler.sample.util,
      memUsedMB: sampler.sample.memUsedMB,
      reason: sampler.sample.warn,
    };
  }

  try {
    const { stdout, stderr } = await runFull('radeontop', ['-d', '-', '-l', '1', '-p', node]);
    const line = stdout.trim().split('\n').filter((l) => /gpu\s+[0-9.]+%/.test(l)).pop() ?? '';
    const parsed = parseRadeontopLine(line);
    const warn = /cannot access gpu registers|failed to open drm node|no vram support/i.exec(stderr)?.[0];
    return {
      util: parsed.util,
      memUsedMB: parsed.bogus ? null : parsed.memUsedMB,
      reason: warn ?? (parsed.bogus ? 'radeontop reported identical 100% readings for every counter, which a working GPU does not do.' : null),
    };
  } catch {
    return { util: null, memUsedMB: null, reason: null };
  }
}

// Leaves no sampler behind if the process is asked to exit.
process.on('exit', stopAmdSampler);
for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, () => {
    stopAmdSampler();
    process.exit(0);
  });
}

/** Like run(), but keeps stderr, which some tools use for the real diagnostic. */
function runFull(cmd: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { timeout: 6000, maxBuffer: 1024 * 1024 }, (err, stdout, stderr) => {
      if (err && !stdout && !stderr) reject(err);
      else resolve({ stdout: String(stdout), stderr: String(stderr) });
    });
  });
}

const has = (cmd: string) =>
  new Promise<boolean>((resolve) => {
    execFile('sh', ['-c', `command -v ${cmd}`], { timeout: 3000 }, (err) => resolve(!err));
  });

async function detectVendor(): Promise<{ vendor: Vendor; name: string | null; count: number; reason: string | null }> {
  if (gpuProbe) return gpuProbe;
  gpuProbe = (async () => {
    if (await has('nvidia-smi')) {
      try {
        const raw = await run('nvidia-smi', ['--query-gpu=index,name', '--format=csv,noheader']);
        const lines = raw.trim().split('\n').filter(Boolean);
        if (lines.length) {
          return { vendor: 'nvidia', name: lines[0].split(',').slice(1).join(',').trim(), count: lines.length, reason: null };
        }
      } catch {
        /* fall through to AMD */
      }
    }
    // ROCm's kernel node is the reliable signal that an AMD GPU exists.
    try {
      await stat('/dev/kfd');
    } catch {
      return { vendor: null, name: null, count: 0, reason: 'no GPU detected' };
    }
    // radeontop is the single AMD reader. rocm-smi and amdsmi were the
    // alternatives and both need a ROCm install and a readable /dev/kfd, which
    // is a large thing to demand for the sake of reading four numbers the
    // amdgpu driver already publishes. radeontop is ~80 KB, is in every
    // distribution's repositories, and reads the same counters.
    if (await has('radeontop')) {
      return {
        vendor: 'amd',
        name: await amdDeviceName(),
        count: await amdCardCount(),
        reason: null,
      };
    }
    return {
      vendor: 'amd',
      name: null,
      count: 0,
      reason: 'AMD GPU present, but radeontop is not installed (pacman -S radeontop)',
    };
  })();
  // Re-probe periodically: a tool can be installed without a reboot.
  setTimeout(() => (gpuProbe = null), 30_000);
  return gpuProbe;
}

/**
 * The AMD card's DRM render node, so radeontop looks at the right device on a
 * multi-GPU box rather than guessing.
 *
 * The node number is not derivable from the card number. Render nodes are
 * allocated from a separate range starting at 128, so `card1` is `renderD128`
 * here and would be `renderD129` on a box where the Intel card enumerated
 * first. Building the name from the card index produced `renderD1`, which does
 * not exist, and the result was a card that was present and working but
 * reported as unreachable. The kernel already publishes the mapping under
 * `device/drm`, so ask it instead of deriving it.
 */
async function amdRenderNode(): Promise<string | null> {
  try {
    const entries = await readdir('/sys/class/drm');
    for (const e of entries) {
      if (!/^card\d+$/.test(e)) continue;
      const vendor = await readFile(`/sys/class/drm/${e}/device/vendor`, 'utf8').catch(() => '');
      if (vendor.trim() !== '0x1002') continue; // AMD's PCI vendor id
      let nodes: string[] = [];
      try {
        nodes = await readdir(`/sys/class/drm/${e}/device/drm`);
      } catch {
        continue;
      }
      const render = nodes.find((n) => /^renderD\d+$/.test(n));
      if (!render) continue;
      const node = `/dev/dri/${render}`;
      // Confirm it is a real device node and not a stale sysfs entry: a card
      // that failed to initialise leaves the mapping behind with nothing in
      // /dev, which is precisely the case that must not look healthy.
      try {
        if ((await stat(node)).isCharacterDevice()) return node;
      } catch {
        /* no node — the driver never created it */
      }
    }
  } catch {
    /* ignore */
  }
  return null;
}

/** How many AMD cards the kernel has actually exposed. */
async function amdCardCount(): Promise<number> {
  try {
    const { readdir, readFile } = await import('node:fs/promises');
    let n = 0;
    for (const e of await readdir('/sys/class/drm')) {
      if (!/^card\d+$/.test(e)) continue;
      const v = await readFile(`/sys/class/drm/${e}/device/vendor`, 'utf8').catch(() => '');
      if (v.trim() === '0x1002') n += 1;
    }
    return n;
  } catch {
    return 0;
  }
}

/** Total VRAM the amdgpu driver reports, in MB. */
async function amdVramTotalMB(): Promise<number | null> {
  try {
    const { readdir } = await import('node:fs/promises');
    for (const e of await readdir('/sys/class/drm')) {
      if (!/^card\d+$/.test(e)) continue;
      try {
        const vendor = await readFile(`/sys/class/drm/${e}/device/vendor`, 'utf8');
        if (vendor.trim() !== '0x1002') continue;
        const raw = await readFile(`/sys/class/drm/${e}/device/mem_info_vram_total`, 'utf8');
        const bytes = Number(raw.trim());
        if (Number.isFinite(bytes) && bytes > 0) return Math.round(bytes / 1048576);
      } catch {
        continue;
      }
    }
  } catch {
    /* ignore */
  }
  return null;
}

/** Human name for the AMD card, best effort. */
async function amdDeviceName(): Promise<string | null> {
  try {
    const { readdir, readFile } = await import('node:fs/promises');
    for (const e of await readdir('/sys/class/drm')) {
      if (!/^card\d+$/.test(e)) continue;
      try {
        const v = await readFile(`/sys/class/drm/${e}/device/vendor`, 'utf8');
        if (v.trim() !== '0x1002') continue;
        // uevent carries the driver and PCI ids but no marketing name, and
        // "amdgpu" is not a card name. Prefer sysfs's device name when the
        // driver exposes one; otherwise stay silent rather than mislead.
        const named = await readFile(`/sys/class/drm/${e}/device/product_name`, 'utf8').catch(() => '');
        if (named.trim()) return named.trim();
      } catch {
        continue;
      }
    }
  } catch {
    /* ignore */
  }
  return null;
}

async function gpuSample() {
  const info = await detectVendor();
  if (!info.vendor) {
    return { ...info, percent: null, memUsedMB: null, memTotalMB: null, available: false };
  }
  if (gpuCache && Date.now() - gpuCache.at < GPU_CACHE_MS) {
    return {
      ...info,
      percent: gpuCache.util,
      memUsedMB: gpuCache.memUsed,
      memTotalMB: gpuCache.memTotal,
      available: true,
      reason: info.reason ?? gpuCache.reason,
    };
  }

  try {
    let util: number | null = null;
    let memUsed: number | null = null;
    let memTotal: number | null = null;
    // Set when the reader could not actually talk to the card. Null counters
    // and "no GPU" are different answers, and this is the reason for both.
    let unreachable: string | null = null;

    if (info.vendor === 'nvidia') {
      const raw = await run('nvidia-smi', [
        '--query-gpu=utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits',
      ]);
      const rows = raw.trim().split('\n').filter(Boolean).map((r) => r.split(',').map((s) => Number(s.trim())));
      util = rows.reduce((a, r) => a + (r[0] || 0), 0) / rows.length;
      memUsed = rows.reduce((a, r) => a + (r[1] || 0), 0);
      memTotal = rows.reduce((a, r) => a + (r[2] || 0), 0);
    } else {
      // radeontop is the only AMD reader. Its dump mode emits one line per
      // interval, which is exactly a poll interval — but it is measured in whole
      // seconds, so it is run once and kept alive rather than per request.
      // See the sampler above.
      const node = await amdRenderNode();
      if (!node) {
        // No render node means the amdgpu driver never finished bringing the
        // card up. radeontop would still print a line here, with every counter
        // at 100% — a plausible-looking lie that reads as a GPU pinned at full
        // load. Saying nothing is the honest answer.
        unreachable = 'The AMD card has no DRM node — the driver could not start it.';
      } else {
        const s = await amdSample(node);
        util = s.util;
        memUsed = s.memUsedMB;
        unreachable = s.reason;
      }
      const total = await amdVramTotalMB();
      if (total !== null) memTotal = total;
    }

    gpuCache = { at: Date.now(), util, memUsed, memTotal, reason: unreachable };
    return {
      ...info,
      percent: util,
      memUsedMB: memUsed,
      memTotalMB: memTotal,
      available: true,
      reason: info.reason ?? unreachable,
    };
  } catch {
    return { ...info, percent: null, memUsedMB: null, memTotalMB: null, available: true };
  }
}

/**
 * The report is the machine's own numbers only.
 *
 * The reachability chips that used to sit at the base of the Resources box —
 * an Ollama ping and a Python-env stat, both polled every second — are gone:
 * the daemon's state belongs to the running-models list and Settings, where
 * it is acted on, and a stat of a directory nobody changes mid-session
 * answered a question no one was asking.
 */
export async function resourceReport(): Promise<ResourceReport> {
  const cpu = cpuSample();
  const totalMB = Math.round(os.totalmem() / 1048576);
  const freeMB = Math.round(os.freemem() / 1048576);
  const usedMB = totalMB - freeMB;
  const gpu = await gpuSample();

  return {
    cpu: {
      percent: cpu.percent === null ? null : Math.round(cpu.percent),
      cores: os.cpus().length,
      loadAvg1: Number(os.loadavg()[0].toFixed(2)),
      perCore: cpu.perCore.map((n) => (n === null ? null : Math.round(n))),
      model: os.cpus()[0]?.model.trim() ?? 'unknown',
    },
    ram: { totalMB, usedMB, freeMB, percent: totalMB ? Math.round((usedMB / totalMB) * 100) : 0 },
    gpu,
    at: Date.now(),
  };
}