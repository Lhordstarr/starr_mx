/**
 * What each execution backend is actually doing, measured rather than declared.
 *
 * The reason this exists: the app used to infer "your GPU is unusable" from
 * whether ROCm could use it, and on this machine that was simply false. Ollama
 * was running a model on a card the runtime page called dead, because it went
 * through Vulkan rather than ROCm. Inference from capability is how that
 * happened — it asks what a backend *could* do instead of what it *is* doing.
 *
 * So nothing here is declared up front. Every backend reports from its own
 * live state:
 *
 *  - Ollama       — `/api/ps` reports `size_vram` against `size`. That ratio is
 *                   not a guess about GPU support, it is the split between what
 *                   is resident in VRAM and what is not, which is also exactly
 *                   the hybrid case: a model bigger than the card shows a
 *                   partial number and is running on both at once.
 *  - Python       — asks torch and ctranslate2 what they resolved to.
 *  - Local servers— whatever their own API reports, with the evidence kept so
 *                   a wrong-looking answer can be argued with.
 */

export type BackendDevice = 'gpu' | 'cpu' | 'hybrid' | 'none' | 'unknown';

export interface BackendReport {
  id: string;
  label: string;
  /** What the backend is doing right now. */
  device: BackendDevice;
  /** Plain English, shown next to the name. */
  detail: string;
  /** Fraction of the current work on the GPU, 0..1, when it can be known. */
  gpuFraction?: number;
  /** Whether this app can change the answer, or only report it. */
  controllable: boolean;
  /** Why it cannot be changed, when it cannot. */
  readonlyReason?: string;
  /** Loaded models and their split, for the backends that load models. */
  models?: Array<{ name: string; sizeBytes: number; vramBytes: number }>;
  /** The raw fact the verdict came from, so nothing is asserted on faith. */
  evidence?: string;
}

const base = () => process.env.OLLAMA_BASE_URL ?? 'http://127.0.0.1:11434';

function getJson<T>(url: string, timeoutMs = 4000): Promise<T | null> {
  return new Promise((resolve) => {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    fetch(url, { signal: ctrl.signal })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null)
      .finally(() => clearTimeout(timer))
      .then((v) => resolve((v as T) ?? null));
  });
}

/**
 * Ollama, read from what it currently has loaded.
 *
 * `size_vram` is the model's bytes resident in video memory and `size` is its
 * total. Equal means entirely on the GPU, zero means entirely on the CPU, and
 * anything between is the split the app cannot otherwise express: a model too
 * large for the card, running partly on each.
 */
async function ollamaBackend(installed = false): Promise<BackendReport | null> {
  const baseUrl = base();
  const ps = await getJson<{ models?: Array<{ name: string; size: number; size_vram: number }> }>(
    `${baseUrl}/api/ps`,
  );
  if (!ps) {
    // An installed daemon that is not answering is a backend state worth
    // seeing. Omitting it made "Ollama is down" look identical to "Ollama was
    // never installed", and the only clue was a missing row.
    if (!installed) return null;
    return {
      id: 'ollama',
      label: 'Ollama',
      device: 'none',
      detail: 'Installed, but the daemon is not answering',
      controllable: true,
      evidence: `${baseUrl}/api/ps did not answer`,
    };
  }

  const models = (ps.models ?? []).map((m) => ({
    name: m.name,
    sizeBytes: m.size ?? 0,
    vramBytes: m.size_vram ?? 0,
  }));

  if (!models.length) {
    return {
      id: 'ollama',
      label: 'Ollama',
      device: 'none',
      detail: 'Running, with no model loaded',
      controllable: true,
      models: [],
      evidence: 'the daemon answered /api/ps with an empty list',
    };
  }

  const total = models.reduce((a, m) => a + m.sizeBytes, 0);
  const vram = models.reduce((a, m) => a + m.vramBytes, 0);
  const frac = total > 0 ? vram / total : 0;

  // A tenth of a percent is not a split, it is rounding. Reporting "hybrid"
  // for a model that is 99.99% on the GPU would be noise dressed as detail.
  const partOnGpu = frac > 0.001;
  const partOnCpu = frac < 0.999;
  const device: BackendDevice =
    partOnGpu && partOnCpu ? 'hybrid' : frac >= 0.999 ? 'gpu' : frac <= 0.001 ? 'cpu' : 'unknown';

  const mb = (n: number) => `${(n / 1048576).toFixed(0)} MB`;
  const detail =
    device === 'hybrid'
      ? `Split across both: ${mb(vram)} of ${mb(total)} in VRAM`
      : device === 'gpu'
        ? `Entirely on the GPU — ${mb(vram)} in VRAM`
        : `Entirely on the CPU — nothing in VRAM (${mb(total)} loaded)`;

  return {
    id: 'ollama',
    label: 'Ollama',
    device,
    detail,
    gpuFraction: frac,
    controllable: true,
    models,
    evidence: `size_vram ${mb(vram)} of size ${mb(total)} across ${models.length} model(s)`,
  };
}

/** A local OpenAI-compatible server the app found on loopback. */
async function serverBackend(
  id: string,
  label: string,
  url: string,
  extraPath = '/v1/models',
): Promise<BackendReport | null> {
  const res = await getJson<{ data?: unknown[] }>(`${url.replace(/\/$/, '')}${extraPath}`);
  if (!res) return null;
  const count = res.data?.length ?? 0;
  return {
    id,
    label,
    // Deliberately 'unknown': an OpenAI-compatible endpoint says nothing about
    // its device, and guessing from the model list would be the same mistake
    // this whole file exists to stop.
    device: 'unknown',
    detail: `Serving ${count} model${count === 1 ? '' : 's'} — this API does not report its device`,
    controllable: false,
    readonlyReason: 'Run by a separate program; its device is set where it was started.',
    evidence: `${extraPath} answered with ${count} entr${count === 1 ? 'y' : 'ies'}`,
  };
}

/** The Python runtime: what torch and ctranslate2 resolved to. */
async function pythonBackend(
  status: {
    venv: boolean;
    torch: { cudaAvailable: boolean; deviceName: string | null };
    /** One entry per environment that has a torch, since they can disagree. */
    torches?: Array<{ label: string; cudaAvailable: boolean; deviceName: string | null }>;
  },
  groups: Array<{ id: string; label: string }>,
): Promise<BackendReport> {
  if (!status.venv) {
    return {
      id: 'python',
      label: 'Python runtime',
      device: 'none',
      detail: 'Not installed — needed for speech-to-text, speech and image generation',
      controllable: false,
      readonlyReason: 'Install it from the list above.',
    };
  }
  const labels = groups.map((g) => g.label).join(', ');
  // Named per environment. The main venv and Kokoro's are built separately and
  // can hold different torch builds, so one reading reported for the machine
  // says GPU on a screen where the thing that speaks is on the CPU.
  const torches = status.torches?.length
    ? status.torches
    : [
        {
          label: 'the Python runtime',
          cudaAvailable: status.torch.cudaAvailable,
          deviceName: status.torch.deviceName,
        },
      ];
  const onCpu = torches.filter((t) => !t.cudaAvailable);
  const onGpu = torches.filter((t) => t.cudaAvailable);

  if (onCpu.length === 0) {
    return {
      id: 'python',
      label: 'Python runtime',
      device: 'gpu',
      detail: `On the GPU via ${onGpu[0]?.deviceName ?? 'torch'}${labels ? ` — Python packages: ${labels}` : ''}`,
      gpuFraction: 1,
      controllable: false,
      readonlyReason:
        'Fixed at install time. Reinstall with a different compute choice to change it.',
      evidence: `${torches.map((t) => t.label).join(' and ')} report is_available() is true`,
    };
  }
  if (onGpu.length === 0) {
    return {
      id: 'python',
      label: 'Python runtime',
      device: 'cpu',
      detail: `On the CPU${labels ? ` — Python packages: ${labels}` : ''}`,
      controllable: false,
      // The actionable half. A CPU-only torch cannot move anything to a card
      // even when one is fitted and working — the build has no torch.cuda at
      // all — and pip will not replace it unless it is asked to upgrade, which
      // is what this app now does on the next install.
      readonlyReason:
        'PyTorch here is the CPU-only build. Reinstall with Compute set to GPU to replace it.',
      evidence: `${torches.map((t) => t.label).join(' and ')} report is_available() is false`,
    };
  }
  return {
    id: 'python',
    label: 'Python runtime',
    // Conservative on purpose: part of this is on the CPU, so a reader told
    // "GPU" would go looking for a speed-up that is not there for that part.
    device: 'cpu',
    detail: `Split — ${onGpu.map((t) => t.label).join(', ')} on the GPU, ${onCpu
      .map((t) => t.label)
      .join(', ')} on the CPU${labels ? ` — Python packages: ${labels}` : ''}`,
    controllable: false,
    readonlyReason: 'Reinstall the CPU-only environment to move it onto the GPU.',
    evidence: `${onGpu.map((t) => t.label).join(', ')} report is_available() is true; ${onCpu
      .map((t) => t.label)
      .join(', ')} report it is false`,
  };
}

/** The live state behind the native runners, gathered by the server. */
export interface NativeBackendState {
  onnx: Array<{ venv: string; label: string; providers: string[] }>;
  llama: {
    installed: boolean;
    running: boolean;
    model: string | null;
    pid: number | null;
    kind: 'cuda' | 'vulkan' | 'cpu' | null;
    /** The card's compute capability, or null when it cannot be read. */
    cudaCompute: number | null;
    /** Whether the pinned CUDA archive has kernels for this card. */
    cudaUsable: boolean;
  };
  transformers: {
    installed: boolean;
    running: boolean;
    model: string | null;
    pid: number | null;
    cuda: boolean;
    deviceName: string | null;
  };
  sd: {
    installed: boolean;
    kind: 'vulkan' | 'cpu' | 'metal' | null;
    activeDevice: string | null;
    activePid: number | null;
  };
  whisper: { installed: boolean; device: 'cpu' | 'gpu' };
  fasterWhisper: { installed: boolean; cudaDevices: number | null };
}

function providerDevice(providers: string[]): BackendDevice {
  if (providers.some((p) => /CUDA|ROCM|TensorRT/i.test(p))) return 'gpu';
  if (providers.includes('CPUExecutionProvider')) return 'cpu';
  return providers.length ? 'unknown' : 'none';
}

const KIND_LABEL: Record<string, string> = {
  cuda: 'CUDA',
  vulkan: 'Vulkan',
  metal: 'Metal',
  cpu: 'CPU',
};

function kindBuild(kind: string | null): string {
  return kind ? `${KIND_LABEL[kind] ?? kind} build` : 'build unknown';
}

/**
 * The runners that are not a Python package and not Ollama.
 *
 * These rows exist because "the Python runtime is on the GPU" is not an answer
 * to "is Stable Diffusion using the card?" — the native binaries carry their
 * own build, and the ONNX wheels live in more than one environment. Each row
 * reports the fact its own component publishes, including the idle ones: a
 * runtime that is installed but not running should not disappear.
 */
export function nativeBackendReports(state: NativeBackendState): BackendReport[] {
  const out: BackendReport[] = [];

  const shared = state.onnx.find((o) => o.venv === 'venv');
  if (shared) {
    const device = providerDevice(shared.providers);
    out.push({
      id: 'onnxruntime',
      label: 'ONNX Runtime',
      device,
      detail:
        device === 'gpu'
          ? `Shared environment — ${shared.providers[0]}`
          : `Shared environment — ${shared.providers.join(', ')}`,
      controllable: false,
      readonlyReason:
        device === 'cpu'
          ? 'This is the CPU wheel. Reinstall ONNX Runtime with Compute set to GPU to replace it.'
          : undefined,
      evidence: `onnxruntime.get_available_providers() = ${JSON.stringify(shared.providers)}`,
    });
  }

  const kokoro = state.onnx.find((o) => o.venv === 'venv-kokoro');
  if (kokoro) {
    const device = providerDevice(kokoro.providers);
    out.push({
      id: 'kokoro-onnx',
      label: 'Kokoro voice',
      device,
      detail:
        device === 'gpu'
          ? `ONNX voice engine — ${kokoro.providers[0]}`
          : `ONNX voice engine — ${kokoro.providers.join(', ')}`,
      controllable: false,
      readonlyReason:
        device === 'cpu'
          ? 'The voice environment has the CPU ONNX wheel; reinstall Kokoro with Compute set to GPU.'
          : undefined,
      evidence: `Kokoro's onnxruntime.get_available_providers() = ${JSON.stringify(kokoro.providers)}`,
    });
  }

  if (state.llama.installed) {
    const build = kindBuild(state.llama.kind);
    // A CUDA build whose fatbin has no kernels for this card is installed but
    // unusable. Calling it a GPU backend repeated the lie that sent a T4 into a
    // crash at the first chat; report it as the CPU fallback it effectively is,
    // and name the reason the card cannot use this archive.
    const brokenCuda = state.llama.kind === 'cuda' && !state.llama.cudaUsable;
    const gpu = !brokenCuda && (state.llama.kind === 'cuda' || state.llama.kind === 'vulkan');
    // The CPU build is the correct install on a card the pinned CUDA archive
    // cannot address. Without this branch the row told a T4 owner to "reinstall
    // with Compute set to GPU", which would choose the same CPU build again.
    const cudaFallback =
      state.llama.kind === 'cpu' && state.llama.cudaCompute !== null && !state.llama.cudaUsable;
    const cudaReason =
      state.llama.cudaCompute !== null
        ? `The pinned Linux CUDA build contains only Blackwell (sm_120) kernels; this card is compute ${state.llama.cudaCompute}. Reinstall llama.cpp to replace it with the CPU build.`
        : 'This CUDA build has no kernels for this card. Reinstall llama.cpp to replace it with the CPU build.';
    const cudaFallbackReason =
      state.llama.cudaCompute !== null
        ? `The pinned Linux CUDA build contains only Blackwell (sm_120) kernels; this card is compute ${state.llama.cudaCompute}. The CPU build is installed because it is the one that works; a CUDA source build is needed to use this card.`
        : 'The pinned Linux CUDA build has no kernels for this card, so the CPU build is installed.';
    if (state.llama.running) {
      out.push({
        id: 'llamacpp',
        label: 'llama.cpp',
        device: gpu ? 'gpu' : 'cpu',
        detail: brokenCuda
          ? `Serving ${state.llama.model ?? 'a model'} — ${build} reported, but it cannot run on this card`
          : `Serving ${state.llama.model ?? 'a model'} — ${build}`,
        controllable: true,
        models: state.llama.model
          ? [{ name: state.llama.model, sizeBytes: 0, vramBytes: 0 }]
          : undefined,
        evidence: `llama-server pid ${state.llama.pid ?? '?'} is running`,
      });
    } else {
      out.push({
        id: 'llamacpp',
        label: 'llama.cpp',
        device: gpu ? 'none' : 'cpu',
        detail: brokenCuda
          ? `Installed — ${build}; this card has no matching kernels, so chats would fail`
          : cudaFallback
            ? `Installed — ${build}; this card has no kernels in the pinned CUDA archive, so chats run on the processor`
            : gpu
              ? `Installed — ${build}, idle; loads a model on the first chat`
              : `Installed — ${build}; chat models will run on the processor`,
        controllable: false,
        readonlyReason: gpu
          ? 'The server starts on demand, so there is nothing to unload until a chat has loaded one.'
          : brokenCuda
            ? cudaReason
            : cudaFallback
              ? cudaFallbackReason
              : 'This build has no GPU backend. Reinstall llama.cpp with Compute set to GPU to replace it.',
        evidence: `no llama-server process; ${build} libraries found`,
      });
    }
  }

  if (state.transformers.installed) {
    const where = state.transformers.cuda
      ? `PyTorch CUDA${state.transformers.deviceName ? ` on ${state.transformers.deviceName}` : ''}`
      : 'PyTorch CPU';
    if (state.transformers.running) {
      out.push({
        id: 'transformers',
        label: 'Transformers',
        device: state.transformers.cuda ? 'gpu' : 'cpu',
        detail: `Serving ${state.transformers.model ?? 'a model'} — ${where}`,
        controllable: true,
        evidence: `chat_transformers.py pid ${state.transformers.pid ?? '?'} is running`,
      });
    } else {
      out.push({
        id: 'transformers',
        label: 'Transformers',
        device: state.transformers.cuda ? 'none' : 'cpu',
        detail: `Installed — ${where}, idle; loads a model on the first chat`,
        controllable: false,
        readonlyReason: state.transformers.cuda
          ? 'The server starts on demand, so there is nothing to unload until a chat has loaded one.'
          : 'The shared PyTorch build is CPU-only. Reinstall with Compute set to GPU to replace it.',
        evidence: `no chat_transformers.py process; torch reports CUDA ${state.transformers.cuda}`,
      });
    }
  }

  if (state.fasterWhisper.installed) {
    const cuda = state.fasterWhisper.cudaDevices;
    const device: BackendDevice =
      typeof cuda === 'number' && cuda > 0 ? 'gpu' : typeof cuda === 'number' ? 'cpu' : 'unknown';
    out.push({
      id: 'faster-whisper',
      label: 'Faster Whisper',
      device,
      detail:
        device === 'gpu'
          ? `CTranslate2 sees ${cuda} CUDA device${cuda === 1 ? '' : 's'} — dictation uses the GPU`
          : device === 'cpu'
            ? 'CTranslate2 reports no CUDA devices — dictation uses the processor'
            : 'Installed — CTranslate2 could not report its CUDA devices',
      controllable: false,
      readonlyReason:
        device === 'cpu'
          ? 'CTranslate2 reports no CUDA devices, so dictation uses the processor. On an NVIDIA machine, reinstall with Compute set to GPU.'
          : undefined,
      evidence: `ctranslate2.get_cuda_device_count() = ${cuda ?? 'unknown'}`,
    });
  }

  if (state.whisper.installed) {
    out.push({
      id: 'whispercpp',
      label: 'whisper.cpp',
      device: state.whisper.device,
      detail:
        state.whisper.device === 'gpu'
          ? 'Native GPU build — transcribes on the card'
          : 'Native CPU build — transcribes on the processor',
      controllable: false,
      readonlyReason:
        state.whisper.device === 'cpu'
          ? 'The Linux archive upstream publishes has no CUDA backend; Windows has one. The CPU build always works.'
          : undefined,
      evidence: `whisper-cli is installed with a ${state.whisper.device.toUpperCase()} backend`,
    });
  }

  if (state.sd.installed) {
    const build = kindBuild(state.sd.kind);
    const active = state.sd.activeDevice;
    if (active) {
      out.push({
        id: 'stable-diffusion-cpp',
        label: 'Stable Diffusion',
        device: active === 'cpu' ? 'cpu' : 'gpu',
        detail: `Generating on ${active} — ${build}`,
        controllable: true,
        evidence: `sd-cli pid ${state.sd.activePid ?? '?'} is running`,
      });
    } else if (state.sd.kind === 'cpu') {
      out.push({
        id: 'stable-diffusion-cpp',
        label: 'Stable Diffusion',
        device: 'cpu',
        detail: 'Installed — CPU build; image generation will run on the processor',
        controllable: false,
        readonlyReason:
          'Upstream publishes no Linux CUDA archive for stable-diffusion.cpp. ' +
          'A Vulkan-capable GPU or the Diffusers route is the way onto a card on Linux.',
        evidence: 'sd-cli has no libggml-vulkan or libggml-cuda beside it',
      });
    } else {
      out.push({
        id: 'stable-diffusion-cpp',
        label: 'Stable Diffusion',
        device: 'none',
        detail: `Installed — ${build}, idle; image generation will use the GPU`,
        controllable: false,
        readonlyReason: 'The CLI starts per generation, so there is nothing to unload while it is idle.',
        evidence: `sd-cli has a ${state.sd.kind ?? 'GPU'} backend and no active process`,
      });
    }
  }

  return out;
}

/** Everything, best-effort: one backend failing must not blank the others. */
export async function backendReports(opts: {
  python: {
    venv: boolean;
    torch: { cudaAvailable: boolean; deviceName: string | null };
    torches?: Array<{ label: string; cudaAvailable: boolean; deviceName: string | null }>;
  };
  groups: Array<{ id: string; label: string }>;
  servers: Array<{ id: string; label: string; baseUrl: string }>;
  /** True when the Ollama group is installed, so a stopped daemon is shown. */
  ollamaInstalled?: boolean;
  /** Native runner state, omitted by tests and callers that only need Python. */
  native?: NativeBackendState;
}): Promise<BackendReport[]> {
  const out: BackendReport[] = [];
  const [ollama, python] = await Promise.all([
    ollamaBackend(Boolean(opts.ollamaInstalled)).catch(() => null),
    pythonBackend(opts.python, opts.groups).catch(() => null),
  ]);
  if (ollama) out.push(ollama);
  if (python) out.push(python);
  if (opts.native) out.push(...nativeBackendReports(opts.native));
  const found = await Promise.all(
    opts.servers.map((s) => serverBackend(s.id, s.label, s.baseUrl).catch(() => null)),
  );
  for (const f of found) if (f) out.push(f);
  return out;
}
