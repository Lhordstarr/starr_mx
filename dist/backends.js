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
const base = () => process.env.OLLAMA_BASE_URL ?? 'http://127.0.0.1:11434';
function getJson(url, timeoutMs = 4000) {
    return new Promise((resolve) => {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), timeoutMs);
        fetch(url, { signal: ctrl.signal })
            .then((r) => (r.ok ? r.json() : null))
            .catch(() => null)
            .finally(() => clearTimeout(timer))
            .then((v) => resolve(v ?? null));
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
async function ollamaBackend() {
    const baseUrl = base();
    const ps = await getJson(`${baseUrl}/api/ps`);
    if (!ps)
        return null;
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
    const device = partOnGpu && partOnCpu ? 'hybrid' : frac >= 0.999 ? 'gpu' : frac <= 0.001 ? 'cpu' : 'unknown';
    const mb = (n) => `${(n / 1048576).toFixed(0)} MB`;
    const detail = device === 'hybrid'
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
async function serverBackend(id, label, url, extraPath = '/v1/models') {
    const res = await getJson(`${url.replace(/\/$/, '')}${extraPath}`);
    if (!res)
        return null;
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
async function pythonBackend(status, groups) {
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
            detail: `On the GPU via ${onGpu[0]?.deviceName ?? 'torch'}${labels ? ` — runs ${labels}` : ''}`,
            gpuFraction: 1,
            controllable: false,
            readonlyReason: 'Fixed at install time. Reinstall with a different compute choice to change it.',
            evidence: `${torches.map((t) => t.label).join(' and ')} report is_available() is true`,
        };
    }
    if (onGpu.length === 0) {
        return {
            id: 'python',
            label: 'Python runtime',
            device: 'cpu',
            detail: `On the CPU${labels ? ` — runs ${labels}` : ''}`,
            controllable: false,
            // The actionable half. A CPU-only torch cannot move anything to a card
            // even when one is fitted and working — the build has no torch.cuda at
            // all — and pip will not replace it unless it is asked to upgrade, which
            // is what this app now does on the next install.
            readonlyReason: 'PyTorch here is the CPU-only build. Reinstall with Compute set to GPU to replace it.',
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
            .join(', ')} on the CPU${labels ? ` — runs ${labels}` : ''}`,
        controllable: false,
        readonlyReason: 'Reinstall the CPU-only environment to move it onto the GPU.',
        evidence: `${onGpu.map((t) => t.label).join(', ')} report is_available() is true; ${onCpu
            .map((t) => t.label)
            .join(', ')} report it is false`,
    };
}
/** Everything, best-effort: one backend failing must not blank the others. */
export async function backendReports(opts) {
    const out = [];
    const [ollama, python] = await Promise.all([
        ollamaBackend().catch(() => null),
        pythonBackend(opts.python, opts.groups).catch(() => null),
    ]);
    if (ollama)
        out.push(ollama);
    if (python)
        out.push(python);
    const found = await Promise.all(opts.servers.map((s) => serverBackend(s.id, s.label, s.baseUrl).catch(() => null)));
    for (const f of found)
        if (f)
            out.push(f);
    return out;
}
