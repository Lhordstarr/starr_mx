# STARR Matrix — handoff

Written at the end of the ONNX speech work and the warm speech engine work.
It is meant to be read by whoever picks this up next, including a future me
who does not remember why half the code is the way it is.

---

## 1. What this project is

**STARR Matrix** is a local-first chat app. It lives in this repository
(`/home/nightmare/StarShip/starr_mx`, remote `github.com/Lhordstarr/starr_mx`,
branch `master`).

It is one server and one browser app that:

- **Chats** with cloud providers (OpenAI, OpenRouter, Groq, Gemini, Anthropic,
  a custom OpenAI-compatible endpoint) and with **local** models (Ollama,
  llama.cpp GGUF, downloaded transformers checkpoints).
- **Runs local models itself** for speech-to-text, text-to-speech and image
  generation, and says honestly when a model it downloaded has *no job* on this
  machine rather than pretending it can be run.
- **Manages the machine** it runs on: a resource manager that lists what is
  actually running, what it holds in RAM/VRAM, and lets it be stopped or
  unloaded.
- **Keeps its own house style**: achromatic, no branding marks, no "runs on
  this PC" pills, no startup chatter in the transcript, no duplicated tool
  lines. The wordmark reads "S T A R R Matrix"; the window and console titles
  stay "STARR Matrix".

Stack: **TypeScript + Express** on the server (`src/`), **plain browser JS,
no framework, no build step** on the client (`public/`), **Python runners**
for the model work (`scripts/py/`), **Electron** shell (`desktop/main.mjs`).
Roughly 17k lines of server TS, 22k lines of browser assets, 1.6k lines of
Python, 11k lines of tests.

### The two rules this project is judged by

1. **Never promise what the machine cannot do.** If a runtime is missing, if a
   model has no runner, if a device cannot be used, the app says so in the
   place the user will read it. The whole history of this codebase is a long
   list of places where it used to promise and was corrected.
2. **Tests must not be vacuous.** A test that passes against the broken code
   is worse than no test. New behaviour gets a test that fails when the
   behaviour is reverted.

### Ground rules that are not negotiable

- **Never ask for, or accept, the user's sudo password.** The app's own
  browser password prompt is the correct path for anything needing root.
- **Electron and the browser both have to keep working.** No packaging work
  beyond what exists.
- **LocalAI adoption stays code-level.** Take the *ideas* from LocalAI, not
  its container or backend model.

---

## 2. What is in the repository

### Top level

| Path | What it is |
|---|---|
| `src/` | The server. TypeScript, ESM, run directly with `tsx` in dev. |
| `public/` | The browser app. Plain JS modules + CSS. No bundler, no framework. |
| `scripts/` | Test suites (`*-selftest.ts` / `*-selftest.mjs`) and the Python runners in `scripts/py/`. |
| `desktop/` | The Electron shell (`main.mjs`). Starts the server, opens a window. |
| `models/` | Downloaded weights. **Git-ignored.** 5.9 GB on this box. |
| `.runtimes/` | Python venvs, pinned llama.cpp/whisper.cpp binaries, generated wavs. **Git-ignored.** 3.3 GB. |
| `README.md` | The user-facing documentation. It is expected to stay true; several of its claims were wrong before this work and were corrected. |
| `package.json` | 24 test scripts (29 in all), `dev`, `build`, `start`, `typecheck`. |

### `src/` — the server

| File | Responsibility |
|---|---|
| `server.ts` | Express app, every route, the SSE plumbing, shutdown handling. |
| `infer.ts` | The one-shot runners (`run()`), and `speak()` / `transcribe()` / image generation. Chooses the runner from the weights. |
| `tts-engine.ts` | **New.** The warm speech engines: keeps a runner loaded, one engine per model, ten-minute idle release, manual unload, fallback to the one-shot path. |
| `runnable.ts` | What this machine can run, by task, with the reason when it cannot. Layout detection for every supported model family. |
| `model-catalog.ts` | The hand-picked model catalogue (what "Get models" offers). |
| `runtimes.ts` | The optional runtimes (torch, transformers, kokoro, onnxruntime, faster-whisper, diffusers, whisper.cpp, llama.cpp, Ollama), their installers, probes, and the GPU story. |
| `processes.ts` | The registry of processes this app started. Job ids, stop, memory and VRAM readings, the `warm` state. |
| `huggingface.ts` | Hub downloads, the download records, the shared HF cache. |
| `transformers-chat.ts` | The persistent transformers chat server. **The model the warm TTS engine follows**: 10-minute idle timeout, one model at a time, a supervisor so nothing is left behind. |
| `llamacpp.ts` | The pinned llama.cpp binary, GPU build selection (CUDA on NVIDIA, Vulkan otherwise), idle shutdown. |
| `ollama-server.ts` | The Ollama daemon: started on demand, killed when the app closes, never listed as a stoppable resource. |
| `providers/` | One file per provider family. |
| `agent.ts`, `tools.ts`, `reasoning.ts`, `sse.ts`, `transfers.ts`, `attachments.ts`, `vram.ts`, `resources.ts`, `store.ts`, `paths.ts`, `backends.ts`, `drivers.ts`, `limits.ts`, `types.ts`, `net.ts`, `imagegen.ts`, `image-cache.ts`, `discovery.ts`, `ollama-models.ts`, `whispercpp.ts`, `assignments.ts` | Supporting modules. |

### `public/` — the browser

`index.html` + `app.js` (chat, speaking bar, composer), `settings.html` +
`settings.js` (models, runtimes, tools), `studio.html` + `studio.js` (image,
transcribe, speak panels), `resources.js` (the meters and the process
manager, mounted by both pages), `audio.js` (browser-side audio conversion),
`attachments.js`, `highlight.js`, `notify.js`, `nav.js`, `shader.js`,
`viewer.js`, plus the CSS.

### `scripts/py/` — the runners

| Script | Does |
|---|---|
| `speak.py` | Kokoro `.pth` — PyTorch. |
| `speak_onnx.py` | Kokoro ONNX export — ONNX Runtime for the acoustic graph. |
| `speak_hf.py` | The transformers `text-to-speech` pipeline (VITS, SpeechT5, Bark, FastSpeech2). |
| `tts_engine.py` | **New.** Holds one of the three loaded and takes one utterance at a time. |
| `transcribe.py` | transformers ASR. |
| `transcribe_onnx.py` | ONNX Whisper (int8 encoder + merged decoder). |
| `whisper_ct2.py` | CTranslate2 / faster-whisper. |
| `generate.py` | diffusers image generation. |
| `chat_transformers.py` | The transformers chat server. |

### The test suites

`npm run test:<name>`. The ones that matter most:

| Suite | What it proves |
|---|---|
| `test:tts-engine` | **New.** The engine loads once, is reused, is warm with a deadline, can be unloaded, releases itself when idle, and falls back when it cannot start. |
| `test:tts-engine-ui` | **New.** Headless Chromium: the row says "Voice engine", "warm", counts down, and Unload really unloads. |
| `test:tts` | Kokoro device honesty (the device reported is read off the model), the runner dispatch, long text, the EPIPE guard. |
| `test:onnx` | **New.** The ONNX catalogue entries are runnable, the layouts are detected, and the real models produce audio and text. |
| `test:onnx`/`test:whisper`/`test:stt-local` | The transcription paths. |
| `test:runnable` | What the machine can run, and the reasons it cannot. |
| `test:runtimes` | The runtime page, the installers, the GPU story. |
| `test:processes`, `test:process-poll`, `test:vram` | The resource manager. |
| `test:speaker-ui`, `test:models-ui`, `test:llamacpp-ui`, `test:reasoning-ui`, `test:ui-tools` | Headless-browser checks of the rendered pages. |

---

## 3. What has been changed (and why)

### 3.1 ONNX speech became real work — `a3e30a3`

Before this, a `.onnx` file was a *format label*. The catalogue carried the
Kokoro ONNX export with no `run` descriptor and **two quantisations listed as
if both were required**, the runtime page said ONNX Runtime was unused, and
there was **no ONNX speech-to-text entry at all**.

- `scripts/py/speak_onnx.py` and `scripts/py/transcribe_onnx.py` are new
  runners.
- `scripts/py/speak.py` was restructured so the Kokoro front end (G2P,
  chunking, the 510-phoneme style table) is shared between the ONNX graph and
  the PyTorch checkpoint. The acoustic model is the ONNX graph; the phonemiser
  is still `kokoro`.
- `src/runnable.ts` gained `findKokoroOnnx()` and `isOnnxWhisper()` and
  generic ONNX layout detection. `driverFor()` now answers `'onnx'` before
  `'transformers'`, because the ONNX Whisper runner also uses transformers for
  its feature extractor and must not be labelled as the PyTorch pipeline.
- `src/infer.ts` resolves the repo root when the weights live under `onnx/`
  and picks the right script.
- `src/model-catalog.ts` has two new runnable entries: the Kokoro ONNX export
  (graph + `voices/af_heart.bin` + `tokenizer.json` + `config.json`, 148 MB)
  and `onnx-community/whisper-tiny.en` (encoder + merged decoder + 9 tokenizer
  sidecars, 43 MB).
- `src/runtimes.ts`: the Kokoro environment now installs `onnxruntime`; the
  ONNX Runtime group is described honestly and unlocks speech-to-text.

### 3.2 The voice stays loaded — `8f35d23`, `203b2c5`

A speech run used to be a process that lived for exactly as long as the
utterance: it loaded the weights, said one thing, and exited. The resource
manager showed a job that was gone before it could be read, and every reply
paid for the model load again.

- The three runners each gained `build_engine()` and `render()`. Their
  command-line form is now the same two calls, so there is one implementation
  of how a voice is made, not two that can drift apart.
- `scripts/py/tts_engine.py` holds one of them loaded and speaks a
  one-JSON-object-per-line protocol: `{"ready": true, "device": "cpu"}` when
  the model is loaded, then one answer per request.
- `src/tts-engine.ts` manages the engines: **one per model per device**, a
  **ten idle minute** release (`TTS_ENGINE_IDLE_MS`), manual release, a
  request timeout (`TTS_ENGINE_REQUEST_MS`), and a fallback to the one-shot
  runner whenever an engine cannot be started.
- `src/processes.ts` learned the `warm` state and an `idleUntil` deadline.
- `public/resources.js` shows a warm engine as *Voice engine*, reads "warm"
  rather than working, counts down the time it has left, and offers **Unload**
  instead of Stop.
- The engines are released when the app closes.

Measured on this box: Kokoro `.pth` first reply 33.7 s → next reply 4.6 s.
Kokoro ONNX 21.0 s → 6.1 s. The same pid answers both.

### 3.3 ONNX on the card — the GPU build of ONNX Runtime

The ONNX models ran on the processor even on a machine with a GPU, because
ONNX Runtime was installed as its **CPU wheel** and nothing ever asked for the
other one. The CPU wheel and the GPU wheel import under the same name, so
"installed" could not tell them apart, and nothing in the app knew which one
it had.

  - `src/runtimes.ts` — the ONNX Runtime group and the Kokoro group now
    resolve their packages from the device: the GPU wheel on an NVIDIA
    machine, the CPU wheel without one. The GPU wheel is pinned to
    `onnxruntime-gpu==1.26.0`, a **CUDA 12** build, because that is the CUDA
    the `cu128` torch build uses.
  - The two builds are never left side by side — they write the same files —
    so the one that is not wanted is uninstalled first.
  - `src/runtimes.ts` — the probe now reads the runtime's own execution
    providers, and the Runtimes page shows *GPU build (CUDA)* or *CPU build*,
    with a one-click **Upgrade to GPU (CUDA)** when a card is there and the
    CPU build is installed.
  - `scripts/py/speak_onnx.py`, `scripts/py/transcribe_onnx.py` — import torch
    **before** ONNX Runtime and call `preload_dlls()`, because the CUDA
    libraries are the ones torch brought in and the runtime has to be told
    where they are. A GPU that will not start falls back to the CPU and says
    so, rather than failing the run.

### 3.4 The models page and the downloads

Three things on the Models page, all reported at once:

  - **A download label no longer pushes its button out of the card.** The pull
    progress line ("Downloading 9a1c3f77e2b0 — 45.9% — 2.2 GB / 4.9 GB") was
    wider than a 330px card, and a flex item's `min-width: auto` stops it
    shrinking — so the button hung 30px out through the card's edge. The
    button now carries the percentage and the tooltip carries the rest, and
    the CSS refuses to let any button in that row exceed its card.
  - **A finished download offers no Resume.** The button was shown for every
    download that was not running, which included the ones that had
    finished. A paused download (the partial file is kept) says **Resume**; a
    failed one says **Retry**; the server now says which happened
    ("paused" / "cancelled") so the label can be honest. The suite holds the
    three states against a stubbed `/api/transfers` — finished offers Delete
    and no Resume, paused says "paused" and offers Resume, failed offers
    Retry — so the old "Resume for anything not running" rule fails it.
  - **One install button for the runtimes the models are waiting on.** Three
    models waiting on three different runtimes produced three buttons that
    all opened the same page. There is one now, and each group still says
    what it is waiting on.
  - **`/api/hf/files` is a poll, and it is now allowed to be one.** The Models
    page re-reads it on every download tick, and the general limit of 30 a
    minute answered it with "Rate limit exceeded" — which the browser reads
    as an empty file list, so the page that lists your models emptied itself.
    It has the same room as the other poll endpoints, and the page keeps the
    last good list rather than showing nothing.

### 3.5 The tools stay armed — a switch beside the picker

The tools were armed for one message and nothing else: a reply finished, the
chips went, and the next message had to go through the picker again. The
composer now carries a **Tools** button (which opens the picker, and says what
is armed) and a **switch** beside it (which says whether it stays armed).

  - **Switch off** — the old behaviour, unchanged: per-message, cleared when a
    reply lands, kept when a send fails so Retry can reuse it.
  - **Switch on** — the tools stay armed for every message, and the chips say
    so ("on for every message"). Turning it on with nothing chosen arms every
    tool the registry has enabled, because the point of a switch is to be the
    one click that does the thing.
  - Both the switch and the selection are remembered between visits, so
    "until you turn it back off" survives a reload.
  - A tool that has since been turned off in the picker is dropped from a saved
    selection, rather than making every message fail with "Tool not enabled".
  - The shared on/off switch moved from `settings.css` to `theme.css`, which
    both pages load, so the two switches are the same switch.
  - The picker has a **close button** in its corner. It could always be
    dismissed by clicking away, but that click lands on whatever was
    underneath, so the thing the person was aiming at is the thing that gets
    clicked. The close button puts the focus back in the input, the way
    Escape does.

### 3.6 The Settings → Tools tab is gone — the switches moved into the picker

The Settings page had a **Tools** tab whose only content was one switch per
tool: whether the model may use that tool at all. The composer then grew its
own Tools button and switch, which answer a different question (what is armed
for this message). Two places to turn a tool off, in two vocabularies, and the
switch that mattered was on the page you were not looking at.

  - The tab's switches are now a small switch on each row of the picker, beside
    the tool they are about. The row still arms the tool for this message; the
    switch says whether the model may use it at all. Turning it off unarms the
    tool and greys the row, so a tool the model cannot use is never silently
    missing.
  - The picker's hint line now carries both sentences the tab used to own: how
    many tools are turned off, and which configured providers support tool
    calls.
  - `TABS` is `['keys', 'models']`; an old `#tools` hash falls back to the keys
    tab the same way any unknown hash does. The `/tools` route redirects to
    `/#tools` and the chat page opens the picker on that hash, so a bookmark
    made when the tab existed still lands on the thing it named.
  - The server's "No tools enabled" notice points at the Tools button instead
    of a Settings tab.
  - `settings.css` lost the `.tool-card` block that only the tab used; the
    chat page kept its own `.tm-*` row styles.

### 3.7 Smaller things, from the recent history

- `d228929` — a bare date question is answered from the clock, before the
  model is even prepared, instead of being sent to Wikipedia.
- `bd0931c` — the model is given the date, and one transcript line per unique
  tool call per run (no duplicates).
- `3566924` — the dual-stack fetch stall is gone and failures are named
  instead of surfacing as a bare `fetch failed`.
- `9785c30` — the diamond brand mark is removed.
- `7f1a231` — `.gitignore` extended.
- `8f35d23` also removed the committed Python `__pycache__` bytecode from
  version control and ignored it.

### 3.8 Stable Diffusion did not work — the dtype and the guidance

Two bugs, both in `scripts/py/generate.py`, and together they read as "local
image generation is broken": the runner produced coloured noise on a fast
machine and appeared to hang on a slow one.

- **bfloat16 on a CPU that cannot run it.** The runner loaded every CPU
  pipeline in bf16 to save memory. PyTorch has real bf16 kernels only where
  the chip has AVX512-BF16 or AMX; everywhere else it converts to fp32 for
  every operation. Measured on the dev box (i5-6200U, AVX2 only): **0.3
  GFLOPS for bf16 against 55 for fp32**, a 170x gap. One 64x64 step took
  **267 s**, and a 256x256 one-step image was still running after twenty
  minutes. The choice is now made from the machine: fp16 on the card; on a
  CPU, bf16 only where it is native or where total RAM is under 7 GB (fp32
  weights are about 4.5 GB, so on a small laptop the smaller build is the
  only one that loads); fp32 otherwise. The dtype and the reason are printed
  on the `{"device": ...}` line the resource manager already reads.
- **Guidance 7.5 on a distilled model.** sd-turbo and sdxl-turbo are trained
  without classifier-free guidance — their cards ask for `guidance_scale=0`.
  The pipeline default is 7.5, so the unconditional half was being built from
  a model that was never taught to use it, and at one step the output was
  noise. The runner now reads the tree's name (`...-turbo`) and passes 0.
- **The text encoder is released after the prompt is encoded.** It runs once;
  its weights are ~1.2 GB of dead weight for the denoiser. Both embedding
  halves are built up front when guidance is on, because the pipeline would
  otherwise reach for the encoder again — after it has been dropped — and
  only one of `prompt`/`prompt_embeds` is forwarded, which the pipeline
  insists on (it raises `Cannot forward both ...` otherwise).

After the fix, on the same box: `512x512`, 4 steps, one image in **about two
minutes** including the 67 s load, and it is a clean "red apple on a wooden
table". A dtype suite (`npm run test:image-dtype`, 4 checks) asks the runner
for its policy and compares it with the machine facts read independently, so
a future edit that hard-codes one dtype fails.

### 3.9 Ollama after a restart — the model list lives on disk

The daemon is this app's own child and is killed when the app exits, by
design. The chat picker got its models from `/api/tags` only, so right after
a restart it was empty — and picking a model is what starts the daemon, so
it could never start. Settings showed "not reachable" instead of the models
that were installed. A real deadlock, and the reason "Ollama is Installed but
does not work" looked true even with models on disk.

- **The list is on disk either way.** Ollama writes one small manifest per
  tag under `OLLAMA_MODELS` (default `~/.ollama/models`), in two layouts that
  coexist: `manifests/<registry>/<namespace>/<name>/<tag>` files, and since
  Ollama 0.12 `manifests-v2/<registry>/<namespace>/<name>/<tag>` symlinks
  into `blobs/`. `modelsOnDisk()` walks both, dedupes by name (the v2 entry
  wins — the daemon reads it too), sums the layer sizes, and reads the config
  blob for `parameter_size`, `quantization_level` and `family` — the exact
  three fields `/api/tags` emits (`model_type`, `file_type`,
  `model_family`). Verified on this box: all 7 models, sizes and details are
  identical with the daemon up and with it killed.
- **`listLocalModels()` falls back to that store** only when the daemon call
  fails, and never when the caller aborted. The chat provider's `listModels()`
  now goes through the same function, so it stays daemon-free on page load
  and still shows everything installed.
- **The 404 wording was lying.** Ollama answers 404 for a model it does not
  have; the provider turned that into "Ollama is not answering … start the
  daemon", which made an unpulled model look like a broken install. It now
  surfaces Ollama's own text: `Ollama: model 'x' not found`.
- **Verified live:** daemon down → 7 models in `/api/providers` and
  `/api/ollama/models`; a chat request starts the daemon, and the picker
  still lists 7. The new checks in `npm run test:ollama-models` use a fake
  store in both layouts with a dead daemon; against the old code the first
  one fails with "not reachable".

### 3.10 The ONNX voice was silent on the card — the fp16 exports

The box with the T4 is the first machine here where ONNX Kokoro could run on
a GPU, and its first synthesis produced a valid 10.43 s WAV that was **all
−32768**. The session said `device: cuda`, the sample count and duration were
right, and nothing in the API, the log or the play button could tell it apart
from speech.

- **The graph, not the request.** The same file on `device: cpu` produced
  8,305 distinct samples (RMS 2161). On CUDA it returned NaNs, which
  libsndfile turns into a full-length constant file — the failure looks like
  a successful run until it is played.
- **Both fp16 exports do it.** q4f16 (154 MB) and fp16 (163 MB) are silent
  on the T4 under `CUDAExecutionProvider`; fp32 `model.onnx` and q4
  `model_q4.onnx` both produce real audio (`device: cuda`, 16,027 and 15,574
  distinct samples, RMS 2370 and 2358). q4f16 is also correct on the CPU,
  which is why the old preference order looked fine everywhere except here.
- **The bundle now ships `model_q4.onnx`** (4-bit weights, fp32 activations,
  305 MB) instead of q4f16, and both `findKokoroOnnx()` and the runner's
  `find_model()` put it first. The fp16 names remain last, so a CPU-only
  machine with an old bundle that has nothing else still runs.
- **The runner refuses silent audio.** `check_audio()` in `speak_onnx.py`
  runs before `sf.write`: a non-finite or constant signal is an error naming
  the graph, the device and the q4 file to download, not a WAV. Warm engines
  call `render()` directly, so they are covered too.
- **Verified live on the second box (2× T4):** q4 on CUDA and on the CPU,
  fp32 on CUDA; the warm engine reports `resolvedDevice: cuda`, 671 MB VRAM
  and 1.96 GB RSS. The test suite now asserts the q4 bundle, a mixed
  directory picks q4 over q4f16 (fails on the old order), and the guard test
  drives `check_audio()` with NaN and constant arrays (fails on the old
  runner).

### 3.11 The cloud image was labelled with the request, not the answer

A 1024×1024 request to Pollinations came back as a 768×768 JPEG. The adapter
returned the width and height it had asked for, so the gallery caption said
1024×1024 and the saved session remembered a size the file never had. The
bytes are the only authority.

- `actualImageSize()` (in `src/image-size.ts`) reads the header of the PNG,
  JPEG, WebP or GIF that came back, and `generateImage()` reports those
  dimensions. An unknown format falls back to the provider's claim rather
  than inventing a size.
- **Verified live:** a 1024×1024 Pollinations request now reports
  **768×768** (15.7 s through the provider call); the local sd-turbo path
  returned a real 512×512 PNG in 19 s with `device: cuda`.
- `npm run test:image-size` builds PNG, JPEG (with segments before the
  frame), WebP and GIF headers, and asserts that a 768 JPEG overrides a
  1024 claim. Six checks; the regression check cannot pass against the old
  adapter because the helper did not exist.

---

## 4. Methods that were tried and did **not** work

This section exists so nobody spends the same afternoon again.

### ONNX Whisper

- **Using the merged decoder's past-key-value cache.** The obvious thing to
  do, and it produces "the, the, the" instead of a transcript — while still
  looking like a successful run. **Do not use the cache branch.** The
  no-cache recomputation is slower and correct; the whole point of the
  no-cache path is that it is right. `transcribe_onnx.py` sets
  `use_cache_branch=false` and feeds empty past tensors.
- **Claiming a split decoder layout.** `decoder_model.onnx` +
  `decoder_with_past_model.onnx` repos are a different layout and are **not**
  claimed by the app. Claiming them produces a failure deep inside ONNX
  Runtime instead of an honest "no job" label. Only the merged decoder counts.
- **Treating the ONNX Kokoro export as a PyTorch-free path.** The `kokoro`
  package is still needed for G2P, and it imports torch. The catalogue
  blurb used to say "no PyTorch at all" and was wrong.

### ONNX Kokoro

- **Listing two quantisations in one bundle** (`model_q4f16.onnx` and
  `model_q4.onnx` as if both were needed). The runner only ever loads one.
  The entry now names exactly one graph.
- **Using the token count as the style-vector index.** The style row is
  `len(phonemes)`, not the number of tokens, and the table has 510 rows; a
  chunk of 510 or more phonemes has no style row and must be refused.
- **Loading a torch checkpoint just to get the phonemiser.** `KPipeline(
  lang_code='a', repo_id='hexgrad/Kokoro-82M', model=False)` gives the front
  end without the weights. First use costs about 4.6 s.

### ONNX on the GPU

- **Installing the latest `onnxruntime-gpu`.** The newest releases (1.27.0 and
  later) are **CUDA 13** builds; 1.26.0 and earlier are CUDA 12. A CUDA 13
  runtime does not run on a CUDA 12 driver, and the torch build this app
  installs is `cu128` — CUDA 12. So the wheel is pinned to `==1.26.0`. Check
  the boundary before changing it: the PyPI metadata for each release says
  which CUDA its `nvidia-*` extras want.
- **Installing the GPU wheel over the CPU one and leaving it there.** They
  are different distributions that install the same import package, so they
  overwrite each other's files and a later "Remove" deletes the build that is
  actually in use. The installer now uninstalls the rival first.
- **Importing onnxruntime before torch.** The CUDA libraries are loaded into
  the process when torch is imported. A runtime that initialises its
  providers before that finds no libraries, loads anyway, and then fails at
  the first session — a much worse place to find out than at import.
- **Expecting the wheel to carry CUDA.** It does not (since 1.19). The
  libraries are the ones torch's own CUDA build brought in, and
  `onnxruntime.preload_dlls()` is what points the runtime at them. It exists
  in 1.31 and is the reason the GPU build works with a pip-installed torch.
- **Offering the ROCm build for AMD.** `onnxruntime-rocm` is a separate wheel
  with its own compatibility story. It is not wired: an AMD machine gets the
  CPU build and is told so, which is the honest answer rather than a promise
  that could not be kept.
- **Reporting the request instead of the answer.** A runner that was asked
  for CUDA and got the CPU has to say *cpu*. The provider reported is read
  off the session (`sess.get_providers()`), never off the request.

### The always-on tools switch

- **Clearing the selection in `releaseTools()` regardless of the switch.** The
  switch was added and the clearing was left in place, so the tools were
  disarmed after every reply and the switch did nothing. The clearing is now
  conditional on the switch being off.
- **Painting the chips before the registry listing arrives.** The labels come
  from `/api/tools`; a chip drawn before that reads as `web_search` — the
  registry's id, not the name a person knows it by. The chips are painted
  again when the listing lands, and a restored selection asks for it.
- **Persisting the selection without pruning it.** The server refuses a
  message that names a tool which is not enabled, so a selection saved before
  a tool was turned off in Settings made every message fail. The selection is
  pruned against the listing.
- **One control doing two things.** Choosing *which* tools and deciding
  *when* they may be used are two questions, and a click that did both is the
  kind of thing a person stops trusting. They are two controls in one group.
- **Putting the switch in the composer row without giving the model name its
  width back.** On a 390px screen the row overflowed and the Send button went
  off the edge. The model trigger gives up the width on narrow screens
  (30vw) and the tools button drops its label.

### The warm engine

- **A separate "keep-alive" flag on the process registry alone.** A process
  that is "running" for ten minutes is a lie about what it is doing. The
  registry needed a real `warm` state and a deadline, and the browser had to
  learn to say "warm" rather than "running".
- **Resolving the start promise from the wrong place.** The first version of
  the engine set `ready = true` in `markReady()` but never resolved the
  waiters, so `waitForReady()` never returned and **every request hung for
  ever**. The symptom was invisible from the server's side: the engine
  process was there, warm, and the HTTP request never finished. Fixed by
  resolving the waiters inside `markReady()`. *If a request ever hangs again,
  check this first.*
- **Trusting the engine's answer without a deadline.** A malformed or missing
  answer left the request waiting for ever. There is now a request timeout
  (`TTS_ENGINE_REQUEST_MS`, 10 minutes) after which the engine is released
  and the one-shot runner takes over.
- **Writing a fake interpreter for the test with `JSON.stringify()` inside a
  double-quoted bash `echo`.** The path lost its quotes, the JSON was
  malformed, and the engine's answer was silently dropped — which looked
  exactly like a hang in the engine and cost real time. Bash quoting inside
  `echo` is the trap; use single quotes and a shell variable.
- **Letting the UI test pick "the first engine row on the page".** With two
  voices warm there are two rows, and the test was unloading somebody else's
  engine. It now matches on the model repo.
- **Assuming the first runnable TTS model is the assigned one.** That was
  true until there were two Kokoro entries. The speaker UI test now follows
  the assignment, like the app does.
- **Leaving `speak_hf.py` out of `SCRIPT_RUNTIMES`.** A process row that
  names no runtime is still a wrong row. The table is now complete.
- **Writing to a response the browser has already left.** The speaking bar's
  Stop button aborts the request while the model is still working; the server
  used to write to the closed response. `withPythonLog()` now checks.

### Environment facts that look like bugs

- `.runtimes/venv/bin/pip` has a broken shebang. Use
  `.runtimes/venv/bin/python -m pip`.
- This dev box has **no NVIDIA GPU**. "GPU 2.0 GB" in the meters is the
  shared/virtualised figure. GPU claims cannot be proven here; they are
  proven by invariant ("the device reported is read off the model").
- The UI suites need the dev server up and `/api/tools` answering before
  they start.
- `tsx watch` restarts the server on every source change, and the restart
  kills the warm engines with it (by design — see `process.on('exit')` in
  `src/tts-engine.ts`). If the server is busy in a long request when a file
  changes, `tsx` waits, then force-kills it ("Previous process hasn't exited
  yet") and the request dies with it. Use `npx tsx src/server.ts` (no watch)
  when a test or a job is going to run for minutes — that is what killed one
  of the first image runs.

---

## 5. State of play right now

- `master` is at the latest commit and **pushed**. Working tree clean.
- The dev server is running in the background on port 3000. On this box it is
  currently `npx tsx src/server.ts` (no watch, log at
  `/tmp/opencode/dev-nowatch.log`) because image runs are long; `npm run dev`
  (watch, log at `/tmp/opencode/dev.log`) is the normal one.
- Models downloaded on this box: Kokoro `.pth`, Kokoro ONNX, ONNX Whisper,
  MMS-TTS, wav2vec2, whisper-small, faster-whisper-tiny, whisper.cpp models,
  `HuggingFaceTB/SmolLM2-135M-Instruct`, sd-turbo, bert-tiny. The SmolLM2
  **GGUF** is no longer in `models/`, so llama.cpp reports "not configured"
  and the two GGUF checks in `test:models-ui` skip — nothing is broken, there
  is simply no GGUF on disk to load.
- Current task assignments in the app: speech-to-text = wav2vec2
  (transformers), speech = **Kokoro ONNX**, image = sd-turbo.
- **A fresh box is live at `https://mutual-welcome-doll-contribute.trycloudflare.com`**
  (`/kaggle/working/starr_mx`, **2× Tesla T4**). It was completely empty:
  no venv and no runtimes. The venv package itself needed root, so the
  runtimes were installed through the browser (never with a password sent
  over the API). All nine runtimes are installed; `torch 2.11.0+cu128`
  reports CUDA available with two T4s.
  - sd-turbo downloaded (12 files, 2.58 GB) and generates 512×512 / 4 steps
    in **19 s** (26 s cold) with `device: cuda` and a clean image.
  - Cloud Pollinations flux returns a cached 768×768 JPEG in **7–9 s**. It
    used to be labelled 1024×1024; §3.11 makes the label follow the bytes.
  - ONNX Kokoro is downloaded and synthesises through `/api/infer/tts`:
    q4 on CUDA (and on the CPU), fp32 on CUDA; q4f16/fp16 are the silent
    files described in §3.10. The box currently has only `model_q4.onnx`
    on disk (the diagnostic variants were deleted), which is the graph the
    new code prefers anyway.
  - Ollama is installed with **0 models**; the daemon stays down until the
    first chat request.
  - `/tools` redirects to `/#tools`. The old box at `/content/starr_mx` is
    current only through the image fix; the Ollama disk listing in §3.9 is
    newer than it.

### Questions that are still open, and are the user's to answer

1. **Speaking bar thickness** — the bar is slim and pinned at the top. No
   verdict yet on whether that is the right weight.
2. ~~**Stable Diffusion**~~ — reproduced and fixed: it was the CPU dtype
   (bfloat16 on a CPU without bf16 kernels) and the guidance default on a
   distilled model. Verified here with sd-turbo at 512x512, 4 steps. The
   **GPU path is the one branch this laptop cannot test** — worth confirming
   on the box with the card, where fp16 runs.
3. ~~**Ollama shows "Installed" but does not work.**~~ Answered: on the live
   box the daemon does start on demand; the box has **no models pulled at
   all** (`/api/tags` is `[]`), so there was nothing to pick and nothing to
   run. Pull one from Settings → Models. The restart deadlock that made this
   worse (an empty picker could not start the daemon) is fixed in §3.9.
4. **The avatar's "generating" motion** — which animation it should use.
5. ~~**ONNX on the GPU**~~ — done and verified on the 2× T4 box: the GPU
   build is installed, the graph runs with `device: cuda`, and the warm
   engine shows its VRAM. The verification found the fp16-silence bug fixed
   in §3.10, which is the reason this item was worth doing on the card
   rather than trusting the CPU result.
6. ~~**The cloud image path reports the requested size, not the returned
   one.**~~ Fixed in §3.11: a 1024×1024 request comes back as a 768×768
   JPEG and is now labelled 768×768. Verified through the provider.

---

## 6. What we do next

Not decided yet — this section is deliberately empty of decisions and is
where the next conversation starts. The shortlist, in the order it seems
worth doing:

1. Answer the five open questions above (they are the user's to answer, not
   ours to guess).
2. Push the ONNX and warm-engine work to the live box and verify it there,
   including that the live box's Kokoro environment is reinstalled so it has
   `onnxruntime`, and that the ONNX models actually land on the T4 — the
   Runtimes page says *GPU build (CUDA)* or *CPU build*, and the resource
   manager's row says which device the run reached.
3. Whatever comes next, the bar is the same as always: it must work on the
   live box, it must be honest, and it must come with a test that fails
   against the old code.
