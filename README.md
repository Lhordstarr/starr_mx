# LLM Chat — local + online

A web chat app that streams from **local Ollama models** and **online providers**
(OpenAI, OpenRouter, Anthropic, Google Gemini, Groq) through one interface.
Switch provider or model mid-conversation; the chat history is preserved.

## Quick start

```bash
npm install
cp .env.example .env      # optional: keys can also be set on the /settings page
npm run dev
```

Open http://localhost:3000, settings at http://localhost:3000/settings.

Ollama is detected automatically at `http://localhost:11434`. Your installed
models appear in the dropdown immediately.

## Settings page

`/settings` lets you add, replace and remove API keys without editing files or
restarting the server.

- Keys are saved to `.keys.json` (gitignored, `chmod 0600`). Change the path
  with `KEYS_FILE`.
- The raw key is never sent to the browser. The page shows only a masked preview
  like `sk-a...wxyz`, and the input is a password field, so a stored key is never
  rendered in the DOM.
- Saving applies immediately. Providers resolve keys per request, so there is no
  restart and no reload.
- Each row shows where its value comes from: **saved** (settings page) or
  **from .env**. The settings page wins when both are set; remove it there to
  fall back to `.env`.
- Empty input removes a key. "Remove all saved keys" clears the file.

Keys in `.env` are read once at startup, so editing that file needs a restart.

## Configuration

Keys may come from either source:

| Variable | Provider | Required |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | Ollama (local) | no, defaults to `http://localhost:11434` |
| `LLAMACPP_PORT` | llama.cpp (local) | no, defaults to `9932` |
| `LLAMACPP_CTX` | llama.cpp (local) | no, prompt context, defaults to `16384` |
| `LLAMACPP_SERVER` | llama.cpp (local) | no, path to a `llama-server` binary; defaults to the pinned build this app installs |
| `OPENAI_API_KEY` | OpenAI | for online use |
| `OPENROUTER_API_KEY` | OpenRouter | for online use |
| `ANTHROPIC_API_KEY` | Anthropic | for online use |
| `GEMINI_API_KEY` | Google Gemini | for online use |
| `GROQ_API_KEY` | Groq | for online use |
| `PORT` | — | no, defaults to `3000` |
| `KEYS_FILE` | — | no, defaults to `./.keys.json` |

Providers without a key still appear in the dropdown, labelled *no key* —
except the local ones, which never need one and say *no models* instead when
they have nothing to run. The app
needs no cloud key at all: chat runs locally through Ollama, or through the
llama.cpp runtime it installs itself — which starts with the GGUF model you
pick (`LLAMACPP_IDLE_MS` overrides the ten-minute idle shutdown) and stops it
again when you are done with it.

Both chat backends are started for you. The llama.cpp server comes up with the
model you pick; the Ollama daemon comes up the first time a request needs it,
because listing models deliberately does not start it — opening the settings
page should not launch anything. Both appear in the same process manager, stop
from the same Stop button, shut down after ten idle minutes
(`OLLAMA_IDLE_MS`), and are killed when the app exits.

llama.cpp is installed as the build that suits the machine: **CUDA** on an NVIDIA
card, **Vulkan** where a render node and `vulkaninfo` are both present, **CPU**
otherwise. Upstream publishes the CUDA 12.8 and 13.4 builds separately, so the
one matching the CUDA runtime already on the machine is chosen — otherwise the
server loads and then fails on the first device. Its CUDA runtime libraries come
down with it (594 MB, once) only when the machine has none of its own, which a
Colab runtime, a CUDA toolkit install or torch's own `nvidia-*` wheels all
already provide. A machine on the CPU build with a GPU it could use is offered a
one-click upgrade to the right one.

### Security

**There is no password and no login.** Anyone who can reach the port can use the
app, spend your API credits, and read or change your keys at `/settings`. It
binds to `127.0.0.1` by default so this is a local-only app unless you change
it. Don't set `HOST=0.0.0.0` without putting authentication in front of it.

**Exposing it.** Put a reverse proxy in front that does auth and terminates TLS:

```bash
HOST=127.0.0.1 TLS_CERT=cert.pem TLS_KEY=key.pem npm start
```

The app warns at startup when bound to `0.0.0.0` without TLS. Behind a reverse
proxy, set `TRUST_PROXY` so rate limiting sees real client IPs.

**What still protects you.** CSP with no inline script or style;
`X-Frame-Options: DENY`, `nosniff`, `no-referrer`, HSTS over TLS; same-origin
checks on all writes, which is the main defence against a malicious page in your
browser reaching the local server; key names validated against an allowlist; API
keys masked in every response and stored `0600`; per-IP rate limiting on `/api`
plus a cap on simultaneous streams; prompt-size limits so one request cannot
drain a paid provider. These are abuse limits, not access control — they stop
runaway loops and cross-site writes, not a determined person with network
access.

## Adding your own provider

The settings page can add any **OpenAI-compatible** endpoint — vLLM, LM Studio,
llama.cpp, LocalAI, Together, Fireworks, DeepSeek, or a private gateway:

- **Name** — what appears in the provider dropdown. The id is derived from it
  (`My LLM Server` → `my-llm-server`) and de-duplicated automatically.
- **Base URL** — must be `http://` or `https://`, and includes the version path
  (`http://localhost:8000/v1`). Trailing slashes are trimmed.
- **API key** — optional. Local servers usually need none; when it's left blank
  no `Authorization` header is sent at all.
- **Supports dictation** — turn on if the endpoint answers
  `/audio/transcriptions`. The STT model is then discovered from its `/models`
  list exactly like a chat model, so nothing is hardcoded.

Providers appear in the chat dropdown, on the welcome cards, and in the mic
picker immediately — no restart. Editing the name, URL, key or dictation flag
takes effect on the next request.

**Scanning a provider.** Each custom provider has a **Scan models** button that
asks its `/models` endpoint what it actually offers and lists the results, split
into chat models and dictation models. It runs on demand rather than on page
load, so opening settings never fires a burst of requests at remote providers.
If the endpoint returns nothing, the scan says so and suggests checking the URL,
the key and reachability rather than silently showing an empty list.

Definitions and keys live in `.providers.json` (`0600`, gitignored). The raw key
is never sent to the browser; the settings page only shows a masked preview,
same as the built-in keys. Remove a provider from the settings page, or delete
the file to reset.

Only OpenAI-compatible endpoints are supported. Anthropic- and Gemini-shaped
APIs would need their own adapter, like the built-ins have.

## Dictation

The settings page has a **Dictation** section listing every provider that can
transcribe speech — built-in and custom alike — with a ready / needs-a-key badge
and a **Scan dictation models** button to see the speech models each one
publishes. Providers with no speech API are simply absent, which is how you can
tell at a glance what the composer mic can use.

A provider flagged for dictation that publishes no speech model is called out
after a scan, since it cannot actually be used for dictation until it does.

Local models join the same list. A Whisper downloaded from the Hub — a
CTranslate2 build, a transformers checkout, or a whisper.cpp one — appears as a
dictation engine once its runtime is installed, and the model assigned under
Settings → **What each job uses** is the one the mic uses; when that assignment
cannot run, the mic says so and falls back to another ready engine instead of
failing silently.

### whisper.cpp

A third local transcription backend, and the one worth reaching for when you
have no Python: **about 10 MB, no root, no PyTorch, no venv**. Install it under
Settings → **Runtimes** and download a `ggml-*.bin` model from **Get models**
(tiny is 42 MB, large-v3-turbo is 834 MB). It needs none of the Python stack,
and on a CPU-only machine it is quicker than the transformers path.

Two things about it that are not obvious:

  - **Its models are `.bin`, not `.gguf`.** whisper.cpp reads the GGML format
    and rejects GGUF outright. There are good GGUF Whisper conversions on the
    Hub, but they are built for *transcribe.cpp* and whisper.cpp cannot open
    them, so the catalogue only lists the ones it can actually load.
  - **A recording is decoded before it is uploaded.** `whisper-cli` opens wav,
    flac, mp3 and ogg — and every browser recording is webm/opus or m4a. So for
    a whisper.cpp model the audio is decoded to 16 kHz mono WAV in the browser
    first. The two Python runners take the original bytes untouched. This costs
    a re-encode and no extra install, which is the trade against ffmpeg.

Upstream publishes no GPU build for Linux, so this one runs on the CPU. The
Windows archive does have CUDA variants, and the installer picks them up if a
future build publishes one for Linux.

### Speech, and the two runners behind it

Kokoro is not the only text-to-speech model this app can run. The second runner
is the transformers `text-to-speech` pipeline — **VITS** (which is what every
MMS-TTS checkpoint is, `facebook/mms-tts-eng` among them), SpeechT5, Bark and
FastSpeech2 — and it uses the shared Python environment, so **nothing extra is
installed to use it**: if speech-to-text or image generation already works, so
does this.

Which runner answers is decided by the weights, not by the caller:

  - A config naming one of those architectures is the transformers pipeline.
  - Anything else is Kokoro, which the `kokoro` package loads from a `.pth`
    beside its `voices/`, in an environment of its own.

Two things these models do not have, said out loud in the run log rather than
silently ignored: **one voice per checkpoint** (the voice picker does not apply)
and **no speed control**. Long replies are split on sentence ends before they
reach the model, because VITS predicts a duration for its whole input at once
and was trained on single utterances.

The same rule as everywhere else applies to the device: it is read off the
loaded model, never off the request, so a run that did not move cannot be
reported as one that did.

Speaking a reply is visible while it happens. The Speak button puts a slim bar
up at the top of the window — under the topbar, out of the conversation's way,
and above everything in the window short of a notification — from the moment the
voice is asked for until the clip ends. It drops out from behind the topbar and
retreats back up when it goes, and it carries only what the run reported: the
first words of the reply being read, the model that is reading it, the device
the runner landed on, and the real position in the clip. While the voice is
still being made the line is indeterminate, because there is no length to show
yet, and the bar's button stops the synthesis; once it plays, the same button
pauses and resumes, and the equalizer holds still while it is paused.

## Attachments

The paperclip in the composer takes **images** (PNG, JPEG, WebP, GIF) by picker,
clipboard paste or drag-and-drop, and **plain-text files** (`.md`, `.txt`,
`.json`, `.csv`, source files and similar).

Text files never reach a provider as anything but text — they are appended to the
prompt inside a fenced block, so they work with every provider and need no new
wire format. Only images use the structured content-parts payload, and each
adapter translates it: `image_url` for OpenAI-compatible endpoints, a separate
`images` array for Ollama, typed source blocks for Anthropic, `inline_data` for
Gemini. SVG is refused deliberately — it is scriptable.

Caps: 4 images per message, 5MB each, 12MB combined. Those three are checked
twice — in the browser for a fast message, and again on the server in
`checkAttachments`, which is the boundary that actually matters.

Text files are capped at 256KB and truncated to 60k characters **in the browser
only**, and that asymmetry is deliberate rather than an oversight. A file's
contents are folded into the prompt as a fenced block and travel as one plain
text part, so by the time a request reaches the server the file text and a long
paste are the same thing — re-checking 60k there would reject exactly the
messages it is meant to permit. The server still bounds the payload, just with a
coarser limit that covers the whole conversation: 200k characters across at most
200 messages, inside a 20MB body.

Image support is detected, never assumed. Ollama answers for real via
`/api/show`'s `capabilities` array, and gateways publishing `input_modalities`
are read from `/models`. Where a provider publishes nothing, the paperclip stays
usable and says so, rather than guessing from the model's name.

**Image bytes are not saved to history.** A 5MB image is ~6.7MB of base64, and
`localStorage` caps out around 5MB for the whole origin, so persisting them would
throw away every saved conversation. A restored turn shows a
`[image attached — not saved]` note instead.

## How it works

```
browser  ──SSE /api/chat──>  server  ──provider-specific stream──>  Ollama / OpenAI / …
   ^                            │
   └────── delta / done / error ┘
```

The server exposes one OpenAI-ish endpoint regardless of backend:

- `GET /api/providers` — every provider, its config state, and its live model list
- `POST /api/chat` — SSE stream of `delta`, then `done`, or `error`
- `POST /api/transcribe` — raw audio body, returns `{ text, model }`
- `GET /api/custom-providers` — user-added providers, keys masked
- `POST /api/custom-providers` — add one (`label`, `baseUrl`, `apiKey?`, `stt?`)
- `PUT /api/custom-providers/:id` · `DELETE /api/custom-providers/:id` — edit/remove
- `GET /api/custom-providers/:id/models` — scan an endpoint for its models
- `GET /api/dictation` — every provider that can transcribe, with config state
- `GET /api/vision?provider=<id>` — models accepting images, plus `known: false`
  when the provider publishes no capability information
- `GET /api/settings` — key fields, masked values, and where each came from
- `PUT /api/settings` — save keys (empty value removes)
- `DELETE /api/settings/:envKey` · `DELETE /api/settings` — remove one or all
- `GET /api/processes` — Ollama's loaded models plus the local jobs this app is
  running, with the runtime each one uses
- `POST /api/processes/:id/stop` · `POST /api/processes/stop-all` — stop one or
  every local job, by the id the server minted for it (never a raw pid)
- `POST /api/ollama/unload` — release a loaded model and free its memory

No endpoint requires a session. If you need access control, put a reverse proxy
in front of the app.

Each provider is a small adapter under `src/providers/` that yields plain text
deltas. OpenAI, OpenRouter and Groq share one OpenAI-compatible adapter. Ollama,
Anthropic and Gemini each have their own, because their wire formats differ —
Gemini, for instance, takes `contents`/`parts`, uses `model` instead of
`assistant` as the assistant role, and puts the system prompt in
`systemInstruction`.

Speech-to-text is also per-provider and opt-in: the adapter sets `stt: true`
and gains `transcribe`/`listSttModels`. OpenAI and Groq are enabled, because
both speak the OpenAI `/audio/transcriptions` format. Providers without it
simply omit those methods, and the mic stays hidden.

STT model ids are **not** hardcoded. They come from the same `/models` list as
chat models, filtered by name shape (matching `whisper`/`transcribe`, excluding
`tts`/embeddings/diarization). A provider that renames a model keeps working,
and one that retires it can't leave the app requesting a dead id.

```
src/
  server.ts               express app, SSE endpoint, settings + transcribe API
  limits.ts               rate limiting and concurrent-stream cap
  sse.ts                  SSE / NDJSON stream parsers
  store.ts                runtime API key storage (masked reads, 0600 file)
  processes.ts            registry of the app's own child processes (id, not pid)
  types.ts                shared message types
  providers/
    types.ts              Provider interface (stream + optional transcribe)
    index.ts              registry + display order (built-ins merged with custom)
    custom.ts             user-added providers: validation + 0600 persistence
    ollama.ts             local, newline-delimited JSON
    openai-compat.ts      shared by OpenAI, OpenRouter, Groq (+ STT)
    anthropic.ts          Messages API (system prompt hoisted to top level)
    gemini.ts             Generative Language API (contents/parts, SSE)
scripts/
  stt-selftest.ts         multipart + model-filter checks against a mock (`npm run test:stt`)
  processes-selftest.ts   job registry: stopping, ids, runtime mapping (`npm run test:processes`)
  tools-selftest.ts       registry, validation, loop, adapters, live executors (`npm run test:tools`)
  ui-tools-selftest.mjs   /tool picker + chips in headless chromium (`npm run test:ui-tools`; skips without a browser or a running server)
  speaker-ui-selftest.mjs the "reading aloud" card, synthesis through playback, in headless chromium (`npm run test:speaker-ui`; skips without a browser, a running server or a voice model)
  runnable-selftest.ts    layout discovery for uncatalogued downloads, runtime reasons (`npm run test:runnable`)
public/                   chat UI (no build step, no framework)
  theme.css               design tokens, per-provider accents
  styles.css              chat layout (history rail, messages, composer, pop-out)
  settings.html/js/css    API key settings page
```

## Features

- Turns alternate sides: your messages sit on the right, the model's on the
  left under its avatar, with the bubble tint and a mirrored corner as
  secondary cues
- History in a fixed left rail: conversations top to bottom, with ＋ New
  conversation pinned to the bottom of the rail (on narrow screens the rail
  becomes a drawer behind the ☰ button)
- The settings button opens a small pop-out from the right with the provider
  selector and a link to the API key page — not a full settings screen
- The model selector lives on the composer, right where you type, so switching model mid-conversation never leaves the input
- The UI recolours itself around the active provider (each has its own accent)
- Token-by-token streaming from all six backends
- No login: opens straight into the chat, local-only by default
- Voice input: the mic records up to two minutes and drops the transcript into
  the composer, so you can edit before sending
- A Dictation section on the settings page listing every provider that can
  transcribe, with a scan for the speech models each one publishes
- Add your own OpenAI-compatible provider (local or hosted), with optional
  dictation support — no restart, no code change
- Local model management on the settings page: pull Ollama models by name
  with live progress, or delete ones you no longer want
- Hugging Face browser on the settings page: trending on open, search with
  sizes, every model with a generated tile — open one for a pop-up with every
  file, its extension and byte size, and per-file downloads (gated repos use
  your saved HF token). Local models, the Hub browser, and your downloads each
  have their own view under Models
- **Which local model does speech-to-text, speech and images is chosen from
  what this machine can genuinely run.** The hand-checked catalog in
  `src/model-catalog.ts` is the download guide, and every surface — the mic,
  the Studio, the pickers on Settings → **What each job uses** — reads one
  endpoint (`/api/runnable`) that combines the catalog with what is on disk
  and installed, each model that cannot run carrying the exact reason. A
  download no catalog entry covers is discovered by its own layout — a
  CTranslate2 Whisper (`model.bin` + `config.json`), a transformers speech
  model (its `architectures`: the CTC heads behind wav2vec2, HuBERT, WavLM and
  MMS, plus Whisper, Speech2Text and Moonshine for transcription, and VITS,
  SpeechT5, Bark and FastSpeech2 for synthesis), a diffusers pipeline
  (`model_index.json`) — and becomes selectable as soon as the runtime it
  needs is installed. Detection mirrors the runner that would load the model,
  so a shape nothing can execute — GGUF diffusion weights, `.onnx` files, an
  encoder with no speech head — is left unoffered rather than promised.
- Stop button — aborts the request upstream, keeps partial output
- **A resource manager**, opened from the Resources box on the chat page. It
  lists everything this machine is actually doing for you and lets you stop it.
  Two kinds of work, kept apart because they are different and need different
  actions:
  - **Ollama models.** A model stays resident after a reply ends, holding VRAM
    and RAM until its keep-alive expires, so a machine can look busy minutes
    after the last token arrived. The list shows what each one holds and how
    much of it is in VRAM, and **Unload** lets it go. Unloading is deliberate
    rather than killing the daemon: releasing a model is reversible, and
    stopping the daemon would take every model down at once and break the app's
    own chat.
  - **Local jobs** — a transcription, a synthesis, an image generation. These are
    real child processes, and they are the reason this exists: image generation
    is allowed to run for an hour and previously nothing could interrupt it from
    the UI. **Stop** sends `SIGTERM`, then `SIGKILL` after four seconds, so a
    runner gets a moment to clean up its own partial output. The caller is told
    plainly that it was stopped, rather than being handed an empty success.

  Each job reports the runtime it actually uses, read off the runner rather than
  guessed: `whisper_ct2.py` is CTranslate2 (`faster-whisper`) and never loads
  torch, `whisper-cli` is the native whisper.cpp build and involves no Python
  at all, while the other three runners build on torch. ONNX Runtime
  never appears, because no bundled runner uses it — and offering a Stop for a
  process that cannot exist would be a lie. PyTorch and Kokoro are libraries
  inside a job, not jobs of their own, so they are reported as what a row uses
  rather than as separate things to stop.

  **Only jobs this app started can be stopped.** They are addressed by an id the
  server minted in `src/processes.ts`, never by a pid: this server has no login,
  so a route accepting an arbitrary pid from the browser would be an
  arbitrary-process killer reachable by anything that can talk to the port.
  Everything else on the machine is out of scope and cannot be touched from here.
- Copy and retry on any assistant reply; est. tokens and elapsed time per reply
- Agent tools on the Settings → Tools tab (wrench icon in the topbar): `web_search`,
  `fetch_page` and `wikipedia`, all keyless. Type `/tool` in the message box to
  open the picker and arm tools for the next message; each pick gets a chip
  ("🌐 Web Search ✕") above the input, removable one by one (the chips reuse the
  app's own inline globe/file/book SVGs). The selection is per-message and
  enforced on the server: the request names the exact subset, only those
  definitions go on the wire, anything else the model invents back is refused
  with "not enabled for this request", and the chips clear once a reply lands
  (a failed send keeps them so Retry reuses them). The model calls tools only
  when the question needs them, and each call streams into the transcript as a
  small activity line ("Web Search — query ✓ completed" / "✗ failed — …") under
  the reply; repeated identical calls reuse the first result instead of hitting
  the service again. Schemas are defined once as LangChain (zod) tools — the JSON schema
  the model sees and the args validator both derive from that source — and
  Ollama's tool rounds run through ChatOllama. Sloppy call shapes — a count sent as `"1"`, a stray extra key — are
  cleaned up before validation, so an approximately-correct call still runs;
  when tools cannot run (none selected, or a model without tool support) the
  model is told so outright, and answers from its own knowledge while saying
  when it cannot check — never narrating a search that never happened. A tool
  call that dies mid-loop (provider error, unreadable tool output) degrades
  the same way: a notice and a direct answer instead of a failed chat, and a
  model that keeps calling instead of answering gets a final tool-free round
  whose stray call JSON is swallowed — the reply always ends in plain text,
  never raw JSON. Works
  with OpenAI-compatible providers, custom endpoints, and Ollama;
  anywhere else the chat answers directly
- Light, dark and follow-the-system themes, remembered across visits; dark is
  pure black (`#000000`) everywhere — background, panels and bubbles — with
  provider-tinted glows and stronger borders providing the only separation. The
  topbar, history rail and settings panel get a liquid-glass treatment
  (translucent white film, hairline edge, blurred backdrop) over three slowly
  drifting colour blobs, which give that blur something to sample. Blobs are
  tinted by the active provider and halt under `prefers-reduced-motion`
- **Streaming output is smoothed.** Providers batch unpredictably — a burst of a
  dozen tokens, a pause, then a wall of text — and painting each chunk on arrival
  reproduces that raggedness, so one model looks smooth and the next stutters.
  Text is coalesced into ~30fps paint steps, and the transcript scroll is instant
  during a stream rather than animated: a smooth scroll restarts on every write
  and measurably fell up to 493px behind the bottom, catching up in visible
  lurches. Scrolling up stops the follow; returning to the bottom resumes it
- Model output is rendered by a small markdown subset — fenced code, headings,
  bold/italic, lists and **links**. Links are allow-listed to `http`/`https`;
  any other scheme (`javascript:`, `data:`, `file:`, protocol-relative) drops the
  href and degrades to plain text, so a model can't inject an executable URL.
  Links open in a new tab with `rel="noopener noreferrer nofollow"`
- The send button gains a conic-gradient ring in the provider's own hue on
  hover, built on a registered `@property` angle. Hover-only on purpose: an
  animating angle re-rasterises the gradient every frame, which isn't worth
  paying for an effect nobody sees until they hover
- Below 900px the history rail becomes a drawer behind the ☰ button, so a
  narrow screen still gives the conversation its full width
- Searchable model dropdown per provider, with a free-text override for anything unlisted
- Every conversation is saved to history and reopened from the left rail — switch
  between them mid-session, or start a new one with the ＋ New conversation button
- Search, arrow-key and Enter navigation in the model dropdown; Escape closes it
- Conversation persists in `localStorage` across reloads
- Model output is HTML-escaped before rendering, so responses can't inject markup

## Notes

- Ollama's first request may be slow while the model loads into memory.
- The chat UI sends only user and assistant turns. The adapters still handle a
  `system` role if one arrives over the API: Anthropic requires it as a top-level
  field rather than in the message array, and Gemini puts it in
  `systemInstruction`.
- Model names come from each provider's API, so they reflect what your key can
  actually access.
- **No model id is hardcoded anywhere.** The set of models depends on the key,
  and a hardcoded default goes stale or 404s — so the app picks from whatever
  `/models` returns, using name-shape signals only (stable over preview, chat
  over media-specific, and for Ollama, most recently pulled first since that one
  is likeliest to be resident in memory). Pick anything from the list, or type
  your own.
- Conversations live in the browser's `localStorage`, so they are per-device and
  per-browser, not shared or synced.