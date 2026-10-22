/**
 * A curated catalog of models that are known to run on a local machine, and
 * the runtime each one actually needs.
 *
 * This is deliberately a hand-checked list rather than a Hub query. A live
 * query returns mirrors, half-finished conversions and repos whose quant files
 * no longer exist; the point of this page is "I want to run TTS locally, what
 * do I download and with what". Every repo, file name and byte count below was
 * verified against the Hub's API, and every `ollamaTag` was confirmed to exist
 * in Ollama's registry.
 *
 * Two facts drive the shape of the data:
 *
 *  - Ollama serves chat models and nothing else. `ollama pull whisper:...`
 *    404s, as do TTS and image repos. So only the text task carries a Pull
 *    button; the rest are download-and-run-elsewhere.
 *  - Ollama pulls from its own registry, never from Hugging Face. The two
 *    actions are unrelated paths to the same model, never one behind the other.
 */

export type CatalogTask = 'text' | 'vision' | 'speech-to-text' | 'text-to-speech' | 'image-generation';

export interface CatalogFile {
  /** Repo-relative path, exactly as the Hub spells it. */
  name: string;
  label: string;
  bytes: number;
  /**
   * Where this file lives when it is not in the entry's main repo.
   *
   * Some quantised pipelines are split across repositories — the diffusion
   * weights in one, the VAE in another — and making the user download those as
   * two unrelated cards produced two half-models. This keeps one card as the
   * unit of "what runs", while the downloader still fetches each file from the
   * repository that actually publishes it.
   */
  repo?: string;
  /**
   * Other file names that satisfy this slot.
   *
   * The Kokoro ONNX repo ships several graph exports with different precision
   * rules. `model_q4.onnx` is the one the catalog downloads because it is safe
   * on CUDA, but a user who fetched `model_q4f16.onnx` from Browse has a graph
   * the runner genuinely supports. Without this list the assignment called
   * that download "missing one file" and refused to run it.
   *
   * The first name remains the download choice; these are presence checks
   * only, so a quantisation is never silently substituted on disk.
   */
  oneOf?: string[];
}

export interface CatalogEntry {
  repo: string;
  label: string;
  task: CatalogTask;
  blurb: string;
  /** What actually executes this model. Not a suggestion — the only runtime that loads these formats. */
  runtime: string;
  /** Present only where the registry tag was verified to exist. */
  ollamaTag?: string;
  files: CatalogFile[];
  /**
   * True when the files are one indivisible model rather than pick-one
   * variants. A CTranslate2 Whisper is `model.bin` plus three small companions;
   * downloading only the weights leaves a directory the loader rejects. When
   * set, the UI fetches all of them together and shows the size as a sum.
   */
  bundle?: boolean;
  /** Total size of a bundle, in MB. The UI shows this as one figure. */
  bundleMB?: number;
  /**
   * Which installed runtimes this entry needs before it can be *run* here, and
   * which bundled runner handles it. Empty or absent means the app cannot run
   * it: llama.cpp, ComfyUI and faster-whisper are external tools, and the UI
   * says so instead of offering a button that cannot work.
   */
  run?: {
    kind: 'vision' | 'stt' | 'tts' | 'image';
    requires: string[];
    /** Sensible starting point for the Run panel; the user can change it. */
    steps?: number;
  };
}

export interface CatalogSection {
  task: CatalogTask;
  title: string;
  /** Why this task has no Pull button, when that's true. */
  note: string;
  entries: CatalogEntry[];
}

const MB = 1024 * 1024;

/**
 * The small companions an ONNX Whisper export needs.
 *
 * The graph files are the weights, but the runner also asks `transformers` for
 * the feature extractor and the tokenizer, and both read these by relative
 * name. They are listed once so the catalog's bundle and the layout detector
 * cannot drift apart on which sidecar is optional — none of them is.
 */
export const ONNX_WHISPER_SIDECARS = [
  'config.json',
  'generation_config.json',
  'preprocessor_config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.json',
  'merges.txt',
  'normalizer.json',
] as const;

/** The four files a CTranslate2 Whisper repository ships. All are required:
 * `model.bin` is the weights, and the other three are the vocabulary and
 * tokenizer CTranslate2 insists on finding next to them.
 */
/**
 * Marks a generated file list as a single indivisible download.
 *
 * Kept as a call rather than a literal flag so the generated block stays valid
 * TypeScript if the shape ever needs another field.
 */
function diffusersBundle(files: CatalogFile[]): CatalogFile[] {
  return files;
}

function ct2Bundle(bytes: number): CatalogFile[] {
  return [
    { name: 'model.bin', label: 'weights', bytes },
    { name: 'config.json', label: 'config', bytes: 2.3 * 1024 },
    { name: 'tokenizer.json', label: 'tokenizer', bytes: 2.2 * MB },
    { name: 'vocabulary.txt', label: 'vocabulary', bytes: 0.46 * MB },
  ];
}

export const CATALOG: CatalogSection[] = [
  {
    task: 'text',
    title: 'Text & chat',
    note:
      'Ollama serves chat models, so these pull straight from your daemon — or download the GGUF and run it on this machine with llama.cpp. ' +
      'Instruct models chat well; base models tend to repeat.',
    entries: [
      {
        repo: 'unsloth/Qwen3-8B-GGUF',
        label: 'Qwen3 8B',
        task: 'text',
        blurb: 'Strong general workhorse; good tool calling and long context for its size.',
        runtime: 'Ollama or llama.cpp',
        ollamaTag: 'qwen3:8b',
        files: [
          { name: 'Qwen3-8B-Q4_K_M.gguf', label: 'Q4_K_M', bytes: 4795 * MB },
          { name: 'Qwen3-8B-Q5_K_M.gguf', label: 'Q5_K_M', bytes: 5580 * MB },
        ],
      },
      {
        repo: 'unsloth/Qwen3-4B-GGUF',
        label: 'Qwen3 4B',
        task: 'text',
        blurb: 'Halves the memory of the 8B for a small drop in quality.',
        runtime: 'Ollama or llama.cpp',
        ollamaTag: 'qwen3:4b',
        files: [{ name: 'Qwen3-4B-Q4_K_M.gguf', label: 'Q4_K_M', bytes: 2382 * MB }],
      },
      {
        repo: 'bartowski/Llama-3.2-3B-Instruct-GGUF',
        label: 'Llama 3.2 3B Instruct',
        task: 'text',
        blurb: 'Very small and quick; fine on CPU-only machines.',
        runtime: 'Ollama or llama.cpp',
        ollamaTag: 'llama3.2:3b',
        files: [{ name: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf', label: 'Q4_K_M', bytes: 1926 * MB }],
      },
      {
        repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF',
        label: 'Llama 3.2 1B Instruct',
        task: 'text',
        blurb: 'Smallest useful option; for quick replies and low-memory boxes.',
        runtime: 'Ollama or llama.cpp',
        ollamaTag: 'llama3.2:1b',
        files: [{ name: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf', label: 'Q4_K_M', bytes: 770 * MB }],
      },
      {
        repo: 'ggml-org/gemma-3-4b-it-GGUF',
        label: 'Gemma 3 4B Instruct',
        task: 'text',
        blurb: 'Compact model with a wide vocabulary and good instruction following.',
        runtime: 'Ollama or llama.cpp',
        ollamaTag: 'gemma3:4b',
        files: [{ name: 'gemma-3-4b-it-Q4_K_M.gguf', label: 'Q4_K_M', bytes: 2374 * MB }],
      },
      {
        repo: 'Qwen/Qwen2.5-7B-Instruct',
        label: 'Qwen2.5 7B Instruct',
        task: 'text',
        blurb: 'Original safetensors release, for running under transformers rather than Ollama.',
        runtime: 'transformers',
        files: [{ name: 'model-00001-of-00004.safetensors', label: 'safetensors shard 1/4', bytes: 3763 * MB }],
      },
      {
        repo: 'HuggingFaceTB/SmolLM2-135M-Instruct',
        label: 'SmolLM2 135M Instruct',
        task: 'text',
        blurb: 'Tiny instruct model in safetensors form, for the transformers runtime.',
        runtime: 'Transformers',
        files: [{ name: 'model.safetensors', label: 'weights', bytes: 269 * MB }],
        bundle: true,
      },
    ],
  },
  {
    task: 'vision',
    title: 'Vision & multimedia',
    note:
      'Multimodal models from Ollama’s registry — no Hugging Face download and ' +
      'no Python stack. Assign one under What each job uses; it turns images ' +
      'and sampled video frames into text for whichever chat model is active. ' +
      'Audio is handled by the Speech to text slot, not here. Sizes are the ' +
      'pulled model and do not include the chat model that will also be loaded.',
    entries: [
      {
        repo: 'ollama/moondream',
        label: 'Moondream 2 (1.9B)',
        task: 'vision',
        blurb:
          'The realistic choice on an 8 GB machine: 1.6 GB, fast enough on a CPU, ' +
          'and good at describing photos, screenshots and diagrams. Images only.',
        runtime: 'Ollama',
        ollamaTag: 'moondream:latest',
        files: [],
        run: { kind: 'vision', requires: [] },
      },
      {
        repo: 'ollama/granite3.2-vision',
        label: 'Granite 3.2 Vision (2B)',
        task: 'vision',
        blurb:
          '2.3 GB. IBM’s document-oriented vision model; charts, tables and dense ' +
          'scans are its strongest case. Images only.',
        runtime: 'Ollama',
        ollamaTag: 'granite3.2-vision:2b',
        files: [],
        run: { kind: 'vision', requires: [] },
      },
      {
        repo: 'ollama/qwen2.5vl',
        label: 'Qwen2.5-VL (3B)',
        task: 'vision',
        blurb:
          '3.0 GB. The strongest all-rounder here, including video understanding. ' +
          'This app samples video into four still frames before asking, so the ' +
          'model sees those frames rather than the original stream.',
        runtime: 'Ollama',
        ollamaTag: 'qwen2.5vl:3b',
        files: [],
        run: { kind: 'vision', requires: [] },
      },
      {
        repo: 'ollama/gemma3',
        label: 'Gemma 3 (4B)',
        task: 'vision',
        blurb:
          '3.2 GB. A stronger general chat model that also sees images; useful if ' +
          'you want one model for both roles. Tight but workable under 8 GB when ' +
          'the chat model is unloaded first.',
        runtime: 'Ollama',
        ollamaTag: 'gemma3:4b',
        files: [],
        run: { kind: 'vision', requires: [] },
      },
    ],
  },
  {
    task: 'speech-to-text',
    title: 'Speech to text',
    note:
      'Ollama has no transcription models — pull returns 404. The CTranslate2 ' +
      'builds, the ggml-*.bin builds and the ONNX export are the three this app ' +
      'can run itself.',
    entries: [
      // One file per entry, and deliberately not a list of quantisations: the
      // runnable check treats every file in an entry as required, so offering
      // two quants here would make the model report "download the whole set"
      // until both were fetched. The repository carries q5_0 and the f16
      // originals too, and the URL is in the note below.
      {
        repo: 'ggerganov/whisper.cpp',
        label: 'Whisper large-v3-turbo (whisper.cpp)',
        task: 'speech-to-text',
        blurb:
          'Best accuracy per megabyte of the Whisper family. Runs on whisper.cpp, ' +
          'so it needs no PyTorch and no Python at all. The repository also has ' +
          'q5_0 (547 MB) and the f16 original if you want a different trade.',
        runtime: 'whisper.cpp',
        files: [{ name: 'ggml-large-v3-turbo-q8_0.bin', label: 'q8_0', bytes: 834 * MB }],
        run: { kind: 'stt', requires: ['whisper-cpp'] },
      },
      {
        repo: 'ggerganov/whisper.cpp',
        label: 'Whisper small (whisper.cpp)',
        task: 'speech-to-text',
        blurb: 'A quarter of the size of turbo; noticeably weaker on accents and noise.',
        runtime: 'whisper.cpp',
        files: [{ name: 'ggml-small-q8_0.bin', label: 'q8_0', bytes: 252 * MB }],
        run: { kind: 'stt', requires: ['whisper-cpp'] },
      },
      {
        repo: 'ggerganov/whisper.cpp',
        label: 'Whisper base (whisper.cpp)',
        task: 'speech-to-text',
        blurb: 'The sensible default. Steady on names and accents at 78 MB.',
        runtime: 'whisper.cpp',
        files: [{ name: 'ggml-base-q8_0.bin', label: 'q8_0', bytes: 78 * MB }],
        run: { kind: 'stt', requires: ['whisper-cpp'] },
      },
      {
        repo: 'ggerganov/whisper.cpp',
        label: 'Whisper tiny (whisper.cpp)',
        task: 'speech-to-text',
        blurb: 'The smallest. Fast enough for live dictation on an old CPU.',
        runtime: 'whisper.cpp',
        files: [{ name: 'ggml-tiny-q8_0.bin', label: 'q8_0', bytes: 42 * MB }],
        run: { kind: 'stt', requires: ['whisper-cpp'] },
      },
      {
        repo: 'Systran/faster-whisper-tiny',
        label: 'Whisper tiny (CTranslate2)',
        task: 'speech-to-text',
        blurb:
          'The smallest of the CTranslate2 builds. Fast enough to transcribe ' +
          'live dictation on an old CPU; expect it to mistype uncommon words.',
        runtime: 'faster-whisper (CTranslate2)',
        files: ct2Bundle(72 * MB),
        bundle: true,
        run: { kind: 'stt', requires: ['faster-whisper'] },
      },
      {
        repo: 'Systran/faster-whisper-base',
        label: 'Whisper base (CTranslate2)',
        task: 'speech-to-text',
        blurb:
          'The sensible default. About twice the size of tiny and noticeably ' +
          'steadier on names, accents and background noise.',
        runtime: 'faster-whisper (CTranslate2)',
        files: ct2Bundle(138 * MB),
        bundle: true,
        run: { kind: 'stt', requires: ['faster-whisper'] },
      },
      {
        repo: 'Systran/faster-whisper-small',
        label: 'Whisper small (CTranslate2)',
        task: 'speech-to-text',
        blurb: 'Clearly more accurate than base, and clearly slower on CPU. Worth it for long dictation.',
        runtime: 'faster-whisper (CTranslate2)',
        files: ct2Bundle(461 * MB),
        bundle: true,
        run: { kind: 'stt', requires: ['faster-whisper'] },
      },
      {
        repo: 'Systran/faster-whisper-large-v3',
        label: 'Whisper large-v3 (CTranslate2)',
        task: 'speech-to-text',
        blurb: 'The most accurate of the family. Wants a GPU or a fast CPU to stay usable.',
        runtime: 'faster-whisper (CTranslate2)',
        files: ct2Bundle(2.8 * 1024 * MB),
        bundle: true,
        run: { kind: 'stt', requires: ['faster-whisper'] },
      },
      {
        repo: 'onnx-community/whisper-tiny.en',
        label: 'Whisper tiny (ONNX, int8)',
        task: 'speech-to-text',
        blurb:
          'Whisper tiny as an ONNX Runtime export, quantized to about 43 MB. ' +
          'No CTranslate2 and no whisper.cpp binary: the shared Python ' +
          'environment runs the encoder and decoder graphs directly. The least ' +
          'accurate Whisper of the family, and the right size for short ' +
          'dictation on an old CPU.',
        runtime: 'onnxruntime',
        files: [
          {
            name: 'onnx/encoder_model_quantized.onnx',
            label: 'encoder (int8)',
            bytes: 10124993,
          },
          {
            name: 'onnx/decoder_model_merged_quantized.onnx',
            label: 'decoder (merged, int8)',
            bytes: 30718858,
          },
          { name: 'config.json', label: 'config.json', bytes: 2197 },
          { name: 'generation_config.json', label: 'generation_config.json', bytes: 1646 },
          { name: 'preprocessor_config.json', label: 'preprocessor_config.json', bytes: 339 },
          { name: 'tokenizer.json', label: 'tokenizer.json', bytes: 2405679 },
          { name: 'tokenizer_config.json', label: 'tokenizer_config.json', bytes: 282662 },
          { name: 'special_tokens_map.json', label: 'special_tokens_map.json', bytes: 2173 },
          { name: 'vocab.json', label: 'vocab.json', bytes: 999186 },
          { name: 'merges.txt', label: 'merges.txt', bytes: 456318 },
          { name: 'normalizer.json', label: 'normalizer.json', bytes: 52666 },
        ],
        bundle: true,
        bundleMB: 43,
        run: { kind: 'stt', requires: ['onnxruntime', 'transformers'] },
      },
      {
        repo: 'openai/whisper-large-v3-turbo',
        label: 'Whisper large-v3-turbo (safetensors)',
        task: 'speech-to-text',
        blurb: 'The original OpenAI release, full precision, for transformers.',
        runtime: 'transformers',
        files: [{ name: 'model.safetensors', label: 'model.safetensors', bytes: 1543 * MB }],
        run: { kind: 'stt', requires: ['torch', 'transformers'] },
      },
    ],
  },
  {
    task: 'text-to-speech',
    title: 'Text to speech',
    note:
      'No Ollama path exists. These run through onnxruntime, the kokoro ' +
      'package, or the shared transformers pipeline.',
    entries: [
      {
        repo: 'onnx-community/Kokoro-82M-v1.0-ONNX',
        label: 'Kokoro 82M (ONNX)',
        task: 'text-to-speech',
        blurb:
          'The same 82M voice model as the PyTorch checkpoint, exported to ONNX ' +
          'Runtime. On an NVIDIA machine the ONNX Runtime in the Kokoro ' +
          'environment is the GPU build, so the graph runs on the card; ' +
          'without one it is the CPU build. The q4 graph — 4-bit weights with ' +
          'floating-point activations — is the export that computes correctly ' +
          'on both. The fp16 exports are not bundled because the CUDA provider ' +
          'turns them into a full-length silent file. The af_heart style table ' +
          'and the phoneme vocabulary come with it; the graph alone cannot ' +
          'pronounce anything.',
        runtime: 'onnxruntime (Kokoro)',
        files: [
          {
            name: 'onnx/model_q4.onnx',
            label: 'q4 weights',
            bytes: 305215966,
            // Browse downloads often land on the smaller q4f16 export. The
            // runner accepts it on CPU (and refuses it on CUDA because that
            // build produces silence), so presence is honest even though the
            // catalog still prefers q4 for a fresh download.
            oneOf: ['onnx/model_q4f16.onnx'],
          },
          { name: 'voices/af_heart.bin', label: 'af_heart voice', bytes: 522240 },
          { name: 'tokenizer.json', label: 'phoneme vocabulary', bytes: 3497 },
          { name: 'config.json', label: 'config.json', bytes: 44 },
        ],
        bundle: true,
        bundleMB: 292,
        run: { kind: 'tts', requires: ['kokoro'] },
      },
      {
        repo: 'hexgrad/Kokoro-82M',
        label: 'Kokoro 82M (pth)',
        task: 'text-to-speech',
        blurb: 'Same model as a PyTorch checkpoint, for the kokoro package. Add a voice file from voices/.',
        runtime: 'kokoro (PyTorch)',
        files: [{ name: 'kokoro-v1_0.pth', label: 'kokoro-v1_0.pth', bytes: 312 * MB }],
        run: { kind: 'tts', requires: ['torch', 'kokoro'] },
      },
      {
        repo: 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
        label: 'Qwen3-TTS 0.6B',
        task: 'text-to-speech',
        blurb: 'Newer and more natural, but several times the size and needs a GPU to be pleasant.',
        runtime: 'transformers',
        files: [{ name: 'model.safetensors', label: 'model.safetensors', bytes: 1728 * MB }],
      },
    ],
  },
  {
    task: 'image-generation',
    title: 'Image generation',
    note:
      'Ollama cannot run diffusion models. The diffusers builds run here once ' +
      'PyTorch and Diffusers are installed, but SD 3.5 Medium’s full pipeline ' +
      'does not fit in 8 GB. Its GGUF build runs through stable-diffusion.cpp ' +
      'from the Runtimes page; that is the route this machine can actually use.',
    entries: [
      {
        repo: 'second-state/stable-diffusion-3.5-medium-GGUF',
        label: 'Stable Diffusion 3.5 Medium (GGUF)',
        task: 'image-generation',
        blurb:
          'The whole SD 3.5 Medium pipeline at 4-bit — UNet, T5-XXL, both CLIP ' +
          'encoders and the VAE — in 5.2 GB. stable-diffusion.cpp keeps the ' +
          'quantised MMDiT and VAE in RAM, streams the 2.75 GB text encoders ' +
          'from disk, and runs the UNet on the Vulkan card. That split is what ' +
          'fits an 8 GB laptop. The VAE is not in the quantised repo, so this ' +
          'card fetches an ungated copy of the official VAE too — no Hugging ' +
          'Face token is required. Expect several minutes per 512×512 image ' +
          'on this hardware.',
        runtime: 'stable-diffusion.cpp',
        files: [
          { name: 'sd3.5_medium-Q4_0.gguf', label: 'UNet Q4_0', bytes: 2079836768 },
          { name: 't5xxl-Q4_0.gguf', label: 'T5-XXL encoder Q4_0', bytes: 2752841312 },
          { name: 'clip_g-Q4_0.gguf', label: 'CLIP-G Q4_0', bytes: 391439360 },
          { name: 'clip_l-Q4_0.gguf', label: 'CLIP-L Q4_0', bytes: 69384832 },
          {
            name: 'vae/diffusion_pytorch_model.safetensors',
            label: 'VAE (ungated mirror of the official release)',
            bytes: 167666902,
            // The official stabilityai repository is gated, and requiring an
            // HF token to run a model that is otherwise downloadable would
            // break the whole point of the local path. This mirror publishes
            // the same 160 MB VAE file without a gate.
            repo: 'adamo1139/stable-diffusion-3.5-medium-ungated',
          },
        ],
        bundle: true,
        bundleMB: 5208,
        run: { kind: 'image', requires: ['stable-diffusion-cpp'], steps: 20 },
      },

  {
    repo: 'stabilityai/sd-turbo',
    label: 'Stable Diffusion Turbo',
    task: 'image-generation',
    blurb:
      'One or two steps to an image, and small enough to be the realistic local ' +
      'choice. The whole diffusers directory is needed, not just the weights.',
    runtime: 'diffusers (PyTorch)',
      files: diffusersBundle([
        { name: "model_index.json", label: "model_index.json", bytes: 616 },
        { name: "scheduler/scheduler_config.json", label: "scheduler/scheduler_config.json", bytes: 553 },
        { name: "text_encoder/config.json", label: "text_encoder/config.json", bytes: 618 },
        { name: "text_encoder/model.fp16.safetensors", label: "text_encoder/model.fp16.safetensors", bytes: 680820392 },
        { name: "tokenizer/merges.txt", label: "tokenizer/merges.txt", bytes: 524619 },
        { name: "tokenizer/special_tokens_map.json", label: "tokenizer/special_tokens_map.json", bytes: 574 },
        { name: "tokenizer/tokenizer_config.json", label: "tokenizer/tokenizer_config.json", bytes: 855 },
        { name: "tokenizer/vocab.json", label: "tokenizer/vocab.json", bytes: 1059962 },
        { name: "unet/config.json", label: "unet/config.json", bytes: 1867 },
        { name: "unet/diffusion_pytorch_model.fp16.safetensors", label: "unet/diffusion_pytorch_model.fp16.safetensors", bytes: 1731904736 },
        { name: "vae/config.json", label: "vae/config.json", bytes: 655 },
        { name: "vae/diffusion_pytorch_model.fp16.safetensors", label: "vae/diffusion_pytorch_model.fp16.safetensors", bytes: 167335342 },
      ]),
      bundle: true,
      bundleMB: 2462,
    run: { kind: 'image', requires: ['torch', 'diffusers'] },
  },
  {
    repo: 'stabilityai/sdxl-turbo',
    label: 'SDXL Turbo',
    task: 'image-generation',
    blurb:
      'Sharper and far better at text in images, at roughly twice the download ' +
      'and noticeably more CPU per step.',
    runtime: 'diffusers (PyTorch)',
      files: diffusersBundle([
        { name: "model_index.json", label: "model_index.json", bytes: 685 },
        { name: "scheduler/scheduler_config.json", label: "scheduler/scheduler_config.json", bytes: 459 },
        { name: "text_encoder_2/config.json", label: "text_encoder_2/config.json", bytes: 575 },
        { name: "text_encoder_2/model.fp16.safetensors", label: "text_encoder_2/model.fp16.safetensors", bytes: 1389382176 },
        { name: "text_encoder/config.json", label: "text_encoder/config.json", bytes: 565 },
        { name: "text_encoder/model.fp16.safetensors", label: "text_encoder/model.fp16.safetensors", bytes: 246144152 },
        { name: "tokenizer_2/merges.txt", label: "tokenizer_2/merges.txt", bytes: 524619 },
        { name: "tokenizer_2/special_tokens_map.json", label: "tokenizer_2/special_tokens_map.json", bytes: 460 },
        { name: "tokenizer_2/tokenizer_config.json", label: "tokenizer_2/tokenizer_config.json", bytes: 855 },
        { name: "tokenizer_2/vocab.json", label: "tokenizer_2/vocab.json", bytes: 1059962 },
        { name: "tokenizer/merges.txt", label: "tokenizer/merges.txt", bytes: 524619 },
        { name: "tokenizer/special_tokens_map.json", label: "tokenizer/special_tokens_map.json", bytes: 586 },
        { name: "tokenizer/tokenizer_config.json", label: "tokenizer/tokenizer_config.json", bytes: 704 },
        { name: "tokenizer/vocab.json", label: "tokenizer/vocab.json", bytes: 1059962 },
        { name: "unet/config.json", label: "unet/config.json", bytes: 1776 },
        { name: "unet/diffusion_pytorch_model.fp16.safetensors", label: "unet/diffusion_pytorch_model.fp16.safetensors", bytes: 5135149760 },
        { name: "vae_decoder/config.json", label: "vae_decoder/config.json", bytes: 607 },
        { name: "vae_encoder/config.json", label: "vae_encoder/config.json", bytes: 607 },
        { name: "vae/config.json", label: "vae/config.json", bytes: 607 },
        { name: "vae/diffusion_pytorch_model.fp16.safetensors", label: "vae/diffusion_pytorch_model.fp16.safetensors", bytes: 167335342 },
      ]),
      bundle: true,
      bundleMB: 6620,
    run: { kind: 'image', requires: ['torch', 'diffusers'] },
  },
    ],
  },
];

/** Flat index for callers that just want every repo mentioned, for validation. */
export function catalogRepoIds(): string[] {
  const repos = new Set<string>();
  for (const section of CATALOG) {
    for (const entry of section.entries) {
      repos.add(entry.repo);
      // Auxiliary repositories are part of a catalog model too. Without them
      // the hub would label the VAE download as a standalone model no job
      // uses, while the image card fails for lack of it.
      for (const file of entry.files ?? []) {
        if (file.repo) repos.add(file.repo);
      }
    }
  }
  return [...repos];
}