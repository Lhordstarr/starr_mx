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

export type CatalogTask = 'text' | 'speech-to-text' | 'text-to-speech' | 'image-generation';

export interface CatalogFile {
  /** Repo-relative path, exactly as the Hub spells it. */
  name: string;
  label: string;
  bytes: number;
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
  run?: { kind: 'stt' | 'tts' | 'image'; requires: string[] };
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
 * The four files a CTranslate2 Whisper repository ships. All are required:
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
    note: 'Ollama serves chat models, so these pull straight from your daemon — or download the GGUF and run it on this machine with llama.cpp.',
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
    ],
  },
  {
    task: 'speech-to-text',
    title: 'Speech to text',
    note: 'Ollama has no transcription models — pull returns 404. The CTranslate2 builds are the ones this app can run itself; the GGUF and safetensors entries need an external tool.',
    entries: [
      {
        repo: 'handy-computer/whisper-large-v3-turbo-gguf',
        label: 'Whisper large-v3-turbo (GGUF)',
        task: 'speech-to-text',
        blurb: 'Best accuracy per megabyte of the Whisper family. The practical local choice.',
        runtime: 'whisper.cpp',
        files: [{ name: 'whisper-large-v3-turbo-Q4_K_M.gguf', label: 'Q4_K_M', bytes: 511 * MB }],
      },
      {
        repo: 'handy-computer/whisper-small-gguf',
        label: 'Whisper small (GGUF)',
        task: 'speech-to-text',
        blurb: 'Roughly a tenth of the size; noticeably weaker on accents and noise.',
        runtime: 'whisper.cpp',
        files: [{ name: 'whisper-small-Q4_K_M.gguf', label: 'Q4_K_M', bytes: 164 * MB }],
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
    note: 'No Ollama path exists. These run through onnxruntime, the kokoro package, or piper.',
    entries: [
      {
        repo: 'onnx-community/Kokoro-82M-v1.0-ONNX',
        label: 'Kokoro 82M (ONNX)',
        task: 'text-to-speech',
        blurb: 'Eighty-two million parameters, sounds far larger, and runs on CPU. This is the ONNX export — the in-app Speak button runs the PyTorch checkpoint (Kokoro 82M (pth)), so this one is for running the export yourself with onnxruntime or transformers.js.',
        runtime: 'onnxruntime or transformers.js',
        files: [
          { name: 'onnx/model.onnx', label: 'fp32', bytes: 310 * MB },
          { name: 'onnx/model_q4f16.onnx', label: 'q4f16', bytes: 147 * MB },
        ],
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
    note: 'Ollama cannot run diffusion models. The diffusers builds run here once PyTorch and Diffusers are installed; the GGUF build needs ComfyUI, which this app does not install.',
    entries: [
      {
        repo: 'city96/stable-diffusion-3.5-medium-gguf',
        label: 'Stable Diffusion 3.5 Medium (GGUF)',
        task: 'image-generation',
        blurb:
          'A single quantised file, so ComfyUI can load it without the whole ' +
          'pipeline tree. Nothing in this app can run it — there is no Run ' +
          'button, because offering one that could not work would be worse ' +
          'than saying so here.',
        runtime: 'ComfyUI (external — not installed by this app)',
        files: [{ name: 'sd3.5_medium-Q4_K_M.gguf', label: 'Q4_K_M', bytes: 1704 * MB }],
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
  return CATALOG.flatMap((s) => s.entries.map((e) => e.repo));
}