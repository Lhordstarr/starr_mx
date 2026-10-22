/**
 * Vision assignment: media replacement, caching, and discovery.
 *
 * The chat route is the only production caller, but its interesting decisions
 * are pure: which assignment id means which provider, when an image becomes
 * text, and what happens to video with no model. These checks exercise those
 * with a fake provider, so they do not need Ollama, ffmpeg or a downloaded
 * model. The runnable half is the same payload Settings reads.
 *
 * Run with: npm run test:vision
 */
import assert from 'node:assert/strict';

import { runnableModels } from '../src/runnable.js';
import { isSdCppRepo, sdCppArgv, sdCppAsset } from '../src/sdcpp.js';
import {
  clearVisionCache,
  parseVisionAssignment,
  preprocessVision,
  type VisionPreprocessOptions,
} from '../src/vision.js';
import type { HFDownload } from '../src/huggingface.js';
import type { Provider } from '../src/providers/types.js';
import type { ChatMessage, ImagePart, VideoPart } from '../src/types.js';

let passed = 0;
const failures: string[] = [];

async function check(name: string, fn: () => void | Promise<void>): Promise<void> {
  try {
    await fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures.push(name);
    console.log(`  FAIL  ${name}\n        ${(err as Error).message}`);
  }
}

const image = (marker: string): ImagePart => ({
  type: 'image',
  mimeType: 'image/png',
  data: Buffer.from(marker).toString('base64'),
});

const video = (): VideoPart => ({
  type: 'video',
  mimeType: 'video/mp4',
  data: Buffer.from('video-bytes').toString('base64'),
});

const frame = (marker: string): ImagePart => image(marker);

function fakeProvider() {
  const calls: Array<{ model: string; prompt: string }> = [];
  const provider = {
    describeImages: async ({ model, prompt }: { model: string; prompt: string }) => {
      calls.push({ model, prompt });
      return `description ${calls.length}`;
    },
  } as unknown as Provider;
  return { provider, calls };
}

function messages(...parts: Array<ImagePart | VideoPart | { type: 'text'; text: string }>): ChatMessage[] {
  return [{ role: 'user', content: parts }];
}

function options(
  assignment: string | null,
  provider: Provider,
  extra: Partial<VisionPreprocessOptions> = {},
): VisionPreprocessOptions {
  return {
    assignment,
    resolveProvider: (id) => (id === 'ollama' ? provider : undefined),
    signal: new AbortController().signal,
    ...extra,
  };
}

console.log('\n=== assignment ids ===');

await check('ollama tags keep their colon', () => {
  assert.deepEqual(parseVisionAssignment('ollama:moondream:latest'), {
    providerId: 'ollama',
    model: 'moondream:latest',
  });
});

await check('local ids are not mistaken for an Ollama model', () => {
  assert.equal(parseVisionAssignment('local:some/repo'), null);
});

console.log('\n=== media replacement ===');

clearVisionCache();
await check('no assignment leaves still images for the native provider', async () => {
  const { provider, calls } = fakeProvider();
  const input = messages({ type: 'text', text: 'look' }, image('a'));
  const result = await preprocessVision(input, options(null, provider));
  assert.equal(result.described, 0);
  assert.equal(calls.length, 0);
  assert.deepEqual(result.messages[0].content, input[0].content);
});

clearVisionCache();
await check('an assigned model replaces the image with its description', async () => {
  const { provider, calls } = fakeProvider();
  const result = await preprocessVision(
    messages({ type: 'text', text: 'what is broken?' }, image('a')),
    options('ollama:moondream:latest', provider),
  );
  assert.equal(calls.length, 1);
  assert.equal(calls[0].model, 'moondream:latest');
  assert.match(calls[0].prompt, /what is broken/);
  assert.equal(result.described, 1);
  const content = result.messages[0].content;
  assert.equal(typeof content, 'string');
  assert.match(String(content), /what is broken\?/);
  assert.match(String(content), /described by ollama:moondream:latest/);
  assert.match(String(content), /description 1/);
  assert.doesNotMatch(String(content), /\[Image 1\].*data:/s);
});

clearVisionCache();
await check('the same image is described once across turns', async () => {
  const { provider, calls } = fakeProvider();
  const picture = image('same');
  const first = messages({ type: 'text', text: 'first' }, picture);
  const second = messages({ type: 'text', text: 'follow-up' }, picture);
  await preprocessVision(first, options('ollama:moondream:latest', provider));
  const result = await preprocessVision(second, options('ollama:moondream:latest', provider));
  assert.equal(calls.length, 1);
  assert.equal(result.cacheHits, 1);
  assert.match(String(result.messages[0].content), /description 1/);
});

clearVisionCache();
await check('video is sampled into frames and described as one block', async () => {
  const { provider, calls } = fakeProvider();
  const result = await preprocessVision(
    messages({ type: 'text', text: 'summarise this clip' }, video()),
    options('ollama:moondream:latest', provider, {
      extractFrames: async () => [
        { image: frame('f1'), at: 0.5 },
        { image: frame('f2'), at: 2.5 },
      ],
    }),
  );
  assert.equal(calls.length, 2);
  assert.match(String(result.messages[0].content), /Video 1 — sampled into 2 frames/);
  assert.match(String(result.messages[0].content), /Frame 1, frame at 0.5s/);
  assert.match(String(result.messages[0].content), /description 2/);
});

clearVisionCache();
await check('video with no vision assignment is refused, not dropped', async () => {
  const { provider } = fakeProvider();
  await assert.rejects(
    preprocessVision(messages(video()), options(null, provider)),
    /Video attachments need an assigned vision model/,
  );
});

await check('a local: vision assignment says it has no runner', async () => {
  const { provider } = fakeProvider();
  await assert.rejects(
    preprocessVision(messages(image('a')), options('local:some/repo', provider)),
    /cannot be used/,
  );
});

console.log('\n=== runnable vision ===');

const download = (repo: string, file: string): HFDownload => ({
  repo,
  file,
  size: 1,
  modifiedAt: '2026-01-01T00:00:00Z',
  location: 'models',
  root: `/tmp/${repo.replace(/\//g, '--')}`,
});

await check('installed Ollama vision models are assignable by tag', async () => {
  const models = await runnableModels('vision', {
    downloads: [],
    installed: {},
    visionModels: ['moondream:latest', 'some-extra:1b'],
  });
  const moon = models.find((m) => m.id === 'ollama:moondream:latest');
  assert.ok(moon, 'catalog moondream entry should be present');
  assert.equal(moon.ready, true);
  const extra = models.find((m) => m.id === 'ollama:some-extra:1b');
  assert.ok(extra, 'an installed model outside the catalog should appear');
  assert.equal(extra.ready, true);
  assert.equal(models.some((m) => m.id?.startsWith('local:')), false);
});

await check('catalog vision models are listed but not runnable before a pull', async () => {
  const models = await runnableModels('vision', {
    downloads: [],
    installed: {},
    visionModels: [],
  });
  assert.ok(models.length >= 2);
  assert.ok(models.every((m) => !m.ready));
  assert.ok(models.every((m) => /Not pulled yet/.test(m.reason)));
});

console.log('\n=== Kokoro graph variants ===');

const kokoroFiles = ['config.json', 'tokenizer.json', 'voices/af_heart.bin'];
await check('a Browse-downloaded q4f16 graph satisfies the catalog q4 slot', async () => {
  const models = await runnableModels('tts', {
    downloads: [
      ...kokoroFiles.map((file) => download('onnx-community/Kokoro-82M-v1.0-ONNX', file)),
      download('onnx-community/Kokoro-82M-v1.0-ONNX', 'onnx/model_q4f16.onnx'),
    ],
    installed: { kokoro: 'installed' },
  });
  const kokoro = models.find((m) => m.repo === 'onnx-community/Kokoro-82M-v1.0-ONNX');
  assert.ok(kokoro, 'Kokoro ONNX should be discovered');
  assert.equal(kokoro.ready, true);
  assert.equal(kokoro.file, 'onnx/model_q4f16.onnx');
});

console.log('\n=== Stable Diffusion 3.5 through stable-diffusion.cpp ===');

const sdRepo = 'second-state/stable-diffusion-3.5-medium-GGUF';
const sdVaeRepo = 'adamo1139/stable-diffusion-3.5-medium-ungated';
const sdBundle = [
  download(sdRepo, 'sd3.5_medium-Q4_0.gguf'),
  download(sdRepo, 't5xxl-Q4_0.gguf'),
  download(sdRepo, 'clip_g-Q4_0.gguf'),
  download(sdRepo, 'clip_l-Q4_0.gguf'),
  download(sdVaeRepo, 'vae/diffusion_pytorch_model.safetensors'),
];

await check('the SD 3.5 bundle is runnable once its runtime is installed', async () => {
  const models = await runnableModels('image', {
    downloads: sdBundle,
    installed: { 'stable-diffusion-cpp': 'installed' },
  });
  const sd = models.find((m) => m.repo === sdRepo);
  assert.ok(sd, 'the catalog SD 3.5 entry should be present');
  assert.equal(sd.ready, true);
  assert.equal(sd.missing.length, 0);
  assert.ok(sd.runtimes['stable-diffusion-cpp']);
  assert.equal(isSdCppRepo(sd.repo), true);
});

await check('a missing VAE is reported instead of claiming the card is complete', async () => {
  const models = await runnableModels('image', {
    downloads: sdBundle.filter((d) => d.repo !== sdVaeRepo),
    installed: { 'stable-diffusion-cpp': 'installed' },
  });
  const sd = models.find((m) => m.repo === sdRepo);
  assert.ok(sd);
  assert.equal(sd.ready, false);
  assert.match(sd.reason, /vae\/diffusion_pytorch_model\.safetensors/);
});

await check('the installer picks the Vulkan archive when the machine has Vulkan', () => {
  const vulkan = sdCppAsset(true);
  assert.ok(vulkan);
  assert.equal(vulkan.kind, 'vulkan');
  assert.match(vulkan.name, /vulkan/);
  const cpu = sdCppAsset(false);
  assert.ok(cpu);
  assert.equal(cpu.kind, 'cpu');
  assert.doesNotMatch(cpu.name, /vulkan/);
});

await check('the SD 3.5 cmd keeps the MMDiT in RAM and streams the encoders', () => {
  const files = {
    model: '/models/unet.gguf',
    t5: '/models/t5.gguf',
    clipL: '/models/clip_l.gguf',
    clipG: '/models/clip_g.gguf',
    vae: '/models/vae.safetensors',
  };
  const argv = sdCppArgv(
    { prompt: 'an apple', negative: 'blurry', steps: 20, width: 512, height: 512 },
    files,
    '/tmp/out.png',
    'Vulkan1',
  );
  const value = (name: string) => argv[argv.indexOf(name) + 1];
  assert.equal(value('--diffusion-model'), files.model);
  assert.equal(value('--t5xxl'), files.t5);
  assert.equal(value('--clip_l'), files.clipL);
  assert.equal(value('--clip_g'), files.clipG);
  assert.equal(value('--vae'), files.vae);
  assert.equal(value('--steps'), '20');
  assert.equal(value('--width'), '512');
  assert.equal(value('--height'), '512');
  // Disk-backed MMDiT weights came out as noise on this build. The encoders
  // may stream, but the diffusion weights must be a real CPU parameter buffer.
  assert.equal(value('--params-backend'), 'diffusion=cpu,te=disk,vae=cpu');
  assert.equal(argv.includes('diffusion=disk'), false);
  assert.equal(argv.includes('--offload-to-cpu'), false);
  assert.equal(value('--backend'), 'diffusion=vulkan1,te=cpu,vae=cpu');

  const cpuArgv = sdCppArgv(
    { prompt: 'an apple', negative: '', steps: 1, width: 64, height: 64 },
    files,
    '/tmp/out.png',
    null,
  );
  assert.equal(cpuArgv[cpuArgv.indexOf('--backend') + 1], 'cpu');
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
