/**
 * Achromatic line icons for providers and models.
 *
 * The app's house style is inline SVG stroke icons, not glyphs, and no brand
 * marks. Provider identity is carried by the label; these are generic role
 * marks (spark, route, chip, cloud) rather than logos. Model ids are mapped by
 * name shape only, so a provider adding a model does not leave a hardcoded
 * list behind.
 *
 * Shared because the chat page, Studio and settings all render the same
 * entities and should not drift apart.
 */

const ICONS = {
  layers:
    '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
  cpu:
    '<rect x="6" y="6" width="12" height="12" rx="2"/>' +
    '<path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  route:
    '<circle cx="5" cy="12" r="2"/><circle cx="19" cy="5" r="2"/><circle cx="19" cy="19" r="2"/>' +
    '<path d="M7 12h4a2 2 0 0 0 2-2V7a2 2 0 0 1 2-2h2"/>' +
    '<path d="M11 12a2 2 0 0 1 2 2v3a2 2 0 0 0 2 2h2"/>',
  spark:
    '<path d="M12 2v4M12 18v4M2 12h4M18 12h4M5 5l2.8 2.8M16.2 16.2 19 19M19 5l-2.8 2.8M7.8 16.2 5 19"/>' +
    '<circle cx="12" cy="12" r="3"/>',
  compass:
    '<circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5 5-2Z"/>',
  twins: '<circle cx="9" cy="12" r="5"/><circle cx="15" cy="12" r="5"/>',
  bolt: '<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z"/>',
  plug:
    '<path d="M9 2v6M15 2v6M6 8h12v3a6 6 0 0 1-12 0V8ZM12 17v5"/>',
  server:
    '<rect x="3" y="4" width="18" height="6" rx="2"/><rect x="3" y="14" width="18" height="6" rx="2"/>' +
    '<path d="M7 7h.01M7 17h.01"/>',
  monitor:
    '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
  cloud:
    '<path d="M17.5 19a4.5 4.5 0 0 0 .5-9 6 6 0 0 0-11.6-1.6A4.5 4.5 0 0 0 7 19h10.5Z"/>',
  box:
    '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9"/>',
  users:
    '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/>' +
    '<path d="M16 4.5a3.5 3.5 0 0 1 0 7M17.5 14a6.5 6.5 0 0 1 4 6"/>',
  sparkles:
    '<path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z"/>' +
    '<path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z"/>',
  chat:
    '<path d="M21 15a2 2 0 0 1-2 2H8l-4 4V5a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2Z"/>',
  code: '<path d="m8 8-4 4 4 4M16 8l4 4-4 4M13 5l-2 14"/>',
  mic:
    '<rect x="9" y="2" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v4"/>' +
    '<path d="M8 22h8"/>',
  speaker:
    '<path d="M4 9v6h4l5 4V5L8 9H4Z"/><path d="M17 9a4 4 0 0 1 0 6"/>' +
    '<path d="M19.5 6.5a8 8 0 0 1 0 11"/>',
  image:
    '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/>' +
    '<path d="m21 15-5-5L5 21"/>',
  grid:
    '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>' +
    '<rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  eye:
    '<path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
  bulb:
    '<path d="M9 18h6M10 22h4M8.5 14.5a6 6 0 1 1 7 0c-.9.7-1.5 1.6-1.5 2.5h-4c0-.9-.6-1.8-1.5-2.5Z"/>',
  audio: '<path d="M3 10v4M7 7v10M11 4v16M15 7v10M19 10v4"/>',
};

/** Exact provider ids and env keys, normalized to lowercase slugs. */
const PROVIDER_KINDS = {
  ollama: 'layers',
  llamacpp: 'cpu',
  'llama-cpp': 'cpu',
  'llamacpp-server': 'cpu',
  'transformers-chat': 'layers',
  openai: 'spark',
  openrouter: 'route',
  anthropic: 'compass',
  gemini: 'twins',
  groq: 'bolt',
  lmstudio: 'monitor',
  vllm: 'server',
  localai: 'server',
  'localai-alt': 'server',
  koboldcpp: 'server',
  jan: 'monitor',
  anythingllm: 'server',
  pollinations: 'sparkles',
  'stable-horde': 'users',
  cloudflare: 'cloud',
  huggingface: 'box',
  'hf-api-token': 'box',
  'cloudflare-api-token': 'cloud',
  'cloudflare-account-id': 'cloud',
  'stable-horde-api-key': 'users',
  local: 'cpu',
  __local__: 'cpu',
};

/** What a provider id looks like, not what the provider sells. */
export function providerIconKind(id) {
  const raw = String(id ?? '').trim().toLowerCase();
  const slug = raw.replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  if (PROVIDER_KINDS[slug]) return PROVIDER_KINDS[slug];

  // Discovered servers and custom ids are user-named, so fall back to the same
  // name-shape rules instead of a single anonymous mark for everything.
  if (slug.includes('llama')) return 'cpu';
  if (slug.includes('transform') || slug.includes('ollama')) return 'layers';
  if (slug.includes('openrouter')) return 'route';
  if (slug.includes('openai')) return 'spark';
  if (slug.includes('anthropic') || slug.includes('claude')) return 'compass';
  if (slug.includes('gemini')) return 'twins';
  if (slug.includes('groq')) return 'bolt';
  if (slug.includes('lmstudio') || slug.includes('jan')) return 'monitor';
  if (slug.includes('vllm') || slug.includes('localai') || slug.includes('kobold')) return 'server';
  if (slug.includes('cloudflare')) return 'cloud';
  if (slug.includes('horde')) return 'users';
  if (slug.includes('hugging') || slug.startsWith('hf-')) return 'box';
  if (slug.includes('pollinations')) return 'sparkles';
  if (slug.includes('local') || slug.includes('pc')) return 'cpu';
  return 'plug';
}

/** Patterns are checked in this order; the first match wins. */
const MODEL_RULES = [
  ['code', /(coder|codex|codestral|starcoder|sqlcoder|sql-coder|deepseek[-_.]?coder|qwen[^/]*coder|wizardcoder)/],
  ['stt', /(whisper|wav2vec|parakeet|speech[-_.]?to[-_.]?text|transcrib|distil[-_.]?whisper|ct2)/],
  ['tts', /(kokoro|bark|vits|speecht5|xtts|piper|tacotron|sovits|rvc|text[-_.]?to[-_.]?speech|(^|[-_/])tts([-_/]|$))/],
  ['image', /(diffus|stable[-_.]?diffusion|sdxl|sd[-_.]?turbo|flux|dall[-_.]?e|midjourney|playground[-_.]?v|kandinsky|imagen|image[-_.]?gen)/],
  ['embedding', /(embed|bge[-_/]|gte[-_/]|e5[-_/]|sentence[-_.]?transformers|nomic|jina|rerank|cross[-_.]?encoder)/],
  ['vision', /(vision|llava|pixtral|moondream|internvl|minicpm[-_.]?v|qwen[^/]*[-_.]?vl|gemma[^/]*[-_.]?vl|video|multimodal)/],
  ['reasoning', /(reason|deepseek[-_.]?r1|(^|[-_/])r1([-_/]|$)|qwq|thinking|magistral)/],
  ['audio', /(audio|music|voice|sound)/],
];

/** What a model id looks like, not a catalogue of every model that exists. */
export function modelIconKind(id) {
  const value = String(id ?? '').toLowerCase();
  for (const [kind, pattern] of MODEL_RULES) {
    if (pattern.test(value)) return kind;
  }
  return 'chat';
}

function iconSvg(kind, className) {
  const inner = ICONS[kind] ?? ICONS.plug;
  const cls = className ? `entity-icon ${className}` : 'entity-icon';
  return (
    `<svg class="${cls}" data-icon="${kind}" viewBox="0 0 24 24" width="16" height="16" ` +
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
    `stroke-linejoin="round" aria-hidden="true" focusable="false">${inner}</svg>`
  );
}

/** Markup for one provider row, chip or heading. */
export function providerIcon(id, className = '') {
  return iconSvg(providerIconKind(id), className);
}

/** Markup for one model row or model pill. */
export function modelIcon(id, className = '') {
  return iconSvg(modelIconKind(id), className);
}
