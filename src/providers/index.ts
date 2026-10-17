import { anthropic } from './anthropic.js';
import { customApiKey, listCustomProviders } from './custom.js';
import { gemini } from './gemini.js';
import { ollama } from './ollama.js';
import { openAICompatible } from './openai-compat.js';
import type { BuiltinProviderId, Provider, ProviderId } from './types.js';

export const openai = openAICompatible({
  id: 'openai',
  label: 'OpenAI',
  // Overridable so a proxy or self-hosted gateway can be pointed at instead.
  baseUrl: process.env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1',
  envKey: 'OPENAI_API_KEY',
  stt: true,
});

export const openrouter = openAICompatible({
  id: 'openrouter',
  label: 'OpenRouter',
  baseUrl: 'https://openrouter.ai/api/v1',
  envKey: 'OPENROUTER_API_KEY',
  hint: 'One key, many models across providers',
  extraHeaders: {
    'HTTP-Referer': 'http://localhost:3000',
    'X-Title': 'STARR Matrix',
  },
});

export const groq = openAICompatible({
  id: 'groq',
  label: 'Groq',
  baseUrl: process.env.GROQ_BASE_URL ?? 'https://api.groq.com/openai/v1',
  envKey: 'GROQ_API_KEY',
  hint: 'Very fast inference, free tier available',
  stt: true,
});

/** The providers compiled into the app. */
export const builtinProviders: Record<BuiltinProviderId, Provider> = {
  ollama,
  openai,
  openrouter,
  anthropic,
  gemini,
  groq,
};

/** Display order for the built-ins, local first. */
export const builtinOrder: BuiltinProviderId[] = [
  'ollama',
  'openai',
  'openrouter',
  'anthropic',
  'gemini',
  'groq',
];

/**
 * Every provider, built-in plus user-added, keyed by id.
 *
 * Custom providers are OpenAI-compatible by construction. Their key is read
 * from the custom store on every call rather than captured, so editing a key
 * on the settings page takes effect without a restart.
 */
export function allProviders(): Record<string, Provider> {
  const merged: Record<string, Provider> = { ...builtinProviders };
  for (const custom of listCustomProviders()) {
    if (merged[custom.id]) continue; // a built-in id is never shadowed
    merged[custom.id] = openAICompatible({
      id: custom.id,
      label: custom.label,
      baseUrl: custom.baseUrl,
      getApiKey: () => customApiKey(custom.id),
      keyOptional: true,
      custom: true,
      stt: custom.stt,
    });
  }
  return merged;
}

/** Built-ins first in their fixed order, then custom providers by name. */
export function providerOrder(): ProviderId[] {
  const custom = listCustomProviders()
    .map((p) => p.id)
    .filter((id) => !(id in builtinProviders))
    .sort((a, b) => a.localeCompare(b));
  return [...builtinOrder, ...custom];
}

export { ollama, anthropic, gemini };
export type { Provider, ProviderId, BuiltinProviderId };