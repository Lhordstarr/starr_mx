import {
  cachedTransformersModels,
  markTransformersBusy,
  markTransformersIdle,
  prepareTransformers,
  transformersChatModelsOnly,
  transformersHint,
  transformersServerState,
} from '../transformers-chat.js';
import { openAICompatible } from './openai-compat.js';
import type { Provider } from './types.js';

const port = () => Number(process.env.TRANSFORMERS_PORT ?? 9933);

let inner: Provider | null = null;
function compat(): Provider {
  if (!inner) {
    inner = openAICompatible({
      id: 'transformers-chat',
      label: 'Transformers (local)',
      baseUrl: `http://127.0.0.1:${port()}/v1`,
      getApiKey: () => '',
      keyOptional: true,
      local: true,
      stt: false,
      extraBody: { repeat_penalty: 1.1 },
    });
  }
  return inner;
}

/** Mirrors the llama.cpp wrapper: warm means loaded and not generating. */
function tracked<T>(inner: AsyncGenerator<T>): AsyncGenerator<T> {
  return (async function* () {
    markTransformersBusy();
    try {
      yield* inner;
    } finally {
      markTransformersIdle();
    }
  })();
}

export const transformersChat: Provider = {
  info: {
    id: 'transformers-chat',
    label: 'Transformers (local)',
    get configured() {
      return transformersServerState().installed && cachedModels().some((m) => m.chat);
    },
    local: true,
    get hint() { return transformersHint(); },
    supportsTools: true,
  },
  async listModels() {
    return (await transformersChatModelsOnly()).map((m) => m.id);
  },
  prepare: (model, onProgress) => prepareTransformers(model),
  stream: (opts) => tracked(compat().stream(opts)),
  chatWithTools: (opts) => tracked(compat().chatWithTools!(opts)),
  listVisionModels: async () => [],
};

function cachedModels() {
  return cachedTransformersModels();
}
