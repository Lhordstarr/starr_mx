/**
 * llama.cpp as a chat provider.
 *
 * The wire format is OpenAI-compatible, so the streaming, tool-calling and
 * vision plumbing is `openAICompatible`'s and nothing here re-implements it.
 * What this adds is the part only this app can know: the model list is the GGUF
 * files on this machine rather than whatever a running server happens to have
 * loaded, and a request brings the server up with the chosen model first
 * (`prepare`) — so picking a downloaded Qwen3 in the composer just works,
 * whether or not llama-server was already running.
 */
import { cachedGgufModels, llamaHint, llamaServerState, ggufChatModels, prepareLlama } from '../llamacpp.js';
import { openAICompatible } from './openai-compat.js';
import type { Provider } from './types.js';

const port = () => Number(process.env.LLAMACPP_PORT ?? 9932);

/**
 * The OpenAI-compatible half. Built once, lazily: the port is read at
 * construction, and reading the environment during module evaluation would
 * make it impossible for a test (or a wrapper script) to set it beforehand.
 */
let inner: Provider | null = null;
function compat(): Provider {
  if (!inner) {
    inner = openAICompatible({
      id: 'llamacpp',
      label: 'llama.cpp',
      baseUrl: `http://127.0.0.1:${port()}/v1`,
      // Loopback, no key: the server this app starts authenticates nothing.
      getApiKey: () => '',
      keyOptional: true,
      local: true,
      stt: false,
      // Small GGUFs (especially base models) tend to loop on their own output.
      // A modest penalty keeps the reply readable without changing the model.
      extraBody: { repeat_penalty: 1.1 },
    });
  }
  return inner;
}

export const llamacpp: Provider = {
  info: {
    id: 'llamacpp',
    label: 'llama.cpp',
    /**
     * Installed runtime and at least one chat model. Both halves matter: the
     * binary without a model has nothing to run, and a model without the
     * binary has nothing to run it — and the hint says which half is missing.
     *
     * Read from the cached listing because this is a synchronous getter while
     * the provider list is assembled; `listModels()` and the server's warm-up
     * fill that cache. Before the first listing this reads as "not ready yet",
     * which is the safe direction — the hint says what to do either way.
     */
    get configured() {
      return llamaServerState().installed && cachedGgufModels().some((m) => m.chat);
    },
    local: true,
    get hint() {
      return llamaHint();
    },
    supportsTools: true,
  },

  /** The downloaded GGUF chat models — the whole point of the provider. */
  async listModels() {
    return (await ggufChatModels()).map((m) => m.id);
  },

  /**
   * Starts the server with this model loaded before the request is sent. The
   * chat turns the progress lines into transcript notices, so a multi-gigabyte
   * load reads as work in progress rather than a hang.
   */
  prepare: prepareLlama,

  stream: (opts) => compat().stream(opts),

  chatWithTools: (opts) => compat().chatWithTools!(opts),

  /**
   * llama.cpp publishes no per-model capability list, and a GGUF can only
   * accept images when it was loaded beside an `mmproj` projector — a fact the
   * model file itself does not carry. Empty means "unknown" to the UI, which is
   * the honest answer; a model that cannot see will say so when it is sent one.
   */
  listVisionModels: async () => [],
};
