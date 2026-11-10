import { transformersHint, transformersServerState, transformersChatModelsOnly, prepareTransformers, cachedTransformersModels } from '../transformers-chat.js';
import { openAICompatible } from './openai-compat.js';
const port = () => Number(process.env.TRANSFORMERS_PORT ?? 9933);
let inner = null;
function compat() {
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
export const transformersChat = {
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
    stream: (opts) => compat().stream(opts),
    chatWithTools: (opts) => compat().chatWithTools(opts),
    listVisionModels: async () => [],
};
function cachedModels() {
    return cachedTransformersModels();
}
