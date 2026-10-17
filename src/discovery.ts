/**
 * Finds locally-hosted model servers that are actually running.
 *
 * The provider list used to be whatever was compiled in plus whatever the user
 * had manually added as a custom endpoint. That made "Local" mean exactly one
 * thing — Ollama — even though llama.cpp, LM Studio, vLLM, LocalAI, KoboldCpp
 * and text-generation-webui all expose an OpenAI-compatible HTTP API on this
 * same machine, usually already running.
 *
 * Discovery is deliberately narrow: loopback only, a fixed table of the
 * well-known ports these tools use, one cheap GET on /v1/models each, in
 * parallel with a short timeout. Nothing is spawned, nothing is installed, and
 * an absent server costs a refused connection rather than a delay.
 */
import { ollama } from './providers/ollama.js';
import { openAICompatible } from './providers/openai-compat.js';
import type { Provider } from './providers/types.js';

/**
 * Ports and paths used by self-hosted OpenAI-compatible servers.
 *
 * `modelsPath` is where the model list lives; a few of these tools answer it
 * without the /v1 prefix, which is why it is stated rather than assumed.
 */
interface LocalCandidate {
  id: string;
  label: string;
  baseUrl: string;
  modelsPath: string;
  /** Extra headers some tools want; none of them require a key by default. */
  keyOptional?: boolean;
}

const CANDIDATES: LocalCandidate[] = [
  { id: 'lmstudio', label: 'LM Studio', baseUrl: 'http://127.0.0.1:1234/v1', modelsPath: '/v1/models' },
  { id: 'llamacpp', label: 'llama.cpp', baseUrl: 'http://127.0.0.1:8080/v1', modelsPath: '/v1/models' },
  { id: 'vllm', label: 'vLLM', baseUrl: 'http://127.0.0.1:8000/v1', modelsPath: '/v1/models' },
  { id: 'localai', label: 'LocalAI', baseUrl: 'http://127.0.0.1:8081/v1', modelsPath: '/v1/models' },
  { id: 'localai-alt', label: 'LocalAI', baseUrl: 'http://127.0.0.1:5000/v1', modelsPath: '/v1/models' },
  { id: 'koboldcpp', label: 'KoboldCpp', baseUrl: 'http://127.0.0.1:5001/v1', modelsPath: '/v1/models' },
  { id: 'jan', label: 'Jan', baseUrl: 'http://127.0.0.1:1337/v1', modelsPath: '/v1/models' },
  { id: 'anythingllm', label: 'AnythingLLM', baseUrl: 'http://127.0.0.1:3001/v1', modelsPath: '/v1/models' },
];

/** Long enough for a local process to answer, short enough not to stall a page. */
const PROBE_TIMEOUT_MS = 1200;

export interface DiscoveredServer {
  id: string;
  label: string;
  baseUrl: string;
  models: string[];
}

/** Whether an OpenAI-compatible server is answering on this port. */
async function probe(c: LocalCandidate): Promise<DiscoveredServer | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
  try {
    const res = await fetch(new URL(c.modelsPath, c.baseUrl).toString(), {
      signal: controller.signal,
      headers: { accept: 'application/json' },
    });
    if (!res.ok) return null;
    const body = (await res.json()) as { data?: { id?: string }[] };
    const models = (body?.data ?? [])
      .map((m) => (typeof m?.id === 'string' ? m.id : ''))
      .filter(Boolean);
    // An endpoint that answers with no models is a server that exists but has
    // nothing loaded. It is still worth showing — the user is about to load
    // something — but it must not masquerade as having models to pick.
    return { id: c.id, label: c.label, baseUrl: c.baseUrl, models };
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

let cached: { at: number; servers: DiscoveredServer[] } | null = null;
const CACHE_MS = 20_000;

/**
 * Local servers currently reachable, cached briefly.
 *
 * The cache matters because the chat page asks for providers on every load and
 * on every model switch; probing eight ports each time would add real latency
 * to a page that has no reason to wait on servers it is not going to use.
 */
export async function discoverLocalServers(force = false): Promise<DiscoveredServer[]> {
  if (!force && cached && Date.now() - cached.at < CACHE_MS) return cached.servers;
  const found = await Promise.all(CANDIDATES.map(probe));
  const servers = found.filter((f): f is DiscoveredServer => f !== null);
  // Two entries can both be LocalAI on different ports; keep both, they are
  // genuinely different services.
  cached = { at: Date.now(), servers };
  return servers;
}

/**
 * A provider backed by a discovered server, in the same shape as a built-in.
 *
 * None of these need a key, so `configured` is true whenever the server
 * answered — which is exactly the test that got us here.
 */
export function discoveredProvider(server: DiscoveredServer): Provider {
  return openAICompatible({
    id: server.id,
    label: server.label,
    baseUrl: server.baseUrl,
    // Loopback services authenticate with nothing, and inventing a key would
    // just add a field the user cannot fill in.
    getApiKey: () => '',
    keyOptional: true,
    custom: true,
    stt: false,
    local: true,
  });
}

export { ollama };