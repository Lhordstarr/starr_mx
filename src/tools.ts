/**
 * Agent tools: things the model can call mid-conversation.
 *
 * Everything here is keyless by design, matching the image providers'
 * free-tier ethos: web search scrapes DuckDuckGo's non-JS endpoint, Wikipedia
 * uses its public API, and page fetching is plain HTTP. No key ever leaves
 * the server for tools, because there are none to configure.
 *
 * Each tool is defined once as a LangChain (zod) tool — the source of truth
 * for both the JSON schema the model sees and the args validator — plus an
 * `execute` implementation (what the server runs). Enable/disable state
 * persists in `.tools.json`, overridable via TOOLS_FILE for tests.
 */
import { lookup as dnsLookup } from 'node:dns/promises';
import * as path from 'node:path';
import { tool as buildLangChainTool } from '@langchain/core/tools';
import type { StructuredToolInterface } from '@langchain/core/tools';
import { z } from 'zod';

/* ---------------- Definitions ---------------- */

/** Scalar types the arg validator understands. */
export type ToolScalar = 'string' | 'number' | 'integer' | 'boolean';

export interface ToolPropertySchema {
  type: ToolScalar;
  description?: string;
  maxLength?: number;
  minimum?: number;
  maximum?: number;
  enum?: (string | number | boolean)[];
}

export interface ToolParameterSchema {
  type: 'object';
  properties: Record<string, ToolPropertySchema>;
  required?: string[];
  additionalProperties: false;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: ToolParameterSchema;
}

/** What an executor receives: parsed args plus the request's abort signal. */
export interface ToolContext {
  signal: AbortSignal;
}

export type ToolExecute = (args: Record<string, unknown>, ctx: ToolContext) => Promise<string>;

export interface Tool {
  definition: ToolDefinition;
  /** Shown on the Tools page; every built-in is keyless. */
  keyless: boolean;
  /** User-facing name for the /tool picker; falls back to the id. */
  label?: string;
  /** Icon id the frontend maps to its own SVG (globe, file-text, book). */
  icon?: string;
  execute: ToolExecute;
}

export interface ToolView extends ToolDefinition {
  enabled: boolean;
  keyless: boolean;
  label?: string;
  icon?: string;
}

/* ---------------- Arg validation ---------------- */

/**
 * Checks model-supplied args against the tool's schema. Models routinely send
 * extra keys or wrong types, so this coerces nothing and reports the first
 * problem as a plain message (which becomes the tool result, not a crash).
 */
export function validateArgs(def: ToolDefinition, args: unknown): string | null {
  if (!args || typeof args !== 'object' || Array.isArray(args)) {
    return `${def.name}: arguments must be an object`;
  }
  const a = args as Record<string, unknown>;
  for (const key of Object.keys(a)) {
    if (!(key in def.parameters.properties)) {
      return `${def.name}: unknown argument "${key}"`;
    }
  }
  for (const key of def.parameters.required ?? []) {
    if (a[key] === undefined || a[key] === null || a[key] === '') {
      return `${def.name}: missing required argument "${key}"`;
    }
  }
  for (const [key, prop] of Object.entries(def.parameters.properties)) {
    const v = a[key];
    if (v === undefined || v === null) continue;
    const t = Array.isArray(v) ? 'array' : typeof v;
    if (prop.type === 'integer') {
      if (t !== 'number' || !Number.isInteger(v)) return `${def.name}: "${key}" must be an integer`;
    } else if (t !== prop.type) {
      return `${def.name}: "${key}" must be a ${prop.type}`;
    }
    if (typeof v === 'string' && prop.maxLength !== undefined && v.length > prop.maxLength) {
      return `${def.name}: "${key}" is too long (max ${prop.maxLength} chars)`;
    }
    if (typeof v === 'number') {
      if (prop.minimum !== undefined && v < prop.minimum) {
        return `${def.name}: "${key}" must be >= ${prop.minimum}`;
      }
      if (prop.maximum !== undefined && v > prop.maximum) {
        return `${def.name}: "${key}" must be <= ${prop.maximum}`;
      }
    }
    if (prop.enum !== undefined && !prop.enum.includes(v as string | number | boolean)) {
      return `${def.name}: "${key}" must be one of ${prop.enum.join(', ')}`;
    }
  }
  return null;
}

/* ---------------- Safe fetching ---------------- */

/** Private/loopback/link-local ranges, v4 and v6. Tools must never reach them. */
const BLOCKED_V4: [number, number][] = [
  [0x00000000, 0xff000000], // 0.0.0.0/8 (this network)
  [0x0a000000, 0xff000000], // 10.0.0.0/8
  [0x7f000000, 0xff000000], // 127.0.0.0/8 (loopback)
  [0xa9fe0000, 0xffff0000], // 169.254.0.0/16 (link-local, cloud metadata)
  [0xac100000, 0xfff00000], // 172.16.0.0/12
  [0xc0a80000, 0xffff0000], // 192.168.0.0/16
  [0xc0000000, 0xffffff00], // 192.0.0.0/24 (IETF assignments)
  [0xc0000200, 0xffffff00], // 192.0.2.0/24 (TEST-NET-1)
  [0xc6336400, 0xffffff00], // 198.51.100.0/24 (TEST-NET-2)
  [0xcb007100, 0xffffff00], // 203.0.113.0/24 (TEST-NET-3)
  [0x64400000, 0xffc00000], // 100.64.0.0/10 (carrier-grade NAT)
  [0xe0000000, 0xf0000000], // 224.0.0.0/4 (multicast)
  [0xf0000000, 0xf0000000], // 240.0.0.0/4 (reserved)
];

function ipv4ToInt(ip: string): number | null {
  const parts = ip.split('.');
  if (parts.length !== 4) return null;
  let n = 0;
  for (const p of parts) {
    if (!/^\d+$/.test(p)) return null;
    const b = Number(p);
    if (b < 0 || b > 255) return null;
    n = (n << 8) + b;
  }
  return n >>> 0;
}

/** True when an IP literal (v4, v6, or v4-mapped v6) must not be fetched. */
function isBlockedIpLiteral(host: string): boolean {
  let h = host.toLowerCase();
  if (h.startsWith('[') && h.endsWith(']')) h = h.slice(1, -1);
  // IPv4-mapped IPv6 (::ffff:1.2.3.4): judge the embedded v4 address.
  const mapped = h.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/);
  if (mapped) h = mapped[1];
  const v4 = ipv4ToInt(h);
  if (v4 !== null) {
    return BLOCKED_V4.some(([net, mask]) => (v4 & mask) === (net & mask));
  }
  // Any other v6 literal: block loopback, unspecified, link-local, unique
  // local, and multicast. A single-label check below already rejects
  // `localhost`; a literal ::1 lands here.
  if (h.includes(':')) {
    if (h === '::1' || h === '::') return true;
    const first = h.split(':')[0];
    if (first === '') return true; // ::/96-ish forms
    const word = parseInt(first, 16);
    if (Number.isNaN(word)) return true;
    if ((word & 0xffc0) === 0xfe80) return true; // fe80::/10 link-local
    if ((word & 0xfe00) === 0xfc00) return true; // fc00::/7 unique local
    if ((word & 0xff00) === 0xff00) return true; // ff00::/8 multicast
    return false;
  }
  return false;
}

const LOCAL_NAMES = new Set(['localhost']);

/**
 * Throws unless `urlString` is a public http(s) URL. Hostnames are resolved
 * and EVERY address checked, so DNS rebinding to 127.0.0.1 fails closed;
 * redirects must be re-checked by the caller (fetchPublic does this).
 */
export async function assertPublicUrl(urlString: string): Promise<URL> {
  let url: URL;
  try {
    url = new URL(urlString);
  } catch {
    throw new Error(`Not a valid URL: ${urlString}`);
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('Only http(s) URLs can be fetched');
  }
  if (url.username || url.password) {
    throw new Error('URLs with credentials cannot be fetched');
  }
  const host = url.hostname.toLowerCase();
  if (!host || LOCAL_NAMES.has(host) || host.endsWith('.localhost') || host.endsWith('.local')) {
    throw new Error('That address is not fetchable');
  }
  if (isBlockedIpLiteral(host)) {
    throw new Error('That address is not fetchable');
  }
  // A bare dotless name (intranet host) can only be judged after resolution,
  // which the lookup below does — but a literal that is neither IP nor a
  // dotted name never resolves publicly, so reject it up front.
  if (!host.includes('.') && !host.includes(':')) {
    throw new Error('That address is not fetchable');
  }
  let addrs;
  try {
    addrs = await dnsLookup(host, { all: true });
  } catch {
    throw new Error(`Could not resolve ${host}`);
  }
  for (const a of addrs) {
    if (isBlockedIpLiteral(a.address)) {
      throw new Error('That address is not fetchable');
    }
  }
  return url;
}

export interface FetchTextOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  maxBytes?: number;
  headers?: Record<string, string>;
  maxRedirects?: number;
}

const TOOL_UA = 'llm-chat-tools/1.0 (local assistant; +https://localhost:3000)';

/**
 * Fetches text over http(s) with SSRF protection, redirect re-validation, a
 * download cap, and a timeout. Returns the decoded body as a string.
 */
/**
 * A transient transport failure worth one more try.
 *
 * On a flaky link these arrive as `ETIMEDOUT`, `ECONNRESET`, `EAI_AGAIN` or
 * undici's bare `fetch failed`, typically in bursts rather than at random, so
 * an immediate retry often lands inside the same bad window and fails again.
 * A short backoff between attempts is what actually clears it.
 */
const TRANSIENT = /fetch failed|ETIMEDOUT|ECONNRESET|ECONNREFUSED|EAI_AGAIN|ENOTFOUND|EPIPE|socket hang up/i;

async function fetchWithRetry(url: string, init: RequestInit, callerSignal?: AbortSignal): Promise<Response> {
  const attempts = 3;
  let lastErr: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fetch(url, init);
    } catch (err) {
      lastErr = err;
      // A real abort (the caller cancelled, or the timeout fired) is a
      // decision, not a glitch. Retrying it would just delay the answer.
      const aborted = callerSignal?.aborted || (err instanceof Error && err.name === 'AbortError');
      if (aborted) throw err;
      if (attempt === attempts || !TRANSIENT.test(String((err as Error)?.message ?? ''))) throw err;
      await new Promise((r) => setTimeout(r, 200 * attempt));
    }
  }
  throw lastErr;
}

/**
 * Turn undici's bare `fetch failed` into a sentence that names the host and
 * the transport problem.
 *
 * The cause is hidden in `err.cause.code`; without unwrapping it the tool
 * result showed only "✗ failed — fetch failed", which tells neither the user
 * nor the model whether the host was missing, refused, timed out, or had a bad
 * certificate. Caller cancellations are returned untouched: those are
 * decisions, and their AbortError carries its own meaning.
 */
export function describeTransportError(url: string, err: unknown): Error {
  if (!(err instanceof Error)) return new Error(String(err));
  if (err.name === 'AbortError') return err;
  let host: string;
  try {
    host = new URL(url).host;
  } catch {
    host = url;
  }
  if (err.name === 'TimeoutError') return new Error(`Timed out reaching ${host}.`);
  const code = (err as { cause?: { code?: string } }).cause?.code;
  const reasons: Record<string, string> = {
    ENOTFOUND: 'the host could not be found',
    EAI_AGAIN: 'the DNS lookup failed temporarily',
    ECONNREFUSED: 'the connection was refused',
    ECONNRESET: 'the connection was reset',
    ETIMEDOUT: 'the connection timed out',
    ENETUNREACH: 'the network is unreachable',
    EHOSTUNREACH: 'the host is unreachable',
    EPIPE: 'the connection closed unexpectedly',
    CERT_HAS_EXPIRED: 'the site certificate has expired',
    DEPTH_ZERO_SELF_SIGNED_CERT: 'the site certificate is self-signed',
    UNABLE_TO_VERIFY_LEAF_SIGNATURE: 'the site certificate could not be verified',
  };
  const reason = code ? reasons[code] : undefined;
  if (reason) return new Error(`Could not reach ${host}: ${reason} (${code}).`);
  const cause = (err as { cause?: unknown }).cause;
  if (err.message === 'fetch failed' && cause instanceof Error && cause.message) {
    return new Error(`Could not reach ${host}: ${cause.message}.`);
  }
  return err;
}

export async function fetchPublicText(urlString: string, opts: FetchTextOptions = {}): Promise<string> {
  const { timeoutMs = 15000, maxBytes = 2_000_000, headers = {}, maxRedirects = 3 } = opts;
  const timeout = AbortSignal.timeout(timeoutMs);
  const signal = opts.signal ? AbortSignal.any([opts.signal, timeout]) : timeout;

  let current = urlString;
  for (let hop = 0; hop <= maxRedirects; hop++) {
    const url = await assertPublicUrl(current);
    const init = {
      headers: { 'user-agent': TOOL_UA, 'accept-language': 'en-US,en;q=0.8', ...headers },
      redirect: 'manual' as const,
      signal,
    };
    let res: Response;
    try {
      res = await fetchWithRetry(url.toString(), init, opts.signal);
    } catch (err) {
      // The retry helper has already exhausted what a fresh socket can fix.
      // Name the host and the cause rather than passing on "fetch failed".
      throw describeTransportError(url.toString(), err);
    }
    if (res.status >= 300 && res.status < 400 && res.headers.get('location')) {
      current = new URL(res.headers.get('location')!, url.toString()).toString();
      continue;
    }
    if (!res.ok) {
      throw new Error(`Fetch failed: HTTP ${res.status}`);
    }
    const announced = Number(res.headers.get('content-length') ?? 0);
    if (announced > maxBytes) {
      throw new Error('That page is too large to read');
    }
    // Stream with an early abort so a lying Content-Length can't OOM us.
    const reader = res.body!.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          total += value.length;
          if (total > maxBytes) throw new Error('That page is too large to read');
          chunks.push(value);
        }
      }
    } finally {
      reader.releaseLock();
    }
    const buf = Buffer.concat(chunks.map((c) => Buffer.from(c)));
    // Respect an explicit charset; default to UTF-8.
    const charset = /charset=([^;]+)/i.exec(res.headers.get('content-type') ?? '')?.[1]?.trim();
    try {
      return new TextDecoder(charset || 'utf-8', { fatal: false }).decode(buf);
    } catch {
      return buf.toString('utf-8');
    }
  }
  throw new Error('Too many redirects');
}

/* ---------------- Small HTML helpers (no parser dependency) ---------------- */

/** Decodes the entities these endpoints actually emit. */
export function decodeEntities(s: string): string {
  return s
    .replace(/&#(\d+);/g, (_, n) => {
      try {
        return String.fromCodePoint(Number(n));
      } catch {
        return '';
      }
    })
    .replace(/&#x([0-9a-fA-F]+);/g, (_, n) => {
      try {
        return String.fromCodePoint(parseInt(n, 16));
      } catch {
        return '';
      }
    })
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;|&apos;/g, "'")
    .replace(/&nbsp;/g, ' ');
}

/** Strips tags, keeping block structure as newlines. */
export function htmlToText(html: string): string {
  return (
    html
      .replace(/<!--[\s\S]*?-->/g, ' ')
      .replace(/<(script|style|noscript|svg|canvas|template)[^>]*>[\s\S]*?<\/\1\s*>/gi, ' ')
      .replace(/<\/(p|div|section|article|h[1-6]|li|tr|blockquote|pre)>/gi, '\n')
      .replace(/<br\s*\/?>/gi, '\n')
      .replace(/<[^>]*>/g, ' ')
      .split('\n')
      .map((line) => decodeEntities(line).replace(/[ \t\r\f\v]+/g, ' ').trim())
      // Drop the one-word crumbs navs leave behind, keep real sentences.
      .filter((line) => line.length > 0)
      .join('\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  );
}

/* ---------------- Tool implementations ---------------- */

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}

/**
 * Pulls the snippet text that follows a result link. The snippet element
 * contains inline markup (<b>, <span>…), so a lazy `.*?-close` regex would
 * stop at the first nested tag — instead the matching close is found by
 * counting same-name opens, which also survives nested divs.
 */
function extractSnippet(after: string): string {
  const open = /<([a-zA-Z][a-zA-Z0-9]*)[^>]*class="result__snippet"[^>]*>/i.exec(after);
  if (!open || open.index === undefined) return '';
  const tag = open[1].toLowerCase();
  const rest = after.slice(open.index + open[0].length);
  const tagRe = new RegExp(`<(/?)${tag}(\\s[^>]*)?/?>`, 'gi');
  let depth = 1;
  let end = rest.length;
  let m: RegExpExecArray | null;
  while ((m = tagRe.exec(rest)) !== null) {
    if (m[0].endsWith('/>')) continue;
    if (m[1]) {
      depth--;
      if (depth === 0) {
        end = m.index;
        break;
      }
    } else {
      depth++;
    }
  }
  return rest.slice(0, end);
}

/**
 * Parses DuckDuckGo's non-JS results page. Kept separate from the fetch so it
 * is unit-testable against a fixture without touching the network.
 */
export function parseDuckDuckGo(html: string): SearchResult[] {
  const out: SearchResult[] = [];
  // <a rel="nofollow" class="result__a" href="...">title</a>
  const linkRe = /<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a\s*>/gi;
  // <a ... href="..." ... class="result__a" ...> (attribute order swapped)
  const linkRe2 = /<a[^>]*href="([^"]+)"[^>]*class="result__a"[^>]*>([\s\S]*?)<\/a\s*>/gi;
  const seen = new Set<string>();
  const blocks: { href: string; titleHtml: string; index: number }[] = [];
  for (const re of [linkRe, linkRe2]) {
    let m: RegExpExecArray | null;
    while ((m = re.exec(html)) !== null) {
      blocks.push({ href: m[1], titleHtml: m[2], index: m.index });
    }
  }
  blocks.sort((a, b) => a.index - b.index);
  for (const b of blocks) {
    let href = decodeEntities(b.href);
    // DDG sometimes wraps outbound links in its own redirector.
    const uddg = /[?&]uddg=([^&]+)/.exec(href)?.[1];
    if (uddg) {
      try {
        href = decodeURIComponent(uddg);
      } catch {
        continue;
      }
    }
    if (href.startsWith('//')) href = `https:${href}`;
    if (!/^https?:\/\//i.test(href) || seen.has(href)) continue;
    seen.add(href);
    // The snippet follows the link in the same .result block.
    const after = html.slice(b.index, b.index + 6000);
    const snip = extractSnippet(after);
    const title = decodeEntities(b.titleHtml.replace(/<[^>]*>/g, ' '))
      .replace(/\s+/g, ' ')
      .trim();
    const snippet = decodeEntities(snip.replace(/<[^>]*>/g, ' '))
      .replace(/\s+/g, ' ')
      .trim();
    if (!title) continue;
    out.push({ title, url: href, snippet });
  }
  return out;
}

async function webSearchExecute(
  args: Record<string, unknown>,
  ctx: ToolContext,
): Promise<string> {
  const query = args.query as string;
  const count = Math.min(Math.max((args.count as number | undefined) ?? 5, 1), 10);
  // Overridable for tests (mirrors OPENAI_BASE_URL); production is always DDG.
  const base = process.env.DDG_BASE_URL ?? 'https://html.duckduckgo.com';
  const html = await fetchPublicText(`${base}/html/?q=${encodeURIComponent(query)}`, {
    signal: ctx.signal,
    headers: { accept: 'text/html' },
  });
  const results = parseDuckDuckGo(html);
  if (!results.length) {
    if (/anomaly|challenge|captcha/i.test(html)) {
      throw new Error(
        'The search engine challenged this request (bot check). Try again later or from another network.',
      );
    }
    return `No results for "${query}".`;
  }
  return results
    .slice(0, count)
    .map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}${r.snippet ? `\n   ${r.snippet}` : ''}`)
    .join('\n');
}

async function fetchPageExecute(
  args: Record<string, unknown>,
  ctx: ToolContext,
): Promise<string> {
  const url = args.url as string;
  const maxChars = Math.min(Math.max((args.max_chars as number | undefined) ?? 12000, 500), 60000);
  const html = await fetchPublicText(url, { signal: ctx.signal });
  const text = htmlToText(html).slice(0, maxChars);
  if (!text) throw new Error('That page had no readable text.');
  const cut = text.length >= maxChars;
  return `Contents of ${url}:\n\n${text}${cut ? '\n\n[…truncated…]' : ''}`;
}

async function wikipediaExecute(
  args: Record<string, unknown>,
  ctx: ToolContext,
): Promise<string> {
  const query = args.query as string;
  const sentences = Math.min(Math.max((args.sentences as number | undefined) ?? 4, 1), 10);
  // Overridable for tests; production is always Wikipedia.
  const base = process.env.WIKI_BASE_URL ?? 'https://en.wikipedia.org';
  const found: unknown = JSON.parse(
    await fetchPublicText(
      `${base}/w/api.php?action=opensearch&search=${encodeURIComponent(query)}&limit=1&format=json`,
      { signal: ctx.signal, headers: { accept: 'application/json' } },
    ),
  );
  const title = Array.isArray(found) && Array.isArray(found[1]) ? String(found[1][0] ?? '') : '';
  if (!title) return `No Wikipedia article found for "${query}".`;
  const summary: any = JSON.parse(
    await fetchPublicText(
      `${base}/w/api.php?action=query&prop=extracts&exsentences=${sentences}` +
        `&explaintext&redirects=1&titles=${encodeURIComponent(title)}&format=json`,
      { signal: ctx.signal, headers: { accept: 'application/json' } },
    ),
  );
  const pages = summary?.query?.pages ?? {};
  const page = Object.values(pages)[0] as { title?: string; extract?: string } | undefined;
  const extract = page?.extract?.trim() ?? '';
  if (!extract) return `Wikipedia has "${page?.title ?? title}" but no summary text.`;
  return `Wikipedia — ${page?.title ?? title}:\n${extract}\nhttps://en.wikipedia.org/wiki/${encodeURIComponent(
    (page?.title ?? title).replace(/ /g, '_'),
  )}`;
}

/* ---------------- Registry + persistence ---------------- */

interface BuiltinSpec {
  name: string;
  /** User-facing name shown in the /tool picker. */
  label: string;
  /** Icon id the frontend maps to its own SVG. */
  icon: string;
  description: string;
  /** The single source of truth: wire schema and validation both derive from it. */
  schema: z.ZodObject<z.ZodRawShape>;
  execute: ToolExecute;
}

const BUILTINS: BuiltinSpec[] = [
  {
    name: 'web_search',
    label: 'Web Search',
    icon: 'globe',
    description:
      'Searches the web for current information: news, docs, prices, recent events. ' +
      'Use when the question needs anything newer than training data or outside it. ' +
      'Returns numbered titles, URLs, and snippets.',
    schema: z.object({
      query: z.string().max(300).describe('The search query.'),
      count: z.number().int().min(1).max(10).describe('How many results (1-10).').optional(),
    }),
    execute: webSearchExecute,
  },
  {
    name: 'fetch_page',
    label: 'Fetch Page',
    icon: 'file-text',
    description:
      'Reads a web page as plain text: follow up on a search result, read docs, ' +
      'check a price. Public http(s) pages only; oversized pages are truncated.',
    schema: z.object({
      url: z.string().max(2000).describe('The page URL.'),
      max_chars: z
        .number()
        .int()
        .min(500)
        .max(60000)
        .describe('Max characters to return (500-60000).')
        .optional(),
    }),
    execute: fetchPageExecute,
  },
  {
    name: 'wikipedia',
    label: 'Wikipedia',
    icon: 'book',
    description:
      'Looks up a topic on Wikipedia and returns the lead summary. Best for ' +
      'stable factual questions: people, places, concepts, history.',
    schema: z.object({
      query: z.string().max(200).describe('The topic to look up.'),
      sentences: z.number().int().min(1).max(10).describe('Summary sentences (1-10).').optional(),
    }),
    execute: wikipediaExecute,
  },
];

/**
 * The wire/validation JSON schema, derived from a tool's zod schema.
 *
 * Deliberately narrow — the built-ins are flat scalars — and throws at module
 * load on anything else rather than shipping a schema nothing can enforce.
 */
function zodParameters(schema: z.ZodObject<z.ZodRawShape>): ToolParameterSchema {
  const json = z.toJSONSchema(schema) as {
    properties?: Record<
      string,
      {
        type?: string;
        description?: string;
        maxLength?: number;
        minimum?: number;
        maximum?: number;
        enum?: unknown[];
      }
    >;
    required?: string[];
  };
  const properties: Record<string, ToolPropertySchema> = {};
  for (const [key, prop] of Object.entries(json.properties ?? {})) {
    const scalar = prop.type;
    if (scalar !== 'string' && scalar !== 'number' && scalar !== 'integer' && scalar !== 'boolean') {
      throw new Error(`Tool schema "${key}" has a type the validator cannot enforce`);
    }
    const out: ToolPropertySchema = { type: scalar };
    if (typeof prop.description === 'string') out.description = prop.description;
    if (typeof prop.maxLength === 'number') out.maxLength = prop.maxLength;
    if (typeof prop.minimum === 'number') out.minimum = prop.minimum;
    if (typeof prop.maximum === 'number') out.maximum = prop.maximum;
    if (Array.isArray(prop.enum)) out.enum = prop.enum as (string | number | boolean)[];
    properties[key] = out;
  }
  return {
    type: 'object',
    properties,
    required: json.required ?? [],
    additionalProperties: false,
  };
}

const langChainTools = new Map<string, StructuredToolInterface>();

/**
 * A built-in as a real LangChain tool (zod schema + executor), or undefined
 * for names this registry doesn't know. ChatOllama.bindTools takes these
 * straight; LangChain converts the zod schema to the wire JSON schema itself.
 */
export function langChainTool(name: string): StructuredToolInterface | undefined {
  const spec = BUILTINS.find((b) => b.name === name);
  if (!spec) return undefined;
  let t = langChainTools.get(name);
  if (!t) {
    t = buildLangChainTool(
      async (input, config) =>
        spec.execute(input as Record<string, unknown>, {
          signal: config?.signal ?? new AbortController().signal,
        }),
      { name: spec.name, description: spec.description, schema: spec.schema },
    );
    langChainTools.set(name, t);
  }
  return t;
}

const REGISTRY: Tool[] = BUILTINS.map((b) => ({
  definition: {
    name: b.name,
    description: b.description,
    parameters: zodParameters(b.schema),
  },
  keyless: true,
  label: b.label,
  icon: b.icon,
  execute: b.execute,
}));

export function toolsFilePath(): string {
  return path.resolve(process.env.TOOLS_FILE ?? '.tools.json');
}

interface ToolsFile {
  disabled?: string[];
}

async function readState(): Promise<ToolsFile> {
  try {
    const { readFile } = await import('node:fs/promises');
    return JSON.parse(await readFile(toolsFilePath(), 'utf-8')) as ToolsFile;
  } catch {
    return {};
  }
}

async function writeState(state: ToolsFile): Promise<void> {
  const { writeFile } = await import('node:fs/promises');
  await writeFile(toolsFilePath(), `${JSON.stringify(state, null, 2)}\n`, 'utf-8');
}

export function toolByName(name: string): Tool | undefined {
  return REGISTRY.find((t) => t.definition.name === name);
}

/** Registry ids — call-shape detection that must recognize a real tool name
 *  (including one not offered for the current request). */
export function toolNames(): string[] {
  return REGISTRY.map((t) => t.definition.name);
}

/** Every tool with its stored on/off flag. Unknown names in state are ignored. */
export async function listTools(): Promise<ToolView[]> {
  const state = await readState();
  const disabled = new Set(state.disabled ?? []);
  return REGISTRY.map((t) => ({
    ...t.definition,
    enabled: !disabled.has(t.definition.name),
    keyless: t.keyless,
    ...(t.label ? { label: t.label } : {}),
    ...(t.icon ? { icon: t.icon } : {}),
  }));
}

/** Tools the model may currently call, in registry order. */
export async function enabledTools(): Promise<Tool[]> {
  const state = await readState();
  const disabled = new Set(state.disabled ?? []);
  return REGISTRY.filter((t) => !disabled.has(t.definition.name));
}

/**
 * Resolves a request's `tools` field against the registry.
 *
 * `true` (the legacy all-or-nothing flag) means every enabled tool; an array
 * names a per-message subset and is validated against the registry *and* the
 * stored on/off state, so a stale or invented id is an error rather than a
 * silent grant. Nothing selected — undefined, false, [] — resolves to no
 * tools: selection is per-message, and no selection is a plain chat.
 */
export function resolveToolSelection(
  requested: unknown,
  enabled: Tool[],
): { tools: Tool[]; error?: string } {
  if (requested === undefined || requested === false) return { tools: [] };
  if (requested === true) return { tools: enabled };
  if (!Array.isArray(requested)) {
    return { tools: [], error: 'tools must be a boolean or an array of tool names' };
  }
  const offered = new Map(enabled.map((t) => [t.definition.name, t]));
  const tools: Tool[] = [];
  for (const id of requested) {
    if (typeof id !== 'string' || !id) {
      return { tools: [], error: 'tools must be a boolean or an array of tool names' };
    }
    const tool = offered.get(id);
    if (!tool) {
      return {
        tools: [],
        error: toolByName(id) ? `Tool not enabled: ${id}` : `Unknown tool: ${id}`,
      };
    }
    if (!tools.includes(tool)) tools.push(tool);
  }
  return { tools };
}

export async function setToolEnabled(name: string, enabled: boolean): Promise<ToolView> {
  const tool = toolByName(name);
  if (!tool) throw new Error(`Unknown tool: ${name}`);
  const state = await readState();
  const disabled = new Set(state.disabled ?? []);
  if (enabled) disabled.delete(name);
  else disabled.add(name);
  // Store only known names so a hand-edited file can't wedge the list.
  state.disabled = REGISTRY.map((t) => t.definition.name).filter((n) => disabled.has(n));
  await writeState(state);
  return {
    ...tool.definition,
    enabled,
    keyless: tool.keyless,
    ...(tool.label ? { label: tool.label } : {}),
    ...(tool.icon ? { icon: tool.icon } : {}),
  };
}
