/**
 * Splitting a model's thinking out of its answer.
 *
 * Thinking models put their scratchpad in the same byte stream as the answer.
 * llama.cpp and Ollama can mark it structurally (`reasoning_content`,
 * `thinking`) and those adapters pass it along as reasoning events — but a
 * tags-in-text model (a raw gateway, or the local transformers server, whose
 * Python side just applies the template) arrives as ordinary content, and the
 * reader should not have scratchpad dumped into the transcript, nor lose it.
 *
 * This splitter walks the live stream once, holding back only the tail that
 * could still be part of a tag, and yields reasoning/content deltas as soon as
 * each is certain. No lookahead, no re-scanning the whole reply per token: a
 * partial opener (`\n<thin`) is held for the next chunk; a tag that completes
 * flips the mode; anything that turns out to be prose flushes immediately.
 *
 * The tag pairs mirror llama.cpp's chat parsers (the same table LocalAI
 * ports), so whichever template a model was trained on, its tags are
 * recognised. A template that *prefills* the opening tag and ships only a
 * closer — `reasoning...`. Text is deliberately NOT guessed at: the runners
 * this app controls either emit structural fields or the open tag, and a
 * guessed closing tag would silently swallow real answers.
 */

/** Longest opening tag here is 18 chars (`<|START_THINKING|>`); a head that
 *  stays undecided past this is whitespace padding, not a tag forming. */
const HOLD_LIMIT = 64;

/** Start/end pairs, longest forms first within each family. Order is only
 *  cosmetic — every check scans the whole table. */
const PAIRS: readonly (readonly [start: string, end: string])[] = [
  ['<|START_THINKING|>', '<|END_THINKING|>'], // Command-R
  ['<|inner_prefix|>', '<|inner_suffix|>'], // Apertus
  ['<seed:think>', '</seed:think>'], // Seed
  ['<think>', '</think>'], // DeepSeek, Granite, ExaOne
  ['<|think|>', '<|end|><|begin|>assistant<|content|>'], // Solar Open
  ['<|channel>thought', '<channel|>'], // Gemma 4
  ['<thinking>', '</thinking>'], // general
  ['[THINK]', '[/THINK]'], // Magistral
];

/** Text of a wire reasoning field: the providers that mark thinking
 *  structurally spell it `reasoning_content` (llama.cpp with its default
 *  `--reasoning-format auto`, DeepSeek, Ollama-as-OpenAI), `reasoning`
 *  (OpenRouter), or a `{content}` / `[{text}]` wrapper. Returns '' for
 *  anything else — an absent field is the common case. */
export function reasoningText(field: unknown): string {
  if (typeof field === 'string') return field;
  if (Array.isArray(field)) {
    return field
      .map((p) => (typeof p === 'string' ? p : ((p as { text?: unknown })?.text as string) ?? ''))
      .join('');
  }
  const content = (field as { content?: unknown })?.content;
  return typeof content === 'string' ? content : '';
}

export interface ReasoningDelta {
  /** New scratchpad text, '' when none. */
  reasoning: string;
  /** New answer text, '' when none. */
  content: string;
}

/** How many trailing chars of `s` form the start of `token` (0 when none). */
function suffixOfToken(s: string, token: string): number {
  const max = Math.min(s.length, token.length - 1);
  for (let n = max; n > 0; n -= 1) {
    if (s.endsWith(token.slice(0, n))) return n;
  }
  return 0;
}

/** Longest trailing run that could grow into any opening tag. */
function trailingOpener(s: string): number {
  let most = 0;
  for (const [start] of PAIRS) most = Math.max(most, suffixOfToken(s, start));
  return most;
}

/** True when position `i` in `s` begins a line (only indentation before it).
 *  A re-entry opener inside prose (`see  thinking`) is being talked *about*,
 *  not emitted, so tags only count at a line start once the head is decided. */
function atLineStart(s: string, i: number): boolean {
  let j = i - 1;
  while (j >= 0 && (s[j] === ' ' || s[j] === '\t')) j -= 1;
  return j < 0 || s[j] === '\n';
}

/**
 * The streaming splitter. One instance per reply streamed; `push` each text
 * delta in arrival order and forward the two returned deltas.
 */
export class ThinkingSplitter {
  private mode: 'head' | 'in' | 'live' = 'head';
  /** Index into PAIRS while `mode` is `in`. */
  private pair = 0;
  /** Text not yet classified — always shorter than the tags it might join. */
  private held = '';

  push(text: string): ReasoningDelta {
    this.held += text;
    const out: ReasoningDelta = { reasoning: '', content: '' };

    for (;;) {
      if (this.mode === 'head') {
        const lead = /^[ \t\r\n]*/.exec(this.held)![0].length;
        const rest = this.held.slice(lead);
        if (rest === '') {
          // Nothing but padding so far: wait for the next chunk, unless it is
          // so much whitespace that no tag is plausibly still arriving.
          if (this.held.length < HOLD_LIMIT) break;
          this.mode = 'live';
          continue;
        }
        const opener = PAIRS.findIndex(([start]) => rest.startsWith(start));
        if (opener >= 0) {
          // Padding before the tag is template noise; drop it rather than
          // painting a blank line ahead of the thinking pane.
          this.mode = 'in';
          this.pair = opener;
          this.held = rest.slice(PAIRS[opener][0].length);
          continue;
        }
        if (PAIRS.some(([start]) => start.startsWith(rest))) break; // still forming
        this.mode = 'live'; // plain prose — flush it all, including the lead
        continue;
      }

      if (this.mode === 'in') {
        const end = PAIRS[this.pair][1];
        const close = this.held.indexOf(end);
        if (close >= 0) {
          out.reasoning += this.held.slice(0, close);
          this.held = this.held.slice(close + end.length);
          this.mode = 'live';
          continue;
        }
        // No closer yet: everything except a possible partial closer is certainly
        // reasoning and can stream now.
        const keep = suffixOfToken(this.held, end);
        out.reasoning += this.held.slice(0, this.held.length - keep);
        this.held = this.held.slice(this.held.length - keep);
        break;
      }

      // live: answer text flows through, but a line-start opener starts another
      // thinking block (models interleave blocks across tool rounds).
      let found = -1;
      let foundPair = 0;
      for (let p = 0; p < PAIRS.length; p += 1) {
        for (let i = 0; ; ) {
          i = this.held.indexOf(PAIRS[p][0], i);
          if (i === -1) break;
          if (atLineStart(this.held, i) && (found === -1 || i < found)) {
            found = i;
            foundPair = p;
          }
          i += 1;
        }
      }
      if (found >= 0) {
        out.content += this.held.slice(0, found);
        this.held = this.held.slice(found + PAIRS[foundPair][0].length);
        this.pair = foundPair;
        this.mode = 'in';
        continue;
      }
      const keep = trailingOpener(this.held);
      out.content += this.held.slice(0, this.held.length - keep);
      this.held = this.held.slice(this.held.length - keep);
      break;
    }

    return out;
  }

  /** Whatever is still held when the stream ends. A tail no longer arriving:
   *  an unfinished tag is not an answer, and reasoned text is not either. */
  flush(): ReasoningDelta {
    const held = this.held;
    this.held = '';
    if (!held) return { reasoning: '', content: '' };
    return this.mode === 'in'
      ? { reasoning: held, content: '' }
      : { reasoning: '', content: held };
  }
}
