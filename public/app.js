import { createAttachments } from './attachments.js';
import { renderCode } from './highlight.js';
import { initShaderBackdrop } from './shader.js';
import { initImageViewer } from './viewer.js';
import { notify, notifySuccess, notifyError, notifyWarning, clearNotifications } from './notify.js';
import { mountNav } from '/nav.js';
import { mountResources } from '/resources.js';

/* Rendered first, before any of this page's own startup. It is three links
   and a toggle, it never awaits anything, and every other module in the
   file is doing network work by this point. */
mountNav();

const els = {
  providerHint: document.getElementById('provider-hint'),
  modelCurrent: document.getElementById('model-current'),
  modelSelect: document.getElementById('model-select'),
  modelTrigger: document.getElementById('model-trigger'),
  modelMenu: document.getElementById('model-menu'),
  modelSearch: document.getElementById('model-search'),
  modelCustom: document.getElementById('model-custom'),
  modelUse: document.getElementById('model-use'),
  modelList: document.getElementById('model-list'),
  composerBox: document.querySelector('.composer-box'),
  composerCount: document.getElementById('composer-count'),
  modelHint: document.getElementById('model-hint'),
  historyList: document.getElementById('history-list'),
  historyHint: document.getElementById('history-hint'),
  messages: document.getElementById('messages'),
  welcome: document.getElementById('welcome'),
  composer: document.getElementById('composer'),
  toLatest: document.getElementById('to-latest'),
  input: document.getElementById('input'),
  send: document.getElementById('send'),
  stop: document.getElementById('stop'),
  mic: document.getElementById('mic'),
  micTime: document.getElementById('mic-time'),
  newChat: document.getElementById('new-chat'),
  themeToggle: document.getElementById('theme-toggle'),
  sidebar: document.getElementById('sidebar') ?? document.querySelector('.sidebar'),
  sidebarToggle: document.getElementById('sidebar-toggle'),
  sidebarClose: document.getElementById('sidebar-close'),
  openPanel: document.getElementById('open-panel'),
  providerPop: document.getElementById('provider-pop'),
  providerList: document.getElementById('provider-list'),
  backdrop: document.getElementById('backdrop'),
  chipProvider: document.getElementById('chip-provider'),
  chipModel: document.getElementById('chip-model'),
  composerHint: document.getElementById('composer-hint'),
  status: document.getElementById('status'),
  statusDot: document.getElementById('status-dot'),
  statusBlob: document.getElementById('status-blob'),
  attach: document.getElementById('attach'),
  attachInput: document.getElementById('attach-input'),
  attachBox: document.getElementById('attach-box'),
  attachList: document.getElementById('attach-list'),
  toolsBox: document.getElementById('tools-box'),
  toolsList: document.getElementById('tools-list'),
  toolMenu: document.getElementById('tool-menu'),
  toolMenuList: document.getElementById('tool-menu-list'),
  toolMenuHint: document.getElementById('tool-menu-hint'),
};

/** @type {{id:string,label:string,models:string[],sttModels:string[],stt:boolean,configured:boolean,hint?:string}[]} */
let providers = [];
/**
 * The provider popup replaced a <select>, so the choice is held here instead.
 * Everything downstream (model list, chip, persistence) reads this.
 */
let activeProviderId = 'ollama';
let conversation = [];
/** Saved conversations. The active one is currentChat(); `conversation` above
 *  is its working message array while it's on screen. */
let chats = [];
let activeId = null;

/** Current model id. Held here because the picker is a custom dropdown, not an input. */
let model = '';
/** Index into the currently filtered option list, for keyboard navigation. */
let menuIndex = -1;
let menuOptions = [];

const STORAGE_KEY = 'llm-chat-state';
const THEME_KEY = 'llm-chat-theme';

const attachments = createAttachments({ els, notifyWarning });

/* ---------------- Theme ---------------- */

function applyTheme(theme) {
  if (theme === 'dark' || theme === 'light') {
    document.documentElement.dataset.theme = theme;
  } else {
    delete document.documentElement.dataset.theme; // follow the OS
  }
}

applyTheme(localStorage.getItem(THEME_KEY));

els.themeToggle.addEventListener('click', () => {
  // Cycles explicit choice -> opposite -> back to following the OS.
  const order = [null, 'light', 'dark'];
  const current = document.documentElement.dataset.theme ?? null;
  const next = order[(order.indexOf(current) + 1) % order.length];

  applyTheme(next);
  if (next) localStorage.setItem(THEME_KEY, next);
  else localStorage.removeItem(THEME_KEY);

  const labels = { null: 'Following system theme', light: 'Light theme', dark: 'Dark theme' };
  notify(labels[next]);
});

/* ---------------- Toast ---------------- */

/* ---------------- Fetch helper ---------------- */

/** Fetch wrapper that attaches the JSON content type for bodies. */
async function api(path, options = {}) {
  return fetch(path, {
    ...options,
    headers: options.body ? { 'content-type': 'application/json' } : undefined,
  });
}

/** GETs a JSON endpoint, with the error message the server sent. */
async function apiJson(path) {
  const res = await fetch(path);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data;
}

/**
 * Posts to an /api/infer/* endpoint and resolves the terminal SSE event.
 *
 * Those routes stream `log` lines while the Python runner works and finish with
 * exactly one `done` or `error`. Posting them with the JSON helper would either
 * mangle the binary upload or try to parse a stream as one document.
 */
async function apiInfer(path, payload) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error ?? `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.search(/\r?\n\r?\n/)) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + (buffer[sep] === '\r' ? 4 : 2));
      let event = 'message';
      let data = '';
      for (const line of raw.split(/\r?\n/)) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (!data) continue;
      let parsed;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue;
      }
      if (event === 'error') throw new Error(parsed.message ?? 'The runner failed.');
      if (event === 'done') return parsed;
    }
  }
  throw new Error('The runner stopped without a result.');
}

/* ---------------- State & history ---------------- */

const currentChat = () => chats.find((c) => c.id === activeId) ?? null;

const genId = () =>
  Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

/**
 * A conversation gets its name from the first thing typed in it.
 *
 * A turn with an attachment carries its text as content parts rather than a
 * plain string, so flatten those before reading a title off it — otherwise
 * `.replace` is called on an array and the whole send throws.
 */
function chatTitle(messages) {
  const first = messages.find((m) => m.role === 'user')?.content;
  const flat = (typeof first === 'string'
    ? first
    : (first ?? [])
        .filter((p) => p.type === 'text')
        .map((p) => p.text)
        .join(' '))
    .replace(/\s+/g, ' ')
    .trim();
  if (!flat) return first ? 'Image conversation' : 'Empty conversation';
  return flat.length > 42 ? `${flat.slice(0, 40)}…` : flat;
}

/** Builds a chat, makes it active, and updates the working message array. */
function createChat({ provider = '', model: initModel = '', conversation: msgs = [] } = {}) {
  const chat = {
    id: genId(),
    title: chatTitle(msgs),
    provider,
    model: initModel,
    conversation: msgs,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  chats.unshift(chat);
  activeId = chat.id;
  return chat;
}

/**
 * Flushes one chat's conversation to storage — not necessarily the open one.
 *
 * An answer that finishes while you are reading a different conversation still
 * has to land: `persist()` below only knows about the chat on screen, which is
 * how a completed reply could sit in memory and vanish on close. Writing the
 * specific chat is what makes background answers durable.
 */
function persistChat(chatId, msgs) {
  const chat = chats.find((c) => c.id === chatId);
  if (!chat) return;
  chat.conversation = msgs.map(attachments.stripForStorage);
  chat.title = chatTitle(msgs);
  chat.updatedAt = Date.now();
  chats.sort((a, b) => b.updatedAt - a.updatedAt);
  saveStored();
  renderHistory();
}

/** Writes the whole store out. Never throws: a full disk must not break the UI. */
function saveStored() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        active: activeId,
        chats,
        // Ambient provider/model: the workbench settings that apply to
        // the unsaved working area, so a pick survives a reload before the
        // first message has created a history entry.
        ambient: { provider: activeProviderId, model },
      }),
    );
  } catch (err) {
    // localStorage is about 5 MB. Rather than throw out of every save after
    // the first full one — which is how old conversations stopped coming back
    // — drop the oldest ones and try again, then say so plainly.
    if (chats.length > 1) {
      chats.length = Math.max(1, Math.floor(chats.length / 2));
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ active: activeId, chats }));
        notifyError('Storage was full — the oldest conversations were dropped to make room.');
        return;
      } catch {
        /* fall through to the honest failure below */
      }
    }
    notifyError(`Could not save conversations: ${err instanceof Error ? err.name : 'storage full'}`);
  }
}

/** Flushes the current chat (including its working array) to storage. */
function persist() {
  const active = currentChat();
  if (active) {
    active.title = chatTitle(conversation);
    active.provider = activeProviderId;
    active.model = model;
    // Image bytes are stripped on the way in: a couple of attachments would
    // exceed localStorage's quota and lose every saved chat with them.
    active.conversation = conversation.map(attachments.stripForStorage);
    active.updatedAt = Date.now();
    chats.sort((a, b) => b.updatedAt - a.updatedAt);
  }
  saveStored();
  renderHistory();
}

/**
 * Loads saved state into `chats`/`conversation`. Returns what the boot path
 * needs to reselect the active conversation, or an empty set when fresh.
 * A pre-history store (a single flat conversation) is promoted to a chat.
 */
/**
 * Deep link from the settings model list: /chat?provider=…&model=…
 *
 * A URL pair wins over whatever was remembered, and unlike a restored
 * conversation it always starts a new one — clicking "Chat" on a model in the
 * list means "let me use this now", not "reopen my last chat with it".
 */
function deepLink() {
  const q = new URLSearchParams(location.search);
  const provider = q.get('provider');
  const model = q.get('model');
  if (!provider && !model) return null;
  return {
    provider: provider ?? undefined,
    model: model ?? undefined,
    conversation: [],
  };
}

function restore() {
  const link = deepLink();
  if (link) return link;
  let ambient = {};
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
    ambient = s.ambient && typeof s.ambient === 'object' ? s.ambient : {};
    if (Array.isArray(s.chats)) {
      chats = s.chats;
      const valid = new Set(chats.map((c) => c.id));
      activeId = valid.has(s.active) ? s.active : chats[0]?.id ?? null;
      const c = currentChat();
      if (c) {
        conversation = Array.isArray(c.conversation) ? c.conversation : [];
        return { provider: c.provider, model: c.model, conversation };
      }
    } else if (Array.isArray(s.conversation) && s.conversation.length) {
      const c = createChat({
        provider: s.provider,
        model: s.model,
        conversation: s.conversation,
      });
      conversation = c.conversation;
      return { provider: c.provider, model: c.model, conversation };
    }
  } catch {
    /* corrupted or absent storage: start fresh below */
  }
  // No saved state: the working area starts unsaved and only becomes a saved
  // conversation once the first message is actually sent.
  activeId = null;
  conversation = [];
  return {
    provider: ambient && ambient.provider,
    model: ambient && ambient.model,
    conversation,
  };
}

/* ---------------- Markdown-ish rendering ---------------- */

const escapeHtml = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/**
 * Decides whether an href may be emitted at all.
 *
 * Everything reaching here has already been through escapeHtml, so markup
 * injection is covered — but escapeHtml does nothing to a `javascript:` URL,
 * which contains no characters needing escaping. So the scheme is checked
 * explicitly, and only http(s) survives.
 *
 * Returns the href to use, or null to degrade the link to plain text.
 */
function safeUrl(raw) {
  const url = raw.trim();
  // Control characters can hide a scheme from the regex below ("java\nscript:").
  if (/[\u0000-\u001f\u007f]/.test(url)) return null;
  // Protocol-relative URLs inherit the page scheme; not worth allowing.
  if (url.startsWith('//')) return null;
  const scheme = /^([a-z][a-z0-9+.-]*):/i.exec(url);
  if (scheme && !/^(https?)$/i.test(scheme[1])) return null;
  return url;
}

/**
 * Minimal renderer: fenced code blocks, then inline formatting, headings and
 * lists. Escapes HTML first so model output can never inject markup.
 *
 * Code blocks are held back as placeholders until the very end. Substituting
 * them earlier looks fine until a code block contains a blank line: the
 * paragraph split then cuts through the <pre>, and the fragments get escaped,
 * so the raw tags render as visible text.
 */
function renderMarkdown(src) {
  const blocks = [];
  // Pull fenced code out first so its contents skip inline formatting. The
  // highlighter returns already-escaped HTML, so the code is not escaped here.
  const text = src.replace(
    /```(\w+)?\n?([\s\S]*?)```/g,
    (_, lang, code) => {
      blocks.push(renderCode(code, lang));
      return `\u0000BLOCK${blocks.length - 1}\u0000`;
    },
  );

  // Inline code is parked in placeholders before anything else runs, for the
  // same reason the fenced blocks are: a URL inside backticks must stay inert
  // text, not become a link, and `<code>` contents must not be re-processed.
  const spans = [];
  const inline = (s) => {
    const parked = escapeHtml(s).replace(/`([^`\n]+)`/g, (_, code) => {
      spans.push(`<code>${code}</code>`);
      return `\u0000SPAN${spans.length - 1}\u0000`;
    });

    const linked = parked
      .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
      // Links last, and only over an allow-listed scheme. If the URL is
      // rejected the label survives as plain text, so nothing is lost but the
      // link itself. noopener matters: target=_blank hands the new tab a
      // window.opener reference back into this app otherwise.
      .replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (_, label, href) => {
        const url = safeUrl(href);
        return url
          ? `<a href="${url}" target="_blank" rel="noopener noreferrer nofollow">${label}</a>`
          : label;
      });

    return linked
      .split(/\u0000SPAN(\d+)\u0000/)
      .map((chunk, i) => (i % 2 ? spans[Number(chunk)] : chunk))
      .join('');
  };

  const ALONE = /^\s*\u0000BLOCK\d+\u0000\s*$/;

  const rendered = text.split(/\n{2,}/).map((block) => {
    // A lone code block must not be wrapped in <p>, which can't hold <pre>.
    if (ALONE.test(block)) return block.trim();

    const lines = block.split('\n');

    if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
      return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*[-*]\s+/, ''))}</li>`).join('')}</ul>`;
    }
    if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
      return `<ol>${lines
        .map((l) => `<li>${inline(l.replace(/^\s*\d+[.)]\s+/, ''))}</li>`)
        .join('')}</ol>`;
    }

    const heading = block.match(/^(#{1,3})\s+(.*)$/s);
    if (heading) {
      const level = heading[1].length;
      return `<h${level + 2}>${inline(heading[2])}</h${level + 2}>`;
    }

    return `<p>${lines.map(inline).join('<br>')}</p>`;
  }).join('');

  // Swap the real markup in last, once nothing can split or escape it.
  return rendered
    .split(/\u0000BLOCK(\d+)\u0000/)
    .map((chunk, i) => (i % 2 ? blocks[Number(chunk)] : chunk))
    .join('');
}

/* ---------------- Message rendering ---------------- */

/**
 * Adds a message. `html` is the rendered body; `raw` is the original text and
 * is what Copy puts on the clipboard, so it must never be the rendered markup.
 */
/** Keeps the "no messages yet" layout centred in sync with the conversation. */
function syncEmptyState() {
  els.messages.classList.toggle('is-empty', conversation.length === 0 && els.welcome.isConnected);
}

/* Very long turns collapse behind a Show more / Show less dropdown. The
   threshold is on the rendered height, and must stay larger than the
   --bubble-clamp clamp in theme.css so only sufficiently long turns get a
   toggle. */
const CLAMP_AT = 560;

/**
 * Collapses a very long turn behind a Show more / Show less toggle.
 *
 * Decided from the rendered height, so it works for prose, code and lists
 * alike, and never for short answers — the toggle adds chrome, so it only
 * exists where it hides something. Safe to call more than once (the second
 * call exits on the existing toggle), so the streaming bubble is simply
 * re-measured once its reply has settled.
 */
function compressLong(bubble) {
  const wrap = bubble.closest('.bubble-wrap');
  if (!wrap || wrap.querySelector('.bubble-toggle')) return;
  if (bubble.scrollHeight <= CLAMP_AT) return;

  bubble.classList.add('clamped');

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'bubble-toggle';
  btn.setAttribute('aria-expanded', 'false');
  btn.innerHTML =
    '<span class="bt-label">Show more</span>' +
    '<svg class="bt-chevron" viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ' +
    'aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>';

  btn.addEventListener('click', () => {
    const expanding = bubble.classList.contains('clamped');
    // Keep the toggle where the eye left it: expanding/collapsing moves it, so
    // correct the scroll by the same delta instead of jumping to the bottom.
    const before = btn.getBoundingClientRect().top;
    bubble.classList.toggle('clamped', !expanding);
    btn.classList.toggle('expanded', expanding);
    btn.querySelector('.bt-label').textContent = expanding ? 'Show less' : 'Show more';
    btn.setAttribute('aria-expanded', String(expanding));
    els.messages.scrollTop += btn.getBoundingClientRect().top - before;
    syncLatestButton();
  });

  // Inside the wrap, below the bubble and above the meta row, so it lines up
  // with the bubble rather than spanning the full column.
  bubble.after(btn);
}

function addMessage(role, html, { meta, raw, images = [], reasoning = '' } = {}) {
  els.welcome.remove();
  syncEmptyState();

  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const body = document.createElement('div');
  body.className = 'message-body';
  body.innerHTML = `
    <div class="bubble-wrap">
      <div class="bubble">${reasoning ? thinkingHtml(reasoning) : ''}${html}</div>
    </div>
    <div class="meta-row">
      ${meta ? `<span class="meta">${escapeHtml(meta)}</span>` : ''}
      <span class="msg-actions"></span>
    </div>`;

  const bubble = body.querySelector('.bubble');

  // Thumbnails sit above the text rather than inside the markdown body, so a
  // reply that is nothing but an image still has a bubble to hang them on.
  if (images.length) {
    const strip = document.createElement('div');
    strip.className = 'msg-images';
    for (const image of images) {
      const img = document.createElement('img');
      img.src = image.url;
      img.alt = image.name;
      img.loading = 'lazy';
      strip.appendChild(img);
    }
    body.querySelector('.bubble-wrap').before(strip);
  }

  // Only the assistant carries an avatar. The user turn is identified by the
  // tint on its bubble, which keeps the conversation on one reading axis.
  if (role === 'assistant') {
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'AI';
    wrap.appendChild(avatar);
  }
  wrap.appendChild(body);
  els.messages.appendChild(wrap);

  // A turn that is still streaming in gets measured again once its reply has
  // settled (see send()'s finally), so this call is a no-op for the empty
  // assistant bubble and a real check for everything already rendered.
  compressLong(bubble);

  const actions = body.querySelector('.msg-actions');

  // The streamed assistant placeholder has no text yet — finish() adds its
  // Copy button once the reply exists. Adding one here would copy an empty
  // string (while still toasting "Copied"), which is exactly a "copy not
  // working" report.
  if (raw) {
    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', () => copyText(raw));
    actions.appendChild(copyBtn);
    // Speak belongs to the AI's side: the ask was to read the model's output
    // aloud, and a Speak button on your own message would spend synthesis on
    // text you just typed.
    if (role === 'assistant') addSpeakButton(actions, raw);
  }

  scrollToBottom({ smooth: true });
  // Content changed without necessarily moving scrollTop (a scrolled-up reader
  // stays put while the gap below them grows), so refresh the pill directly.
  syncLatestButton();
  return {
    wrap,
    bubble,
    actions,
    meta: body.querySelector('.meta'),
  };
}


/* ---------------- Model selector ---------------- */

/** Repaints the option list from the active provider's models. */
function renderModelMenu(query = '') {
  const models = currentProvider()?.models ?? [];
  const q = query.trim().toLowerCase();
  const matches = q ? models.filter((m) => m.toLowerCase().includes(q)) : models;

  menuOptions = matches;
  menuIndex = matches.length ? 0 : -1;

  if (!matches.length) {
    els.modelList.innerHTML = `<li class="mm-empty">${
      models.length ? 'No models match that search' : 'No models available yet'
    }</li>`;
  } else {
    els.modelList.innerHTML = matches
      .map(
        (m) =>
          `<li class="mm-option" role="option" data-model="${escapeHtml(m)}"
               aria-selected="${m === model}"><span>${escapeHtml(m)}</span></li>`,
      )
      .join('');
  }
  highlightOption();

  // Offer the raw text as a model id when it isn't already in the list.
  const exact = models.some((m) => m.toLowerCase() === q);
  const showCustom = Boolean(q) && !exact;
  els.modelCustom.hidden = !showCustom;
  if (showCustom) {
    els.modelUse.querySelector('span').textContent = els.modelSearch.value.trim();
  }
}

function highlightOption() {
  const items = els.modelList.querySelectorAll('.mm-option');
  items.forEach((n, i) => n.classList.toggle('active', i === menuIndex));
  items[menuIndex]?.scrollIntoView({ block: 'nearest' });
}

function openModelMenu() {
  els.modelMenu.hidden = false;
  els.modelTrigger.setAttribute('aria-expanded', 'true');
  els.modelSearch.value = '';
  renderModelMenu();
  els.modelSearch.focus();
}

function closeModelMenu() {
  els.modelMenu.hidden = true;
  els.modelTrigger.setAttribute('aria-expanded', 'false');
}

/** Commits a model id and refreshes everything that depends on it. */
function setModel(next) {
  model = next;
  els.modelCurrent.textContent = next || 'no model selected';
  updateTopbar();
  // Re-labels the attach button against the new model without re-probing.
  attachments.setContext(activeProviderId, next);
  persist();
}

function commitModel(next) {
  setModel(next);
  closeModelMenu();
  els.input.focus();
}

els.modelTrigger.addEventListener('click', () => {
  if (els.modelMenu.hidden) openModelMenu();
  else closeModelMenu();
});

els.modelSearch.addEventListener('input', () => renderModelMenu(els.modelSearch.value));

els.modelSearch.addEventListener('keydown', (e) => {
  const items = els.modelList.querySelectorAll('.mm-option');
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (!items.length) return;
    menuIndex =
      (menuIndex + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
    highlightOption();
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (!els.modelCustom.hidden) commitModel(els.modelSearch.value.trim());
    else if (items[menuIndex]) commitModel(items[menuIndex].dataset.model);
  } else if (e.key === 'Escape') {
    e.preventDefault();
    e.stopPropagation();
    closeModelMenu();
    els.modelTrigger.focus();
  }
});

els.modelList.addEventListener('click', (e) => {
  const item = e.target.closest('.mm-option');
  if (item) commitModel(item.dataset.model);
});

els.modelUse.addEventListener('click', () => commitModel(els.modelSearch.value.trim()));

document.addEventListener('click', (e) => {
  if (!els.modelMenu.hidden && !els.modelSelect.contains(e.target)) closeModelMenu();
});

/* ---------------- Providers ---------------- */

/** The composer's keyboard hint lives in the HTML; this keeps a clean copy. */
const COMPOSER_HINT_HTML = els.composerHint.innerHTML;

const currentProvider = () => providers.find((p) => p.id === activeProviderId);

/* ---------------- Picking a default model ---------------- */

// Name-shape signals only. No model id is hardcoded: which models exist depends
// entirely on the key, and a hardcoded id goes stale or 404s.
const MEDIA_ONLY = /(embed|tts|whisper|transcrib|moderation|rerank|robot|image|dall|lyria|audio|video|vision|omni|deep-research|computer-use|search)/;
const UNSTABLE = /(preview|experimental|exp-|[-.]beta|[-.]rc\d|alpha)/;

function scoreModel(id) {
  const s = id.toLowerCase();
  let score = 0;
  if (MEDIA_ONLY.test(s)) score -= 100; // can serve generateContent but won't chat
  if (UNSTABLE.test(s)) score -= 20;
  if (/latest|stable/.test(s)) score += 5; // aliases track the newest release
  if (/chat|instruct/.test(s)) score += 5;
  return score;
}

/**
 * Picks the best model from what the provider returned. Ties keep the server's
 * order, which for Ollama is most-recently-pulled first.
 */
function pickDefault(models) {
  if (!models.length) return null;
  let best = models[0];
  let bestScore = scoreModel(best);
  for (const model of models.slice(1)) {
    const score = scoreModel(model);
    if (score > bestScore) {
      best = model;
      bestScore = score;
    }
  }
  return best;
}

function updateTopbar() {
  const p = currentProvider();

  // Recolour the whole UI around the active provider.
  document.documentElement.dataset.provider = p?.id ?? '';

  els.chipProvider.textContent = p?.label ?? '—';
  els.chipModel.textContent = model || 'no model selected';

  // The hint line is guidance by default; only borrow it to warn about a
  // provider that can't answer without a key.
  const unready = p && !p.configured;
  els.composerBox.classList.toggle('warn', Boolean(unready));
  els.composerHint.innerHTML = unready
    ? `<span class="hint-warn">${escapeHtml(p.label)} needs an API key — add one in settings</span>`
    : COMPOSER_HINT_HTML;
}

function applyProvider(preferredModel) {
  const p = currentProvider();
  els.providerHint.textContent = p?.hint ?? '';
  els.providerHint.classList.toggle('warn', !!p && !p.configured);

  const models = p?.models ?? [];
  els.modelHint.textContent = models.length
    ? `${models.length} model${models.length > 1 ? 's' : ''} available`
    : p?.configured
      ? 'No models listed — search for one to type it manually'
      : 'Add an API key to load this provider’s models';

  // A remembered model is only honoured while the provider still offers it.
  // Models get retired or are never valid for a given key, and sending a stale
  // one is a guaranteed 404. With no list to check against (no key, or the
  // provider is down) the remembered value is kept so it can still be typed.
  const remembered = [preferredModel, model].filter(Boolean);
  const stillOffered = remembered.find((m) => !models.length || models.includes(m));
  const next = stillOffered ?? pickDefault(models) ?? remembered[0] ?? '';

  // Only complain when there was a list to contradict the remembered value.
  const stale =
    remembered[0] && models.length && !models.includes(remembered[0]) ? remembered[0] : null;

  setModel(next);
  if (stale && next !== stale) {
    notifyWarning(`${stale} isn’t available — using ${next}`);
  }

  // Dictation availability depends on the selected provider, so re-evaluate it.
  refreshMic();

  // Image support is per-provider and per-model; re-probe on provider change.
  attachments.setContext(p?.id ?? '', next);

  // Refresh the dropdown if it's already open, since the list just changed.
  if (!els.modelMenu.hidden) renderModelMenu(els.modelSearch.value);
}

async function loadProviders(saved) {
  const res = await api('/api/providers');
  if (!res.ok) return;
  const data = await res.json();
  providers = data.providers;

  // Each provider is a row in the popup rather than an <option>: it can carry
  // its state as visible text, which an <option> can only hint at in a string
  // the user has to parse.
  // Grouped, because "Local" and "an API key" are different kinds of thing and
  // a flat list with Ollama wedged between OpenAI and Anthropic hides which
  // ones run on this machine and cost nothing.
  //
  // The flag is about a missing credential, so it only belongs on a provider
  // that needs one. `configured` means "can be used", which for a local
  // provider means "has models" — so llama.cpp or Transformers with nothing
  // downloaded yet was being told it was missing a key it never needed, and a
  // row that says nothing at all explains nothing, so those say why they are
  // empty instead.
  const renderRow = (p) => `
    <button type="button" class="popover__item" role="option" data-id="${p.id}"
            aria-selected="${p.id === saved.provider}">
      <span class="popover__item-label">${escapeHtml(p.label)}</span>
      ${p.configured ? '' : `<span class="popover__flag">${p.local ? 'no models' : 'no key'}</span>`}
    </button>`;

  const local = providers.filter((p) => p.local);
  const cloud = providers.filter((p) => !p.local);
  els.providerList.innerHTML =
    local.map(renderRow).join('') +
    (cloud.length
      ? `<div class="popover__group">Cloud &amp; API keys</div>` +
        cloud.map(renderRow).join('')
      : '');

  activeProviderId = providers.some((p) => p.id === saved.provider) ? saved.provider : 'ollama';
  paintProviderSelection();

  // Dictation engines are read separately from the provider list, because a
  // model in ./models is a perfectly good engine with no provider entry at all.
  // Awaited so the mic is correct as soon as the page is usable.
  await refreshSttEngines();

  applyProvider(saved.model);
}

/** Marks the active row without rebuilding the list. */
function paintProviderSelection() {
  for (const item of els.providerList.querySelectorAll('.popover__item')) {
    item.setAttribute('aria-selected', String(item.dataset.id === activeProviderId));
  }
}

/** Adopts a provider and everything that depends on it. */
function setProvider(id) {
  if (!providers.some((p) => p.id === id) || id === activeProviderId) return;
  activeProviderId = id;
  paintProviderSelection();
  applyProvider();
  persist();
}

/* ---------------- Dictation (speech-to-text) ---------------- */

/**
 * Recording state. Only one recording can exist at a time, so the pieces live
 * at module scope rather than in the click handler.
 */
let recorder = null;
let recorderStream = null;
let recorderChunks = [];
let recorderTimer = null;
let recorderStartedAt = 0;
/** Set while an upload is in flight so the button can't be double-fired. */
let transcribing = false;

/** Longest single recording, to keep uploads (and provider bills) bounded. */
const MAX_RECORDING_MS = 120_000;

/**
 * Speech-to-text engines this machine can actually use right now.
 *
 * @typedef {{kind:'cloud', id:string, label:string, ready:boolean, hint:string}
 *          |{kind:'local', repo:string, label:string, file:string, ready:boolean, hint:string}} SttEngine
 *
 * There is deliberately no picker here. Which model transcribes your voice is
 * set once, in Settings -> Your models, and this page just follows it.
 */

/** @type {SttEngine[]} */
let sttEngines = [];

/**
 * Re-reads which engines exist. Called on load and whenever models change, so
 * dictation is never stale about what has been downloaded since.
 */
async function refreshSttEngines() {
  try {
    const data = await apiJson('/api/dictation');
    const engines = [];
    for (const p of data.providers ?? []) {
      const ready = Boolean(p.configured ?? p.hasKey);
      engines.push({
        kind: 'cloud',
        id: p.id,
        label: p.label,
        ready,
        hint: ready ? 'configured' : 'needs an API key',
      });
    }
    for (const m of data.local ?? []) {
      engines.push({
        kind: 'local',
        repo: m.repo,
        label: m.label,
        file: m.file,
        // Which bundled runner the server says will handle it. Carried through
        // because it decides the upload's audio format — whisper-cli cannot read
        // a browser recording, the two Python runners can.
        driver: m.driver,
        ready: Boolean(m.ready),
        hint: m.ready ? 'downloaded and runnable here' : m.reason || 'not runnable',
      });
    }
    sttEngines = engines;

    // The assignment lives in Settings -> Your models. Fall back to the first
    // ready engine when nothing is assigned, so dictation still works for
    // someone who never visits that page.
    assignedSttId = data.assigned?.stt ?? '';
    refreshMic();
    window.dispatchEvent(new CustomEvent('stt-assigned'));
  } catch {
    sttEngines = [];
  }
}

function sttEngineId(e) {
  if (typeof e === 'string') return e;
  return e?.kind === 'local' ? `local:${e.repo}` : `cloud:${e?.id}`;
}

/** The speech-to-text model chosen in Settings -> Your models. */
let assignedSttId = '';

/** True when the engine in use was inferred rather than assigned. */
let sttUsingFallback = false;
/** Set when an assignment exists but its model cannot run right now. */
let sttAssignedBroken = false;

/**
 * The engine dictation will use.
 *
 * The assignment in Settings wins. When there is none, a model is still used —
 * refusing to dictate because nobody filled in a dropdown would be worse — but
 * the fact is recorded so the mic can say so. Silently ignoring an explicit
 * "not set" is the thing this whole change was meant to remove.
 */
function sttEngine() {
  sttAssignedBroken = false;
  if (assignedSttId) {
    const assignedEntry = sttEngines.find((e) => sttEngineId(e) === assignedSttId);
    const hit = sttEngines.find((e) => sttEngineId(e) === assignedSttId && e.ready);
    if (hit) {
      sttUsingFallback = false;
      return hit;
    }
    // An assignment that exists but cannot run is a different situation from no
    // assignment at all, and saying "no model is assigned" when one plainly is
    // sends the user to fix something that is not broken.
    if (assignedEntry) sttAssignedBroken = true;
  }
  sttUsingFallback = true;
  // Prefer a configured cloud provider, because that is the path that was
  // always available; fall back to whatever local model can run.
  return (
    sttEngines.find((e) => e.kind === 'cloud' && e.ready) ??
    sttEngines.find((e) => e.kind === 'local' && e.ready) ??
    null
  );
}

/**
 * The provider used for dictation. Prefers the one currently selected for
 * chat when it can transcribe, so the key you already have is the one used.
 *
 * Only consulted when a cloud engine was picked; local transcription goes
 * through /api/infer/stt instead.
 */
function sttProvider() {
  const active = currentProvider();
  if (active?.stt && active.configured && active.sttModels?.length) return active;
  return providers.find((p) => p.stt && p.configured && p.sttModels?.length) ?? null;
}

function setMicState(state) {
  els.mic.classList.toggle('recording', state === 'recording');
  els.mic.classList.toggle('busy', state === 'busy');
  els.mic.setAttribute('aria-pressed', state === 'recording' ? 'true' : 'false');
  els.mic.disabled = state === 'busy';
}

/**
 * Shows the mic when anything can transcribe.
 *
 * This used to hide itself unless a *cloud* provider was configured, so on a
 * machine with no API keys the mic simply was not there — even with a
 * perfectly good Whisper sitting in ./models. A downloaded model counts now.
 */
function refreshMic() {
  const e = sttEngine();
  els.mic.hidden = !e;
  if (!e) {
    els.mic.title = sttEngines.length
      ? 'No speech-to-text engine is ready'
      : 'No speech-to-text engine available';
    return;
  }
  const name = e.kind === 'local' ? `local ${e.label}` : e.label;
  if (sttAssignedBroken) {
    const why = sttEngines.find((x) => sttEngineId(x) === assignedSttId)?.hint ?? 'not runnable here';
    els.mic.title =
      `Record a message — using ${name} instead, because the model assigned in ` +
      `Settings is ${why}. Fix or change it under "What each job uses".`;
    return;
  }
  els.mic.title = sttUsingFallback
    ? `Record a message — using ${name}, because no model is assigned. ` +
      'Set one in Settings, Your models.'
    : `Record a message (transcribed by ${name})`;
}

function stopRecorderTracks() {
  recorderStream?.getTracks().forEach((t) => t.stop());
  recorderStream = null;
}

function formatElapsed(ms) {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

async function startRecording() {
  if (recorder || transcribing) return;

  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
    notifyError('Recording is not supported in this browser');
    return;
  }

  const engine = sttEngine();
  if (!engine) {
    notifyError(
      sttEngines.length
        ? 'No speech-to-text engine is ready'
        : 'No speech-to-text engine available',
    );
    return;
  }

  try {
    recorderStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    // Denied permission and missing hardware are both user-actionable.
    notifyError(
      err?.name === 'NotAllowedError' ? 'Microphone permission denied' : 'No microphone available',
    );
    return;
  }

  // Pick a container the provider recognises. Most browsers record webm/opus;
  // Safari gives mp4/aac, which the transcription APIs also accept.
  const mimeType = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/ogg;codecs=opus',
  ].find((t) => MediaRecorder.isTypeSupported?.(t));

  try {
    recorder = mimeType ? new MediaRecorder(recorderStream, { mimeType }) : new MediaRecorder(recorderStream);
  } catch {
    stopRecorderTracks();
    notifyError('This browser cannot record audio');
    return;
  }

  recorderChunks = [];
  recorder.addEventListener('dataavailable', (e) => {
    if (e.data?.size) recorderChunks.push(e.data);
  });
  recorder.addEventListener('stop', () => void finishRecording(engine));
  recorder.start();

  recorderStartedAt = Date.now();
  recorderTimer = setInterval(() => {
    const elapsed = Date.now() - recorderStartedAt;
    els.micTime.textContent = formatElapsed(elapsed);
    if (elapsed >= MAX_RECORDING_MS) stopRecording();
  }, 250);

  setMicState('recording');
  setStatus('recording');
}

function stopRecording() {
  clearInterval(recorderTimer);
  recorderTimer = null;
  els.micTime.textContent = '';
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  recorder = null;
  stopRecorderTracks();
  setStatus('ready');
}

async function finishRecording(engine) {
  setMicState('busy');
  const blob = new Blob(recorderChunks, { type: recorder?.mimeType || 'audio/webm' });
  recorderChunks = [];

  if (blob.size === 0) {
    setMicState('idle');
    notifyWarning('Nothing was recorded');
    return;
  }

  transcribing = true;
  setStatus('transcribing');

  try {
    let data;
    if (engine.kind === 'local') {
      // Local path: stage the recording, then run it through the same runner
      // the Settings page uses, with the chosen model named explicitly.
      //
      // `engine.driver` is which bundled runner the server says will handle
      // this model. whisper-cli reads wav/flac/mp3/ogg and neither webm/opus
      // nor m4a, which is what MediaRecorder actually produces, so that one
      // runner needs the recording decoded to 16 kHz WAV on the way up. The
      // two Python runners want the original bytes.
      const { audioForUpload } = await import('/audio.js');
      const upload = await audioForUpload(blob, engine.driver);
      const upRes = await fetch('/api/infer/upload', {
        method: 'POST',
        headers: { 'content-type': upload.type || 'application/octet-stream' },
        body: upload,
      });
      const uploaded = await upRes.json().catch(() => ({}));
      if (!upRes.ok) throw new Error(uploaded.error ?? 'Could not stage the recording.');
      data = await apiInfer('/api/infer/stt', {
        repo: engine.repo,
        file: engine.file,
        audio: uploaded.file,
        device: 'auto',
      });
    } else {
      const params = new URLSearchParams({ provider: engine.id });
      const res = await fetch(`/api/transcribe?${params}`, {
        method: 'POST',
        headers: { 'content-type': blob.type || 'application/octet-stream' },
        body: blob,
      });
      data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
    }

    // Land the text in the box rather than sending it: dictation is for
    // composing a message, and the user may want to edit before sending.
    if (data.text) {
      const existing = els.input.value.trim();
      els.input.value = existing ? `${existing} ${data.text}` : data.text;
      autosize();
      els.input.focus();
      notifySuccess('Transcribed — edit or send as-is');
    } else {
      notifyWarning('No speech detected');
    }
  } catch (err) {
    notifyError(err.message);
  } finally {
    transcribing = false;
    setMicState('idle');
    setStatus('ready');
  }
}

els.mic.addEventListener('click', () => {
  if (transcribing) return;
  if (recorder) stopRecording();
  else void startRecording();
});

/* ---------------- Smooth streaming ---------------- */

/**
 * Streamed text is painted on a fixed cadence rather than as each chunk lands.
 *
 * Providers batch unpredictably: a burst of a dozen tokens, a pause, then a
 * wall of text. Painting on arrival faithfully reproduces that raggedness, so a
 * reply can look smooth for one model and stuttery for the next. Coalescing
 * into ~30fps steps makes the reveal even, and costs nothing: the text still
 * appears as fast as the model produces it, just in steady increments rather
 * than lumps.
 *
 * Re-rendering per tick is deliberately left as a full markdown re-parse. It
 * measured at ~0.09ms per delta (83ms across a 900-delta reply), which is not a
 * bottleneck, and appending only the new text would risk splitting a markdown
 * construct that is still mid-token — a half-written `**bold` renders as
 * literal asterisks and then pops.
 */
const STREAM_PAINT_MS = 32;

/**
 * Streams in flight, keyed by chat id.
 *
 * One per chat, and independent of which chat is on screen — that is what
 * stops switching conversations from killing an answer. The views ask
 * `activeStream()` for the one they should reflect, so the composer's busy
 * state and the Stop button always describe the conversation on screen.
 *
 * A stream carries everything its answer needs: the chat and the message
 * array it belongs to, its bubble, and the paint state. With all of that on
 * the object, an answer keeps working after the reader walks away.
 */
const streams = new Map();

/** The stream belonging to the conversation currently open, if any. */
function activeStream() {
  return activeId ? (streams.get(activeId) ?? null) : null;
}

/** The text a stream has received so far, for painting. */
const streamingAnswer = (s) => s.text;

/**
 * The scratchpad pane: thinking models' reasoning, above the answer.
 *
 * Rendered as plain HTML with a `data-open` attribute rather than a native
 * `<details>`, because the live bubble's innerHTML is replaced on every paint
 * and a `<details>` would slam shut under the reader mid-stream. The click
 * handler flips the attribute in place; for the live bubble it also records
 * the choice on the stream (s.thinkUser) so the next repaint honours it.
 */
function thinkingHtml(reasoning, { open = false } = {}) {
  return (
    `<div class="thinking"${open ? ' data-open="1"' : ''}>` +
    '<button type="button" class="thinking-toggle" aria-expanded="' +
    (open ? 'true' : 'false') +
    '">' +
    '<svg class="thinking-chevron" viewBox="0 0 24 24" width="12" height="12" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ' +
    'aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>' +
    '<span class="thinking-label">Thinking</span>' +
    '</button>' +
    `<div class="thinking-body">${renderMarkdown(reasoning)}</div>` +
    '</div>'
  );
}

/**
 * Whether the live thinking pane should be open.
 *
 * Open while the model is still thinking (reasoning and no answer yet, which
 * is the moment the pane is the only thing happening), closed as soon as the
 * answer starts — and never re-opened behind the reader's back. An explicit
 * click always wins.
 */
function thinkOpen(s) {
  if (s.thinkUser) return Boolean(s.thinkOpen);
  return Boolean(s.reasoning) && !s.text.trim();
}

/** The live bubble's HTML: thinking pane first, then the answer so far. */
function liveBubbleHtml(s) {
  const think = s.reasoning ? thinkingHtml(s.reasoning, { open: thinkOpen(s) }) : '';
  return think + renderMarkdown(streamingAnswer(s));
}

/**
 * Keeps the canvas honest: busy iff the open chat has an answer coming.
 *
 * A background answer is deliberately not allowed to dim the composer — you
 * can send into the chat you are looking at while another one finishes.
 */
function syncBusy() {
  toggleBusy(Boolean(activeStream()));
}

/** Bubbles a background answer's progress into the conversation's own row. */
function refreshChatRow(chatId) {
  if (chatId !== activeId) return;
  const row = chats.find((c) => c.id === chatId);
  if (!row) return;
  row.updatedAt = Date.now();
  chats.sort((a, b) => b.updatedAt - a.updatedAt);
  renderHistory();
}

/** Paints a delta into the live bubble, when the answer is still on screen. */
function queuePaint(s, text) {
  s.paintQueue += text;
  if (s.paintTimer) return;
  const wait = Math.max(0, STREAM_PAINT_MS - (performance.now() - s.lastPaint));
  s.paintTimer = setTimeout(() => runPaint(s), wait);
}

function runPaint(s) {
  s.paintTimer = 0;
  s.lastPaint = performance.now();
  if (!s.paintQueue || !s.bubble?.isConnected) {
    s.paintQueue = '';
    return;
  }
  s.bubble.innerHTML = liveBubbleHtml(s);
  scrollToBottom();
  syncLatestButton();
}

/** Paints everything buffered right now and cancels any pending tick. */
function flushPaint(s) {
  if (s.paintTimer) {
    clearTimeout(s.paintTimer);
    s.paintTimer = 0;
  }
  if (!s.paintQueue || !s.bubble?.isConnected) {
    s.paintQueue = '';
    return;
  }
  s.lastPaint = performance.now();
  s.bubble.innerHTML = liveBubbleHtml(s);
  s.paintQueue = '';
}

/** ---------------- Streaming ---------------- */

async function send(text, { isRetry = false } = {}) {
  const provider = currentProvider();
  if (!provider) return;
  // One answer at a time per conversation; other chats may be streaming.
  if (activeStream()) return;

  // The working area only becomes a saved conversation once something is
  // actually sent, so an untouched page never leaves a phantom chat behind.
  if (!currentChat()) {
    conversation = createChat({
      provider: provider.id,
      model,
    }).conversation;
  }

  // Pin this request to the exact conversation it started with, and to the id
  // that conversation has. openChat swaps the module-level `conversation` when
  // you walk away mid-answer; the answer keeps going and lands in the chat it
  // belongs to, not the one that happened to be on screen when it finished.
  const convo = conversation;
  const chatId = activeId;

  if (!isRetry) {
    // Text files fold into the message text; images become content parts. With
    // nothing attached this returns the plain string, so text-only turns send
    // exactly the payload they always did.
    const content = attachments.buildContent(text);
    // Render the folded text, not just what was typed, so a file attachment is
    // visible in the transcript rather than silently appended to the payload.
    const shown = attachments.foldedText(text);
    conversation.push({ role: 'user', content });
    addMessage('user', renderMarkdown(shown), {
      raw: shown,
      images: attachments.items().filter((i) => i.kind === 'image'),
    });
    attachments.clear();
    els.input.value = '';
    autosize();
    persist();
    // Sending is an explicit "show me the answer": pull to the newest turn
    // even if the reader had scrolled up, so the reply streams into view.
    // (addMessage's own scroll respects the reading position and would not.)
    scrollToBottom({ force: true });
  }

  // Tools are per-message: snapshot the current selection so a mid-stream
  // change to the chips can't alter this request's permissions.
  const requestTools = selectedTools.slice();
  const payload = {
    provider: provider.id,
    model,
    messages: conversation,
    tools: requestTools,
  };

  const { bubble, actions, meta } = addMessage('assistant', '');

  // Declared before the stream starts and outside the try, so the abort path
  // can reuse whatever arrived.
  let answer = '';
  // The model's scratchpad, when it streams one: kept apart from the answer so
  // the saved turn, the copy button and the speaking bar all carry the answer
  // alone.
  let reasoning = '';
  // Tool-call activity lines, in arrival order, so each result lands on the
  // line its call created.
  const pendingToolLines = [];

  const started = performance.now();
  // Everything this answer needs, on one object, registered before the request
  // goes out — so a Stop click arriving mid-flight finds it.
  const s = {
    chatId,
    convo,
    bubble,
    actions,
    meta,
    text: answer,
    reasoning,
    ctl: new AbortController(),
    paintQueue: '',
    paintTimer: 0,
    lastPaint: 0,
  };
  streams.set(chatId, s);
  bubble.classList.add('cursor');
  toggleBusy(true);

  try {
    const res = await api('/api/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal: s.ctl.signal,
    });

    if (!res.ok || !res.body) {
      const { error } = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      throw new Error(error);
    }

    // Named SSE events, so parse by hand rather than using EventSource.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buffer.search(/\r?\n\r?\n/)) !== -1) {
        const raw = buffer.slice(0, sep);
        buffer = buffer.slice(sep + (buffer[sep] === '\r' ? 4 : 2));

        let event = 'message';
        let data = '';
        for (const line of raw.split(/\r?\n/)) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) data += line.slice(5).trim();
        }
        if (!data) continue;

        const parsed = JSON.parse(data);
        if (event === 'delta') {
          answer += parsed.text;
          s.text = answer;
          queuePaint(s, parsed.text);
        } else if (event === 'reasoning') {
          // Scratchpad in its own lane: painted into the thinking pane, never
          // appended to `answer`.
          reasoning += parsed.text;
          s.reasoning = reasoning;
          queuePaint(s, parsed.text);
        } else if (event === 'tool_call') {
          pendingToolLines.push(addSideLine(bubble, `${toolMeta(parsed.name).label} — ${parsed.args}`, 'tool-call'));
        } else if (event === 'tool_result') {
          const line = pendingToolLines.shift();
          if (line) {
            line.textContent += parsed.ok
              ? ' ✓ completed'
              : ` ✗ failed — ${String(parsed.detail ?? '').replace(/^Error: /, '')}`;
            if (!parsed.ok) line.classList.add('tool-error');
          }
        } else if (event === 'notice') {
          addSideLine(bubble, parsed.message, 'notice-line');
        } else if (event === 'error') {
          throw new Error(parsed.message);
        } else if (event === 'done') {
          // Paint whatever is still buffered before finishing, so the saved
          // transcript and the copy button both include the whole reply.
          flushPaint(s);
          finish(s, started);
          // The reply landed, so the per-message selection has served its
          // purpose; a failure path below deliberately keeps it for Retry.
          releaseTools(requestTools);
          return;
        }
      }
    }

    flushPaint(s);
    finish(s, started);
    releaseTools(requestTools);
  } catch (err) {
    flushPaint(s);
    const visible = s.bubble.isConnected;
    if (err.name === 'AbortError') {
      s.bubble.classList.remove('cursor');
      if (visible) {
        s.bubble.innerHTML += `<p class="error"><em>stopped</em></p>`;
        syncEmptyState();
        if (s.meta) s.meta.textContent = `stopped · ${answer.length} chars`;
      }
      // Keep whatever arrived: a stopped answer is still the user's text, and
      // it belongs to this chat whether or not anyone is watching it.
      if (answer.trim()) {
        s.convo.push(
          s.reasoning
            ? { role: 'assistant', content: answer, reasoning: s.reasoning }
            : { role: 'assistant', content: answer },
        );
      }
      persistChat(s.chatId, s.convo);
      refreshChatRow(s.chatId);
    } else {
      s.bubble.classList.remove('cursor');
      if (visible) {
        s.bubble.innerHTML = `<p class="error">${escapeHtml(err.message)}</p>`;
        setStatus(err.message, true);
      }
      notifyError(err.message);
      refreshChatRow(s.chatId);
    }
  } finally {
    if (s.paintTimer) {
      clearTimeout(s.paintTimer);
      s.paintTimer = 0;
    }
    s.paintQueue = '';
    s.bubble.classList.remove('cursor');
    // The full reply is now in the bubble (finish/appended above, or a short
    // error), so measure it: a very long answer collapses behind the dropdown.
    if (s.bubble.isConnected) compressLong(s.bubble);
    streams.delete(s.chatId);
    // Coming home to a finished answer should show it: the live bubble went
    // with the old view, so the chat is repainted from what was saved.
    if (s.chatId === activeId && !s.bubble.isConnected) {
      conversation = s.convo;
      renderConversation();
    }
    syncBusy();
  }
}

/**
 * Records the finished answer and shows timing plus copy/retry actions.
 *
 * Takes the stream rather than loose parts because the answer can land in a
 * chat that is not on screen — then it is saved and a progress dot left in
 * the history row, and the next time that chat is opened the full reply is
 * there.
 */
function finish(s, started) {
  const answer = s.text;
  const { meta, actions, convo, bubble } = s;
  const visible = bubble.isConnected;
  if (!answer.trim()) {
    if (visible && meta) meta.textContent = 'no response';
    return;
  }
  // Append to the conversation this request belongs to, and save that
  // conversation by id — not whatever chat happens to be open when the last
  // token lands. The scratchpad rides along so reopening the chat can show it
  // (collapsed); it is never sent back to a provider.
  convo.push(
    s.reasoning
      ? { role: 'assistant', content: answer, reasoning: s.reasoning }
      : { role: 'assistant', content: answer },
  );
  persistChat(s.chatId, convo);
  if (!visible) {
    refreshChatRow(s.chatId);
    return;
  }
  syncEmptyState();

  if (meta) {
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    const tokens = Math.round(answer.length / 4);
    meta.textContent = `${tokens} est. tokens · ${seconds}s`;
  }

  // Only the rendered text, so copying gives back what the user sees.
  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  copyBtn.textContent = 'Copy';
  copyBtn.addEventListener('click', () => copyText(answer));
  actions.appendChild(copyBtn);
  addSpeakButton(actions, answer);

  const retry = document.createElement('button');
  retry.type = 'button';
  retry.textContent = 'Retry';
  retry.addEventListener('click', regenerate);
  actions.appendChild(retry);
}

/**
 * Re-attaches an answer that kept running while another chat was open.
 *
 * `renderConversation()` rebuilt the view from saved turns, so the bubble the
 * stream was painting into is detached. This points the stream at a new bubble
 * and fills it with everything received so far; from here the deltas land on
 * screen again.
 */
function reattachStream(chatId) {
  const s = streams.get(chatId);
  if (!s) return;
  // The stored array is a sanitized copy; the stream holds the live one. Adopt
  // it, so the answer that lands next goes into the same array this view draws
  // from — two arrays for one chat is how a finished reply goes missing.
  conversation = s.convo;
  const { bubble, actions, meta } = addMessage(
    'assistant',
    s.text || s.reasoning ? liveBubbleHtml(s) : '',
  );
  bubble.classList.add('cursor');
  s.bubble = bubble;
  s.actions = actions;
  s.meta = meta;
  flushPaint(s);
  scrollToBottom({ force: true });
}

/** Drops the last assistant turn and asks again with the same settings. */
function regenerate() {
  if (activeStream()) return;
  if (conversation.at(-1)?.role === 'assistant') conversation.pop();

  const nodes = els.messages.querySelectorAll('.message.assistant');
  nodes[nodes.length - 1]?.remove();
  if (!conversation.length) return;

  // The last user turn may be content parts rather than a string, and send()
  // expects the text on its own; the attachments are re-sent from the stored
  // message rather than re-read from the composer, which is already empty.
  const last = conversation.at(-1);
  if (last?.role !== 'user') return;
  const text =
    typeof last.content === 'string'
      ? last.content
      : last.content.filter((p) => p.type === 'text').map((p) => p.text).join(' ');
  void send(text, { isRetry: true });
}

/** Download extension per canonical language (see highlight.js LANGS);
 *  anything else saves as .txt. */
const CODE_EXT = {
  javascript: 'js', typescript: 'ts', python: 'py', shell: 'sh', json: 'json',
  css: 'css', html: 'html', sql: 'sql', go: 'go', rust: 'rs', java: 'java',
  glsl: 'glsl', ruby: 'rb', php: 'php', graphql: 'gql', powershell: 'ps1',
  lua: 'lua', r: 'r', perl: 'pl', elixir: 'ex', haskell: 'hs', clojure: 'clj',
  swift: 'swift', dart: 'dart', zig: 'zig', julia: 'jl', fsharp: 'fs',
  erlang: 'erl', solidity: 'sol', objectivec: 'm', latex: 'tex', asm: 'asm',
  protobuf: 'proto', crystal: 'cr', nim: 'nim', ocaml: 'ml', diff: 'diff',
  markdown: 'md',
};
/** Languages whose conventional filename has no extension. */
const CODE_SPECIAL = { dockerfile: 'Dockerfile', makefile: 'Makefile' };

/** Saves a code block as a file named for its language. */
function downloadCode(text, lang) {
  const name = CODE_SPECIAL[lang] ?? `snippet-${lang || 'text'}.${CODE_EXT[lang] ?? 'txt'}`;
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  notifySuccess(`Saved ${name}`);
}

/** Briefly confirms a per-block Copy on the button itself. */
function flashCodeButton(btn, label) {
  const prev = btn.textContent;
  btn.textContent = label;
  clearTimeout(btn._flash);
  btn._flash = setTimeout(() => { btn.textContent = prev; }, 1200);
}

// Tapping a bubble toggles its Copy/Retry row: touch has no hover, so the
// tap is the explicit reveal there (and on desktop it pins the row open).
// Single selection — opening one message closes the others.
els.messages.addEventListener('click', (e) => {
  const bubble = e.target.closest('.bubble');
  if (!bubble) return;
  // Controls handle themselves: buttons, links, and the clamp toggle must not
  // flip the row, and a click that selected text is reading, not tapping.
  if (e.target.closest('button, a, input, textarea, select')) return;
  if (window.getSelection()?.toString()) return;
  const wrap = bubble.closest('.message');
  if (!wrap) return;
  const wasOpen = wrap.classList.contains('show-actions');
  for (const m of els.messages.querySelectorAll('.message.show-actions')) {
    m.classList.remove('show-actions');
  }
  if (!wasOpen) wrap.classList.add('show-actions');
});

// Tapping anywhere outside a message dismisses the open row.
document.addEventListener('click', (e) => {
  if (e.target.closest('.message')) return;
  for (const m of els.messages.querySelectorAll('.message.show-actions')) {
    m.classList.remove('show-actions');
  }
});

// Per-code-block Copy / Download. Delegated on #messages so the buttons
// survive streaming (bubble.innerHTML is replaced per delta) and replay.
els.messages.addEventListener('click', (e) => {
  const btn = e.target.closest('.code-copy, .code-dl');
  if (!btn) return;
  const code = btn.closest('.codeblock')?.querySelector('pre code');
  if (!code) return;
  // textContent is the raw code: highlighting only added spans, and the header
  // buttons live outside the <pre>, so nothing else leaks in.
  if (btn.classList.contains('code-copy')) {
    copyText(code.textContent);
    flashCodeButton(btn, 'Copied');
  } else {
    downloadCode(code.textContent, code.dataset.lang || '');
  }
});

// The thinking pane's toggle. Delegated for the same reason as the code
// buttons, plus one more: while a reply streams, the live bubble's innerHTML
// is rebuilt on every paint. Flipping the attribute here is instant; if the
// pane belongs to a live stream, the choice is recorded on the stream so the
// next repaint honours it instead of reverting to the automatic state.
els.messages.addEventListener('click', (e) => {
  const btn = e.target.closest('.thinking-toggle');
  if (!btn) return;
  const box = btn.closest('.thinking');
  const open = box.getAttribute('data-open') !== '1';
  box.setAttribute('data-open', open ? '1' : '');
  btn.setAttribute('aria-expanded', String(open));
  const bubble = box.closest('.bubble');
  for (const s of streams.values()) {
    if (s.bubble === bubble) {
      s.thinkUser = true;
      s.thinkOpen = open;
      break;
    }
  }
});

async function copyText(text) {
  // The async Clipboard API needs a secure context (localhost counts, a LAN
  // address does not) plus a permission grant, so it is tried first and the
  // legacy execCommand path covers everywhere else instead of failing.
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      notifySuccess('Copied');
      return;
    } catch {
      /* fall through to the legacy path */
    }
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;';
    document.body.appendChild(ta);
    ta.select();
    ta.setSelectionRange(0, ta.value.length);
    const done = document.execCommand('copy');
    ta.remove();
    if (!done) throw new Error('execCommand returned false');
    notifySuccess('Copied');
  } catch {
    notifyError('Could not copy');
  }
}

/* ---------------- Reading a reply aloud ----------------

   Every AI reply carries a Speak button. The text goes through the same local
   voice pipeline the studio's voice panel uses — POST /api/infer/tts, the
   voice and speed that panel defaults to, the assigned model — so this adds no
   server surface and inherits its honest errors verbatim. Which runner speaks,
   Kokoro or the transformers one, is decided by the weights and not here. */

/** The reply as speech: markdown noise removed and code skipped. A fenced
 *  block is dropped whole (nobody wants keywords and braces read aloud), links
 *  keep their label, images and bare URLs are dropped — spoken, those are
 *  punctuation soup. Italics are left alone: the underscore rule would eat
 *  words like `snake_case_names` from the inside. */
function speakableText(md) {
  return String(md)
    .replace(/```[\s\S]*?(?:```|$)/g, ' ')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/<https?:\/\/[^>\s]+>/g, ' ')
    .replace(/\bhttps?:\/\/\S+/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s*>\s?/gm, '')
    .replace(/^\s*(?:[-*+]|\d+[.)])\s+/gm, '')
    .replace(/\s*\|\s*/g, ' ')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

/** The clip currently playing, if any, and the button that started it. */
let speakAudio = null;
let speakButton = null;

/* ---------------- The card that says what is being read ----------------

   Speech is invisible: the only sign a reply is being read aloud is that it
   sounds different. This is the sign — it drops out from under the topbar
   while a voice is being made and while it plays, and retreats back up when it
   is done. It carries only what the run itself reported: which model is
   speaking, the device the runner landed on, and the real position in the
   clip. */

/** The card currently up, if any. */
let speaker = null;

const SPEAKER_ICON = {
  // Filled shapes: a triangle and two bars read as icons at 16px.
  pause: { body: '<path d="M8 5h3v14H8zM13 5h3v14h-3z"/>', paint: 'fill="currentColor"' },
  play: { body: '<path d="M8.5 5.4v13.2L19 12z"/>', paint: 'fill="currentColor"' },
  // A cross is a stroke, not a filled shape: drawn with fill alone it has no
  // area at all and the button comes out blank.
  cancel: {
    body: '<path d="M6 6l12 12M18 6L6 18"/>',
    paint: 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"',
  },
};

/** Builds the card and puts it up. `title` and `sub` are text, never markup:
 *  the title is a slice of the model's own reply. */
function openSpeaker(title, sub) {
  // A clip replacing a clip must not leave two cards stacked in one slot.
  for (const old of document.querySelectorAll('.speaker-slot')) old.remove();

  const slot = document.createElement('div');
  slot.className = 'speaker-slot';

  const card = document.createElement('div');
  card.className = 'speaker';
  card.setAttribute('role', 'group');
  card.setAttribute('aria-label', 'Reading a reply aloud');

  const art = document.createElement('span');
  art.className = 'speaker__art';
  art.setAttribute('aria-hidden', 'true');
  art.innerHTML =
    '<svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H3v6h3l5 4z"/><path d="M15.4 8.6a4.8 4.8 0 0 1 0 6.8"/><path d="M18.3 6a8.7 8.7 0 0 1 0 12"/></svg>';

  const meta = document.createElement('div');
  meta.className = 'speaker__meta';
  const titleEl = document.createElement('p');
  titleEl.className = 'speaker__title';
  titleEl.textContent = title;
  const subEl = document.createElement('p');
  subEl.className = 'speaker__sub';
  subEl.textContent = sub;
  const bar = document.createElement('span');
  bar.className = 'bar speaker__line';
  const fill = document.createElement('i');
  bar.appendChild(fill);
  meta.append(titleEl, subEl, bar);

  const eq = document.createElement('span');
  eq.className = 'speaker__eq';
  eq.setAttribute('aria-hidden', 'true');
  for (let i = 0; i < 4; i += 1) eq.appendChild(document.createElement('b'));

  const btn = document.createElement('button');
  btn.type = 'button'; // nothing here submits anything, and nothing here should start
  btn.className = 'speaker__btn';
  const icon = document.createElement('span');
  btn.appendChild(icon);

  card.append(art, meta, eq, btn);
  slot.appendChild(card);
  // On the window, not in the composer: it belongs to the top of the screen,
  // out of the conversation's way, where the composer's growth cannot move it.
  document.body.appendChild(slot);

  // One frame with the offset applied, then clear it, so the transition has a
  // start value to animate from. The timer is the safety net: a backgrounded
  // tab may not run rAF at all, and a card stuck at opacity 0 while audio
  // plays would be worse than one that skipped its entrance.
  const reveal = () => {
    if (speaker?.card === card) card.classList.add('is-in');
  };
  requestAnimationFrame(reveal);
  setTimeout(reveal, 120);

  let mode = 'waiting'; // 'waiting' = making the voice, 'clip' = playing one
  let cancel = null; // what the button does while there is no clip yet

  const setIcon = (name, label) => {
    const { body, paint } = SPEAKER_ICON[name];
    icon.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" ${paint}>${body}</svg>`;
    btn.setAttribute('aria-label', label);
    btn.title = label;
  };
  setIcon('cancel', 'Stop making this voice');

  btn.addEventListener('click', () => {
    if (mode === 'waiting') {
      cancel?.();
      return;
    }
    if (!speakAudio) return;
    if (speakAudio.paused) speakAudio.play().catch(() => {});
    else speakAudio.pause();
  });

  speaker = {
    card,
    /** The button's job before there is a clip: stop making it. */
    onCancel(fn) {
      cancel = fn;
    },
    setSub(text) {
      subEl.textContent = text;
    },
    /** No figure exists yet, so the line says "working" instead of inventing one. */
    setWaiting() {
      bar.classList.add('bar--live');
      fill.style.width = '';
    },
    /** A clip exists: the line becomes its real position, the bars come alive. */
    setPlaying(playing) {
      mode = 'clip';
      bar.classList.remove('bar--live');
      card.classList.toggle('is-playing', playing);
      setIcon(playing ? 'pause' : 'play', playing ? 'Pause reading' : 'Resume reading');
    },
    setProgress(fraction) {
      fill.style.width = `${(Math.max(0, Math.min(1, fraction)) * 100).toFixed(1)}%`;
    },
  };
  return speaker;
}

/** Takes the card down the way it came up. */
function closeSpeaker() {
  const up = speaker;
  speaker = null;
  if (!up) return;
  up.card.classList.remove('is-in');
  up.card.classList.add('is-out');
  const slot = up.card.parentElement;
  const remove = () => slot?.remove();
  up.card.addEventListener('transitionend', remove, { once: true });
  // transitionend never fires if the tab is backgrounded mid-transition, so
  // guarantee the removal — the same net the toasts use.
  setTimeout(remove, 400);
}

/** Silences the clip that is playing. The card is left alone: playClip takes
 *  the one that is already up rather than blinking a new one into its place. */
function stopAudio() {
  if (!speakAudio) return;
  speakAudio.pause();
  speakAudio.removeAttribute('src');
  speakAudio = null;
  if (speakButton) {
    speakButton.textContent = 'Speak';
    speakButton.disabled = false;
    speakButton = null;
  }
}

/** Stops playback and puts that button back to "Speak". */
function stopSpeaking() {
  stopAudio();
  closeSpeaker();
}

function playClip(btn, src, info) {
  stopAudio();
  const audio = new Audio(src);
  speakAudio = audio;
  speakButton = btn;
  btn.textContent = 'Stop';
  // The card may already be up from the synthesis that produced this clip:
  // reusing it means no flicker between making the voice and playing it.
  const card =
    speaker ?? openSpeaker(info?.title ?? 'Reading the reply aloud', info?.sub ?? '');
  if (info?.sub) card.setSub(info.sub);
  card.setPlaying(true);
  // Real position, not a decorative sweep: the clip's own clock when the
  // browser knows it, and the runner's reported length until then.
  const progress = () => {
    const total =
      Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : info?.seconds ?? 0;
    if (total > 0) card.setProgress(audio.currentTime / total);
  };
  audio.addEventListener('timeupdate', progress);
  audio.addEventListener('play', () => card.setPlaying(true));
  audio.addEventListener('pause', () => card.setPlaying(false));
  const done = () => {
    if (speakAudio === audio) stopSpeaking();
  };
  audio.addEventListener('ended', done);
  audio.addEventListener('error', () => {
    done();
    notifyError('Could not play the audio.');
  });
  audio.play().catch(() => {
    // A browser may refuse playback with no gesture attached; the clip is
    // already on disk, so the button asks for one (Play) instead of failing.
    done();
    btn.textContent = 'Play';
  });
}

/** Runs an /api/infer/* stream and resolves its terminal event. */
async function inferStream(path, payload, signal) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error ?? `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.search(/\r?\n\r?\n/)) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + (buffer[sep] === '\r' ? 4 : 2));
      let event = 'message';
      let data = '';
      for (const line of raw.split(/\r?\n/)) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (!data) continue;
      let parsed;
      try { parsed = JSON.parse(data); } catch { continue; }
      if (event === 'error') throw new Error(parsed.message ?? 'The runner failed.');
      if (event === 'done') return parsed;
    }
  }
  throw new Error('The runner stopped without a result.');
}

async function speakReply(btn, text) {
  if (speakButton === btn && speakAudio) {
    stopSpeaking();
    return;
  }
  if (btn.dataset.clip) {
    playClip(btn, btn.dataset.clip, { title: btn.dataset.clipTitle, sub: btn.dataset.clipSub });
    return;
  }
  btn.disabled = true;
  btn.textContent = 'Saying…';
  let abort = null;
  try {
    const res = await fetch('/api/runnable?task=tts');
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
    const models = (data.tts ?? []).filter((m) => m.ready);
    const assigned = data.assigned?.tts;
    const model = models.find((m) => assigned === `local:${m.repo}`) ?? models[0];
    if (!model) {
      throw new Error('No voice model is runnable here — get one in Settings → Models.');
    }
    // The endpoint refuses over 20k; cut at a word edge first so a very long
    // reply is read as far as it goes instead of refused outright.
    const say = text.length > 19_000 ? `${text.slice(0, 19_000).replace(/\S+$/, '')}…` : text;
    // Up before the wait, not after it: making the voice is the slow half — a
    // long reply is seconds of synthesis — and until now the only sign of it
    // was a disabled button. The title is what is being read, so a card left
    // over a clip still says which reply it belongs to.
    const label = model.label ?? model.repo;
    const title = say.slice(0, 64);
    const card = openSpeaker(title, `${label} · making the voice…`);
    card.setWaiting();
    abort = new AbortController();
    card.onCancel(() => abort.abort());
    const out = await inferStream(
      '/api/infer/tts',
      { repo: model.repo, file: model.file, text: say, voice: 'af_heart', speed: 1, device: 'auto' },
      abort.signal,
    );
    // The runner answers {path, samples, seconds, device}; the serving route
    // wants the bare filename out of that path.
    const name = String(out?.path ?? '').split('/').pop();
    if (!name) throw new Error(out?.message ?? 'The voice runner finished without audio.');
    btn.dataset.clip = `/api/infer/out/${encodeURIComponent(name)}`;
    // The device is the runner's own report, not the request's wish — the
    // request says 'auto', and only the run knows where it landed.
    const sub = out?.device ? `${label} · on ${out.device}` : `${label} · speaking`;
    btn.dataset.clipTitle = title;
    btn.dataset.clipSub = sub;
    btn.disabled = false;
    playClip(btn, btn.dataset.clip, { title, sub, seconds: Number(out?.seconds) || 0 });
  } catch (err) {
    closeSpeaker();
    btn.disabled = false;
    btn.textContent = 'Speak';
    // The card's button stopping the synthesis is a decision, not a fault.
    if (abort?.signal.aborted) return;
    notifyError(err instanceof Error ? err.message : String(err));
  }
}

/** Adds Speak to an actions row. A reply that is nothing but code gets no
 *  button: after the stripping above there is nothing to read, and a dead
 *  button is worse than none. */
function addSpeakButton(actions, text) {
  const spoken = speakableText(text);
  if (!spoken) return;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.textContent = 'Speak';
  btn.title = 'Read this reply aloud';
  btn.addEventListener('click', () => speakReply(btn, spoken));
  actions.appendChild(btn);
}

/* ---------------- Tool selection (/tool) ---------------- */

/** Tools picked for the next send, by registry id. Per-message: a finished
 *  reply clears them, and a failed one keeps them so Retry can reuse them. */
let selectedTools = [];

/** Last /api/tools listing (id, label, icon, description, enabled) and when
 *  it was fetched — reused for reopen-and-pick and for activity-line labels. */
let toolsMeta = null;
let toolsMetaAt = 0;

/** The registry listing, or null while unreachable (the picker then opens
 *  with an error line rather than doing nothing). */
async function toolsList() {
  const now = Date.now();
  if (toolsMeta && now - toolsMetaAt < 60_000) return toolsMeta;
  try {
    const res = await api('/api/tools');
    if (res.ok) {
      const { tools } = await res.json();
      if (Array.isArray(tools)) {
        toolsMeta = tools;
        toolsMetaAt = now;
      }
    }
  } catch {
    /* keep whatever we had; the picker explains when there is nothing */
  }
  return toolsMeta;
}

/** Inline SVGs for registry icon ids, in the same hand-inlined stroke style
 *  as the rest of the app (constant strings; no user data enters this map). */
const TOOL_ICONS = {
  globe:
    '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>' +
    '<path d="M2 12h20"/>',
  'file-text':
    '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>' +
    '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>',
  book: '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
  wrench:
    '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 ' +
    '6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76Z"/>',
};

function toolIconSvg(icon, size = 13) {
  const inner = TOOL_ICONS[icon] ?? TOOL_ICONS.wrench;
  return (
    `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor"` +
    ` stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${inner}</svg>`
  );
}

/** Listing entry for a tool id; before the first fetch, the id stands in. */
function toolMeta(name) {
  return toolsMeta?.find((t) => t.name === name) ?? { name, label: name, icon: 'wrench' };
}

/** Paints the indicator chips above the textarea; hidden when none. */
function renderToolChips() {
  els.toolsBox.hidden = selectedTools.length === 0;
  els.toolsList.innerHTML = '';
  for (const name of selectedTools) {
    const meta = toolMeta(name);
    const chip = document.createElement('span');
    chip.className = 'tool-chip';
    chip.innerHTML = `${toolIconSvg(meta.icon)}<span class="tc-name"></span>`;
    chip.querySelector('.tc-name').textContent = meta.label;
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'tool-chip-remove';
    rm.setAttribute('aria-label', `Remove ${meta.label}`);
    rm.textContent = '✕';
    rm.addEventListener('click', (e) => {
      // Adjusting chips shouldn't count as clicking away from an open picker.
      e.stopPropagation();
      toggleTool(name);
    });
    chip.appendChild(rm);
    els.toolsList.appendChild(chip);
  }
}

/** Adds or removes a tool for the next send. Idempotent per id, so selecting
 *  the same tool twice can never produce a second chip or payload entry. */
function toggleTool(name) {
  selectedTools = selectedTools.includes(name)
    ? selectedTools.filter((x) => x !== name)
    : [...selectedTools, name];
  renderToolChips();
  syncToolRows();
}

/** Flips the selected marks on the rows already on screen. Rebuilding the
 *  list here would detach the clicked row mid-event, and the outside-click
 *  guard would then read the click as landing outside the composer box. */
function syncToolRows() {
  for (const row of els.toolMenuList.querySelectorAll('.tm-row')) {
    row.setAttribute('aria-selected', selectedTools.includes(row.dataset.name) ? 'true' : 'false');
  }
}

/** Rebuilds the picker from the last listing. Tools turned off in Settings
 *  show greyed with the reason, so their absence is never a mystery. */
function renderToolRows() {
  els.toolMenuList.innerHTML = '';
  const tools = toolsMeta;
  if (!tools?.length) {
    els.toolMenuHint.hidden = false;
    els.toolMenuHint.textContent = tools
      ? 'No tools are registered.'
      : 'Could not load tools — is the server reachable?';
    return;
  }
  for (const t of tools) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'tm-row';
    row.dataset.name = t.name;
    row.setAttribute('role', 'option');
    row.setAttribute('aria-selected', selectedTools.includes(t.name) ? 'true' : 'false');
    if (!t.enabled) {
      row.disabled = true;
      row.title = 'Turned off in Settings → Tools';
    }
    row.innerHTML =
      toolIconSvg(t.icon) +
      '<span class="tm-meta"><span class="tm-name"></span><span class="tm-desc"></span></span>' +
      '<span class="tm-check" aria-hidden="true">✓</span>';
    row.querySelector('.tm-name').textContent = t.label || t.name;
    row.querySelector('.tm-desc').textContent = t.enabled
      ? t.description
      : 'Turned off in Settings → Tools';
    els.toolMenuList.appendChild(row);
  }
  const off = tools.filter((t) => !t.enabled).length;
  els.toolMenuHint.hidden = off === 0;
  els.toolMenuHint.textContent = `${off} tool${off === 1 ? ' is' : 's are'} turned off in Settings → Tools.`;
}

function openToolMenu() {
  renderToolRows();
  els.toolMenu.hidden = false;
}

function closeToolMenu() {
  els.toolMenu.hidden = true;
}

/** "/tool" was typed: fetch the listing, then open the picker. */
async function openToolPicker() {
  await toolsList();
  openToolMenu();
}

/** Per-message selection, cleared once the reply finishes — unless the user
 *  already picked tools again mid-stream, whose newer choice stands. A failed
 *  send keeps the selection so Retry (and the composer) can reuse it. */
function releaseTools(requestTools) {
  const unchanged =
    selectedTools.length === requestTools.length &&
    requestTools.every((t, i) => selectedTools[i] === t);
  if (!unchanged) return;
  selectedTools = [];
  renderToolChips();
}

els.toolMenuList.addEventListener('click', (e) => {
  const row = e.target.closest('.tm-row');
  if (row && !row.disabled) {
    // Keep the click from reaching the document-level outside-click guard;
    // toggling re-paints the row, and a detached target would read as
    // "outside" and close the picker being used.
    e.stopPropagation();
    toggleTool(row.dataset.name);
  }
});

// Arrow keys walk the rows once focus is inside the picker; the input's own
// handler covers the moment right after "/tool" (focus is still there).
els.toolMenuList.addEventListener('keydown', (e) => {
  const items = [...els.toolMenuList.querySelectorAll('.tm-row:not(:disabled)')];
  if (e.key === 'Escape') {
    e.preventDefault();
    closeToolMenu();
    els.input.focus();
    return;
  }
  if ((e.key === 'Enter' || e.key === ' ') && e.target?.classList?.contains('tm-row')) {
    // Explicit activation: deterministic for real and synthetic key events
    // alike (and Space can't scroll the list instead of picking).
    e.preventDefault();
    e.target.click();
    return;
  }
  if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && items.length) {
    e.preventDefault();
    const at = items.indexOf(document.activeElement);
    const dir = e.key === 'ArrowDown' ? 1 : -1;
    items[(at + dir + items.length) % items.length]?.focus();
  }
});

document.addEventListener('click', (e) => {
  if (!els.toolMenu.hidden && !els.composerBox.contains(e.target)) closeToolMenu();
});

/**
 * Appends a small activity line under the bubble (tool calls, server
 * notices). textContent only — tool args are model-generated, so markup here
 * would be an injection hole. Sibling of the bubble, never inside it, so
 * streaming rewrites, copying, and the length clamp all ignore it.
 */
function addSideLine(bubble, text, cls) {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = String(text).slice(0, 300);
  bubble.after(line);
  return line;
}

/**
 * Pins the transcript to the newest text.
 *
 * Uses an instant scroll by default: during a stream this runs once per delta,
 * and an animated scroll restarts on every write and never catches up. Pass
 * `smooth` for a discrete jump, such as opening a different conversation.
 *
 * Also stops following when the reader has scrolled up to look at something —
 * yanking them back down mid-sentence is its own kind of jumpiness. The
 * threshold is generous enough to forgive a little drift near the bottom.
 */
function scrollToBottom({ smooth = false, force = false } = {}) {
  const el = els.messages;
  if (!force && !atBottom()) return;
  if (smooth) {
    el.classList.add('smooth-scroll');
    el.scrollTop = el.scrollHeight;
    el.classList.remove('smooth-scroll');
  } else {
    el.scrollTop = el.scrollHeight;
  }
}

/** True when the transcript is close enough to the bottom to keep following. */
function atBottom(slack = 120) {
  const el = els.messages;
  return el.scrollHeight - el.scrollTop - el.clientHeight <= slack;
}

/** Whether the jump-to-latest pill is currently shown. */
let latestVisible = false;

/**
 * Shows the jump-to-latest pill once the reader has scrolled meaningfully up
 * away from new content, and hides it again near the bottom. The two
 * thresholds are hysteresis: with a single cut-off the pill flickered while
 * streaming kept the gap hovering around it.
 */
function syncLatestButton() {
  if (!els.toLatest) return;
  const el = els.messages;
  const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
  latestVisible = gap > 240 ? true : gap <= 120 ? false : latestVisible;
  els.toLatest.classList.toggle('show', latestVisible);
}

els.messages.addEventListener('scroll', syncLatestButton, { passive: true });

els.toLatest?.addEventListener('click', () => {
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  scrollToBottom({ smooth: !reduceMotion, force: true });
});

function setStatus(text, isError = false) {
  els.status.textContent = text;
  els.statusDot.className = `dot${isError ? ' error' : ''}`;
}

function toggleBusy(busy) {
  els.send.disabled = busy;
  // The input stays enabled while a reply streams, so the box is always ready
  // for the next message. Disabling it also stole focus mid-thought (the
  // browser drops focus from a disabled control), which is what made the
  // composer feel "closed" during generation. Sending is still gated on
  // !activeStream() in the submit handler below — the open conversation's own
  // answer, not someone else's running in the background.
  els.stop.hidden = !busy;
  els.statusDot.className = `dot${busy ? ' busy' : ''}`;
  if (busy) els.status.textContent = 'generating';
  else els.status.textContent = 'ready';
  // The thinking blob lives next to the status text in the topbar, so it stays
  // visible even when the transcript is scrolled up — unlike the old overlay,
  // which sat on the reply bubble itself.
  els.statusBlob?.classList.toggle('on', busy);
}

/* ---------------- Input ---------------- */

function autosize() {
  els.input.style.height = 'auto';
  els.input.style.height = `${Math.min(els.input.scrollHeight, 220)}px`;

  // Swap the keyboard hint for a character count once there's something typed,
  // so the status line is useful instead of decorative.
  const len = els.input.value.length;
  els.composerBox.classList.toggle('typing', len > 0);
  els.composerCount.textContent = len > 1999 ? `${len.toLocaleString()} chars` : '';
}

els.input.addEventListener('input', () => {
  autosize();
  // "/tool" is a command, not a message: clear it and open the picker, so
  // the box is immediately free for the message itself. Exact match only —
  // "/tools" or a sentence containing "/tool" is ordinary text.
  if (els.input.value.trim() === '/tool') {
    els.input.value = '';
    autosize();
    void openToolPicker();
  }
});

els.input.addEventListener('keydown', (e) => {
  // While the picker is open it owns navigation: arrows walk the rows,
  // Escape closes, and Enter on an empty box closes instead of sending.
  if (!els.toolMenu.hidden) {
    if (e.key === 'Escape') {
      e.preventDefault();
      closeToolMenu();
      return;
    }
    if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && document.activeElement === els.input) {
      e.preventDefault();
      const items = [...els.toolMenuList.querySelectorAll('.tm-row:not(:disabled)')];
      if (!items.length) return;
      items[e.key === 'ArrowDown' ? 0 : items.length - 1]?.focus();
      return;
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && !els.input.value.trim()) {
      e.preventDefault();
      closeToolMenu();
      return;
    }
  }
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

els.composer.addEventListener('submit', (e) => {
  e.preventDefault();
  closeToolMenu();
  const text = els.input.value.trim();
  // A file on its own is a complete message: text files fold into the body and
  // images become content parts, so there is no reason to demand typed text.
  if (!text && !attachments.items().length) return;
  if (activeStream()) return;
  send(text);
});

els.stop.addEventListener('click', () => activeStream()?.ctl.abort());

els.providerList.addEventListener('click', (e) => {
  const item = e.target.closest('.popover__item');
  if (!item) return;
  setProvider(item.dataset.id);
  closeProviderPop();
});

els.newChat.addEventListener('click', newChat);

function newChat() {
  // A running answer keeps running: it is tied to its own chat, not to this
  // canvas. The next message here becomes a new conversation.
  if (recorder) stopRecording();
  // Detach from any saved conversation. The next message becomes a new one.
  activeId = null;
  conversation = [];
  renderConversation();
  persist();
  closeProviderPop();
  // The button lives in the sidebar, which is a drawer on narrow screens.
  closeSidebar();
  setStatus('ready');
  syncBusy();
  notify('New conversation');
  els.input.focus();
}

/* ---------------- Chat history ---------------- */

/**
 * Draws the conversation list into the panel. Called from persist(), so it
 * stays in sync with every message and model change without extra work.
 */
function renderHistory() {
  const items = chats
    .slice()
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .map(
      (c) => `
      <li class="history-item${c.id === activeId ? ' active' : ''}" data-id="${c.id}">
        <button type="button" class="hi-title" data-act="${c.id}"
                title="${escapeHtml(c.title || '')}">${escapeHtml(c.title || '')}</button>
        <button type="button" class="hi-del" data-del="${c.id}"
                aria-label="Delete conversation">✕</button>
      </li>`,
    )
    .join('');

  els.historyList.innerHTML = items || '<li class="history-empty">No conversations yet</li>';
  els.historyHint.textContent = chats.length
    ? `${chats.length} conversation${chats.length > 1 ? 's' : ''} saved on this device`
    : 'Send a message and it will appear here.';
}

/** Repaints the chat area from the working `conversation` array. */
function renderConversation() {
  els.messages.innerHTML = '';
  els.messages.appendChild(els.welcome);
  for (const m of conversation) {
    // A turn can hold content parts; render its text, and show a placeholder
    // where an image was. Stored history never has image bytes (persist strips
    // them), so this only runs for the live in-session conversation.
    const isParts = typeof m.content !== 'string';
    const text = isParts
      ? m.content
          .filter((p) => p.type === 'text')
          .map((p) => p.text)
          .join(' ')
      : m.content;
    const images = isParts
      ? m.content
          .filter((p) => p.type === 'image')
          .map((p) => ({ name: 'image', url: `data:${p.mimeType};base64,${p.data}` }))
      : [];
    addMessage(m.role, renderMarkdown(text || (images.length ? '' : m.content)), {
      raw: text,
      images,
      reasoning: typeof m.reasoning === 'string' ? m.reasoning : '',
    });
  }
  syncEmptyState();
  // Land on the newest turn. addMessage() scrolled per message, but those calls
  // run before the final layout, and fonts / highlighted code can grow the
  // transcript afterwards — which is exactly how the last message ends up below
  // the fold. Pin now, then again after the next frame has settled.
  scrollToBottom({ force: true });
  requestAnimationFrame(() => requestAnimationFrame(() => scrollToBottom({ force: true })));
  syncLatestButton();
}

/** Switches the app to another saved conversation. */
function openChat(id) {
  const c = chats.find((x) => x.id === id);
  if (!c) return;
  if (c.id === activeId) {
    closeProviderPop();
    return;
  }
  // Deliberately no abort here. An answer in flight belongs to the chat it
  // was sent from; leaving to read another conversation must not kill it.
  persist(); // capture the conversation we're leaving
  clearNotifications(); // the view changed underneath any visible status

  activeId = c.id;
  conversation = Array.isArray(c.conversation) ? c.conversation : [];
  if (activeProviderId !== c.provider) {
    activeProviderId = c.provider;
    paintProviderSelection();
  }
  applyProvider(c.model);
  renderConversation();
  // Coming back mid-answer: the live bubble went with the old view, so give
  // this answer a fresh one and pour the text received so far into it.
  reattachStream(id);
  persist(); // bump updatedAt so the reopened chat rises to the top
  closeProviderPop();
  updateTopbar();
  syncBusy();
  els.input.focus();
}

/** Removes a conversation. Deleting the active one opens the newest survivor. */
function deleteChat(id) {
  // An answer still streaming into this conversation stops with it — there is
  // nothing left for it to write into.
  streams.get(id)?.ctl.abort();
  chats = chats.filter((x) => x.id !== id);
  if (id === activeId) {
    activeId = chats[0]?.id ?? null;
    conversation = activeId
      ? Array.isArray(chats[0].conversation) ? chats[0].conversation : []
      : [];
  }
  renderConversation();
  persist();
  closeProviderPop();
  els.input.focus();
}

// History item clicks are delegated so per-item buttons don't each need a
// listener. Deleting is a two-step "✕ → Sure?" so a slip doesn't lose a chat.
els.historyList.addEventListener('click', (e) => {
  const del = e.target.closest('[data-del]');
  if (del) {
    if (del.dataset.armed !== '1') {
      del.dataset.armed = '1';
      del.textContent = 'Sure?';
      setTimeout(() => {
        del.dataset.armed = '';
        del.textContent = '✕';
      }, 2600);
    } else {
      deleteChat(del.dataset.del);
    }
    return;
  }
  const act = e.target.closest('[data-act]');
  if (act) openChat(act.dataset.act);
});

/* ---------------- Resource usage ----------------
   The meters and the resource manager now live in resources.js, mounted once
   from a `[data-resource-mount]` placeholder. The Studio mounts the same box,
   and two copies of this would be two things to keep in step. */
const resourcePanel = mountResources();
/* ---------------- Provider popup ---------------- */

/**
 * A small anchored popover instead of the old full-height slide-over: the sheet
 * only ever held provider choice and a link, which fits in a rounded card the
 * width of the chip. It takes no scrim — clicking away dismisses it — so the
 * conversation underneath stays readable while you pick.
 */
function openProviderPop() {
  closeSidebar();
  els.providerPop.hidden = false;
  els.openPanel.setAttribute('aria-expanded', 'true');
  // Land focus on the current provider so the popup is keyboard-usable at once.
  els.providerList.querySelector('[aria-selected="true"]')?.focus();
}

function closeProviderPop() {
  if (els.providerPop.hidden) return;
  els.providerPop.hidden = true;
  els.openPanel.setAttribute('aria-expanded', 'false');
}

function toggleProviderPop() {
  if (els.providerPop.hidden) openProviderPop();
  else closeProviderPop();
}

els.openPanel.addEventListener('click', toggleProviderPop);

// Clicking anywhere outside the popup closes it. The chip and its popover are
// siblings inside .chip-wrap, so one containment test covers both.
document.addEventListener('click', (e) => {
  if (els.providerPop.hidden) return;
  if (e.target.closest('.chip-wrap')) return;
  closeProviderPop();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !els.providerPop.hidden) {
    closeProviderPop();
    els.openPanel.focus();
  }
});

els.backdrop.addEventListener('click', () => {
  closeSidebar();
});

/* ---------------- History drawer (narrow screens) ---------------- */

// Above 900px the sidebar is a permanent rail and the toggle is hidden by CSS.
// Below it, the same element slides in over the conversation. One list in the
// DOM either way, so there is nothing to keep in sync.
const SIDEBAR_BREAKPOINT = 900;

function sidebarIsDrawer() {
  return window.matchMedia(`(max-width: ${SIDEBAR_BREAKPOINT}px)`).matches;
}

function openSidebar() {
  if (!sidebarIsDrawer()) return;
  closeProviderPop();
  els.sidebar.classList.add('open');
  els.backdrop.classList.add('show');
  els.sidebarToggle.setAttribute('aria-expanded', 'true');
  // The resource box lives in the drawer here, so this is the first moment it
  // can be seen. Catches up immediately instead of waiting out the interval.
  resourcePanel?.refresh();
}

function closeSidebar() {
  els.sidebar.classList.remove('open');
  els.sidebarToggle.setAttribute('aria-expanded', 'false');
  // The provider popup takes no scrim, so the drawer owns the backdrop outright.
  els.backdrop.classList.remove('show');
}

els.sidebarToggle.addEventListener('click', () => {
  if (els.sidebar.classList.contains('open')) closeSidebar();
  else openSidebar();
});
els.sidebarClose?.addEventListener('click', closeSidebar);

// Picking a conversation on a phone should get the drawer out of the way —
// but arming a delete must NOT close it, or the "Sure?" tap would land on a
// closed drawer and the delete could never complete.
els.historyList.addEventListener('click', (e) => {
  if (e.target.closest('[data-del]')) return;
  closeSidebar();
});

// Leaving the narrow breakpoint with the drawer open would strand it open.
window.matchMedia(`(max-width: ${SIDEBAR_BREAKPOINT}px)`).addEventListener('change', (e) => {
  if (!e.matches) closeSidebar();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && els.sidebar.classList.contains('open')) {
    closeSidebar();
    els.input.focus();
  }
  // Escape while recording cancels instead of sending anything.
  if (e.key === 'Escape' && recorder) {
    stopRecording();
    notify('Recording cancelled');
  }
});

// A mic left open would keep recording after the page goes away.
window.addEventListener('pagehide', () => {
  if (recorder) stopRecording();
  stopRecorderTracks();
});

/* ---------------- Boot ---------------- */

/* Give the glass something to blur. Deferred until after first paint so the
   shader's compile and first frame never delay the chat being usable — the
   canvas stays hidden until shader.js marks it ready. */
requestAnimationFrame(() => {
  const canvas = document.getElementById('shader-bg');
  if (!canvas) return;
  const backdrop = initShaderBackdrop(canvas);
  if (backdrop) canvas.dataset.ready = '';
  else canvas.remove(); // no WebGL: drop the element rather than leave a dead box
});

const saved = restore();
// Drop the query once honoured: a reload should resume the chat, not re-open
// the deep link on top of whatever has happened since.
if (deepLink()) {
  history.replaceState(null, '', location.pathname);
}
renderHistory();
renderConversation();

// Wires the picker, paste and drop handlers. The capability probe runs from
// applyProvider() once a provider is actually chosen.
attachments.init();

// Click-to-enlarge + download for images in the transcript (attachments now,
// anything added later via the observer).
initImageViewer();

await loadProviders(saved);

autosize();
els.input.focus();