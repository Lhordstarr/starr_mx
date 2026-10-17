import { notifySuccess, notifyError } from './notify.js';
import { mountNav } from './nav.js';
import { initShaderBackdrop } from './shader.js';
import { initImageViewer } from './viewer.js';
import { mountResources } from '/resources.js';

/* The same rail the chat page carries — primary navigation and the resource
   box — from the same modules, so the two pages cannot drift apart. */
mountNav();
mountResources();


const els = {
  form: document.getElementById('composer'),
  prompt: document.getElementById('input'),
  go: document.getElementById('go'),
  status: document.getElementById('status'),
  statusBlob: document.getElementById('status-blob'),
  messages: document.getElementById('messages'),
  welcome: document.getElementById('welcome'),
  provider: document.getElementById('studio-img-provider'),
  size: document.getElementById('studio-img-size'),
  modelSelect: document.getElementById('model-select'),
  modelTrigger: document.getElementById('model-trigger'),
  modelMenu: document.getElementById('model-menu'),
  modelSearch: document.getElementById('model-search'),
  modelList: document.getElementById('model-list'),
  modelCurrent: document.getElementById('model-current'),
  modelHint: document.getElementById('model-hint'),
  imgStatus: document.getElementById('img-status'),
  imageSettings: document.getElementById('studio-image-settings'),
  chipProvider: document.getElementById('chip-provider'),
  chipModel: document.getElementById('chip-model'),
  historyList: document.getElementById('history-list'),
  historyHint: document.getElementById('history-hint'),
  newChat: document.getElementById('new-chat'),
  sidebar: document.getElementById('sidebar'),
  sidebarToggle: document.getElementById('sidebar-toggle'),
  sidebarClose: document.getElementById('sidebar-close'),
  backdrop: document.getElementById('backdrop'),
};

const STORAGE_KEY = 'llm-img-state';

/**
 * Sessions are the Studio's answer to the chat's conversation history: one
 * entry per batch of work, listed in the same sidebar with the same title and
 * delete affordances. `turns` is the working set of the active session — or of
 * the unsaved one a fresh page starts with, which only becomes a session once
 * it has a result worth keeping.
 */
let sessions = [];
let activeId = null;

/** In-memory turns. Assistant turns carry a transient `dataUrl` while they are
 *  on screen; persist() strips it so history never stores image bytes. */
let turns = [];

/** The tab on screen. */
let studioTask = 'image';

/**
 * Which task the working `turns` belong to.
 *
 * Kept apart from `studioTask` on purpose: when the tab changes, the canvas
 * being left has to be saved under the task it belonged to, not the one being
 * opened. After `switchTaskCanvas` runs the two are the same again. Declared
 * up here with the other state because `restore()` — which runs long before the
 * tab machinery below — reads and writes it.
 */
let canvasTask = 'image';

/** Image providers from the server: { id, label, keyless, configured, models, defaultModel, sizes }. */
let providers = [];

/** Local image models from /api/runnable. Declared with the other boot-time
 *  state: the boot block calls loadLocalImage() before later declarations
 *  execute, which used to leave it in the temporal dead zone. */
let localImageModels = [];

/** The model the composer pill points at: a cloud id, or a local repo. */
let imageModel = '';

/** The pill's filtered options and the highlighted row, as the chat keeps them. */
let imageMenuOptions = [];
let imageMenuIndex = -1;

/* ---------------- Theme ---------------- */

const THEME_KEY = 'llm-chat-theme';

/**
 * Sentinel id for "generate on this machine" in the image provider dropdown.
 * Declared with the other constants because `setBusy` reads it during boot;
 * leaving it below its first use put it in the temporal dead zone.
 */
const LOCAL_IMAGE_ID = '__local__';

function applyTheme(theme) {
  if (theme === 'dark' || theme === 'light') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

applyTheme(localStorage.getItem(THEME_KEY));

document.getElementById('theme-toggle').addEventListener('click', () => {
  const order = [null, 'light', 'dark'];
  const current = document.documentElement.dataset.theme ?? null;
  const next = order[(order.indexOf(current) + 1) % order.length];
  applyTheme(next);
  if (next) localStorage.setItem(THEME_KEY, next);
  else localStorage.removeItem(THEME_KEY);
});

/* ---------------- Shader backdrop ---------------- */

/* Same as the other pages: deferred so the shader's first frame never delays
   the page being usable — the canvas stays hidden until ready. */
requestAnimationFrame(() => {
  const canvas = document.getElementById('shader-bg');
  if (!canvas) return;
  const backdrop = initShaderBackdrop(canvas);
  if (backdrop) canvas.dataset.ready = '';
  else canvas.remove(); // no WebGL: drop the element rather than leave a dead box
});

/* ---------------- Providers ---------------- */

async function loadProviders() {
  try {
    const res = await fetch('/api/image/providers');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { providers: list } = await res.json();
    providers = list ?? [];
  } catch {
    providers = [];
  }
  if (!providers.length) {
    els.go.disabled = true;
    els.imgStatus.textContent = 'No image providers available.';
    renderImagePicker();
    return;
  }
  populate();
}

function sizeLabel(size) {
  return size.w === size.h ? `${size.w} × ${size.w}` : `${size.w} × ${size.h}`;
}

function populate() {
  const sel = els.provider;
  sel.textContent = '';
  let preferred = '';
  for (const p of providers) {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = p.configured ? p.label : `${p.label} (needs setup)`;
    opt.disabled = !p.configured;
    if (p.configured && !preferred) preferred = p.id;
    sel.appendChild(opt);
  }
  sel.value = preferred;
  syncControls();
  syncChips();
}

function syncControls() {
  // The local entry has no provider record behind it — it is the diffusers
  // runner — so it is handled before the lookup. Without this, choosing it
  // disabled Generate outright and left the cloud model list showing.
  if (localImageSelected()) {
    els.go.disabled = false;
    els.imgStatus.textContent = '';
    renderImagePicker();
    return;
  }
  const p = providers.find((x) => x.id === els.provider.value);
  els.go.disabled = !p || !p.configured;
  els.imgStatus.textContent = p && !p.configured ? 'Set this provider up in Settings to use it.' : '';
  if (!p) {
    renderImagePicker();
    return;
  }

  els.size.textContent = '';
  for (const s of p.sizes) {
    const opt = document.createElement('option');
    opt.value = `${s.w}x${s.h}`;
    opt.textContent = sizeLabel(s);
    els.size.appendChild(opt);
  }
  els.size.selectedIndex = 0;

  renderImagePicker();
}

/**
 * Mirrors the rail's provider and model into the topbar chips, so the bar
 * reads the same as the chat's — whose chip is also just "what is answering".
 */
function syncChips() {
  const local = localImageSelected();
  const p = providers.find((x) => x.id === els.provider.value);

  if (els.chipProvider) {
    els.chipProvider.textContent = local ? 'On this PC' : p?.label ?? 'No provider';
  }
  if (!els.chipModel) return;
  if (local) {
    els.chipModel.textContent =
      localImageModels.find((m) => m.repo === imageModel)?.label ?? 'No model runnable';
  } else if (p) {
    els.chipModel.textContent = imageModel || p.defaultModel || '—';
  } else {
    els.chipModel.textContent = '—';
  }
}

els.provider.addEventListener('change', () => {
  syncControls();
  syncChips();
});

/* ---------------- The composer's model pill ---------------- */

/**
 * The chat page's model selector, moved to where the Studio keeps its model
 * choice. It lists whatever the rail's provider can answer with: a cloud
 * provider's models, or the diffusers repos on this machine.
 */
function imageOptions() {
  if (localImageSelected()) {
    return localImageModels.map((m) => ({
      id: m.repo,
      label: m.label,
      ready: m.ready,
      reason: m.reason,
    }));
  }
  const p = providers.find((x) => x.id === els.provider.value);
  if (!p) return [];
  const ids = p.models.length ? p.models : p.defaultModel ? [p.defaultModel] : [];
  return ids.map((id) => ({ id, label: id, ready: true }));
}

/**
 * Repaints the pill, its hint and (when open) the menu from the active
 * provider. The selected model survives a repaint whenever it is still on
 * offer; otherwise the provider's default — or the first runnable local
 * model — takes over.
 */
function renderImagePicker() {
  const options = imageOptions();
  if (!options.some((o) => o.id === imageModel && o.ready)) {
    imageModel = options.find((o) => o.ready)?.id ?? options[0]?.id ?? '';
  }
  els.modelCurrent.textContent =
    options.find((o) => o.id === imageModel)?.label ?? 'no model selected';

  const ready = options.filter((o) => o.ready).length;
  if (localImageSelected()) {
    els.modelHint.textContent = !options.length
      ? 'No local image models — see Get models'
      : ready
        ? `${ready} of ${options.length} ready on this machine`
        : 'None runnable here — download one in Get models and its runtime in Runtimes';
  } else {
    els.modelHint.textContent = options.length
      ? `${options.length} model${options.length > 1 ? 's' : ''} available`
      : providers.length
        ? 'No models listed for this provider'
        : 'No image providers available';
  }

  if (!els.modelMenu.hidden) renderImageMenu(els.modelSearch.value);
  syncChips();
}

/** Repaints the menu's rows from the active provider's models. */
function renderImageMenu(query = '') {
  const options = imageOptions();
  const q = query.trim().toLowerCase();
  const matches = q
    ? options.filter((o) => `${o.label} ${o.id}`.toLowerCase().includes(q))
    : options;
  imageMenuOptions = matches;
  imageMenuIndex = matches.findIndex((o) => o.ready);

  if (!matches.length) {
    const empty = document.createElement('li');
    empty.className = 'mm-empty';
    empty.textContent = options.length
      ? 'No models match that search'
      : 'No models available yet';
    els.modelList.replaceChildren(empty);
  } else {
    els.modelList.replaceChildren(
      ...matches.map((o) => {
        const item = document.createElement('li');
        item.className = 'mm-option';
        item.setAttribute('role', 'option');
        item.dataset.model = o.id;
        item.setAttribute('aria-selected', String(o.id === imageModel));
        // Unrunnable models are listed but not selectable: hiding them would
        // read as "nothing exists", which is the one answer that never helps.
        if (!o.ready) {
          item.setAttribute('aria-disabled', 'true');
          item.title = o.reason ?? 'Not runnable here';
        }
        const name = document.createElement('span');
        name.textContent = o.label;
        item.append(name);
        if (!o.ready) {
          const why = document.createElement('span');
          why.className = 'mm-reason';
          why.textContent = o.reason ?? 'not runnable here';
          item.append(why);
        }
        return item;
      }),
    );
  }
  highlightImageOption();
}

function highlightImageOption() {
  const items = [...els.modelList.querySelectorAll('.mm-option')];
  items.forEach((n, i) => n.classList.toggle('active', i === imageMenuIndex));
  items[imageMenuIndex]?.scrollIntoView({ block: 'nearest' });
}

/** Moves the highlight, skipping models that cannot be chosen. */
function stepImageMenu(delta) {
  const items = [...els.modelList.querySelectorAll('.mm-option')];
  if (!items.length) return;
  let i = imageMenuIndex;
  for (let n = 0; n < items.length; n++) {
    i = (i + delta + items.length) % items.length;
    if (items[i].getAttribute('aria-disabled') !== 'true') break;
  }
  imageMenuIndex = i;
  highlightImageOption();
}

function openImageMenu() {
  els.modelMenu.hidden = false;
  els.modelTrigger.setAttribute('aria-expanded', 'true');
  els.modelSearch.value = '';
  renderImageMenu();
  els.modelSearch.focus();
}

function closeImageMenu() {
  els.modelMenu.hidden = true;
  els.modelTrigger.setAttribute('aria-expanded', 'false');
}

/** Commits a model and refreshes everything that depends on it. */
function commitImageModel(id) {
  const option = imageOptions().find((o) => o.id === id);
  if (!option?.ready) return;
  imageModel = id;
  closeImageMenu();
  renderImagePicker();
  els.prompt.focus();
}

els.modelTrigger.addEventListener('click', () => {
  if (els.modelMenu.hidden) openImageMenu();
  else closeImageMenu();
});

els.modelSearch.addEventListener('input', () => renderImageMenu(els.modelSearch.value));

els.modelSearch.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    stepImageMenu(e.key === 'ArrowDown' ? 1 : -1);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    const option = imageMenuOptions[imageMenuIndex];
    if (option?.ready) commitImageModel(option.id);
  } else if (e.key === 'Escape') {
    e.preventDefault();
    e.stopPropagation();
    closeImageMenu();
    els.modelTrigger.focus();
  }
});

els.modelList.addEventListener('click', (e) => {
  const item = e.target.closest('.mm-option');
  if (item && item.getAttribute('aria-disabled') !== 'true') commitImageModel(item.dataset.model);
});

document.addEventListener('click', (e) => {
  if (!els.modelMenu.hidden && !els.modelSelect.contains(e.target)) closeImageMenu();
});

/* ---------------- Transcript ---------------- */

function messageEl(role) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;
  const body = document.createElement('div');
  body.className = 'message-body';
  if (role === 'assistant') {
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'AI';
    wrap.appendChild(avatar);
  }
  wrap.appendChild(body);
  return { wrap, body };
}

function bubble(text) {
  const bw = document.createElement('div');
  bw.className = 'bubble-wrap';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  bw.appendChild(b);
  return bw;
}

function renderUser(text) {
  const { wrap, body } = messageEl('user');
  body.appendChild(bubble(text));
  els.messages.appendChild(wrap);
}

function renderAssistant(turn, index) {
  const { wrap, body } = messageEl('assistant');
  // The download chips resolve their turn by position in `turns`; renderAll
  // walks the same array in the same order.
  if (index != null) wrap.dataset.turn = String(index);

  // `dataUrl` is a base64 payload from a cloud provider; `url` points at a file
  // the local runner wrote into the output directory. One renderer, both.
  const src = turn.dataUrl || turn.url;
  if (src) {
    const strip = document.createElement('div');
    strip.className = 'msg-images';
    const img = document.createElement('img');
    img.src = src;
    img.alt = turn.prompt ? `Generated: ${turn.prompt}` : 'Generated image';
    // Downgrades to the alt text if there's no prompt; names the download.
    if (turn.prompt) img.dataset.name = turn.prompt;
    img.loading = 'lazy';
    strip.appendChild(img);
    body.appendChild(strip);
  }

  const bw = bubble(turn.text || '');
  if (!src && turn.kind !== 'transcript') {
    const note = document.createElement('span');
    note.className = 'img-note';
    note.textContent = '[image — not saved]';
    bw.firstChild.appendChild(note);
  }
  body.appendChild(bw);

  if (turn.meta || turn.kind === 'transcript') {
    const row = document.createElement('div');
    row.className = 'meta-row';
    if (turn.meta) {
      const meta = document.createElement('span');
      meta.className = 'meta';
      meta.textContent = turn.meta;
      row.appendChild(meta);
    }
    if (turn.kind === 'transcript') row.appendChild(transcriptTools(turn));
    body.appendChild(row);
  }

  els.messages.appendChild(wrap);
}

/** The chips under a transcript: Copy first — pasting beats a download for the
 *  usual reason you want a transcript — then one per shape the result supports.
 *  SRT and VTT need cue timings, so a runner that returned none just does not
 *  offer them rather than writing a subtitles file that cannot play. */
function transcriptTools(turn) {
  const tools = document.createElement('span');
  tools.className = 'tr-tools';
  const shapes = [
    ['copy', 'Copy', 'Copy the transcript text'],
    ['txt', 'TXT', 'Download the plain text (.txt)'],
    ['srt', 'SRT', 'Download SubRip subtitles (.srt)'],
    ['vtt', 'VTT', 'Download WebVTT subtitles (.vtt)'],
    ['json', 'JSON', 'Download the whole result (.json)'],
  ];
  for (const [format, label, title] of shapes) {
    if ((format === 'srt' || format === 'vtt') && !turn.segments?.length) continue;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'tr-dl';
    button.dataset.fmt = format;
    button.textContent = label;
    button.title = title;
    tools.appendChild(button);
  }
  return tools;
}

function renderAll() {
  els.messages.textContent = '';
  // The welcome node survives in `els`; it is simply detached and re-attached
  // based on whether there is anything to show.
  if (!turns.length) els.messages.appendChild(els.welcome);
  turns.forEach((t, index) => {
    if (t.role === 'user') renderUser(t.text);
    else renderAssistant(t, index);
  });
  els.messages.scrollTop = els.messages.scrollHeight;
}

/* ---------------- Persistence and session history ----------------
   Modelled on the chat's conversation store: sessions saved on this device,
   drawn into the same sidebar list, opened and deleted the same way. */

function genId() {
  return `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;
}

/** One-line summary for the list, the same shape as a conversation's title. */
function sessionTitle(list) {
  const first = list.find((t) => t.role === 'user')?.text ?? '';
  const flat = String(first).replace(/\s+/g, ' ').trim();
  if (!flat) return 'New session';
  return flat.length > 42 ? `${flat.slice(0, 40)}…` : flat;
}

function escapeHtml(value) {
  return String(value).replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );
}

/**
 * Draws the session list into the panel — only the sessions of the canvas on
 * screen, which is what makes the two tabs read as two workspaces. Called from
 * persist(), so it stays in step with every result without extra work.
 */
function renderHistory() {
  const mine = sessions.filter((s) => (s.task ?? 'image') === canvasTask);
  const items = mine
    .map(
      (s) => `
      <li class="history-item${s.id === activeId ? ' active' : ''}" data-id="${s.id}">
        <button type="button" class="hi-title" data-act="${s.id}"
                title="${escapeHtml(s.title || '')}">${escapeHtml(s.title || '')}</button>
        <button type="button" class="hi-del" data-del="${s.id}"
                aria-label="Delete session">✕</button>
      </li>`,
    )
    .join('');

  els.historyList.innerHTML = items || '<li class="history-empty">No sessions yet</li>';
  const word = canvasTask === 'image' ? 'image session' : 'transcript session';
  els.historyHint.textContent = mine.length
    ? `${mine.length} ${word}${mine.length > 1 ? 's' : ''} saved on this device`
    : canvasTask === 'image'
      ? 'Generate something and it will appear here.'
      : 'Transcribe something and it will appear here.';
}

/**
 * Writes the working turns to the active session, minting one on the first
 * result. Image bytes never reach localStorage — every generated image is
 * served from the on-disk cache by URL, so what is stored here is a path and a
 * few words of text.
 *
 * The write is guarded: localStorage is ~5 MB and a quota error here used to
 * throw out of every save after it, which is how old sessions stopped coming
 * back. Failing loudly in a toast is worse than keeping the canvas, so the
 * write is the thing that gets dropped, and the reason is said out loud.
 */
function persist() {
  let active = sessions.find((s) => s.id === activeId);
  if (!active && turns.length) {
    active = { id: genId(), task: canvasTask, title: '', turns: [], createdAt: Date.now(), updatedAt: Date.now() };
    sessions.unshift(active);
    activeId = active.id;
  }
  if (active) {
    active.task = canvasTask;
    active.title = sessionTitle(turns);
    active.turns = turns.map((t) =>
      t.dataUrl ? { role: t.role, text: t.text, meta: t.meta, prompt: t.prompt } : t,
    );
    active.updatedAt = Date.now();
    sessions.sort((a, b) => b.updatedAt - a.updatedAt);
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ active: activeId, sessions }));
  } catch (err) {
    notifyError(`Could not save this session to disk: ${err instanceof Error ? err.name : 'storage full'}`);
  }
  renderHistory();
}

function restore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed.sessions)) {
      sessions = parsed.sessions.filter(
        (s) => s && typeof s.id === 'string' && Array.isArray(s.turns),
      );
      for (const s of sessions) {
        s.title = typeof s.title === 'string' ? s.title : sessionTitle(s.turns);
        s.updatedAt = Number(s.updatedAt) || Date.now();
        // Sessions written before the tabs split: whatever a transcript sits
        // in, that session belongs to the Transcribe canvas.
        if (typeof s.task !== 'string') {
          s.task = s.turns.some((t) => t && t.kind === 'transcript') ? 'stt' : 'image';
        }
      }
      sessions.sort((a, b) => b.updatedAt - a.updatedAt);
      activeId = sessions.some((s) => s.id === parsed.active) ? parsed.active : sessions[0]?.id ?? null;
    } else if (Array.isArray(parsed.turns) && parsed.turns.length) {
      // The single flat turn list this stored before there were sessions,
      // promoted to one rather than dropped.
      const list = parsed.turns.filter(
        (t) => t && typeof t === 'object' && typeof t.text === 'string',
      );
      if (list.length) {
        const s = { id: genId(), title: sessionTitle(list), turns: list,
                    createdAt: Date.now(), updatedAt: Date.now() };
        sessions = [s];
        activeId = s.id;
      }
    }
    turns = sessions.find((s) => s.id === activeId)?.turns ?? [];
    canvasTask = sessions.find((s) => s.id === activeId)?.task ?? 'image';
  } catch {
    sessions = [];
    activeId = null;
    turns = [];
  }
}

/** Opens a saved session; the working turns become that session's. */
function openSession(id) {
  const s = sessions.find((x) => x.id === id);
  if (!s || s.id === activeId) return;
  persist(); // capture the session we're leaving
  activeId = s.id;
  turns = s.turns;
  renderAll();
  persist(); // bump updatedAt so the reopened session rises to the top
  els.prompt.focus();
}

/** Removes a session. Deleting the active one opens the newest survivor. */
function deleteSession(id) {
  sessions = sessions.filter((s) => s.id !== id);
  if (id === activeId) {
    // The replacement stays on this canvas; the other task's sessions are not
    // ours to open just because the active one is gone.
    const mine = sessions
      .filter((s) => (s.task ?? 'image') === canvasTask)
      .sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0));
    activeId = mine[0]?.id ?? null;
    turns = mine[0]?.turns ?? [];
  }
  renderAll();
  persist();
  els.prompt.focus();
}

/** Detaches from any session: the next result starts a new one. */
function newSession() {
  activeId = null;
  turns = [];
  renderAll();
  persist();
  closeSidebar();
  notifySuccess('Started a new session');
  els.prompt.focus();
}

els.newChat.addEventListener('click', newSession);

// History item clicks are delegated so per-item buttons don't each need a
// listener. Deleting is a two-step "✕ → Sure?" so a slip doesn't lose a session.
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
      deleteSession(del.dataset.del);
    }
    return;
  }
  const act = e.target.closest('[data-act]');
  if (act) openSession(act.dataset.act);
});

/* ---------------- Generation ---------------- */

function setBusy(busy) {
  // The local entry has no provider record behind it, so it has to be treated
  // as usable explicitly — otherwise choosing it disabled Generate outright.
  const local = els.provider.value === LOCAL_IMAGE_ID;
  const p = providers.find((x) => x.id === els.provider.value);
  const usable = local || Boolean(p && p.configured);
  els.go.disabled = busy || !usable;
  els.prompt.disabled = busy;
  els.status.textContent = busy ? 'generating' : 'ready';
  els.statusBlob.classList.toggle('on', busy);
}

function appendGenerated(data, prompt) {
  turns.push({ role: 'user', text: prompt });
  turns.push({
    role: 'assistant',
    text: `Generated with ${data.label}${data.model ? ` (${data.model})` : ''}`,
    meta: `${data.label} · ${data.width}×${data.height}`,
    prompt,
    // Exactly one of these is set: a served path for local output, and now
    // also for a cloud provider — its bytes are written to the on-disk image
    // cache by the server, so this is a URL that still resolves after a
    // reload. The data-URL branch is the fallback for when that write failed.
    dataUrl: data.data ? `data:${data.mimeType};base64,${data.data}` : undefined,
    url: data.url,
  });
  renderAll();
  persist();
}

/**
 * Generates an image with a diffusers model on this machine.
 *
 * `/api/infer/image` streams progress as SSE and answers with a file in the
 * output directory, not a base64 payload like the cloud route, so the two are
 * handled separately rather than folded into one response shape.
 */
async function generateLocalImage(prompt) {
  const model = localImageModels.find((m) => m.repo === imageModel);
  if (!model) {
    els.imgStatus.textContent = 'Pick a local image model first.';
    return;
  }
  if (!model.ready) {
    els.imgStatus.textContent = model.reason || 'That model is not runnable here.';
    return;
  }
  const [width, height] = els.size.value.split('x').map(Number);
  setBusy(true);
  els.imgStatus.textContent = `Generating on this PC with ${model.label}… this is slow on CPU.`;
  try {
    const out = await studioInfer('/api/infer/image', {
      repo: model.repo,
      file: model.file,
      prompt,
      steps: 4,
      width,
      height,
      device: 'auto',
    });
    if (out?.name) {
      appendGenerated(
        { label: model.label, mimeType: 'image/png', data: '', url: `/api/infer/out/${encodeURIComponent(out.name)}`, width, height },
        prompt,
      );
      notifySuccess('Image generated on this PC');
    }
    els.imgStatus.textContent = '';
  } catch (err) {
    els.imgStatus.textContent = '';
    notifyError(err.message);
  } finally {
    setBusy(false);
  }
}

async function generate() {
  const prompt = els.prompt.value.trim();
  if (!prompt) {
    els.imgStatus.textContent = 'Describe the image first.';
    els.prompt.focus();
    return;
  }

  // Local generation first: it needs no provider entry, and letting it fall
  // through to the cloud path produced "No image provider selected" for the one
  // option that actually runs on this machine.
  if (localImageSelected()) {
    await generateLocalImage(prompt);
    return;
  }

  const p = providers.find((x) => x.id === els.provider.value);
  if (!p) {
    els.imgStatus.textContent = 'No image provider selected.';
    return;
  }
  if (!p.configured) {
    els.imgStatus.textContent = 'Set this provider up in Settings to use it.';
    return;
  }

  const [width, height] = els.size.value.split('x').map(Number);
  setBusy(true);
  els.imgStatus.textContent = `Generating with ${p.label}… free services can take up to a minute.`;
  try {
    const res = await fetch('/api/image/generate', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        provider: p.id,
        model: imageModel || p.defaultModel,
        prompt,
        width,
        height,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    appendGenerated(data, prompt);
    els.prompt.value = '';
    els.imgStatus.textContent = '';
    notifySuccess('Image added to the conversation');
  } catch (err) {
    els.imgStatus.textContent = '';
    notifyError(err.message);
  } finally {
    setBusy(false);
  }
}

els.form.addEventListener('submit', (e) => {
  e.preventDefault();
  void generate();
});

// Enter generates, Shift+Enter starts a new line — the contract the hint
// prints, and the same one the chat composer keeps. isComposing keeps IME
// composition (CJK input) from firing a generate half-way through.
els.prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    els.form.requestSubmit();
  }
});

/* ---------------- Boot ---------------- */

initImageViewer();
// Saved sessions first: the sidebar list and the transcript both read them.
restore();
renderHistory();
// Local image models are discovered once the cloud list exists, so the
// dropdown is populated before anything can select from it.
loadProviders().then(loadLocalImage);
renderAll();
els.prompt.focus();
/* ---------------- Studio: image / transcribe / speak ---------------- */

/**
 * The Studio is three capabilities behind one page, because they share the
 * same question — "what can this machine actually run?" — and answering it once
 * here is clearer than making the user visit Settings to discover why a tab is
 * empty. Every panel is populated from /api/runnable, so a model downloaded a
 * moment ago shows up without a reload, and one that cannot run here says why
 * instead of offering a button that fails.
 */
const studioTabs = [...document.querySelectorAll('.studio-tab')];
const composer = document.getElementById('composer');
const promptInput = document.getElementById('input');
const welcomeTitle = document.getElementById('welcome-title');
const welcomeSub = document.getElementById('welcome-sub');

const COPY = {
  image: {
    title: 'What should we make?',
    sub: 'Describe an image and hit Generate. Pollinations and Stable Horde need no ' +
         'account; Cloudflare and Hugging Face work off a free token from Settings.',
  },
  stt: {
    title: 'What did you say?',
    sub: 'Record or drop in an audio file. Runs on this machine using a Whisper model ' +
         'from Get models — no API key, nothing leaves the computer.',
  },
  };

function selectStudioTask(task, { focus = false } = {}) {
  const leaving = studioTask;
  studioTask = task;
  for (const tab of studioTabs) {
    const on = tab.dataset.task === task;
    tab.setAttribute('aria-selected', String(on));
    // Roving tabindex: one stop in the tab order, arrows move within it. Without
    // this the whole strip is unreachable and reads as three loose buttons.
    tab.tabIndex = on ? 0 : -1;
    if (on && focus) tab.focus();
  }
  for (const name of ['image', 'stt']) {
    const panel = document.getElementById(`panel-${name}`);
    if (panel) panel.hidden = name !== task;
  }
  // The image composer is only meaningful on the Image tab; leaving it
  // enabled elsewhere invited "Generate" doing nothing visible.
  if (composer) composer.hidden = task !== 'image';
  if (promptInput) promptInput.placeholder = 'A red panda in a spacesuit, studio lighting…';

  // Image settings live on the rail when the Image tab is live, so they sit
  // with the model pickers instead of down in the composer.
  if (els.imageSettings) els.imageSettings.hidden = task !== 'image';

  // The image model is the composer pill's job now; the rail's picker belongs
  // to the Transcribe tab only.
  const modelsSection = document.getElementById('studio-models-section');
  if (modelsSection) modelsSection.hidden = task === 'image';

  const copy = COPY[task] ?? COPY.image;
  if (welcomeTitle) welcomeTitle.textContent = copy.title;
  if (welcomeSub) welcomeSub.textContent = copy.sub;

  if (task === 'stt') void loadRunnable('stt');

  if (task !== leaving) switchTaskCanvas(task);
}

/**
 * Loads the working conversation for the task just opened: its newest session,
 * or an empty canvas. Sessions belonging to the other task are untouched in
 * storage — they are simply not on this canvas, which is the whole point of
 * the split: a run of image generations and a run of transcriptions no longer
 * share one list, one title, or one download row.
 */
function switchTaskCanvas(task) {
  persist(); // capture whatever the tab we are leaving was holding
  const mine = sessions
    .filter((s) => (s.task ?? 'image') === task)
    .sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0));
  activeId = mine[0]?.id ?? null;
  turns = mine[0]?.turns ?? [];
  canvasTask = task;
  renderAll();
  renderHistory();
}

for (const tab of studioTabs) {
  tab.addEventListener('click', () => selectStudioTask(tab.dataset.task));
}

/** Arrow keys move along the strip, as a tablist is expected to. */
document.querySelector('.studio-tabs')?.addEventListener('keydown', (e) => {
  const delta = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
  if (!delta) return;
  e.preventDefault();
  const i = studioTabs.findIndex((t) => t.dataset.task === studioTask);
  const next = studioTabs[(i + delta + studioTabs.length) % studioTabs.length];
  selectStudioTask(next.dataset.task, { focus: true });
});

/** Reads a JSON endpoint, surfacing the server's own error message. */
async function studioApi(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: options.body ? { 'content-type': 'application/json' } : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data;
}

/** Runs an /api/infer/* stream and resolves its terminal event. */
async function studioInfer(path, payload) {
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
      try { parsed = JSON.parse(data); } catch { continue; }
      if (event === 'error') throw new Error(parsed.message ?? 'The runner failed.');
      if (event === 'done') return parsed;
    }
  }
  throw new Error('The runner stopped without a result.');
}

/**
 * Renders the model picker for a task.
 *
 * Unrunnable models are listed but not selectable. Hiding them would leave an
 * empty panel that reads as "nothing exists", which is the one answer that
 * never helps.
 */
const PICK_HOST = { stt: 'stt-pick' };

function renderStudioPick(task, models, assignedId) {
  const host = document.getElementById(PICK_HOST[task]);
  if (!host) return;
  host.replaceChildren();
  for (const [key, id] of Object.entries(PICK_HOST)) {
    const other = document.getElementById(id);
    if (other) other.hidden = key !== task;
  }

  const label = document.createElement('div');
  label.className = 'studio-pick__label';
  label.textContent = 'Speech-to-text model';
  host.append(label);

  const select = document.createElement('select');
  select.className = 'studio-pick__select';
  select.id = `${task}-model`;
  const ready = models.filter((m) => m.ready);
  for (const m of models) {
    const opt = document.createElement('option');
    opt.value = m.repo;
    // Pre-select whatever is assigned in Settings -> Your models, so the two
    // screens never disagree about which model is in use.
    if (assignedId === `local:${m.repo}`) opt.selected = true;
    opt.textContent = m.ready ? m.label : `${m.label} — ${m.reason}`;
    opt.disabled = !m.ready;
    select.append(opt);
  }
  if (!models.length) {
    const opt = document.createElement('option');
    opt.textContent = 'No models in the catalog for this task';
    opt.disabled = true;
    select.append(opt);
  } else if (!ready.length) {
    const opt = document.createElement('option');
    opt.textContent = 'None runnable here — see Get models';
    opt.disabled = true;
    select.append(opt);
  }
  host.append(select);

  const note = document.createElement('p');
  note.className = 'studio-pick__note';
  note.textContent = ready.length
    ? `${ready.length} of ${models.length} ready on this machine.`
    : 'Download a model from Get models, and install its runtime in Runtimes.';
  host.append(note);
}

async function loadRunnable(task) {
  try {
    const data = await studioApi(`/api/runnable?task=${task}`);
    renderStudioPick(task, data[task] ?? [], data.assigned?.[task]);
  } catch (err) {
    const host = document.getElementById('stt-pick');
    if (host) host.textContent = err.message;
  }
}

/* ---- Transcription ---- */

const sttFile = document.getElementById('stt-file');
const sttAttach = document.getElementById('stt-attach');
const sttAttachBox = document.getElementById('stt-attach-box');
const sttAttachList = document.getElementById('stt-attach-list');
const sttRecord = document.getElementById('stt-record');
const sttStatus = document.getElementById('stt-status');

/** Byte size the way the chat's attachment chips say it. */
function fmtAudioSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '';
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

/**
 * The chosen audio as a chip, in the same strip the chat draws over its
 * composer: icon, name, size, remove. The thing that opens the dialog is a
 * paperclip button, not a file input parked in the row — an input prints "No
 * file chosen" forever, keeps no memory of what is staged, and gives the
 * remove affordance nowhere to live.
 */
function renderSttPick() {
  sttAttachBox.hidden = !sttPending;
  sttAttachList.replaceChildren();
  if (sttGo) sttGo.disabled = !sttPending;
  if (!sttPending) return;

  const chip = document.createElement('div');
  chip.className = 'attach-chip attach-audio';

  const icon = document.createElement('span');
  icon.className = 'attach-icon';
  icon.textContent = '♪';
  chip.append(icon);

  const meta = document.createElement('span');
  meta.className = 'attach-meta';
  const name = document.createElement('span');
  name.className = 'attach-name';
  name.textContent = sttPending.name;
  name.title = sttPending.name;
  const size = document.createElement('span');
  size.className = 'attach-size';
  size.textContent = fmtAudioSize(sttPending.size);
  meta.append(name, size);
  chip.append(meta);

  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'attach-remove';
  remove.textContent = '✕';
  remove.setAttribute('aria-label', `Remove ${sttPending.name}`);
  remove.addEventListener('click', () => {
    sttPending = null;
    renderSttPick();
    sttStatus.textContent = '';
  });
  chip.append(remove);

  sttAttachList.append(chip);
}

/* ---- Transcript formats ----
   One result card per run, downloadable in the shapes a transcript is used in:
   plain text, SubRip and WebVTT subtitles, and the whole result as JSON. The
   chat's code blocks offered Copy/Download per block; this is the same idea
   with the extension chosen after the fact. */

/** `00:00:05,120` (SubRip) or `00:00:05.120` (WebVTT). Works in whole ms so
 *  rounding can never print `00:00:05,1000`. */
function cueStamp(seconds, separator) {
  let ms = Math.max(0, Math.round((Number(seconds) || 0) * 1000));
  const rest = ms % 1000;
  ms = (ms - rest) / 1000;
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(Math.floor(ms / 3600))}:${pad(Math.floor(ms / 60) % 60)}:${pad(ms % 60)}${separator}${String(rest).padStart(3, '0')}`;
}

/** Timed cues out of whichever shape the runner answered with — `segments`
 *  from the CTranslate2 path, `chunks` from the transformers one — dropping
 *  the cues the transformers path sometimes leaves untimed. */
function transcriptCues(result) {
  return (result.segments ?? result.chunks ?? [])
    .map((c) => ({
      start: Number(c?.start),
      end: Number(c?.end),
      text: String(c?.text ?? '').replace(/\s+/g, ' ').trim(),
    }))
    .filter((c) => c.text && Number.isFinite(c.start) && Number.isFinite(c.end));
}

/** A download name from the audio's: `interview.mp3` -> `interview`. */
function transcriptBase(label) {
  const stem = String(label || '')
    .replace(/\.[^.]*$/, '')
    .replace(/[^\w-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return stem || 'transcript';
}

/** The transcript in one of the offered shapes. */
function transcriptFile(turn, format) {
  const base = turn.base || 'transcript';
  const cues = turn.segments ?? [];
  if (format === 'srt') {
    const cuesText = cues
      .map((c, i) => `${i + 1}\n${cueStamp(c.start, ',')} --> ${cueStamp(c.end, ',')}\n${c.text}\n`)
      .join('\n');
    return { name: `${base}.srt`, mime: 'application/x-subrip;charset=utf-8', text: cuesText };
  }
  if (format === 'vtt') {
    const cuesText = cues
      .map((c) => `${cueStamp(c.start, '.')} --> ${cueStamp(c.end, '.')}\n${c.text}\n`)
      .join('\n');
    return { name: `${base}.vtt`, mime: 'text/vtt;charset=utf-8', text: `WEBVTT\n\n${cuesText}` };
  }
  if (format === 'json') {
    return {
      name: `${base}.json`,
      mime: 'application/json;charset=utf-8',
      text: JSON.stringify(
        {
          text: turn.text,
          language: turn.language ?? null,
          languageProbability: turn.languageProbability ?? null,
          duration: turn.duration ?? null,
          segments: cues,
        },
        null,
        2,
      ),
    };
  }
  return { name: `${base}.txt`, mime: 'text/plain;charset=utf-8', text: turn.text };
}

/** Saves text under a name — the chat's code-block Download, for transcripts. */
function downloadText(name, mime, text) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

// Delegated on the transcript host, like the code blocks' buttons, so every
// renderAll() rebuild keeps working without re-attaching anything.
els.messages.addEventListener('click', (e) => {
  const btn = e.target.closest('.tr-dl');
  if (!btn) return;
  const turn = turns[Number(btn.closest('[data-turn]')?.dataset.turn)];
  if (!turn || turn.kind !== 'transcript') return;
  if (btn.dataset.fmt === 'copy') {
    transcriptCopy(turn.text ?? '', btn);
    return;
  }
  const file = transcriptFile(turn, btn.dataset.fmt);
  downloadText(file.name, file.mime, file.text);
});

/** Puts the transcript on the clipboard and confirms it on the button. The
 *  chat's copyText lives on the other page; this is the same ladder — the async
 *  Clipboard API needs a secure context plus a permission grant, and
 *  execCommand covers everything else instead of failing outright. */
async function transcriptCopy(text, btn) {
  const prev = btn.textContent;
  const flash = (label) => {
    btn.textContent = label;
    clearTimeout(btn._flash);
    btn._flash = setTimeout(() => { btn.textContent = prev; }, 1200);
  };
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      flash('Copied');
      return;
    }
  } catch {
    /* fall through to the legacy path */
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
    flash('Copied');
  } catch {
    flash('Copy failed');
  }
}

/** Stages a Blob in _uploads and returns the path the runner expects. */
async function stageAudio(blob) {
  const res = await fetch('/api/infer/upload', {
    method: 'POST',
    headers: { 'content-type': blob.type || 'application/octet-stream' },
    body: blob,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error ?? 'Could not stage the audio.');
  return data.file;
}

function selectedModel(task) {
  const sel = document.getElementById(`${task}-model`);
  if (!sel || !sel.value) throw new Error('Pick a model first.');
  return sel.value;
}

async function runTranscription(blob, label) {
  const status = sttStatus;
  status.textContent = `Transcribing ${label}…`;
  sttRecord.disabled = true;
  try {
    // Resolve the weights file first. `/api/infer/stt` needs both halves — the
    // model and the audio — and asking it to guess which file is the weights is
    // how an earlier version of this called the runner twice and threw an empty
    // file path at it.
    const models = (await studioApi('/api/runnable?task=stt')).stt ?? [];
    const repo = selectedModel('stt');
    const model = models.find((x) => x.repo === repo);
    if (!model) throw new Error('That model is not in the catalog any more.');

    const audio = await stageAudio(blob);
    const result = await studioInfer('/api/infer/stt', {
      repo: model.repo,
      file: model.file,
      audio,
      device: 'auto',
    });
    const text = (result.text ?? '').trim();
    if (!text) {
      status.textContent = 'No speech detected.';
      return;
    }

    // The result becomes a turn with its timings and language kept whole: the
    // download shapes are built from this, and persistence only strips image
    // bytes, so the chips keep working after a reload.
    const segments = transcriptCues(result);
    const duration = Number(result.duration) || segments.at(-1)?.end || null;
    const probability = Number(result.languageProbability);
    const meta = [
      result.language ? String(result.language).toUpperCase() : '',
      Number.isFinite(probability) ? `${Math.round(probability * 100)}%` : '',
      duration ? `${duration.toFixed(1)}s` : '',
    ]
      .filter(Boolean)
      .join(' · ');

    turns.push({ role: 'user', text: `Transcribe ${label}` });
    turns.push({
      role: 'assistant',
      kind: 'transcript',
      text,
      meta,
      base: transcriptBase(label),
      language: result.language ?? null,
      languageProbability: Number.isFinite(probability) ? probability : null,
      duration,
      segments,
    });
    renderAll();
    persist();
    status.textContent = segments.length
      ? 'Transcript added — copy it or download it under the bubble.'
      : 'Transcript added — no timings came back, so text and JSON only.';
  } catch (err) {
    status.textContent = err.message;
  } finally {
    sttRecord.disabled = false;
    if (sttAttach) sttAttach.disabled = false;
    if (sttGo) sttGo.disabled = !sttPending;
  }
}

const sttGo = document.getElementById('stt-go');
let sttPending = null;

/**
 * Choosing a file stages it as a chip and arms the button; pressing the button
 * runs the job. The file used to live as "No file chosen" text inside a native
 * input, which hid both what was staged and how to unstage it — a paperclip
 * plus a chip says both, the way the chat's attachments do.
 */
sttAttach?.addEventListener('click', () => sttFile.click());
sttFile?.addEventListener('change', () => {
  sttPending = sttFile.files?.[0] ?? null;
  // Reset so re-picking the same file fires change again.
  sttFile.value = '';
  renderSttPick();
  sttStatus.textContent = sttPending ? 'Ready to transcribe.' : '';
});

sttGo?.addEventListener('click', () => {
  if (!sttPending) return;
  sttGo.disabled = true;
  if (sttAttach) sttAttach.disabled = true;
  void runTranscription(sttPending, sttPending.name);
});

let studioRecorder = null;
let studioChunks = [];
let studioStream = null;

sttRecord?.addEventListener('click', async () => {
  if (studioRecorder) {
    studioRecorder.stop();
    return;
  }
  try {
    studioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    sttStatus.textContent = err?.name === 'NotAllowedError'
      ? 'Microphone permission denied'
      : 'No microphone available';
    return;
  }
  const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
    .find((t) => MediaRecorder.isTypeSupported?.(t));
  studioRecorder = new MediaRecorder(studioStream, mimeType ? { mimeType } : undefined);
  studioChunks = [];
  studioRecorder.addEventListener('dataavailable', (e) => {
    if (e.data?.size) studioChunks.push(e.data);
  });
  studioRecorder.addEventListener('stop', () => {
    studioStream?.getTracks().forEach((t) => t.stop());
    studioStream = null;
    const blob = new Blob(studioChunks, { type: studioRecorder?.mimeType || 'audio/webm' });
    studioChunks = [];
    studioRecorder = null;
    sttRecord.textContent = 'Record';
    if (blob.size) void runTranscription(blob, 'recording');
  });
  studioRecorder.start();
  sttRecord.textContent = 'Stop';
  sttStatus.textContent = 'Recording… press Stop when done.';
});

// Open on the canvas that was last in use, not always the first tab.
selectStudioTask(canvasTask);

/* Sidebar toggle for narrow screens — same drawer pattern as the chat page. */
const SIDEBAR_BREAKPOINT = 900;
function sidebarIsDrawer() {
  return window.matchMedia(`(max-width: ${SIDEBAR_BREAKPOINT}px)`).matches;
}
function openSidebar() {
  els.sidebar.classList.add('open');
  els.backdrop.classList.add('show');
  els.sidebarToggle.setAttribute('aria-expanded', 'true');
}
function closeSidebar() {
  els.sidebar.classList.remove('open');
  els.backdrop.classList.remove('show');
  els.sidebarToggle.setAttribute('aria-expanded', 'false');
}
els.sidebarToggle.addEventListener('click', () => {
  if (sidebarIsDrawer()) {
    if (els.sidebar.classList.contains('open')) closeSidebar();
    else openSidebar();
  }
});
els.sidebarClose.addEventListener('click', closeSidebar);
els.backdrop.addEventListener('click', closeSidebar);
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && els.sidebar.classList.contains('open')) closeSidebar();
});

/* ---------------- Local image generation ---------------- */

/**
 * A pseudo-provider for images made on this machine.
 *
 * The Image tab used to list only cloud services, so a locally installed
 * diffusers model — which this app can genuinely run through
 * /api/infer/image — was invisible from the one place generation happens. The
 * entry is only added when the catalog actually has such a model; otherwise it
 * would be a permanently broken option in the same list.
 */

/** Loads local image models and reflects them in the provider dropdown. */
async function loadLocalImage() {
  let models = [];
  let assigned = null;
  try {
    const data = await studioApi('/api/runnable?task=image');
    models = data.image ?? [];
    assigned = data.assigned ?? null;
  } catch {
    models = [];
  }
  localImageModels = models;

  const select = els.provider;
  if (!select) return;
  // Drop any previous local entry so a refresh cannot accumulate them.
  for (const opt of [...select.options]) {
    if (opt.value === LOCAL_IMAGE_ID) opt.remove();
  }

  const ready = models.filter((m) => m.ready);
  if (models.length) {
    const opt = document.createElement('option');
    opt.value = LOCAL_IMAGE_ID;
    opt.textContent = ready.length
      ? 'On this PC (diffusers)'
      : 'On this PC — needs setup';
    select.prepend(opt);
  }

  // Only claim the local runner when one is actually assigned. Defaulting to it
  // made the Studio pick a model the user had not chosen.
  const assignedRepo =
    assigned?.image?.startsWith('local:') ? assigned.image.slice('local:'.length) : '';
  if (assignedRepo && models.some((m) => m.repo === assignedRepo)) {
    select.value = LOCAL_IMAGE_ID;
    imageModel = assignedRepo;
  } else if (ready.length) {
    select.value = select.options[1]?.value ?? select.value;
  }
  syncControls();
  syncChips();
}

/** True when the dropdown is currently pointing at the local runner. */
function localImageSelected() {
  return els.provider?.value === LOCAL_IMAGE_ID;
}
