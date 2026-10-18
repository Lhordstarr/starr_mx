import { notifySuccess, notifyError, notifyWarning } from './notify.js';
import { initShaderBackdrop } from './shader.js';


const form = document.getElementById('keys');
const clearAllBtn = document.getElementById('clear-all');
const themeBtn = document.getElementById('theme-toggle');

const THEME_KEY = 'llm-chat-theme';

/* ---------------- Theme ---------------- */

function applyTheme(theme) {
  if (theme === 'dark' || theme === 'light') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

applyTheme(localStorage.getItem(THEME_KEY));

themeBtn.addEventListener('click', () => {
  const order = [null, 'light', 'dark'];
  const current = document.documentElement.dataset.theme ?? null;
  const next = order[(order.indexOf(current) + 1) % order.length];
  applyTheme(next);
  if (next) localStorage.setItem(THEME_KEY, next);
  else localStorage.removeItem(THEME_KEY);
});

/* ---------------- Tabs (API Keys / Tools / Models) ---------------- */

const TABS = ['keys', 'tools', 'models'];
/** Sub-views of the models tab, mirrored in the URL as "#models/<subview>". */
const SUBVIEWS = ['local', 'catalog', 'downloads', 'browse', 'runtimes'];
const tabBtns = Object.fromEntries(TABS.map((n) => [n, document.getElementById(`tab-${n}`)]));
const tabPanels = Object.fromEntries(TABS.map((n) => [n, document.getElementById(`panel-${n}`)]));
const loadedTabs = new Set();

function tabFromHash() {
  const h = location.hash.replace(/^#/, '').split('/')[0];
  return TABS.includes(h) ? h : 'keys';
}

/** Shows one tab; loads its data on first show so hidden tabs cost nothing. */
async function selectTab(name, push = true) {
  if (!TABS.includes(name)) name = 'keys';
  for (const n of TABS) {
    const active = n === name;
    tabBtns[n].setAttribute('aria-selected', String(active));
    tabBtns[n].tabIndex = active ? 0 : -1;
    tabPanels[n].hidden = !active;
  }
  // Carry the current subview across a tab change so returning to models does
  // not silently reset it to "Your models".
  const want = `#${name}${name === 'models' ? `/${currentSubview}` : ''}`;
  if (push && location.hash !== want) location.hash = want;
  if (name === 'models') {
    // Every entry, not just the first. `loadedTabs` made the models tab load
    // once and then trust itself: leave it, delete something, come back, and
    // the page still showed it. A tab that displays mutable state has to
    // re-read it on entry; the lazy-load saving is one request the user is
    // already paying for by being on the tab.
    await syncDerived();
    await refreshHfFiles();
  } else if (!loadedTabs.has(name)) {
    loadedTabs.add(name);
    if (name === 'tools') await refreshTools();
  }
}

for (const [i, n] of TABS.entries()) {
  tabBtns[n].addEventListener('click', () => selectTab(n));
  // Roving tabindex with arrow keys, per the tabs pattern.
  tabBtns[n].addEventListener('keydown', (e) => {
    let j = null;
    if (e.key === 'ArrowRight') j = (i + 1) % TABS.length;
    else if (e.key === 'ArrowLeft') j = (i - 1 + TABS.length) % TABS.length;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = TABS.length - 1;
    if (j !== null) {
      e.preventDefault();
      tabBtns[TABS[j]].focus();
      selectTab(TABS[j]);
    }
  });
}

window.addEventListener('hashchange', () => {
  const { tab, sub } = subviewFromHash();
  selectTab(tab, false);
  // Only when it disagrees, or a plain tab change would bounce the subview back.
  if (tab === 'models' && sub && sub !== currentSubview) setSubview(sub, false);
});

/* ---------------- Notifications ----------------
   Every feedback path below uses the typed helpers from notify.js directly, so
   there is no local wrapper here. This page previously carried its own
   single-slot toast, duplicated from the chat page, where a second message
   arriving before the first expired replaced it — saving a key and then hitting
   a validation error on the next field would lose one of the two. */

/* Give the glass something to blur, exactly as on the chat page. Deferred
   until after first paint so the shader's compile and first frame never delay
   the settings being usable — the canvas stays hidden until ready. */
requestAnimationFrame(() => {
  const canvas = document.getElementById('shader-bg');
  if (!canvas) return;
  const backdrop = initShaderBackdrop(canvas);
  if (backdrop) canvas.dataset.ready = '';
  else canvas.remove(); // no WebGL: drop the element rather than leave a dead box
});

/* ---------------- API ---------------- */

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: options.body ? { 'content-type': 'application/json' } : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data;
}

const escapeHtml = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/**
 * The direct Hugging Face download URL for one file in a repo.
 *
 * The same resolve path the app's own downloader builds, so a link copied from
 * here fetches exactly what a re-download would. Each path segment is encoded
 * individually — a filename with a space has to resolve as a path, not as one
 * escaped blob.
 */
function hfFileUrl(repo, file) {
  const p = String(file)
    .split('/')
    .map((seg) => encodeURIComponent(seg))
    .join('/');
  return `https://huggingface.co/${repo}/resolve/main/${p}`;
}

/** Puts text on the clipboard, with a legacy fallback for non-secure contexts. */
async function copyToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      /* fall through to execCommand */
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
    return done;
  } catch {
    return false;
  }
}

/** Source badge: saved here, inherited from .env, or not set. */
function badge(field) {
  if (!field.configured) return '<span class="badge warn">not set</span>';
  if (field.source === 'settings') return '<span class="badge ok">saved</span>';
  return '<span class="badge info">from .env</span>';
}

/**
 * Built-in providers as one compact list instead of a stack of full-width
 * cards. Each row is the whole entry point; the key input lives in a popup, so
 * six providers cost six short lines instead of six tall forms. The rows stay
 * real buttons, so this is reachable by keyboard and announced as a dialog.
 */
const keyFields = new Map();

function renderProviderRow(field) {
  keyFields.set(field.envKey, field);

  const row = document.createElement('button');
  row.type = 'button';
  row.className = 'key-row';
  row.setAttribute('aria-haspopup', 'dialog');
  row.innerHTML = `
    <span class="key-row__label">${escapeHtml(field.label)}</span>
    ${badge(field)}
    <span class="key-row__env">${escapeHtml(field.envKey)}</span>
    <svg class="key-row__chev" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="m9 6 6 6-6 6" />
    </svg>`;

  row.addEventListener('click', () => openKeyModal(field));
  return row;
}

/* ---------------- API key popup ---------------- */

const keyModal = document.getElementById('key-modal-backdrop');
const keyModalTitle = document.getElementById('key-modal-title');
const keyModalEnv = document.getElementById('key-modal-env');
const keyModalNote = document.getElementById('key-modal-note');
const keyModalInput = document.getElementById('key-modal-input');
const keyModalSave = document.getElementById('key-modal-save');
const keyModalRemove = document.getElementById('key-modal-remove');
const keyModalClose = document.getElementById('key-modal-close');
/** Which provider the popup is currently editing; null while closed. */
let keyModalField = null;

function openKeyModal(field) {
  keyModalField = field;
  keyModalTitle.textContent = field.label;
  keyModalEnv.textContent = field.envKey;
  const fromEnv = field.source === 'env';
  keyModalInput.value = '';
  keyModalInput.placeholder = field.masked
    ? `${field.masked}   (${fromEnv ? 'override' : 'current'})`
    : field.placeholder;
  keyModalNote.textContent = fromEnv
    ? 'Currently read from .env. Saving here overrides it for this server.'
    : '';
  keyModalNote.hidden = !fromEnv;
  // Nothing to remove unless this one came from our own storage.
  keyModalRemove.disabled = field.source !== 'settings';
  keyModalSave.disabled = false;
  keyModal.hidden = false;
  keyModalInput.focus();
}

function closeKeyModal() {
  keyModal.hidden = true;
  // Don't leave a pasted secret sitting in a detached input.
  keyModalInput.value = '';
  keyModalField = null;
}

keyModalClose.addEventListener('click', closeKeyModal);
// Clicking the scrim closes; clicks inside the dialog must not bubble into it.
keyModal.addEventListener('click', (e) => {
  if (e.target === keyModal) closeKeyModal();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !keyModal.hidden) closeKeyModal();
});

keyModalSave.addEventListener('click', async () => {
  const field = keyModalField;
  if (!field) return;
  const value = keyModalInput.value.trim();
  if (!value) {
    notifyWarning('Enter a key first');
    keyModalInput.focus();
    return;
  }
  keyModalSave.disabled = true;
  try {
    await api('/api/settings', {
      method: 'PUT',
      body: JSON.stringify({ [field.envKey]: value }),
    });
    closeKeyModal();
    await refresh();
    notifySuccess(`${field.label} key saved`);
  } catch (err) {
    notifyError(err.message);
    keyModalSave.disabled = false;
  }
});

keyModalRemove.addEventListener('click', async () => {
  const field = keyModalField;
  if (!field) return;
  keyModalRemove.disabled = true;
  try {
    await api(`/api/settings/${encodeURIComponent(field.envKey)}`, { method: 'DELETE' });
    closeKeyModal();
    await refresh();
    notifySuccess(`${field.label} key removed`);
  } catch (err) {
    notifyError(err.message);
    keyModalRemove.disabled = false;
  }
});
async function refresh() {
  const data = await api('/api/settings');
  form.innerHTML = '';
  keyFields.clear();
  for (const field of data.fields) form.append(renderProviderRow(field));
}

clearAllBtn.addEventListener('click', async () => {
  if (!confirm('Remove every key saved on this server? Keys in .env are not affected.')) return;
  clearAllBtn.disabled = true;
  try {
    await api('/api/settings', { method: 'DELETE' });
    await refresh();
    notifySuccess('All saved keys removed');
  } catch (err) {
    notifyError(err.message);
  } finally {
    clearAllBtn.disabled = false;
  }
});

/* ---------------- Custom providers ---------------- */

const customForm = document.getElementById('custom-form');
const customList = document.getElementById('custom-list');
const cpSubmit = document.getElementById('cp-submit');

customForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    label: document.getElementById('cp-label').value.trim(),
    baseUrl: document.getElementById('cp-url').value.trim(),
    apiKey: document.getElementById('cp-key').value.trim(),
    stt: document.getElementById('cp-stt').checked,
  };
  if (!payload.label || !payload.baseUrl) {
    notifyWarning('Name and base URL are required');
    return;
  }

  cpSubmit.disabled = true;
  try {
    await api('/api/custom-providers', { method: 'POST', body: JSON.stringify(payload) });
    customForm.reset();
    await refreshCustom();
    await refreshDictation();
    notifySuccess(`${payload.label} added — it is in the chat now`);
  } catch (err) {
    notifyError(err.message);
  } finally {
    cpSubmit.disabled = false;
  }
});

/** Renders one saved custom provider with its edit and delete controls. */
function renderCustomProvider(p) {
  const row = document.createElement('div');
  row.className = 'row custom-row';
  row.innerHTML = `
    <div class="row-top">
      <span class="provider">${escapeHtml(p.label)}</span>
      ${p.hasKey ? '<span class="badge ok">key saved</span>' : '<span class="badge info">no key</span>'}
      ${p.stt ? '<span class="badge ok">dictation</span>' : ''}
    </div>
    <p class="env-name">${escapeHtml(p.baseUrl)}</p>
    <div class="input-row">
      <input type="text" spellcheck="false" autocomplete="off"
             placeholder="Name" aria-label="Provider name" value="${escapeHtml(p.label)}" />
      <input type="url" spellcheck="false" autocomplete="off"
             placeholder="Base URL" aria-label="Base URL" value="${escapeHtml(p.baseUrl)}" />
    </div>
    <div class="input-row">
      <input type="password" spellcheck="false" autocomplete="off"
             placeholder="${p.masked ? `Key ${escapeHtml(p.masked)} — leave blank to keep` : 'API key (optional)'}"
             aria-label="API key" />
      <label class="checkbox-label inline">
        <input type="checkbox" ${p.stt ? 'checked' : ''} />
        <span>Dictation</span>
      </label>
    </div>
    <div class="input-row">
      <button type="button" class="btn-primary">Save</button>
      <button type="button" class="btn-ghost btn-scan">Scan models</button>
      <button type="button" class="btn-danger">Remove</button>
    </div>
    <div class="scan-result" hidden></div>`;

  // Query by aria-label rather than position: the row contains several
  // .input-row blocks, and structural selectors like nth-of-type count every
  // div sibling.
  const nameInput = row.querySelector('input[aria-label="Provider name"]');
  const urlInput = row.querySelector('input[aria-label="Base URL"]');
  const keyInput = row.querySelector('input[aria-label="API key"]');
  const sttBox = row.querySelector('input[type="checkbox"]');
  const [saveBtn, scanBtn, removeBtn] = row.querySelectorAll('button');
  const scanResult = row.querySelector('.scan-result');

  /**
   * Asks the provider's own /models endpoint what it has. Runs on demand
   * rather than on page load so opening settings never fires a burst of
   * requests at remote providers.
   */
  async function scan() {
    scanBtn.disabled = true;
    scanBtn.textContent = 'Scanning…';
    scanResult.hidden = false;
    scanResult.textContent = 'Asking the provider for its models…';
    scanResult.className = 'scan-result pending';
    try {
      const data = await api(`/api/custom-providers/${encodeURIComponent(p.id)}/models`);
      if (!data.ok) {
        scanResult.className = 'scan-result error';
        scanResult.textContent = data.note;
        return;
      }
      const stt = data.sttModels ?? [];
      const chat = data.models.filter((m) => !stt.includes(m));
      scanResult.className = 'scan-result';
      scanResult.innerHTML = `
        <p class="scan-note">${escapeHtml(data.note)}</p>
        ${renderModelList('Chat models', chat)}
        ${stt.length ? renderModelList('Dictation models', stt) : ''}`;

      // A provider whose endpoint offers no STT model shouldn't keep claiming
      // it supports dictation, so offer to fix the flag.
      if (p.stt && stt.length === 0) {
        const warn = document.createElement('p');
        warn.className = 'scan-note warn';
        warn.textContent =
          'Dictation is switched on but the provider lists no speech-to-text model. It will not appear for dictation until it does.';
        scanResult.appendChild(warn);
      }
    } catch (err) {
      scanResult.className = 'scan-result error';
      scanResult.textContent = err.message;
    } finally {
      scanBtn.disabled = false;
      scanBtn.textContent = 'Scan models';
    }
  }

  scanBtn.addEventListener('click', scan);

  saveBtn.addEventListener('click', async () => {
    saveBtn.disabled = true;
    try {
      const patch = {
        label: nameInput.value.trim(),
        baseUrl: urlInput.value.trim(),
        stt: sttBox.checked,
      };
      // Omitted when blank so the stored key survives an unrelated edit.
      const key = keyInput.value.trim();
      if (key) patch.apiKey = key;
      await api(`/api/custom-providers/${encodeURIComponent(p.id)}`, {
        method: 'PUT',
        body: JSON.stringify(patch),
      });
      keyInput.value = '';
      await refreshCustom();
      await refreshDictation();
      notifySuccess('Provider updated');
    } catch (err) {
      notifyError(err.message);
      saveBtn.disabled = false;
    }
  });

  removeBtn.addEventListener('click', async () => {
    if (!confirm(`Remove "${p.label}"? Any conversation using it stays, but chat will fail until you pick another provider.`)) {
      return;
    }
    removeBtn.disabled = true;
    try {
      await api(`/api/custom-providers/${encodeURIComponent(p.id)}`, { method: 'DELETE' });
      await refreshCustom();
      await refreshDictation();
      notifySuccess(`${p.label} removed`);
    } catch (err) {
      notifyError(err.message);
      removeBtn.disabled = false;
    }
  });

  customList.appendChild(row);
}

/** Collapsible chip list of model ids, shown by a scan. */
function renderModelList(title, models) {
  if (!models.length) return '';
  return `
    <details class="model-list-block" open>
      <summary>${escapeHtml(title)} (${models.length})</summary>
      <div class="model-chips">${models
        .map((m) => `<code class="chip-model-id">${escapeHtml(m)}</code>`)
        .join('')}</div>
    </details>`;
}

async function refreshCustom() {
  const { providers } = await api('/api/custom-providers');
  customList.innerHTML = '';

  if (!providers.length) {
    const empty = document.createElement('p');
    empty.className = 'section-sub';
    empty.textContent = 'No custom providers yet.';
    customList.appendChild(empty);
    return;
  }
  providers.forEach(renderCustomProvider);
}

/* ---------------- Dictation providers ---------------- */

const dictationList = document.getElementById('dictation-list');

/**
 * Lists every provider that can transcribe, so it's obvious which one the mic
 * will actually use. Capability comes from the server's own provider flags, so
 * this renders instantly and costs nothing; the model list is scanned on demand
 * exactly like a custom provider's.
 */
function renderDictationProvider(p) {
  const ready = p.configured;
  const row = document.createElement('div');
  row.className = 'row dictation-row';
  row.innerHTML = `
    <div class="row-top">
      <span class="provider">
        <svg class="mic-mini" viewBox="0 0 24 24" width="13" height="13" fill="none"
             stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
          <rect x="9" y="2" width="6" height="11" rx="3" />
          <path d="M5 11a7 7 0 0 0 14 0" />
          <path d="M12 18v4" />
        </svg>
        ${escapeHtml(p.label)}
      </span>
      <span>
        ${p.custom ? '<span class="badge info">custom</span> ' : ''}
        ${ready
          ? '<span class="badge ok">ready</span>'
          : '<span class="badge warn">needs a key</span>'}
      </span>
    </div>
    <div class="input-row">
      <button type="button" class="btn-ghost btn-scan-stt">Scan dictation models</button>
    </div>
    <div class="scan-result" hidden></div>`;

  const scanBtn = row.querySelector('.btn-scan-stt');
  const scanResult = row.querySelector('.scan-result');

  scanBtn.addEventListener('click', async () => {
    scanBtn.disabled = true;
    scanBtn.textContent = 'Scanning…';
    scanResult.hidden = false;
    scanResult.className = 'scan-result pending';
    scanResult.textContent = 'Asking the provider for its speech-to-text models…';
    try {
      // Built-in providers are scanned through the providers listing, which
      // already filters STT models out of the chat list.
      const data = await api(`/api/providers`);
      const entry = data.providers.find((x) => x.id === p.id);
      const models = entry?.sttModels ?? [];
      scanResult.className = 'scan-result';
      if (!models.length) {
        scanResult.innerHTML =
          '<p class="scan-note warn">No speech-to-text models listed. This provider cannot be used for dictation until it publishes one.</p>';
      } else {
        scanResult.innerHTML = renderModelList('Dictation models', models);
      }
    } catch (err) {
      scanResult.className = 'scan-result error';
      scanResult.textContent = err.message;
    } finally {
      scanBtn.disabled = false;
      scanBtn.textContent = 'Scan dictation models';
    }
  });

  dictationList.appendChild(row);
}

async function refreshDictation() {
  const { providers } = await api('/api/dictation');
  dictationList.innerHTML = '';

  if (!providers.length) {
    const empty = document.createElement('p');
    empty.className = 'section-sub';
    empty.textContent = 'No provider offers speech-to-text.';
    dictationList.appendChild(empty);
    return;
  }

  const intro = document.createElement('p');
  intro.className = 'section-sub';
  const ready = providers.filter((p) => p.configured).length;
  intro.textContent = ready
    ? `${ready} of ${providers.length} ready. The composer mic uses your current provider when it can, otherwise the first ready one.`
    : 'Add a key to one of these to enable dictation.';
  dictationList.appendChild(intro);

  providers.forEach(renderDictationProvider);
}

/* ---------------- Local Ollama models ---------------- */

const localModelsEl = document.getElementById('local-models');
/** The model count lives in the collapsible section's heading, so the folded
 *  state still says how many models are installed. */
const ollamaCountEl = document.getElementById('ollama-count');
const pullForm = document.getElementById('pull-form');
const pullName = document.getElementById('pull-name');
const pullSubmit = document.getElementById('pull-submit');
const pullStatus = document.getElementById('pull-status');
const pullProgress = document.getElementById('pull-progress');
const pullBar = document.getElementById('pull-bar');

/** Bytes to "1.9 GB" — the daemon reports raw byte counts. */
function fmtSize(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = bytes;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v >= 100 ? Math.round(v) : v.toFixed(1)} ${units[u]}`;
}

/** A pull already running from this page (the server guards too, with a 503). */
let pulling = false;

function renderLocalModel(m) {
  const row = document.createElement('div');
  row.className = 'row local-model-row';
  const meta = [m.parameterSize, m.quantization, m.family]
    .filter(Boolean)
    .join(' · ');
  const when = m.modifiedAt ? new Date(m.modifiedAt).toLocaleDateString() : '';
  row.innerHTML = `
    <div class="row-top">
      <span class="provider model-name">${escapeHtml(m.name)}</span>
      <span>
        <span class="badge info">${escapeHtml(fmtSize(m.size))}</span>
        <button type="button" class="btn-ghost btn-chat-model" title="Start a chat with this model">Chat</button>
        <button type="button" class="btn-ghost btn-del-model">Delete</button>
      </span>
    </div>
    ${meta || when ? `<div class="row-note">${escapeHtml([meta, when].filter(Boolean).join(' · '))}</div>` : ''}`;

  row.querySelector('.row-top').prepend(modelTile(m.name));

  // "Chat" is the whole point of having the model: hand the chat page this
  // provider+model pair and start a new conversation on it. Without this the
  // list only ever offered Delete, which made it look like a filing cabinet.
  row.querySelector('.btn-chat-model').addEventListener('click', () => {
    // In place, not a new tab: "Chat" means go use this model, and opening a
    // second window leaves the model list behind for no reason.
    window.location.href = `/?provider=ollama&model=${encodeURIComponent(m.name)}`;
  });

  // Two-step delete, same "✕ → Sure?" shape as the chat history.
  const delBtn = row.querySelector('.btn-del-model');
  delBtn.addEventListener('click', async () => {
    if (delBtn.dataset.armed !== '1') {
      delBtn.dataset.armed = '1';
      delBtn.textContent = 'Sure?';
      setTimeout(() => {
        delBtn.dataset.armed = '';
        delBtn.textContent = 'Delete';
      }, 2600);
      return;
    }
    delBtn.disabled = true;
    try {
      await api('/api/ollama/delete', {
        method: 'POST',
        body: JSON.stringify({ name: m.name }),
      });
      await refreshLocalModels();
    } catch (err) {
      delBtn.disabled = false;
      delBtn.dataset.armed = '';
      delBtn.textContent = 'Delete';
      pullStatus.hidden = false;
      pullStatus.className = 'scan-result error';
      pullStatus.textContent = err instanceof Error ? err.message : String(err);
    }
  });

  localModelsEl.appendChild(row);
}

/** One refresh at a time: two overlapping runs would each clear, then both
 *  append, and the section would show every model twice. */
let localModelsFetch = null;

async function refreshLocalModels() {
  if (localModelsFetch) return localModelsFetch;
  localModelsFetch = (async () => {
    localModelsEl.innerHTML = '';
    if (ollamaCountEl) ollamaCountEl.textContent = '';
    // The daemon can take a moment to answer. An empty section with no note
    // reads as "no models" — say what is happening instead.
    localModelsEl.append(hfNote('Reading your Ollama models…'));
    let models;
    try {
      ({ models } = await api('/api/ollama/models'));
    } catch (err) {
      localModelsEl.innerHTML = '';
      const p = document.createElement('p');
      p.className = 'section-sub';
      p.textContent = `Ollama daemon unreachable: ${err instanceof Error ? err.message : String(err)}`;
      localModelsEl.append(p);
      return;
    }
    localModelsEl.innerHTML = '';
    if (!models?.length) {
      const p = document.createElement('p');
      p.className = 'section-sub';
      p.textContent = 'No local models. Pull one by name below.';
      localModelsEl.append(p);
      return;
    }
    if (ollamaCountEl) ollamaCountEl.textContent = ` (${models.length})`;
    models.forEach(renderLocalModel);
  })().finally(() => {
    localModelsFetch = null;
  });
  return localModelsFetch;
}

/** Daemon status lines into plain sentences, with a short digest + percent. */
function describePull(p) {
  if (typeof p.total === 'number' && p.total > 0 && typeof p.completed === 'number') {
    const pct = Math.min(100, (p.completed / p.total) * 100).toFixed(1);
    const short = typeof p.digest === 'string' ? ` ${p.digest.replace(/^sha256:/, '').slice(0, 12)}` : '';
    // The bytes as well as the percentage: a percentage alone cannot tell you
    // whether the thing taking four minutes is 400 MB or 4 GB, and that is the
    // difference between leaving it running and going to do something else.
    const size = ` — ${fmtSize(p.completed)} / ${fmtSize(p.total)}`;
    return `Downloading${short} — ${pct}%${size}`;
  }
  switch (p.status) {
    case 'pulling manifest':
      return 'Fetching manifest…';
    case 'verifying sha256 digest':
      return 'Verifying…';
    case 'writing manifest':
      return 'Writing…';
    case 'removing any unused layers':
      return 'Cleaning up…';
    case 'success':
      return 'Done.';
    default:
      return p.status;
  }
}

/**
 * Reads a named-event SSE body, the shape both the Ollama pull and the Hub
 * download streams use (event:/data: fields, blank-line framed). One parser for
 * both, rather than a second copy of the frame splitting next to the first.
 */
async function sseStream(body, onEvent) {
  const reader = body.getReader();
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
      // A throw here must reach the caller, so it leaves the reader locked up
      // on purpose rather than swallowing the frame.
      await onEvent(event, parsed);
    }
  }
}

pullForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = pullName.value.trim();
  if (!name || pulling) return;
  pulling = true;
  pullSubmit.disabled = true;
  pullStatus.hidden = false;
  pullStatus.className = 'scan-result pending';
  pullStatus.textContent = `Pulling ${name}…`;
  pullProgress.hidden = false;
  pullBar.style.width = '0%';
  try {
    const res = await fetch('/api/ollama/pull', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error ?? `HTTP ${res.status}`);
    }
    // Already pulling this model server-side (a reload, or another tab). The
    // ollama daemon is mid-transfer either way; follow it rather than refuse.
    if ((res.headers.get('content-type') ?? '').includes('application/json')) {
      const data = await res.json();
      if (data.attached) {
        beginTransfer('pull', `ollama:${name}`, {}).update({ state: 'running' });
        void syncTransfersSoon();
        pullStatus.textContent = `Already pulling ${name} — see Downloads`;
        pull.disabled = false;
        return;
      }
      throw new Error(data.error ?? 'Pull refused');
    }
    const transfer = beginTransfer('pull', `ollama:${name}`, {});
    let finished = false;
    await sseStream(res.body, async (event, parsed) => {
      if (event === 'progress') {
        transfer.update({
          received: typeof parsed.completed === 'number' ? parsed.completed : 0,
          total: typeof parsed.total === 'number' && parsed.total > 0 ? parsed.total : null,
          percent:
            typeof parsed.total === 'number' && parsed.total > 0 && typeof parsed.completed === 'number'
              ? (parsed.completed / parsed.total) * 100
              : 0,
        });
        if (typeof parsed.total === 'number' && parsed.total > 0 &&
            typeof parsed.completed === 'number') {
          pullBar.style.width = `${Math.min(100, (parsed.completed / parsed.total) * 100).toFixed(1)}%`;
        }
        pullStatus.textContent = describePull(parsed);
      } else if (event === 'error') {
        throw new Error(parsed.message);
      } else if (event === 'done') {
        finished = true;
        transfer.finish('done');
        pullBar.style.width = '100%';
        pullStatus.className = 'scan-result';
        pullStatus.textContent = `Pulled ${parsed.model}.`;
        pullName.value = '';
        await refreshLocalModels();
      }
    });
    if (finished) return;
  } catch (err) {
    pullStatus.className = 'scan-result error';
    pullStatus.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    pulling = false;
    pullSubmit.disabled = false;
  }
});

/* ---------------- Agent tools (moved from the old /tools page) ---------------- */

const toolsListEl = document.getElementById('tools-list');
const toolsProvidersNote = document.getElementById('tools-providers');

/** One-line summary of a tool's arguments for the details block. */
function toolParamsSummary(tool) {
  const props = tool.parameters?.properties ?? {};
  const required = new Set(tool.parameters?.required ?? []);
  return Object.entries(props).map(([name, p]) => {
    const bits = [p.type];
    if (required.has(name)) bits.push('required');
    if (p.description) bits.push(p.description);
    const li = document.createElement('li');
    const code = document.createElement('code');
    code.textContent = name;
    li.append(code, ` — ${bits.join(' · ')}`);
    return li;
  });
}

function renderTool(tool) {
  const card = document.createElement('div');
  card.className = 'tool-card';

  const head = document.createElement('div');
  head.className = 'tool-head';

  const name = document.createElement('span');
  name.className = 'tool-name';
  name.textContent = tool.name;

  const badges = document.createElement('span');
  badges.className = 'tool-badges';
  if (tool.keyless) {
    const b = document.createElement('span');
    b.className = 'badge ok';
    b.textContent = 'keyless';
    badges.append(b);
  }

  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'tool-toggle';
  toggle.setAttribute('role', 'switch');
  toggle.setAttribute('aria-checked', String(tool.enabled));
  toggle.setAttribute('aria-label', `Enable ${tool.name}`);
  toggle.dataset.on = tool.enabled ? '1' : '';
  toggle.addEventListener('click', async () => {
    const next = !tool.enabled;
    toggle.disabled = true;
    try {
      const res = await fetch(`/api/tools/${encodeURIComponent(tool.name)}`, {
        method: 'PUT',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ enabled: next }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      tool.enabled = data.tool?.enabled ?? next;
      toggle.dataset.on = tool.enabled ? '1' : '';
      toggle.setAttribute('aria-checked', String(tool.enabled));
      toggle.setAttribute('aria-label', `${tool.enabled ? 'Disable' : 'Enable'} ${tool.name}`);
      notifySuccess(`${tool.name} ${tool.enabled ? 'enabled' : 'disabled'}`);
    } catch (err) {
      notifyError(err instanceof Error ? err.message : String(err));
    } finally {
      toggle.disabled = false;
    }
  });

  head.append(name, badges, toggle);

  const desc = document.createElement('p');
  desc.className = 'tool-desc';
  desc.textContent = tool.description;

  const details = document.createElement('details');
  details.className = 'tool-args';
  const summary = document.createElement('summary');
  summary.textContent = 'Arguments';
  const ul = document.createElement('ul');
  for (const li of toolParamsSummary(tool)) ul.append(li);
  details.append(summary, ul);

  card.append(head, desc, details);
  return card;
}

async function refreshTools() {
  let tools = null;
  try {
    const res = await fetch('/api/tools');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    ({ tools } = await res.json());
  } catch {
    tools = null;
  }
  toolsListEl.innerHTML = '';
  if (!tools?.length) {
    const p = document.createElement('p');
    p.className = 'section-sub';
    p.textContent = 'Could not load tools. Is the server running?';
    toolsListEl.append(p);
  } else {
    for (const tool of tools) toolsListEl.append(renderTool(tool));
  }

  try {
    const res = await fetch('/api/providers');
    if (!res.ok) throw new Error();
    const { providers } = await res.json();
    const capable = (providers ?? []).filter((p) => p.supportsTools).map((p) => p.label);
    toolsProvidersNote.textContent = capable.length
      ? `Tool calls work with: ${capable.join(', ')}. Anywhere else the chat answers directly.`
      : 'No configured provider supports tool calls right now.';
  } catch {
    toolsProvidersNote.textContent = 'Could not load provider support.';
  }
}

/* ---------------- Optional local runtimes ---------------- */

const rtStatus = document.getElementById('rt-status');
const rtList = document.getElementById('rt-list');
const rtInstall = document.getElementById('rt-install');
/* ---------------- Driver installer ----------------
   One button that runs the script and streams what it prints. It opens on a
   dry run: the script installs system packages and needs root, so the safe
   reading has to be the default one, and applying has to be asked for
   deliberately. */

const rtUpdateDrivers = document.getElementById('rt-update-drivers');
const rtDriverLog = document.getElementById('rt-driver-log');

function driverLog(line) {
  if (!rtDriverLog) return;
  rtDriverLog.hidden = false;
  const atBottom =
    rtDriverLog.scrollTop + rtDriverLog.clientHeight >= rtDriverLog.scrollHeight - 24;
  rtDriverLog.textContent += `${line}\n`;
  // Follow the tail only if the reader was already at it. Yanking the view back
  // down while someone is reading an earlier error is its own annoyance.
  if (atBottom) rtDriverLog.scrollTop = rtDriverLog.scrollHeight;
}

/** Runs the installer. `apply` is the only choice, so nothing user-supplied
 *  reaches the command line. */
async function runDriverInstaller(apply, password) {
  // Fresh run from a button press: it has not supplied a password yet, so a
  // privilege request from the server is allowed to open the prompt.
  driverPassTried = Boolean(password);
  if (!rtUpdateDrivers) return;
  const label = rtUpdateDrivers.querySelector('span');
  const was = label?.textContent;
  if (rtDriverLog) rtDriverLog.textContent = '';
  rtUpdateDrivers.disabled = true;
  if (label) label.textContent = apply ? 'Installing…' : 'Checking…';
  try {
    const res = await fetch('/api/drivers/update', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      // In the body, never the query string: a URL would be written to whatever
      // sits in front of this server.
      body: JSON.stringify({ apply, password }),
    });
    if (!res.ok || !res.body) {
      driverLog(`Could not reach the installer (HTTP ${res.status}).`);
      return;
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const frames = buf.split('\n\n');
      buf = frames.pop() ?? '';
      for (const frame of frames) {
        const dataLine = frame.split('\n').find((l) => l.startsWith('data: '));
        if (!dataLine) continue;
        let payload;
        try {
          payload = JSON.parse(dataLine.slice(6));
        } catch {
          continue;
        }
        if (payload.line) driverLog(payload.line);
        else if (payload.message) driverLog(payload.message);
        else if (payload.command) {
          // The server reached the point where it needed a password and had
          // none. This is the only thing that should ever open the prompt:
          // an install that never needs root (a cached sudo, a passwordless
          // account, a no-op) just finishes without asking.
          driverLog('That step needs your password.');
          if (driverPassTried) {
            // Already supplied one and it still asked: the password was
            // wrong or sudo is refusing non-interactively. Don't loop.
            driverLog('It is still asking for a password — run this in a terminal to see why:');
            driverLog(`  ${payload.command}`);
          } else {
            openPassPrompt();
          }
        }
      }
    }
    // Whatever changed on the system is worth re-reading.
    await loadRuntimes({ force: true });
    await refreshBackends();
  } catch (err) {
    driverLog(`Failed: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    rtUpdateDrivers.disabled = false;
    if (label) label.textContent = was ?? 'Update GPU drivers';
  }
}

/**
 * Whether the current driver-install run has already offered (or supplied) a
 * password, so a second privilege request does not re-open the prompt forever.
 */
let driverPassTried = false;

/**
 * What the password prompt currently does with what it collects, and which
 * button Escape should hand focus back to.
 */
let passPromptRun = null;
let passPromptReturn = null;

const rtPass = document.getElementById('rt-pass');
const rtPassInput = document.getElementById('rt-pass-input');
const rtPassGo = document.getElementById('rt-pass-go');
const rtPassCancel = document.getElementById('rt-pass-cancel');

/** Closes the prompt and makes sure the secret does not outlive it. */
function closePassPrompt() {
  if (rtPass) rtPass.hidden = true;
  if (rtPassInput) rtPassInput.value = '';
}

/**
 * Opens the prompt. The password is collected here rather than in the terminal
 * because the installer needs root and a server has no terminal to ask on.
 *
 * `run` is what the password is handed to. It is passed in rather than assumed
 * because two different installers ask for one — the GPU driver update and a
 * runtime install that turns out to need root — and a box wired to the wrong
 * one would send the password to the wrong command.
 */
function openPassPrompt(run, returnTo) {
  passPromptRun = run ?? ((value) => runDriverInstaller(true, value || undefined));
  passPromptReturn = returnTo ?? (run ? null : rtUpdateDrivers);
  if (!rtPass) {
    // No prompt in the DOM: this build cannot ask, and starting over without a
    // password would just ask again. Say where to type it instead.
    rtLogBody.textContent +=
      'This build cannot ask for a password. Run the command above in a terminal.\n';
    return;
  }
  rtPass.hidden = false;
  rtPassInput?.focus();
}

rtPassCancel?.addEventListener('click', closePassPrompt);

/**
 * Sends the password once and clears it immediately — it is not held in a
 * variable past this point, not stored, and never written to the transcript.
 */
async function submitPassPrompt() {
  const value = rtPassInput?.value ?? '';
  const run = passPromptRun ?? ((v) => runDriverInstaller(true, v || undefined));
  closePassPrompt();
  await run(value.length ? value : undefined);
}

rtPassGo?.addEventListener('click', () => void submitPassPrompt());

rtPassInput?.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') {
    ev.preventDefault();
    void submitPassPrompt();
  } else if (ev.key === 'Escape') {
    closePassPrompt();
    (passPromptReturn ?? rtUpdateDrivers)?.focus();
  }
});

rtUpdateDrivers?.addEventListener('click', async () => {
  // First press explains, second starts it. A button that installs system
  // packages should never be the thing you hit by aiming slightly high.
  if (rtUpdateDrivers.dataset.armed !== '1') {
    rtUpdateDrivers.dataset.armed = '1';
    driverLog('This will install or update GPU drivers using your package manager,');
    driverLog('and add your user to the video and render groups. Some steps need root.');
    driverLog('');
    driverLog('Press "Update GPU drivers" again to continue.');
    return;
  }
  delete rtUpdateDrivers.dataset.armed;
  // Start without a password. The installer only asks when a step actually
  // needs root, and the server tells us that when it happens — so the prompt
  // appears then, not on every install.
  await runDriverInstaller(true);
});

/* ---------------- Backend report ----------------
   Rendered from /api/backends, which asks each backend where it is running.
   Nothing is inferred from the compute selector above: that choice governs
   which wheels an install downloads, and has nothing to say about what a
   backend already running is doing with them. */

const rtBackendsEl = document.getElementById('rt-backends');
const rtCheck = document.getElementById('rt-check');

/** One row per backend, with the GPU/CPU split when it is measurable. */
function renderBackend(b) {
  const row = document.createElement('div');
  row.className = 'rt-backend';

  const name = document.createElement('span');
  name.className = 'rt-backend__name';
  name.textContent = b.label;

  const badge = document.createElement('span');
  badge.className = 'rt-backend__badge';
  // "unknown" is its own state on purpose. An OpenAI-compatible endpoint says
  // nothing about its device, and showing it as idle would be a guess the user
  // would reasonably take as a measurement.
  badge.textContent =
    b.device === 'gpu' ? 'GPU'
      : b.device === 'cpu' ? 'CPU'
        : b.device === 'hybrid' ? 'GPU + CPU'
          : b.device === 'none' ? 'idle'
            : 'unknown';

  const detail = document.createElement('span');
  detail.className = 'rt-backend__detail';
  detail.textContent = b.detail;

  // The split bar only means something when both halves are in play, or when
  // the answer is a clean zero. A full-width bar for a 100%-GPU backend adds a
  // graphic that says nothing a sentence did not already say.
  if (typeof b.gpuFraction === 'number' && (b.device === 'hybrid' || b.gpuFraction < 0.999)) {
    const bar = document.createElement('span');
    bar.className = 'rt-backend__split';
    const gpu = document.createElement('i');
    gpu.style.width = `${(b.gpuFraction * 100).toFixed(1)}%`;
    const cpu = document.createElement('i');
    cpu.style.width = `${((1 - b.gpuFraction) * 100).toFixed(1)}%`;
    cpu.style.background = 'transparent';
    bar.append(gpu, cpu);

    const legend = document.createElement('span');
    legend.className = 'rt-backend__legend';
    const l = document.createElement('span');
    l.textContent = `GPU ${(b.gpuFraction * 100).toFixed(0)}%`;
    const r = document.createElement('span');
    r.textContent = `CPU ${((1 - b.gpuFraction) * 100).toFixed(0)}%`;
    legend.append(l, r);

    const wrap = document.createElement('div');
    wrap.style.flexBasis = '100%';
    wrap.append(bar, legend);
    detail.append(wrap);
  }

  if (b.evidence) {
    const ev = document.createElement('span');
    ev.className = 'rt-backend__evidence';
    ev.textContent = b.evidence;
    detail.append(ev);
  }
  if (b.readonlyReason) {
    const why = document.createElement('span');
    why.className = 'rt-backend__evidence';
    why.textContent = b.readonlyReason;
    detail.append(why);
  }

  row.append(name, badge, detail);
  return row;
}

async function refreshBackends() {
  if (!rtBackendsEl) return;
  let data = null;
  try {
    data = await api('/api/backends');
  } catch {
    rtBackendsEl.innerHTML = '';
    rtBackendsEl.append(hfNote('Could not reach the backends.'));
    return;
  }
  const list = data?.backends ?? [];
  rtBackendsEl.innerHTML = '';
  if (!list.length) {
    rtBackendsEl.append(hfNote('No execution backends are running.'));
    return;
  }
  for (const b of list) rtBackendsEl.append(renderBackend(b));
}

/**
 * The single "Check" button reads everything this tab claims about the machine:
 * what is installed, what each backend is actually doing, and what it can run.
 *
 * These used to be two buttons — "Check" and "Re-check" — 200px apart, with
 * labels that implied they checked different things. They were re-reading the
 * same machine on the same visit, so one now does both.
 */
rtCheck?.addEventListener('click', async () => {
  rtCheck.disabled = true;
  try {
    await loadRuntimes({ force: true });
    await refreshBackends();
  } finally {
    rtCheck.disabled = false;
  }
});

const rtRemove = document.getElementById('rt-remove');
const rtLog = document.getElementById('rt-log');
const rtLogBody = document.getElementById('rt-log-body');
const rtWarning = document.getElementById('rt-warning');
/** Set when the GPU choice cannot actually help the local Python runtimes. */
let gpuInertReason = '';
/** The server's own install-time warning, kept so it can be re-rendered. */
let rtServerWarning = '';

/**
 * Renders the one warning line. Both sources of warning are shown together
 * rather than one replacing the other, because they can be true at once and
 * showing only the newest is how the other one gets lost.
 */
function showRuntimeWarning(serverWarning) {
  const parts = [];
  if (serverWarning) parts.push(serverWarning);
  if (rtDevice === 'auto' && gpuInertReason) parts.push(gpuInertReason);
  rtWarning.hidden = parts.length === 0;
  rtWarning.textContent = parts.join(' ');
}
const rtGpuNote = document.getElementById('rt-gpu-note');
const rtTotal = document.getElementById('rt-total');

/** Which radio in the device group is picked. Starts on CPU and upgrades to
 *  GPU once the server reports a card, so a machine with no GPU never silently
 *  commits to a multi-gigabyte wheel index. */
let rtDevice = 'cpu';
/**
 * Runtimes the user has chosen to install. Nothing is pre-selected.
 *
 * This used to be seeded with ONNX Runtime on every fresh page, which made it
 * read "always on" and put an 18 MB download in front of people who had not
 * asked for it — for a runtime whose own blurb says none of the bundled runners
 * use it. A default that costs a download has to be one that unlocks something.
 *
 * What is left of the old code is the part that was genuinely load-bearing: a
 * pick has to be reconciled against what is actually installable when the list
 * renders, because a group that is already installed is disabled, and a disabled
 * checkbox cannot be cleared. Without that, a stale pick survives for the life
 * of the page, the total reads "1 selected", and Install tries to reinstall what
 * is already there.
 */
let rtPicked = new Set();
/** The groups from the last loadRuntimes, so "Install all" can select them. */
let rtCurrentGroups = [];
/** True once the user has clicked a device, so auto-detection stops second-guessing them. */
let rtDeviceChosen = false;

function rtSelectedDevice() {
  return document.querySelector('.rt-device__opt[aria-checked="true"]')?.dataset.device ?? 'cpu';
}

/**
 * Keeps exactly one radio in the tab order.
 *
 * A radiogroup is a single tab stop; the others are reached with the arrow
 * keys. Both options defaulting to tabindex 0 made the group two tab stops,
 * which is worse than not being a radiogroup at all — and the second stop was
 * the disabled GPU button, so keyboard users landed on something unclickable.
 */
function applyRovingTabindex() {
  const opts = [...document.querySelectorAll('.rt-device__opt')];
  const chosen = opts.find((o) => o.getAttribute('aria-checked') === 'true') ?? opts[0];
  for (const o of opts) o.tabIndex = o === chosen ? 0 : -1;
}

for (const opt of document.querySelectorAll('.rt-device__opt')) {
  opt.addEventListener('click', () => {
    if (opt.disabled) return;
    for (const o of document.querySelectorAll('.rt-device__opt')) {
      o.setAttribute('aria-checked', String(o === opt));
      o.tabIndex = o === opt ? 0 : -1;
    }
    rtDevice = rtSelectedDevice();
    rtDeviceChosen = true;
    applyRovingTabindex();
    try {
      localStorage.setItem(RT_DEVICE_KEY, rtDevice);
    } catch {
      /* private mode: the choice simply will not survive a reload */
    }
    loadRuntimes();
  });
}

// A radiogroup is one tab stop with arrow keys between the options. Without
// this the GPU option was reachable only by tabbing past it, and since it is
// the one that changes wheel indexes by gigabytes it needed to be properly
// navigable, not merely clickable.
document.querySelector('.rt-device__opts')?.addEventListener('keydown', (e) => {
  const opts = [...document.querySelectorAll('.rt-device__opt')].filter((o) => !o.disabled);
  const delta = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1
    : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
  if (!delta || !opts.length) return;
  e.preventDefault();
  const i = opts.findIndex((o) => o.getAttribute('aria-checked') === 'true');
  const next = opts[(i + delta + opts.length) % opts.length];
  next.click();
  next.focus();
});

applyRovingTabindex();

/** The last explicit choice, so a reload does not silently reset to CPU. */
const RT_DEVICE_KEY = 'llm-runtime-device';
function restoreDevice() {
  try {
    const saved = localStorage.getItem(RT_DEVICE_KEY);
    if (saved !== 'cpu' && saved !== 'auto') return;
    rtDevice = saved;
    rtDeviceChosen = true;
    for (const o of document.querySelectorAll('.rt-device__opt')) {
      o.setAttribute('aria-checked', String(o.dataset.device === saved));
    }
    applyRovingTabindex();
  } catch {
    /* ignore */
  }
}

function renderRuntimeGroups(groups, installed, blocked = [], upgrades = {}, ollamaVulkan = false) {
  const blockedBy = new Map(blocked.map((b) => [b.id, b]));

  // Reconcile the selection with reality. Anything already installed, blocked
  // by the interpreter, or gone from the table cannot be installed, so leaving
  // it selected would inflate the total and produce a doomed install.
  const selectable = new Set(
    groups.filter((g) => !installed[g.id] && !blockedBy.has(g.id)).map((g) => g.id),
  );
  for (const id of [...rtPicked]) {
    if (!selectable.has(id)) rtPicked.delete(id);
  }

  rtList.innerHTML = '';
  for (const g of groups) {
    const isIn = Boolean(installed[g.id]);
    const stop = blockedBy.get(g.id);
    // Carried on the entry so updateRuntimeTotal can tell "installable" from
    // "already there" without re-deriving it.
    g.__installed = isIn;
    g.__blocked = Boolean(stop);
    // A div, not a label. It was a label so the whole row was a hit target for
    // the install tick — but a Remove button inside a label toggles that tick
    // too, so the two controls have to be siblings. The label wrapper below
    // keeps the "click the text to tick it" behaviour for free.
    const row = document.createElement('div');
    row.className = `rt-row${isIn ? ' is-installed' : ''}${stop ? ' is-blocked' : ''}`;

    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = rtPicked.has(g.id);
    // A group this interpreter cannot install is not selectable, and says why
    // rather than failing later at the pip resolver.
    box.disabled = isIn || Boolean(stop);
    box.addEventListener('change', () => {
      if (box.checked) rtPicked.add(g.id);
      else rtPicked.delete(g.id);
      updateRuntimeTotal(groups);
    });

    const text = document.createElement('div');
    text.className = 'rt-row__text';
    const title = document.createElement('div');
    title.className = 'rt-row__title';
    title.textContent = g.label;
    const blurb = document.createElement('p');
    blurb.className = 'rt-row__blurb';
    blurb.textContent = g.blurb;
    text.append(title, blurb);
    if (stop) {
      const why = document.createElement('p');
      why.className = 'rt-row__blocked';
      why.textContent = stop.reason;
      text.append(why);
    }

    const meta = document.createElement('div');
    meta.className = 'rt-row__meta';
    const state = document.createElement('span');
    state.className = `rt-state${isIn ? ' is-on' : ''}${stop ? ' is-blocked' : ''}`;
    state.textContent = stop ? `Needs Python ${stop.maxPython}` : isIn ? 'Installed' : `≈ ${g.approxMB} MB`;
      // Only when it has something to say. Every row got one unconditionally,
      // so any package without an `unlocks` list left an empty span in the
      // document — invisible, but markup that means nothing.
      meta.append(state);
      // Which backend Ollama will actually use. Without it, the only way to tell
      // whether the card is being used is to read the service's own logs, and
      // "Installed" says nothing about where it runs.
      if (g.id === 'ollama' && isIn) {
        const vk = document.createElement('span');
        vk.className = 'rt-unlocks';
        vk.textContent = ollamaVulkan ? 'Vulkan' : 'default backend';
        meta.append(vk);
      }
      // A better build is available for this machine (e.g. Vulkan instead of
      // the CPU build). Offer a one-click reinstall that swaps it in.
      if (isIn && upgrades[g.id]) {
        const up = document.createElement('button');
        up.type = 'button';
        up.className = 'btn-ghost btn-tiny';
        up.textContent = `Upgrade to ${upgrades[g.id] === 'vulkan' ? 'GPU (Vulkan)' : upgrades[g.id]}`;
        up.addEventListener('click', () => {
          rtPicked = new Set([g.id]);
          updateRuntimeTotal(groups);
          void runRuntimeInstall();
        });
        meta.append(up);
      }
      if (Array.isArray(g.unlocks) && g.unlocks.length) {
        const unlocks = document.createElement('span');
        unlocks.className = 'rt-unlocks';
        unlocks.textContent = g.unlocks.join(' · ');
        meta.append(unlocks);
      }

    // Remove, for a group that is actually installed. Deliberately not the tick
    // box: that one means "install this", and a checkbox that silently uninstalls
    // when you mis-click it is how people lose a working environment. Two
    // controls, two meanings, both visible.
    let removeBtn = null;
    if (isIn) {
      removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'rt-row__remove';
      removeBtn.textContent = 'Remove';
      removeBtn.title = `Uninstall ${g.label} and leave the rest of the environment alone`;
      removeBtn.addEventListener('click', (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        removeRuntimeGroup(g);
      });
      removeBtn.dataset.group = g.id;
    }

    const label = document.createElement('label');
    label.className = 'rt-row__label';
    label.append(box, text);
    row.append(label, meta);
    if (removeBtn) row.append(removeBtn);
    rtList.append(row);
  }
  updateRuntimeTotal(groups);
}

/**
 * Uninstalls one runtime group, after a confirmation that names what goes.
 *
 * Confirming matters here in a way it does not for most buttons in this app:
 * torch is half a gigabyte and removing it breaks every local model until it is
 * installed again. The dialog says which packages are about to be removed
 * rather than just naming the row, so there is nothing to guess at.
 */
function removeRuntimeGroup(group) {
  const pkgs = Array.isArray(group.pkgs) && group.pkgs.length ? group.pkgs.join(', ') : null;
  const what = pkgs ? `${group.label} (${pkgs})` : group.label;
  const ok = window.confirm(
    pkgs
      ? `Remove ${what}?\n\nEverything else in the environment is left alone. You can install it again afterwards.`
      : `Remove ${what}?`,
  );
  if (!ok) return;
  driverLog(`Removing ${group.label}…`);
  streamPost('/api/runtimes/remove-group', { id: group.id }, {
    onLog: (line) => driverLog(line),
    onDone: () => {
      void loadRuntimes({ force: true });
      void refreshBackends();
    },
    onError: (message) => driverLog(`Failed: ${message}`),
  });
}

/**
 * POSTs a JSON body and reads an SSE transcript back, in the shape the install
 * and remove routes both answer with.
 */
function streamPost(url, body, { onLog, onDone, onError }) {
  fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.error ?? `HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const frames = buf.split('\n\n');
        buf = frames.pop() ?? '';
        for (const frame of frames) {
          const dataLine = frame.split('\n').find((l) => l.startsWith('data: '));
          if (!dataLine) continue;
          let payload;
          try {
            payload = JSON.parse(dataLine.slice(6));
          } catch {
            continue;
          }
          if (payload.line || payload.message) onLog(payload.line ?? payload.message);
          else if (payload.error) onError(payload.error);
        }
      }
      onDone?.();
    })
    .catch((err) => onError(err instanceof Error ? err.message : String(err)));
}

function updateRuntimeTotal(groups) {
  // Only count what will genuinely be fetched. A stale entry here is what made
  // the counter look stuck at "1 selected" that could not be cleared.
  const pick = [...rtPicked].filter((id) => groups.some((g) => g.id === id));
  const needTorch = pick.some(
    (id) => id === 'torch' || groups.find((g) => g.id === id)?.needsTorch,
  );
  const mb = pick.reduce((sum, id) => sum + (groups.find((x) => x.id === id)?.approxMB ?? 0), 0);

  if (!pick.length) {
    // "0 selected · ≈ 0 MB" reads like a broken counter. Say what is true.
    const anyInstallable = groups.some(
      (g) => !g.__installed && !g.__blocked,
    );
    rtTotal.textContent = anyInstallable ? 'Nothing selected' : 'Everything available is already installed';
  } else {
    const parts = [`${pick.length} selected`, `≈ ${mb} MB`];
    // torch comes along as a dependency of another group, so it counts toward
    // the download without being a separate tick.
    if (needTorch && !pick.includes('torch')) parts.push('includes PyTorch');
    rtTotal.textContent = parts.join(' · ');
  }

  const install = document.getElementById('rt-install');
  if (install) {
    install.disabled = pick.length === 0;
    install.title = pick.length ? '' : 'Tick at least one runtime to install';
  }
}

function renderRuntimeStatus(r) {
  rtStatus.innerHTML = '';
  const row = document.createElement('div');
  row.className = 'rt-status__row';

  if (!r.venv) {
    row.textContent = 'No environment yet. Everything here is optional — install only what you need.';
  } else {
    // No path. The venv's location is the same class of disclosure as the key
    // store's: it tells a reader where the writable, gitignored part of this
    // project lives, and nothing on this screen needs to act on it.
    const bits = ['Environment ready'];
    if (r.torch?.version) {
      bits.push(`PyTorch ${r.torch.version}`);
      bits.push(
        r.torch.cudaAvailable
          ? `GPU ready — ${r.torch.deviceName ?? 'CUDA device'}`
          : 'no GPU detected, running on CPU',
      );
    }
    row.textContent = bits.join(' · ');
  }
  rtStatus.append(row);

  // Only offered once there is something to remove.
  rtRemove.hidden = !r.venv;
}

let rtDeviceRestored = false;

/**
 * Re-probes the machine.
 *
 * The panel used to load exactly once for the life of the page, so installing
 * something by hand — or finishing an install in another tab, or fixing a GPU
 * driver — left the list stale with no way to refresh it short of a reload.
 */
async function checkRuntimes() {
  const btn = document.getElementById('rt-check');
  if (btn) {
    btn.disabled = true;
    btn.classList.add('is-busy');
  }
  const status = document.getElementById('rt-status');
  if (status) status.textContent = 'checking…';
  try {
    await loadRuntimes({ force: true });
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove('is-busy');
    }
  }
}

document.getElementById('rt-check')?.addEventListener('click', () => void checkRuntimes());

/**
 * True while a load is in flight, so a rapid double-click on Check does not
 * fire two probes and let the slower one overwrite the newer result.
 */
let rtLoadInFlight = null;

/**
 * Pending confirmation read for a "busy" status, so a label that may already
 * be out of date gets checked rather than left on screen.
 */
let rtBusyConfirm = null;

async function loadRuntimes({ force = false } = {}) {
  // Restore the saved choice once, before the first fetch, so the very first
  // request already asks for the right wheel index.
  if (!rtDeviceRestored) {
    rtDeviceRestored = true;
    restoreDevice();
  }
  // The newest request wins. Two probes racing — Check double-clicked, or a
  // refresh landing while an earlier one is still open — let the slower
  // response paint last and show a stale answer with no way to tell.
  const token = {};
  rtLoadInFlight = token;
  try {
    const res = await fetch(`/api/runtimes?device=${encodeURIComponent(rtDevice)}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
    if (rtLoadInFlight !== token) return;   // superseded
    const r = data.runtimes;

    // The install-time warning, plus a second one that only applies once the
    // user picks GPU. Those are different moments and they are separate
    // sentences: "your card cannot run these wheels" is a property of the
    // machine, "these wheels will not use your card" is a property of the
    // choice they just made.
    // Short on purpose. The paragraph above already explains the card in full;
    // repeating it here made the warning a wall of text that said the same
    // thing twice. This only has to name the consequence of the choice just
    // made, which the paragraph does not.
    gpuInertReason = data.gpu?.usable === false && data.gpu?.vulkan?.device
      ? 'These wheels will not use your GPU — it runs chat through Vulkan ' +
        'instead. Speech, image and text models here will run on the CPU.'
      : '';
    rtServerWarning = data.warning ?? '';
    showRuntimeWarning(rtServerWarning);

    // First load: select GPU by default only when one can actually run.
    // `usable` is false for a card present but unsupported (an Intel iGPU, or
    // an AMD card with amdgpu unloaded), and defaulting to Auto there sent the
    // user into a ROCm install for hardware that could never use it.
    if (!rtDeviceChosen && data.gpu?.vendor && data.gpu.usable !== false) {
      rtDevice = 'auto';
      for (const o of document.querySelectorAll('.rt-device__opt')) {
        o.setAttribute('aria-checked', String(o.dataset.device === 'auto'));
      }
    }

    const drv = document.getElementById('rt-driver-state');
    if (drv && data.gpu) {
      drv.textContent = data.gpu.vendor
        ? `Detected ${data.gpu.vendor === 'amd' ? 'AMD' : 'NVIDIA'}: ${data.gpu.detail}` +
          (data.runtimes?.torch?.cudaAvailable ? ' — CUDA/ROCm is active.' : '')
        : data.gpu.detail;
    }

    // Say what was found, in one line. This used to reprint the card's entire
    // description — which the GPU drivers block directly below already prints
    // in full, verbatim. The same 400-odd characters twice on one screen made
    // the tab read as noise and buried the thing that changes: the compute
    // choice.
    if (rtGpuNote) {
      const g = data.gpu;
      if (!g) {
        rtGpuNote.textContent = '';
      } else if (!g.vendor) {
        rtGpuNote.textContent = g.detail ?? '';
      } else {
        const vk = g.vulkan?.device;
        const kind = g.vendor === 'amd' ? 'AMD' : 'NVIDIA';
        rtGpuNote.textContent = g.usable === false && vk
          ? `${kind} GPU detected — works for chat through Vulkan. The Python wheels ` +
            'installed here need CUDA or ROCm, so they will run on the CPU.'
          : `${kind} GPU detected.`;
      }
      // The Vulkan answer, when there is one. On a card too old for ROCm this is
      // the difference between "your GPU is unusable" and "your GPU is already
      // running your model" — which is the situation on this machine.
      const vkNote = document.getElementById('rt-gpu-vulkan');
      if (vkNote) {
        const v = g?.vulkan;
        vkNote.textContent =
          v?.available && v.device
            ? `Vulkan: ${v.device} · ${v.driver ?? 'unknown driver'}` +
              `${v.discrete ? ' · discrete' : ''}`
            : v?.available
              ? 'Vulkan: no AMD device — the loader sees only other vendors'
              : '';
      }

      // What the kernel driver reports about the card itself: architecture, VRAM,
      // revision. None of this is visible anywhere else in the app, and it is
      // what makes "it can't use my GPU" answerable without leaving the page.
      const hw = g?.hardware;
      const hwNote = document.getElementById('rt-gpu-hw');
      if (hwNote) {
        // VRAM is deliberately absent: the sentence directly above already
        // states it, and printing "2.0 GB" twice on one screen is the same
        // noise the rest of this cleanup is about. This line carries only what
        // the prose does not.
        hwNote.textContent = hw
          ? [
              hw.arch,
              hw.driver ? `${hw.driver} driver` : null,
              hw.revision ? `rev ${hw.revision}` : null,
            ]
              .filter(Boolean)
              .join(' · ')
          : '';
      }
    }

    // Whether the GPU choice is dead, and whether it merely cannot help this
    // particular install, are different questions and used to be conflated into
    // one greyed-out button.
    //
    // This selector chooses which wheels a *local Python runtime* downloads. A
    // card below ROCm's floor cannot run those and always could not — that is
    // not the card being unusable. When a working Vulkan backend is present the
    // card is in active use by chat, so greying it out read as "your GPU does
    // nothing" while the Backends panel above showed it at 100%.
    //
    // So: disabled only when there is no GPU path at all. Otherwise selectable,
    // with the real consequence stated.
    const gpuOpt = document.querySelector('.rt-device__opt[data-device="auto"]');
    if (gpuOpt) {
      const vk = data.gpu?.vulkan?.device;
      const noPath = Boolean(data.gpu && data.gpu.usable === false && !vk);
      gpuOpt.disabled = noPath;
      gpuOpt.classList.toggle('is-unusable', noPath);
      gpuOpt.title = noPath ? (data.gpu.detail ?? '') : (data.gpu?.detail ?? '');
      const em = gpuOpt.querySelector('em');
      if (em) {
        em.textContent = data.gpu?.faulted
          ? 'Detected, but the card is not responding — a driver fault'
          : noPath
            ? 'Detected, but this card cannot run CUDA or ROCm'
            : vk
              // Selectable, and honest: the card works, these wheels will not use it.
              ? 'GPU present (Vulkan works) — but Python models here need CUDA or ROCm'
              : 'NVIDIA CUDA or AMD ROCm, detected automatically';
      }
      applyRovingTabindex();
    }
    renderRuntimeStatus(r);
    runtimeBlockedIds = new Set((data.blocked ?? []).map((b) => b.id));
    rtCurrentGroups = data.groups ?? [];
    renderRuntimeGroups(
      rtCurrentGroups,
      r.installed ?? {},
      data.blocked ?? [],
      data.upgrades ?? {},
      r.ollamaVulkan === true,
    );
    // renderRuntimeGroups already disabled this when nothing is ticked; that
    // decision was being overwritten by the busy check alone, so an empty
    // selection looked installable.
    rtInstall.disabled = Boolean(r.busy) || rtPicked.size === 0;
    rtInstall.textContent = r.busy ? 'Installing…' : 'Install selected';
    if (r.busy) {
      // This read says an install is running, but it cannot say how long that
      // has been true — the install may have finished microseconds before the
      // server answered, in which case this label is already a lie and nothing
      // would correct it. One confirmation read a moment later turns "believed"
      // into "known", which is what stops the button sitting on "Installing…"
      // with nothing behind it.
      clearTimeout(rtBusyConfirm);
      rtBusyConfirm = setTimeout(() => void loadRuntimes({ force: true }), 2500);
    }
  } catch (err) {
    rtStatus.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'scan-result error';
    p.textContent = err instanceof Error ? err.message : String(err);
    rtStatus.append(p);
  }
}

rtInstall.addEventListener('click', () => void runRuntimeInstall());
const rtInstallAll = document.getElementById('rt-install-all');
rtInstallAll?.addEventListener('click', () => {
  rtPicked = new Set(
    rtCurrentGroups
      .filter((g) => !g.__installed && !g.__blocked)
      .map((g) => g.id),
  );
  if (!rtPicked.size) {
    notifyWarning('Everything installable is already installed');
    return;
  }
  void runRuntimeInstall();
});

/**
 * What the current (or last) install was asked to install, so a retry after a
 * password prompt installs the same selection even if the list re-rendered.
 */
let pendingInstallGroups = [];

/**
 * Waits for the server's install lock to clear.
 *
 * One install at a time is deliberate — pip into the same venv from two
 * processes is the corruption the lock prevents — so a retry that arrives
 * before the previous attempt has finished is told to wait rather than to
 * give up. Bounded: after this the answer is genuinely no.
 */
async function waitForInstallIdle(timeoutMs = 20000) {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    try {
      const data = await fetch('/api/runtimes').then((r) => r.json());
      if (!data?.runtimes?.busy) return true;
    } catch {
      // Cannot tell; assume idle rather than block the user on a guess.
      return false;
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  return false;
}

/**
 * Installs the picked runtime groups, streaming the log.
 *
 * `password` is only ever non-undefined when an installer on the previous
 * attempt actually asked for one. Nothing here prompts for a password up
 * front: pip installs need none, and on macOS and Windows the Ollama install
 * needs none either, so an install that never needs root never shows a box
 * asking for a password nobody is going to give it.
 */
async function runRuntimeInstall(password) {
  // The groups being installed, held outside the picker on purpose. A failed
  // attempt re-renders the list, which un-ticks anything already installed —
  // so by the time the password prompt is answered the selection can be empty
  // and the retry would install nothing. The retry installs what the user
  // asked for, not whatever the list happens to hold afterwards.
  const groups = rtPicked.size ? [...rtPicked] : pendingInstallGroups;
  if (!groups.length) {
    notifyWarning('Pick at least one runtime first');
    return;
  }
  pendingInstallGroups = groups;
  rtInstall.disabled = true;
  rtInstall.textContent = 'Installing…';
  rtLog.hidden = false;
  rtLogBody.textContent += password ? 'Retrying with your password…\n' : '';
  rtLog.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Set only after the server asks. A second ask means the password was
  // refused or sudo will not run non-interactively, and asking again in a
  // loop would trap the user in a dialog they cannot leave.
  let privilegeAsked = Boolean(password);
  let needsPassword = false;

  try {
    const post = () =>
      fetch('/api/runtimes/install', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ groups, device: rtDevice, password }),
      });
    let res = await post();
    if (res.status === 409 && password) {
      // The attempt that asked for the password is still tearing down and
      // still holding the one install lock. It releases in milliseconds, so
      // wait for it rather than telling the user their password did nothing.
      await waitForInstallIdle();
      res = await post();
    }
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error ?? `HTTP ${res.status}`);
    }
    // Already installing server-side — after a reload, or a second tab. The
    // pip run continues regardless; this tab just follows it on the downloads
    // page instead of starting a duplicate.
    if ((res.headers.get('content-type') ?? '').includes('application/json')) {
      const data = await res.json();
      if (data.attached) {
        rtInstall.textContent = 'Install selected';
        rtLogBody.textContent +=
          'Already running on the server — watch it on the Downloads page.\n';
        setSubview('downloads');
        void syncTransfers();
        return;
      }
      throw new Error(data.error ?? 'Install refused');
    }
    await sseStream(res.body, async (event, parsed) => {
      if (event === 'progress') {
        rtLogBody.textContent += `${parsed}\n`;
      } else if (event === 'privilege') {
        // An installer really did hit a password prompt. This is the only
        // thing in the app that opens the box for a runtime install.
        rtLogBody.textContent += `This step needs your password: ${parsed.command}\n`;
        needsPassword = true;
        if (!privilegeAsked) {
          privilegeAsked = true;
          openPassPrompt((value) => runRuntimeInstall(value || undefined), rtInstall);
        }
      } else if (event === 'error') {
        throw new Error(parsed.message);
      } else if (event === 'done') {
        // Newly installed groups leave the selection so a second click can't
        // reinstall them.
        for (const id of Object.keys(parsed.runtimes?.installed ?? {})) {
          if (parsed.runtimes.installed[id]) rtPicked.delete(id);
        }
        pendingInstallGroups = [];
        notifySuccess('Runtimes installed');
        // The list just changed underneath us; re-read rather than guess.
        await loadRuntimes({ force: true });
      }
    });
  } catch (err) {
    notifyError(err instanceof Error ? err.message : String(err));
  } finally {
    rtInstall.textContent = 'Install selected';
    // Held disabled until the refresh below says otherwise: that read knows
    // whether an install is still running, and re-enabling the button first
    // would leave a window where a second click starts a duplicate install —
    // which is what the server answers with "already running on the server".
    rtInstall.disabled = true;
    await loadRuntimes();
    // An install can turn a "needs X" note into a Run button.
    await refreshCatalogRunState();
    // The prompt stays open when it is still the useful thing to do; a closed
    // one would leave the user with a failed install and no way to retry it.
    if (!rtInstall.disabled && !needsPassword) rtInstall.focus();
  }
}

// Two-step, like the delete guards elsewhere: an accidental click must not
// take out a multi-gigabyte environment.
let rtArmed = false;
rtRemove.addEventListener('click', async () => {
  if (!rtArmed) {
    rtArmed = true;
    rtRemove.textContent = 'Really remove?';
    rtRemove.dataset.armed = '1';
    setTimeout(() => {
      if (!rtArmed) return;
      rtArmed = false;
      rtRemove.textContent = 'Remove environment';
      delete rtRemove.dataset.armed;
    }, 4000);
    return;
  }
  rtArmed = false;
  rtRemove.disabled = true;
  try {
    const res = await fetch('/api/runtimes/remove', { method: 'POST' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
    rtRemove.textContent = 'Remove environment';
    delete rtRemove.dataset.armed;
    rtLog.hidden = true;
    notifySuccess('Environment removed');
    // Removing an environment changes what is runnable, which is what the hub
    // groups by and what the assignment dropdowns offer. Without this the page
    // kept advertising runtimes that had just been deleted.
    await syncDerived();
  } catch (err) {
    notifyError(err instanceof Error ? err.message : String(err));
  } finally {
    rtRemove.disabled = false;
    await loadRuntimes();
  }
});

/* ---------------- Which runtime can run this file ---------------- */

/**
 * Maps a downloaded file to the runtime that can execute it, by weight format.
 *
 * The catalog only knows about sixteen hand-picked repos, but a user can
 * download anything from the Hub. Without this, an unrecognised download was
 * simply "no local runtime" with nothing to act on — the whole point of the
 * hub is to answer "what runs this, and how do I get it running".
 *
 * The format is the reliable signal. A `.gguf` is llama.cpp, an `.onnx` is
 * onnxruntime, a `.safetensors`/`.bin` is transformers, a `.ckpt` is a
 * diffusers-era checkpoint.
 */
const FORMAT_RUNTIME = [
  {
    id: 'onnxruntime',
    ext: /\.(onnx|ort)$/i,
    label: 'ONNX Runtime',
    requires: ['onnxruntime'],
    blurb: 'ONNX weights. Runs under onnxruntime, with no PyTorch at all.',
  },
  {
    id: 'transformers',
    ext: /\.(safetensors|bin|pth|pt)$/i,
    label: 'Transformers',
    requires: ['torch', 'transformers'],
    blurb: 'Torch weights. Chat models run here through transformers; others need their own program.',
  },
  {
    id: 'diffusers',
    ext: /\.(ckpt)$/i,
    label: 'Diffusers',
    requires: ['torch', 'diffusers'],
    blurb: 'A diffusers-era checkpoint.',
  },
  {
    id: 'llamacpp',
    ext: /\.(gguf)$/i,
    label: 'llama.cpp',
    requires: [],
    // A GGUF this app cannot chat with: whisper GGUFs belong to whisper.cpp and
    // diffusion GGUFs to ComfyUI, and neither is a text model. The chat models
    // sharing the same extension are pulled out of this path by
    // `chatRuntimeFor` below, which is why the note here can be blunt.
    external: true,
    blurb: 'A quantised GGUF, but not a chat model — whisper and diffusion GGUFs need their own program.',
  },
];

/**
 * The runtime that can chat with a downloaded GGUF, when it is a chat model.
 *
 * The app installs llama.cpp itself and offers it as a chat provider, so most
 * GGUFs on this page are runnable here — the exception being the whisper and
 * diffusion files that share the format. Which is which is decided once, on the
 * server (`/api/hf/files` marks the chat models with `llama`), because that is
 * the same list the chat page's model picker shows: two answers here would mean
 * a Chat button pointing at a model the provider has never heard of.
 */
const LLAMA_CHAT_RUNTIME = {
  id: 'llama-cpp',
  label: 'llama.cpp',
  requires: ['llama-cpp'],
  blurb: 'A quantised GGUF chat model. Runs on this machine through llama.cpp.',
};

const TRANSFORMERS_CHAT_RUNTIME = {
  id: 'transformers',
  label: 'Transformers',
  requires: ['torch', 'transformers'],
  blurb: 'A transformers chat model. Runs on this machine through torch + transformers.',
};

/** `null` when this row is not a chat model. */
function chatRuntimeFor(d) {
  if (d?.llama) return { ...LLAMA_CHAT_RUNTIME, provider: d.llama.provider, model: d.llama.model };
  if (d?.transformers) return { ...TRANSFORMERS_CHAT_RUNTIME, provider: d.transformers.provider, model: d.transformers.model };
  return null;
}

/** What it takes to run this file, based on its extension. */
function runtimeForFile(fileName) {
  // A `.pt` under voices/ is a voice pack, not weights, and `.cache/` is the
  // downloader's own bookkeeping. Counting them as weights turned every
  // Kokoro install into a second, always-incomplete "model" row.
  if (fileName.startsWith('voices/') || fileName.startsWith('.cache/')) return null;
  for (const r of FORMAT_RUNTIME) {
    if (r.ext.test(fileName)) return r;
  }
  return null;
}

/** Short extension label for the chip, e.g. GGUF / safetensors. */
function formatChip(fileName) {
  const m = String(fileName).match(/\.(gguf|safetensors|onnx|bin|ckpt|pth|pt)$/i);
  return m ? m[1].toUpperCase() : 'weights';
}

/**
 * Runtime groups this machine's Python cannot install, as reported by
 * /api/runtimes. Cached from the last fetch so the hub can ask about a runtime
 * without another round trip.
 */
let runtimeBlockedIds = new Set();

/** True when any of `ids` is known to be un-installable here. */
function runtimeBlocked(ids) {
  return (ids ?? []).some((id) => runtimeBlockedIds.has(id));
}

/**
 * The runtime descriptor for a downloaded row.
 *
 * A catalog bundle is authoritative about its own runtime — a CTranslate2
 * Whisper's `model.bin` extension alone would say "Transformers", which is
 * both wrong and would send the user off to install 529 MB of PyTorch for a
 * runtime that needs about 40 MB of CTranslate2.
 */
function runtimeForRow(d, resolved) {
  // A chat model first: the GGUF extension alone says "external tool", which is
  // true of whisper GGUFs and false of the ones this app's own provider runs.
  const chat = chatRuntimeFor(d);
  if (chat) return chat;
  if (d.entry?.run?.requires?.length) {
    const reqs = d.entry.run.requires;
    return {
      id: reqs[0],
      label: d.entry.runtime || reqs.join(' + '),
      requires: reqs,
      blurb:
        d.entry.blurb ||
        `Run by ${d.entry.runtime || reqs.join(' + ')} from the model catalogue.`,
    };
  }
  return resolved ?? runtimeForFile(d.file);
}

/** Whether `runtime`'s packages are all present, ignoring blocked groups. */
function runtimeReady(runtime, installed) {
  if (!runtime || runtime.external) return false;
  return runtime.requires.every((g) => Boolean(installed[g]));
}

/* ---------------- Task assignments ---------------- */

/**
 * Where each job's model is chosen.
 *
 * The composer used to carry a speech-to-text picker, which put model choice
 * in the message bar and left speech and images with nowhere to be assigned at
 * all. Everything is decided here now, once, against what this machine can
 * actually run — and every other surface (the composer, the Studio) reads the
 * result rather than asking again.
 */
const TASK_LABELS = {
  stt: { title: 'Speech to text', verb: 'Transcribes your dictation', task: 'stt' },
  tts: { title: 'Speech', verb: 'Reads text aloud', task: 'tts' },
  image: { title: 'Image generation', verb: 'Draws images in the Studio', task: 'image' },
};

/**
 * Renders one assignment row per job.
 *
 * Only models that can run are offered. A downloaded model with no runtime
 * installed is listed but unselectable with the reason attached, because "the
 * model you downloaded is not wired to anything" is worth surfacing here
 * rather than leaving a silently ignored download.
 */
async function renderAssignments() {
  const host = document.getElementById('task-assignments');
  if (!host) return;
  host.innerHTML = '';
  host.append(hfNote('Loading what this machine can run…'));

  let data = {};
  try {
    data = await fetch('/api/runnable').then((r) => r.json());
  } catch {
    host.innerHTML = '';
    host.append(hfNote('Could not read what is runnable.'));
    return;
  }
  const assigned = data.assigned ?? {};
  host.innerHTML = '';

  for (const key of ['stt', 'tts', 'image']) {
    const meta = TASK_LABELS[key];
    const models = data[key] ?? [];
    const ready = models.filter((m) => m.ready);

    const row = document.createElement('div');
    row.className = 'assign-row';

    const text = document.createElement('div');
    text.className = 'assign-row__text';
    const title = document.createElement('div');
    title.className = 'assign-row__title';
    title.textContent = meta.title;
    const sub = document.createElement('p');
    sub.className = 'assign-row__sub';
    sub.textContent = ready.length
      ? `${meta.verb} · ${ready.length} model${ready.length === 1 ? '' : 's'} available`
      : `Nothing runnable here for ${meta.title.toLowerCase()} yet`;
    text.append(title, sub);

    const select = document.createElement('select');
    select.className = 'assign-row__select';
    select.id = `assign-${key}`;
    select.setAttribute('aria-label', `Model used for ${meta.title.toLowerCase()}`);

    const none = document.createElement('option');
    none.value = '';
    none.textContent = ready.length ? 'Not set' : 'Nothing available';
    none.disabled = !ready.length;
    select.append(none);

    for (const m of models) {
      const opt = document.createElement('option');
      opt.value = `local:${m.repo}`;
      opt.textContent = m.ready ? m.label : `${m.label} — ${m.reason}`;
      opt.disabled = !m.ready;
      if (assigned[key] === `local:${m.repo}`) opt.selected = true;
      select.append(opt);
    }

    select.addEventListener('change', async () => {
      try {
        await fetch('/api/assignments', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ task: key, model: select.value || null }),
        });
        notifySuccess(`${meta.title}: ${select.value ? select.selectedOptions[0].textContent : 'not set'}`);
          // The choice is shown in the hub and on the run panels, both of which
          // read the assignment store; they were left showing the previous one.
          await syncDerived();
      } catch (err) {
        notifyError(err.message instanceof Error ? err.message : String(err));
      }
    });

    row.append(text, select);
    host.append(row);
  }
}

/* ---------------- Model hub, grouped by runtime ---------------- */

const hubRuntimesEl = document.getElementById('hub-runtimes');
const downloadActivityEl = document.getElementById('download-activity');

/**
 * Every transfer this page has started, live. The catalog, the file list and
 * the local-downloads buttons all report here, so "where is my download up to"
 * has exactly one answer rather than one per list.
 */
const transfers = new Map();

function transferKey(kind, repo, file) {
  return `${kind}:${repo ?? ''}::${file ?? ''}`;
}

/** Creates (or refreshes) a row and returns its update function. */
function beginTransfer(kind, label, meta = {}) {
  const key = transferKey(kind, meta.repo, meta.file);
  const existing = transfers.get(key);
  if (existing) {
    existing.label = label;
    return existing;
  }
  const t = {
    key, kind, label, ...meta,
    state: 'running',
    percent: 0,
    received: 0,
    total: null,
    startedAt: Date.now(),
    update(patch) {
      Object.assign(this, patch);
      renderDownloads();
    },
    finish(state) {
      this.state = state;
      this.endedAt = Date.now();
      renderDownloads();
    },
  };
  transfers.set(key, t);
  renderDownloads();
  return t;
}

/**
 * Re-reads transfer state from the server.
 *
 * The page's own Map is a live mirror, not the source of truth — after a reload
 * it is empty, but the downloads did not stop. Pulling the server's view on load
 * is what makes reloading safe: in-flight work reappears instead of vanishing.
 */
async function syncTransfers() {
  try {
    const res = await fetch('/api/transfers');
    if (!res.ok) return;
    const { active: running = [], recent = [] } = await res.json();
    transfers.clear();
    for (const t of [...running, ...recent]) {
      transfers.set(t.id, {
        key: t.id,
        kind: t.kind,
        label: t.label,
        repo: t.repo,
        file: t.file,
        state: t.state,
        percent: t.percent ?? 0,
        received: t.received ?? 0,
        total: t.total ?? null,
        startedAt: t.startedAt,
        error: t.error,
      });
    }
    hfSyncFails = 0;
    if (!hfActionInFlight) renderDownloads();
    // Anything still running keeps ticking without a page refresh — and the
    // ticker is torn down once it finishes, so an idle page holds no timer.
    if (running.length && !hfTransferTicker) {
      hfTransferTicker = setInterval(() => void syncTransfers(), 2000);
    } else if (!running.length && hfTransferTicker) {
      clearInterval(hfTransferTicker);
      hfTransferTicker = null;
    }
  } catch {
    // Leaving the row untouched here looked exactly like progress: the
    // indeterminate stripe animates forever, so a download whose server had
    // died appeared to be working indefinitely. Say the connection dropped,
    // and keep trying in case it comes back.
    hfSyncFails += 1;
    if (hfSyncFails >= 2) {
      for (const t of transfers.values()) {
        if (t.state === 'running') t.stale = true;
      }
      renderDownloads();
      if (!hfTransferTicker) {
        hfTransferTicker = setInterval(() => void syncTransfers(), 4000);
      }
    }
  }
}

let hfTransferTicker = null;
/** Consecutive failed syncs. Two in a row is a lost connection, not a blip. */
let hfSyncFails = 0;

let hfSyncPending = false;
/** While a download button is mid-action, re-rendering the list would destroy
 *  the button under the cursor and make the click look like it did nothing. */
let hfActionInFlight = false;
/** Coalesces progress re-reads so a fast download cannot spam the endpoint. */
async function syncTransfersSoon() {
  if (hfSyncPending) return;
  hfSyncPending = true;
  setTimeout(() => {
    hfSyncPending = false;
    void syncTransfers();
  }, 700);
}

const speedSamples = new Map();

/** A short label for a repo: the catalog's name if it has one, else the id. */
function catalogLabel(repo) {
  if (!repo) return '';
  const e = catalogData.find((c) => c.repo === repo);
  return e?.label ?? repo;
}

function renderDownloads() {
  if (!downloadActivityEl) return;
  const list = [...transfers.values()].sort(
    (a, b) => (b.startedAt ?? 0) - (a.startedAt ?? 0),
  );
  downloadActivityEl.innerHTML = '';
  if (!list.length) {
    downloadActivityEl.append(
      hfNote('Nothing yet. Pull or download a model and it will show up here.'),
    );
    return;
  }

  // One row per model: a model with twelve files used to be twelve rows, each
  // claiming to be "the download", and the only way to remove it was a dozen
  // clicks. Grouping by repo (and by transfer id for non-model work, like pip
  // installs) keeps the list readable and makes a whole model one unit again.
  const groups = new Map();
  for (const t of list) {
    const key = t.kind === 'download' && t.repo ? `repo:${t.repo}` : `one:${t.key}`;
    let g = groups.get(key);
    if (!g) {
      g = { key, kind: t.kind, repo: t.repo, files: [], startedAt: 0 };
      groups.set(key, g);
    }
    g.files.push(t);
    g.startedAt = Math.max(g.startedAt, t.startedAt ?? 0);
  }
  const ordered = [...groups.values()].sort((a, b) => (b.startedAt ?? 0) - (a.startedAt ?? 0));

  for (const g of ordered) {
    const row = document.createElement('div');
    const running = g.files.some((t) => t.state === 'running');
    const errored = g.files.some((t) => t.state === 'error');
    const stale = g.files.some((t) => t.stale);
    const state = stale ? 'running' : (running ? 'running' : (errored ? 'error' : 'done'));
    row.className = `dl-row dl-row--${state}${stale ? ' is-stale' : ''}`;

    const top = document.createElement('div');
    top.className = 'dl-row__top';

    const kind = document.createElement('span');
    kind.className = 'dl-row__kind';
    kind.textContent = g.kind === 'pull' ? 'Pull' : 'Download';
    top.append(kind);

    const name = document.createElement('span');
    name.className = 'dl-row__name';
    name.textContent = g.kind === 'download' ? catalogLabel(g.repo) : g.files[0]?.label ?? 'Transfer';
    name.title = g.repo ? `${g.repo}` : name.textContent;
    top.append(name);

    const status = document.createElement('span');
    status.className = 'dl-row__status';

    if (stale) {
      status.textContent = 'lost contact with the server — rechecking…';
    } else if (running) {
      const received = g.files.reduce((a, t) => a + (t.received ?? 0), 0);
      const total = g.files.reduce((a, t) => a + (t.total ?? 0), 0);
      const doneCount = g.files.filter((t) => t.state === 'done').length;
      const of = g.files.length;
      let rate = null;
      if (total > 0) {
        const sample = speedSamples.get(g.key);
        const now = Date.now();
        if (sample && now - sample.at >= 400) {
          const dt = (now - sample.at) / 1000;
          const dr = (received - sample.received) / dt;
          if (Number.isFinite(dr) && dr > 0) rate = dr;
        }
        speedSamples.set(g.key, { at: now, received });
      }
      status.textContent =
        total > 0
          ? `${doneCount > 0 ? `${doneCount}/${of} files · ` : `${of} file${of > 1 ? 's' : ''} · `}${fmtSize(received)} / ${fmtSize(total)} (${Math.round((received / total) * 100)}%)` +
            (rate ? ` · ${fmtSize(rate)}/s` : '')
          : received > 0
            ? `${fmtSize(received)}${rate ? ` · ${fmtSize(rate)}/s` : ''}`
            : 'starting…';
    } else if (errored) {
      const first = g.files.find((t) => t.state === 'error');
      const msg = first?.error ?? 'failed';
      status.textContent = /abort/i.test(msg) ? 'cancelled' : msg;
    } else {
      const total = g.files.reduce((a, t) => a + (t.total ?? t.received ?? 0), 0);
      status.textContent = total ? fmtSize(total) : 'done';
    }
    top.append(status);

    // Pause is offered while the model is still coming down: it aborts the
    // transfer but keeps the partial file, so Resume can continue from it.
    if (g.kind === 'download' && running && g.repo) {
      const pause = document.createElement('button');
      pause.type = 'button';
      pause.className = 'btn-ghost btn-tiny';
      pause.title = 'Pause this download (progress is kept)';
      pause.textContent = 'Pause';
      pause.addEventListener('click', async () => {
        pause.disabled = true;
        hfActionInFlight = true;
        try {
          await api('/api/hf/download/cancel', {
            method: 'POST',
            body: JSON.stringify({ repo: g.repo }),
          });
          await syncTransfers();
          await syncDerived();
        } catch (err) {
          notifyError(err instanceof Error ? err.message : String(err));
        } finally {
          hfActionInFlight = false;
          pause.disabled = false;
          renderDownloads();
        }
      });
      top.append(pause);

      // Cancel discards the partial file as well.
      const cancel = document.createElement('button');
      cancel.type = 'button';
      cancel.className = 'btn-ghost btn-tiny';
      cancel.title = 'Cancel and discard the partial download';
      cancel.textContent = 'Cancel';
      cancel.addEventListener('click', async () => {
        cancel.disabled = true;
        hfActionInFlight = true;
        try {
          await api('/api/hf/download/cancel', {
            method: 'POST',
            body: JSON.stringify({ repo: g.repo }),
          });
          await api('/api/hf/download/discard', {
            method: 'POST',
            body: JSON.stringify({ repo: g.repo }),
          });
          await syncTransfers();
          await syncDerived();
        } catch (err) {
          notifyError(err instanceof Error ? err.message : String(err));
        } finally {
          hfActionInFlight = false;
          cancel.disabled = false;
          renderDownloads();
        }
      });
      top.append(cancel);
    }

    // Resume (and retry) for a paused or failed download: re-running the
    // download skips files already on disk and resumes partials via Range.
    if (g.kind === 'download' && !running && g.repo) {
      const resume = document.createElement('button');
      resume.type = 'button';
      resume.className = 'btn-primary btn-tiny';
      resume.title = errored ? 'Retry this download' : 'Resume this download';
      resume.textContent = errored ? 'Retry' : 'Resume';
      resume.addEventListener('click', async () => {
        resume.disabled = true;
        hfActionInFlight = true;
        resume.textContent = '…';
        try {
          const endpoint = g.files.length === 1 ? '/api/hf/download' : '/api/hf/download-repo';
          const body =
            g.files.length === 1
              ? { repo: g.repo, file: g.files[0].file }
              : { repo: g.repo };
          await fetch(endpoint, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(body),
          });
          // Give the server a moment to register the new transfer before polling.
          await new Promise((r) => setTimeout(r, 300));
          await syncTransfers();
          await syncDerived();
        } catch (err) {
          notifyError(err instanceof Error ? err.message : String(err));
        } finally {
          hfActionInFlight = false;
          resume.disabled = false;
          resume.textContent = errored ? 'Retry' : 'Resume';
          renderDownloads();
        }
      });
      top.append(resume);
    }

    // One-click removal of a whole model: only when nothing is still running,
    // and only for files this app owns (a HF-cache row is someone else's).
    if (g.kind === 'download' && !running && g.repo) {
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'btn-ghost btn-tiny';
      del.title = `Remove every file of ${g.repo} from this machine`;
      armThenDelete(del, `repo::${g.repo}`, async () => {
        try {
          await api('/api/hf/repo/delete', {
            method: 'POST',
            body: JSON.stringify({ repo: g.repo }),
          });
          notifySuccess(`${g.repo} removed`);
          await syncDerived();
        } catch (err) {
          notifyError(err instanceof Error ? err.message : String(err));
        }
      });
      top.append(del);
    }

    row.append(top);

    if (running) {
      const bar = document.createElement('div');
      bar.className = 'dl-row__bar';
      const fill = document.createElement('i');
      const total = g.files.reduce((a, t) => a + (t.total ?? 0), 0);
      const received = g.files.reduce((a, t) => a + (t.received ?? 0), 0);
      if (total > 0) {
        fill.style.width = `${Math.min(100, (received / total) * 100)}%`;
      } else {
        fill.classList.add('is-indeterminate');
      }
      // No stripe when stale: see the note in the status text above.
      if (stale) fill.classList.remove('is-indeterminate');
      bar.append(fill);
      row.append(bar);
    }
    downloadActivityEl.append(row);
  }
}

/**
 * Repos at least one job can load, read from the same endpoint the job
 * pickers read, so the hub can never mark a row usable while no picker offers
 * it. The mapping only changes with the catalog, so one fetch serves every
 * render; null means the answer is unknown and rows stay untagged rather than
 * guessed at.
 */
let jobReposPromise = null;
function jobRepos() {
  if (!jobReposPromise) {
    jobReposPromise = fetch('/api/runnable')
      .then((r) => r.json())
      .then((d) => {
        const repos = new Set();
        for (const task of ['stt', 'tts', 'image']) {
          for (const m of d[task] ?? []) repos.add(m.repo);
        }
        return repos;
      })
      .catch(() => {
        // Not worth a banner — the tag is an extra, and the next render
        // retries for free.
        jobReposPromise = null;
        return null;
      });
  }
  return jobReposPromise;
}

/**
 * The hub: Ollama models, then one section per runtime listing the weights on
 * disk that runtime can run. A runtime that is missing but has models waiting
 * on it gets an install button, which is the whole point of grouping this way.
 */
async function refreshHub() {
  if (!hubRuntimesEl) return;
  // The catalog decides what a download is. Without it a bundle is four
  // anonymous files and nothing knows what runs them.
  await ensureCatalog().catch(() => {});
  // One request for runtimes, not two: /api/runtimes carries both the installed
  // map and the group table, and it costs a Python probe on the server.
  const runtimesFetch = fetch('/api/runtimes')
    .then((r) => r.json())
    .catch(() => null);
  const [runtimesRes, filesRes, usableRepos] = await Promise.all([
    runtimesFetch,
    fetch('/api/hf/files').then((r) => r.json()).catch(() => null),
    jobRepos(),
  ]);

  const installed = runtimesRes?.runtimes?.installed ?? {};
  const groups = runtimesRes?.groups ?? [];
  const files = filesRes?.files ?? [];

  hubRuntimesEl.innerHTML = '';

  // Group every download under the runtime that can run it. The catalog knows
  // the exact task for its sixteen entries; anything else is grouped by weight
  // format, so an unrecognised repo still lands in a sensible section instead
  // of being dropped or lumped into "unknown".
  const byRuntime = new Map();
  const unroutableGroups = new Map();

  // A downloaded bundle is one model, not one row per file. A CTranslate2
  // Whisper is `model.bin` plus `config.json`, `tokenizer.json` and
  // `vocabulary.txt`; listing those as four separate entries gave three of
  // them "unrecognised format" and made the hub disagree with the catalog,
  // which knows perfectly well what the set is.
  const byRepo = new Map();
  for (const file of files) {
    const entry = catalogEntryForRepo(file.repo);
    // `file.repo`, not `entry.repo`: most downloads have no catalog entry at
    // all, and dereferencing the entry in the non-bundle branch threw a
    // TypeError that silently emptied the whole hub.
    const key = entry?.bundle ? `${entry.repo}#bundle` : file.repo;
    if (!byRepo.has(key)) byRepo.set(key, { entry, members: [] });
    byRepo.get(key).members.push(file);
  }

  // The weights, not whichever file happened to sort first — "config.json" is
  // not what the user recognises as the model. The largest file is the weights
  // in every format here, the same rule the server-side resolver uses.
  const largestOf = (list) =>
    list.length > 1
      ? list.slice().sort((a, b) => (b.size ?? 0) - (a.size ?? 0))[0] ?? list[0]
      : list[0];

  // Rows are models, not files. A bundle is one model by catalog definition.
  // Otherwise the weight files are the models and the sidecars that make them
  // work — tokenizer JSONs, configs, vocabularies, demo clips — ride along
  // with the largest weight. Listing every file separately is what filled the
  // hub with "no local runtime" rows for tokenizer.json and sample wavs. A
  // repo with no weight at all keeps a row per file, so nothing on disk
  // silently disappears from the page.
  const grouped = [];
  for (const { entry, members } of byRepo.values()) {
    if (entry?.bundle) {
      grouped.push({ entry, members, anchor: largestOf(members) });
      continue;
    }
    const weights = members.filter((m) => runtimeForFile(m.file));
    if (!weights.length) {
      const anchor = largestOf(members);
      const repoKey = anchor.repo;
      if (!unroutableGroups.has(repoKey)) {
        unroutableGroups.set(repoKey, { entry, members: [], anchor });
      }
      const g = unroutableGroups.get(repoKey);
      g.members.push(...members);
      if ((anchor.size ?? 0) > (g.anchor.size ?? 0)) g.anchor = anchor;
      if (!g.entry) g.entry = entry;
      continue;
    }
    weights.sort((a, b) => (b.size ?? 0) - (a.size ?? 0));
    const sidecars = members.filter((m) => !runtimeForFile(m.file));
    weights.forEach((w, i) => {
      grouped.push({ entry, members: i === 0 ? [w, ...sidecars] : [w], anchor: w });
    });
  }

  for (const { entry, members, anchor } of grouped) {
    // A catalog entry with a runner is authoritative — it knows the task.
    const inferred = entry?.bundle ? null : chatRuntimeFor(anchor) ?? runtimeForFile(anchor.file);
    const key = entry?.run?.requires[0] ?? inferred?.id ?? null;
    const present = members.map((m) => m.file);
    const wanted = entry?.files?.map((f) => f.name) ?? [];
    const missing = wanted.filter((w) => !present.includes(w));

    const row = {
      ...anchor,
      // Where the bytes live: this app's downloads dir, or the shared HF
      // cache another tool wrote to. The row says which, and only the app's
      // own copies may be deleted from here.
      location: anchor.location ?? 'models',
      // True when no job in the app loads this repo — a forecasting model, a
      // tuner, a TTS architecture nothing here speaks. The row says so
      // instead of leaving a runtime label that reads as usable. A GGUF chat
      // model is not jobless: the chat page's llama.cpp provider loads it.
      jobless: anchor.llama || anchor.transformers ? false : usableRepos ? !usableRepos.has(anchor.repo) : false,
      entry: entry ?? null,
      // One model can span several files; the runner is pointed at the weights.
      file: anchor.file,
      files: present,
      missing,
      complete: !missing.length,
      // The API field is `size`; reading `bytes` zeroed every row's size.
      // A bundle reports the sum of its parts, which is what occupies disk.
      size: members.reduce((a, m) => a + (m.size ?? 0), 0),
    };

    if (!key) {
      // Already accumulated above by repo — nothing to do here.
      continue;
    }
    if (!byRuntime.has(key)) byRuntime.set(key, { entry, inferred, files: [] });
    byRuntime.get(key).files.push(row);
  }

  // Collapse the loose files into one row per repo, so a folder of configs and
  // sample clips no longer reads as a wall of "unrecognised format" rows. The
  // row keeps the largest file as its anchor and lists the rest as members, so
  // Delete still clears the whole set in one go.
  const unroutable = [...unroutableGroups.values()].map(({ entry, members, anchor }) => {
    const present = members.map((m) => m.file);
    const wanted = entry?.files?.map((f) => f.name) ?? [];
    const missing = wanted.filter((w) => !present.includes(w));
    return {
      ...anchor,
      location: anchor.location ?? 'models',
      jobless: usableRepos ? !usableRepos.has(anchor.repo) : false,
      entry: entry ?? null,
      file: anchor.file,
      files: present,
      missing,
      complete: !missing.length,
      size: members.reduce((a, m) => a + (m.size ?? 0), 0),
      loose: true,
    };
  });

  const groupsWithModels = [...byRuntime.entries()];
  const title = document.createElement('h3');
  title.className = 'hub-title';
  title.textContent = 'Downloaded weights · by runtime';
  hubRuntimesEl.append(title);

  if (!groupsWithModels.length && !unroutable.length) {
    const p = document.createElement('p');
    p.className = 'section-sub';
    p.textContent =
      'No downloaded weights yet. Get models pulls and downloads land here, grouped by the runtime that runs them.';
    hubRuntimesEl.append(p);
    return;
  }

  for (const [runtimeId, { entry, files: groupFiles }] of groupsWithModels) {
    const group = groups.find((g) => g.id === runtimeId);
    const inferred = byRuntime.get(runtimeId)?.inferred ?? null;
    const external = Boolean(inferred?.external);
    // What this group needs to run: the extension map's requirements, plus the
    // catalog entry's. A bundle has no extension-based runtime at all, so
    // reading only the former reported an installed faster-whisper as missing.
    const requires = inferred?.requires ?? [];
    const catalogRequires = groupFiles.flatMap((f) => f.entry?.run?.requires ?? []);
    const allRequires = [...new Set([...requires, ...catalogRequires])];
    const isIn = external
      ? false
      : allRequires.length > 0 && allRequires.every((g) => Boolean(installed[g]));
    const needed = new Set(allRequires);

    const section = document.createElement('section');
    section.className = `hub-group${isIn ? '' : ' is-missing'}`;

    const head = document.createElement('div');
    head.className = 'hub-group__head';

    const h = document.createElement('h4');
    h.className = 'hub-group__title';
    h.textContent = group?.label ?? inferred?.label ?? runtimeId;
    head.append(h);

    // The state tag appears only when something is missing. A green "Installed"
    // on every group was decoration — the one state that needs acting on has
    // to stand out, not blend into a wall of pills.
    if (!isIn) {
      const state = document.createElement('span');
      state.className = 'hub-group__state';
      state.textContent = 'Not installed';
      head.append(state);
    }
    section.append(head);

    const blurbText = group?.blurb ?? inferred?.blurb;
    if (blurbText) {
      const blurb = document.createElement('p');
      blurb.className = 'hub-group__blurb';
      blurb.textContent = blurbText;
      section.append(blurb);
    }

    if (external) {
      const note = document.createElement('p');
      note.className = 'hub-group__note';
      note.textContent =
        `${groupFiles.length} GGUF file${groupFiles.length === 1 ? '' : 's'}, none of them chat models. ` +
        'Nothing to install here — these need their own program (whisper.cpp, ComfyUI).';
      const bar = document.createElement('div');
      bar.className = 'hub-group__actions';
      bar.append(note);
      section.append(bar);
    } else if (!isIn) {
      // The exact thing that is missing, with the button to fetch it. This is
      // what grouping by runtime buys: the gap and its fix are adjacent.
      const missing = [...needed].filter((g) => !installed[g]);
      const bar = document.createElement('div');
      bar.className = 'hub-group__actions';
      const note = document.createElement('span');
      note.className = 'hub-group__note';
      note.textContent = `${groupFiles.length} model${groupFiles.length === 1 ? '' : 's'} waiting on ${missing.join(' + ') || runtimeId}.`;
      bar.append(note);
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn-primary';
      btn.textContent = `Install ${missing.join(' + ') || runtimeId}`;
      btn.addEventListener('click', () => {
        for (const g of missing) rtPicked.add(g);
        setSubview('runtimes');
        loadRuntimes();
      });
      bar.append(btn);
      section.append(bar);
    }

    const list = document.createElement('div');
    list.className = 'hub-files';
    for (const f of groupFiles) list.append(renderLocalDownload(f, byRuntime.get(runtimeId)?.inferred ?? null));
    section.append(list);
    hubRuntimesEl.append(section);
  }

  if (unroutable.length) {
    // Downloaded, present, and not something this app can execute. Said
    // plainly rather than omitted, because "where did my file go" is worse
    // than "this one needs another program".
    const section = document.createElement('section');
    section.className = 'hub-group';
    const head = document.createElement('div');
    head.className = 'hub-group__head';
    const h = document.createElement('h4');
    h.className = 'hub-group__title';
    h.textContent = 'On disk · no local runtime';
    const state = document.createElement('span');
    state.className = 'hub-group__state';
    state.textContent = `${unroutable.length} file${unroutable.length === 1 ? '' : 's'}`;
    head.append(h, state);
    section.append(head);

    const blurb = document.createElement('p');
    blurb.className = 'hub-group__blurb';
    blurb.textContent =
      'Downloaded and intact, but this app has no runner for them. Use them with their own tooling — llama.cpp, ComfyUI, faster-whisper — or find a catalog entry that maps to a runtime here.';
    section.append(blurb);

    const list = document.createElement('div');
    list.className = 'hub-files';
    for (const f of unroutable) list.append(renderLocalDownload(f, null));
    section.append(list);
    hubRuntimesEl.append(section);
  }
}

/* ---------------- Local runtime catalog ---------------- */

/**
 * Renders the curated local-runtime catalog.
 *
 * Two kinds of action, and the difference is not cosmetic: Ollama pulls chat
 * models from its own registry by name and 404s on anything else. So a Pull
 * button only ever appears on a text entry carrying a verified tag, and every
 * other entry gets Download buttons that fetch a named file from the Hub
 * through the existing downloader.
 */
const FORMAT_LABEL = {
  gguf: 'GGUF',
  safetensors: 'safetensors',
  onnx: 'ONNX',
  bin: 'bin',
  pth: 'pth',
};

let catalogData = [];
/** "repo::file" keys of what's on disk, and the installed runtime groups. Both
 *  gate whether a card offers Run; neither is guessed. */
const catalogDownloaded = new Set();
let catalogInstalled = {};

function catalogFormats(entry) {
  const seen = new Set();
  for (const f of entry.files ?? []) {
    const m = f.name.match(/\.(gguf|safetensors|onnx|bin|pth)$/i);
    if (m) seen.add(m[1].toLowerCase());
  }
  return [...seen];
}

/** One model: what it is, which runtime loads it, and what you can do now. */
function renderCatalogEntry(entry) {
  const row = document.createElement('article');
  row.className = 'cat-entry';
  row.dataset.repo = entry.repo;
  row.dataset.catalogTask = entry.task;

  // Clicking anywhere on the card — not just a tiny button — is what people
  // actually try, and it used to do nothing at all. It now expands the actions
  // in place, which is one click fewer than hunting for the right button.
  const head = document.createElement('button');
  head.type = 'button';
  head.className = 'cat-entry__head';
  head.setAttribute('aria-expanded', 'false');

  const title = document.createElement('div');
  title.className = 'cat-entry__title';
  title.textContent = entry.label;
  head.append(title);

  const badges = document.createElement('div');
  badges.className = 'cat-entry__badges';
  for (const f of catalogFormats(entry)) {
    const b = document.createElement('span');
    b.className = `cat-badge cat-badge--${f}`;
    b.textContent = FORMAT_LABEL[f] ?? f.toUpperCase();
    badges.append(b);
  }
  // The runtime is the load-bearing fact on this card, so it reads as the
  // primary tag rather than as one more format chip.
  const rt = document.createElement('span');
  rt.className = 'cat-badge cat-badge--runtime';
  rt.textContent = entry.runtime;
  badges.append(rt);
  head.append(badges);
  row.append(head);

  const blurb = document.createElement('p');
  blurb.className = 'cat-entry__blurb';
  blurb.textContent = entry.blurb;
  row.append(blurb);

  const repo = document.createElement('p');
  repo.className = 'cat-entry__repo';
  repo.textContent = entry.repo;
  row.append(repo);

  const actions = document.createElement('div');
  actions.className = 'cat-entry__actions';
  // Hidden until the card is opened, so a grid of 16 cards doesn't read as 16
  // walls of buttons. Still in the DOM, so the Run gate stays live.
  actions.hidden = true;

  head.addEventListener('click', () => {
    actions.hidden = !actions.hidden;
    row.classList.toggle('is-open', !actions.hidden);
    head.setAttribute('aria-expanded', String(!actions.hidden));
  });

  if (entry.ollamaTag) {
    const pull = document.createElement('button');
    pull.type = 'button';
    pull.className = 'btn-primary cat-pull';
    const pullLabel = `Pull ${entry.ollamaTag}`;
    pull.textContent = pullLabel;
    pull.title = 'Pulls from Ollama’s own registry into your local daemon.';
    pull.addEventListener('click', async () => {
      pull.disabled = true;
      pull.textContent = 'Pulling…';
      try {
        const res = await fetch('/api/ollama/pull', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ name: entry.ollamaTag }),
        });
        if (!res.ok || !res.body) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error ?? `HTTP ${res.status}`);
        }
        if ((res.headers.get('content-type') ?? '').includes('application/json')) {
          const data = await res.json();
          if (data.attached) {
            beginTransfer('pull', `ollama:${entry.ollamaTag}`, {}).update({ state: 'running' });
            void syncTransfersSoon();
            pull.textContent = 'Already pulling';
            pull.disabled = false;
            notifyWarning(`${entry.ollamaTag} is already pulling — see Downloads`);
            return;
          }
          throw new Error(data.error ?? 'Pull refused');
        }
        let pulled = null;
        await sseStream(res.body, async (event, parsed) => {
          if (event === 'error') throw new Error(parsed.message);
          if (event === 'progress') pull.textContent = describePull(parsed);
          if (event === 'done') pulled = parsed.model ?? entry.ollamaTag;
        });
        await refreshLocalModels();
        const installed = pulled ?? entry.ollamaTag;
        pull.textContent = `${installed} pulled`;
        notifySuccess(`${installed} installed`);
      } catch (err) {
        notifyError(err.message);
        pull.textContent = pullLabel;
        pull.disabled = false;
      }
    });
    actions.append(pull);
  }

  // A bundle is one model split across several files. Downloading them one at
  // a time leaves a half-populated directory the loader refuses — a CTranslate2
  // Whisper without its config and tokenizer is exactly that — so they are
  // fetched together behind a single button, in size order with the weights
  // last so the progress reads sensibly.
  if (entry.bundle && entry.files?.length) {
    const total = entry.files.reduce((a, f) => a + (f.bytes ?? 0), 0);
    const have = entry.files.filter((f) => catalogDownloaded.has(`${entry.repo}::${f.name}`)).length;
    const complete = have === entry.files.length;

    if (complete && runIsReady(entry, catalogInstalled)) {
      const run = document.createElement('button');
      run.type = 'button';
      run.className = 'btn-primary cat-run';
      run.textContent = 'Run';
      run.title = `${runLabel(entry.run.kind)} on this machine`;
      run.addEventListener('click', () => openRunPanel(entry, entry.files[0]));
      actions.append(run);
    }

    const dl = document.createElement('button');
    dl.type = 'button';
    dl.className = complete ? 'btn-ghost cat-dl' : 'btn-primary cat-dl';
    const small = entry.files.slice().sort((a, b) => (a.bytes ?? 0) - (b.bytes ?? 0));
    dl.textContent = complete
      ? `Re-download · ${fmtSize(total)}`
      : `Download all ${entry.files.length} files · ${fmtSize(total)}`;
    dl.title = complete
      ? `Overwrite: ${entry.files.map((f) => f.name).join(', ')}`
      : `${small.map((f) => `${f.name} (${fmtSize(f.bytes)})`).join(', ')} — all are required`;
    actions.append(dl);
    dl.addEventListener('click', async () => {
      dl.disabled = true;
      const original = dl.textContent;
      // Whole-model progress, not per-file. A CTranslate2 Whisper is four files
      // and the weights are 97% of it, so a per-file figure spends most of the
      // download reading "0 / 4" and then jumps at the end. The catalog knows
      // every file's size up front, which is exactly the denominator a
      // percentage needs.
      const totalBytes = entry.files.reduce((a, f) => a + (f.bytes ?? 0), 0);
      let doneBytes = 0;
      const paint = (got, file) => {
        if (!totalBytes) {
          dl.textContent = `Downloading ${file ?? ''}…`;
          return;
        }
        const pct = Math.min(100, (got / totalBytes) * 100);
        dl.textContent = `${pct.toFixed(0)}% · ${fmtSize(got)} / ${fmtSize(totalBytes)}`;
      };
      try {
        for (const f of small) {
          dl.title = `Downloading ${f.name}…`;
          await downloadHfFile(entry.repo, f.name, dl, null, {
            onTick: (received) => paint(doneBytes + received, f.name),
          });
          doneBytes += f.bytes ?? 0;
          paint(doneBytes, f.name);
        }
        notifySuccess(`${entry.label} saved under ./models/`);
      } catch (err) {
        notifyError(err.message);
      } finally {
        dl.textContent = original;
        dl.disabled = false;
        await refreshCatalogRunState();
      }
    });
  }

  for (const f of entry.bundle ? [] : entry.files ?? []) {
    // Run only appears once BOTH halves are real: the runtime this entry needs
    // is installed, and these exact weights are already on disk. Either half
    // missing and there is nothing to run, so no button is drawn.
    const downloaded = catalogDownloaded.has(`${entry.repo}::${f.name}`);
    if (downloaded && runIsReady(entry, catalogInstalled)) {
      const run = document.createElement('button');
      run.type = 'button';
      run.className = 'btn-primary cat-run';
      run.textContent = 'Run';
      run.title = `${runLabel(entry.run.kind)} on this machine`;
      run.addEventListener('click', () => openRunPanel(entry, f));
      actions.append(run);
    }

    const dl = document.createElement('button');
    dl.type = 'button';
    dl.className = 'btn-ghost cat-dl';
    const dlLabel = `${f.label} · ${fmtSize(f.bytes)}`;
    dl.textContent = dlLabel;
    dl.title = `${f.name} — saves under ./models/`;
    // The same download path the Hugging Face file list uses, so progress,
    // the overwrite guard and the Downloads refresh all behave identically.
    const status = document.createElement('span');
    status.className = 'cat-entry__none';
    status.textContent = '';
    actions.append(dl);
    dl.addEventListener('click', async () => {
      dl.disabled = true;
      try {
        await downloadHfFile(entry.repo, f.name, dl, status);
        notifySuccess(`${f.name} saved under ./models/`);
      } finally {
        dl.textContent = dlLabel;
        dl.disabled = false;
        await refreshCatalogRunState();
      }
    });
  }

  if (!actions.childElementCount) {
    const none = document.createElement('span');
    none.className = 'cat-entry__none';
    none.textContent = 'No single-file download — follow the repo for setup.';
    actions.append(none);
  }
  row.append(actions);
  return row;
}

function renderCatalog() {
  const host = document.getElementById('catalog-list');
  host.innerHTML = '';
  let shown = 0;
  for (const section of catalogData) {
    if (!section.entries?.length) continue;
    shown += section.entries.length;

    const wrap = document.createElement('section');
    wrap.className = 'cat-section';
    wrap.dataset.catalogTask = section.task;

    const h = document.createElement('h3');
    h.className = 'cat-section__title';
    h.textContent = section.title;
    const n = document.createElement('span');
    n.className = 'cat-section__count';
    n.textContent = String(section.entries.length);
    h.append(n);
    wrap.append(h);

    // Why this task can't be pulled. The reader should never have to guess
    // whether a missing Pull button is a bug.
    if (section.note) {
      const note = document.createElement('p');
      note.className = 'cat-section__note';
      note.textContent = section.note;
      wrap.append(note);
    }

    const list = document.createElement('div');
    list.className = 'cat-list';
    for (const entry of section.entries) list.append(renderCatalogEntry(entry));
    wrap.append(list);
    host.append(wrap);
  }
  if (!shown) {
    const p = document.createElement('p');
    p.className = 'section-sub';
    p.textContent = 'The catalog is empty.';
    host.append(p);
  }
}

/** Static data shipped with the server, so this never fails with the Hub down. */
let catalogLoad = null;

/**
 * Loads the catalog once, whichever surface asks first.
 *
 * The hub needs it to know a download's task and whether its files are a
 * bundle, but the hub is reachable without ever visiting Get models — so the
 * data was empty and every downloaded bundle silently lost its runtime.
 */
function ensureCatalog() {
  if (catalogData.length) return Promise.resolve(catalogData);
  if (!catalogLoad) {
    catalogLoad = fetch('/api/catalog')
      .then((res) => res.json().then((d) => {
        if (!res.ok) throw new Error(d.error ?? `HTTP ${res.status}`);
        catalogData = d.catalog ?? [];
        return catalogData;
      }))
      .finally(() => {
        catalogLoad = null;
      });
  }
  return catalogLoad;
}

async function loadCatalog() {
  if (catalogData.length) {
    renderCatalog();
    return;
  }
  const host = document.getElementById('catalog-list');
  host.innerHTML = '';
  host.append(hfNote('Loading the catalog…'));
  try {
    await ensureCatalog();
    // Gating Run needs both facts, so gather them before the first paint.
    await refreshCatalogRunState();
    renderCatalog();
  } catch (err) {
    host.innerHTML = '';
    host.append(hfNote(err instanceof Error ? err.message : String(err)));
  }
}


/**
 * Re-reads what's installed and what's downloaded, then redraws the catalog so
 * Run buttons appear (or disappear) as soon as either side changes — after an
 * install, or after a download finishes.
 */
async function refreshCatalogRunState() {
  try {
    const [rt, files] = await Promise.all([
      fetch('/api/runtimes').then((r) => r.json()).catch(() => null),
      fetch('/api/hf/files').then((r) => r.json()).catch(() => null),
    ]);
    catalogInstalled = rt?.runtimes?.installed ?? {};
    catalogDownloaded.clear();
    for (const f of files?.files ?? []) catalogDownloaded.add(`${f.repo}::${f.file}`);
  } catch {
    /* leave the previous state; the buttons simply stay as they were */
  }
  if (catalogData.length) renderCatalog();
}

/* ---------------- Running a downloaded model locally ---------------- */

/**
 * The last link in the chain: a runtime is installed, a model is downloaded,
 * and this turns that into an actual result. Nothing here decides *whether* a
 * model can run — that comes from the catalog's `run` block, so the button only
 * appears for entries a bundled runner genuinely handles, and an entry that
 * needs llama.cpp or ComfyUI says so instead of offering a dead button.
 */

const runPanel = document.getElementById('run-panel');
const runTitle = document.getElementById('run-title');
const runBody = document.getElementById('run-body');
const runClose = document.getElementById('run-close');
const runLog = document.getElementById('run-log');

let runBusy = false;

function closeRunPanel() {
  runPanel.hidden = true;
  runBody.innerHTML = '';
  runLog.textContent = '';
}

/** Streams an /api/infer/* response, logging progress and resolving the result. */
async function streamInference(url, body, onProgress) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error ?? `HTTP ${res.status}`);
  }
  let result = null;
  let failure = null;
  await sseStream(res.body, async (event, parsed) => {
    if (event === 'log') {
      runLog.textContent += `${parsed}\n`;
      runLog.scrollTop = runLog.scrollHeight;
    } else if (event === 'error') {
      failure = parsed.message;
    } else if (event === 'done') {
      result = parsed;
    }
  });
  if (failure) throw new Error(failure);
  return result;
}

function runLabel(kind) {
  return kind === 'stt' ? 'Transcribe audio' : kind === 'tts' ? 'Speak text' : 'Generate image';
}

/** Builds the form for one run kind and wires its submit. */
function openRunPanel(entry, file) {
  if (!entry.run) return;
  runPanel.hidden = false;
  runTitle.textContent = `${runLabel(entry.run.kind)} — ${entry.label}`;
  runBody.innerHTML = '';
  runLog.textContent = '';
  runPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  const form = document.createElement('div');
  form.className = 'run-form';
  const out = document.createElement('div');
  out.className = 'run-out';

  const add = (labelText, input) => {
    const wrap = document.createElement('label');
    wrap.className = 'run-field';
    const span = document.createElement('span');
    span.textContent = labelText;
    wrap.append(span, input);
    form.append(wrap);
    return input;
  };

  const go = document.createElement('button');
  go.type = 'button';
  go.className = 'btn-primary';
  go.textContent = runLabel(entry.run.kind);

  const inputs = {};
  if (entry.run.kind === 'stt') {
    const audio = document.createElement('input');
    audio.type = 'file';
    audio.accept = 'audio/*';
    inputs.audio = add('Audio file', audio);
  } else if (entry.run.kind === 'tts') {
    const text = document.createElement('textarea');
    text.rows = 3;
    text.placeholder = 'What should it say?';
    inputs.text = add('Text', text);
    const voice = document.createElement('input');
    voice.type = 'text';
    voice.value = 'af_heart';
    inputs.voice = add('Voice', voice);
  } else {
    const prompt = document.createElement('textarea');
    prompt.rows = 3;
    prompt.placeholder = 'Describe the image';
    inputs.prompt = add('Prompt', prompt);
    const steps = document.createElement('input');
    steps.type = 'number';
    steps.min = '1';
    steps.max = '50';
    steps.value = '4';
    inputs.steps = add('Steps', steps);
    const size = document.createElement('input');
    size.type = 'text';
    size.value = '512x512';
    inputs.size = add('Size', size);
  }

  form.append(go);
  runBody.append(form, out);

  go.addEventListener('click', async () => {
    if (runBusy) return;
    const device = document.querySelector('.rt-device__opt[aria-checked="true"]')?.dataset.device ?? 'cpu';
    runBusy = true;
    go.disabled = true;
    go.textContent = 'Working…';
    out.innerHTML = '';
    runLog.textContent = '';

    try {
      let result;
      if (entry.run.kind === 'stt') {
        const picked = inputs.audio.files?.[0];
        if (!picked) throw new Error('Choose an audio file first.');
        // The runner takes a path, and only accepts a file the downloader
        // wrote, so a browser file upload is staged next to it server-side.
        const uploaded = await uploadAudio(picked);
        result = await streamInference(
          '/api/infer/stt',
          // `file` names the model; `audio` is the freshly uploaded recording.
          { repo: entry.repo, file: file.name, audio: uploaded, device },
          null,
        );
      } else if (entry.run.kind === 'tts') {
        result = await streamInference(
          '/api/infer/tts',
          {
            repo: entry.repo,
            file: file.name,
            text: inputs.text.value,
            voice: inputs.voice.value.trim() || 'af_heart',
            speed: 1,
            device,
          },
          null,
        );
      } else {
        const [w, h] = String(inputs.size.value).toLowerCase().split('x').map((n) => Number(n) || 512);
        result = await streamInference(
          '/api/infer/image',
          {
            repo: entry.repo,
            file: file.name,
            prompt: inputs.prompt.value,
            steps: Number(inputs.steps.value) || 4,
            width: w,
            height: h,
            device,
          },
          null,
        );
      }
      renderRunResult(entry.run.kind, result, out);
    } catch (err) {
      out.innerHTML = '';
      const p = document.createElement('p');
      p.className = 'scan-result error';
      p.textContent = err instanceof Error ? err.message : String(err);
      out.append(p);
    } finally {
      runBusy = false;
      go.disabled = false;
      go.textContent = runLabel(entry.run.kind);
    }
  });
}

function renderRunResult(kind, result, out) {
  out.innerHTML = '';
  if (!result) {
    const p = document.createElement('p');
    p.className = 'scan-result warn';
    p.textContent = 'The runner finished without returning a result.';
    out.append(p);
    return;
  }
  if (kind === 'stt') {
    const text = document.createElement('p');
    text.className = 'run-text';
    text.textContent = result.text || '(no speech detected)';
    out.append(text);
    const chunks = result.chunks ?? [];
    if (chunks.length) {
      const detail = document.createElement('details');
      detail.className = 'run-chunks';
      const sum = document.createElement('summary');
      sum.textContent = `${chunks.length} timed segment${chunks.length === 1 ? '' : 's'}`;
      detail.append(sum);
      for (const c of chunks) {
        const row = document.createElement('div');
        row.className = 'run-chunk';
        row.textContent = `${c.start ?? '?'}–${c.end ?? '?'}s  ${c.text}`;
        detail.append(row);
      }
      out.append(detail);
    }
    return;
  }
  const name = String(result.path ?? '').split('/').pop();
  if (kind === 'tts') {
    const audio = document.createElement('audio');
    audio.controls = true;
    audio.src = `/api/infer/out/${encodeURIComponent(name)}`;
    out.append(audio);
  } else {
    const img = document.createElement('img');
    img.className = 'run-image';
    img.alt = 'Generated image';
    img.src = `/api/infer/out/${encodeURIComponent(name)}`;
    out.append(img);
  }
  const meta = document.createElement('p');
  meta.className = 'run-meta';
  meta.textContent = [
    result.seconds ? `${result.seconds}s audio` : null,
    result.width ? `${result.width}×${result.height}` : null,
    result.device ? `on ${result.device}` : null,
  ]
    .filter(Boolean)
    .join(' · ');
  if (meta.textContent) out.append(meta);
}

/**
 * Stages a browser-picked audio file under ./models/_uploads so the runner's
 * "must already be a downloaded file" rule still holds.
 *
 * The body is the File itself, not multipart: the server's uploader parses a raw
 * audio/* payload (the same one dictation already uses), so wrapping it in
 * FormData arrives as an empty body.
 */
async function uploadAudio(file) {
  const res = await fetch('/api/infer/upload', {
    method: 'POST',
    headers: { 'content-type': file.type || 'application/octet-stream' },
    body: file,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data.file;
}

runClose?.addEventListener('click', closeRunPanel);

/** True when every runtime this entry needs is installed. */
function runIsReady(entry, installed) {
  if (!entry.run?.requires?.length) return false;
  return entry.run.requires.every((g) => Boolean(installed[g]));
}

/* ---------------- Hugging Face browser ---------------- */

/**
 * Deterministic tile image per model id (gradient picked by hash, initial
 * letter on top). The Hub exposes no artwork for models and CSP forbids hot-
 * linking any, so every model gets a generated visual instead of a broken
 * image box. Same id always yields the same tile, across all three views.
 */
const TILE_GRADIENTS = [
  ['#7c5cff', '#00c2ff'],
  ['#ff5c8a', '#ff9a3c'],
  ['#00c2a8', '#7c5cff'],
  ['#f5b301', '#ef4444'],
  ['#38bdf8', '#818cf8'],
  ['#c084fc', '#f472b6'],
  ['#34d399', '#0ea5e9'],
  ['#fb7185', '#a855f7'],
];

function modelTile(id, size = '') {
  const s = document.createElement('span');
  s.className = `model-tile ${size}`.trim();
  let h = 0;
  for (const ch of id) h = (h * 31 + ch.codePointAt(0)) >>> 0;
  const [a, b] = TILE_GRADIENTS[h % TILE_GRADIENTS.length];
  s.style.background = `linear-gradient(135deg, ${a}, ${b})`;
  const short = (id.split('/').pop() || id).replace(/^[^A-Za-z0-9]+/, '');
  s.textContent = (short[0] || '?').toUpperCase();
  s.setAttribute('aria-hidden', 'true');
  return s;
}

/**
 * Re-renders everything that is derived from what is on disk.
 *
 * Four separate views read the same two endpoints — /api/hf/files and
 * /api/runnable — and each one used to be refreshed by whichever action
 * happened to remember it. Deleting a file refreshed the flat download list
 * and left the runtime-grouped hub and "What each job uses" showing a model
 * that had just been deleted, which is worse than not updating at all: the two
 * halves of the same page disagree and there is no way to tell which is right.
 *
 * One function, called after every mutation, is the only way this stays true.
 * The individual refreshers are still used directly where one view genuinely
 * needs to load on its own.
 */
let syncing = null;
async function syncDerived() {
  // Overlapping triggers — a finished download and a click on another tab — used
  // to interleave their writes into the same containers and leave a half-built
  // list. Collapsing them into one in-flight run is cheaper and safer.
  if (syncing) return syncing;
  syncing = (async () => {
    // The runtime inventory behind the hub is the slow part when a probe is
    // cold on the server (a full Python + torch import). The Ollama list and
    // the task pickers do not depend on it, and sequenced behind it they were
    // the last two things to paint — the Chat models section sat empty for as
    // long as that probe took. Start everything, await everything.
    const inventory = refreshCatalogRunState()
      .then(refreshLocalDownloads)
      .then(refreshHub)
      .catch(() => {});
    await Promise.all([
      inventory,
      renderAssignments().catch(() => {}),
      refreshLocalModels().catch(() => {}),
    ]);
  })().finally(() => {
    syncing = null;
  });
  return syncing;
}

/* Models-tab sub-views: local list, Hub browser, downloads. */
const subBtns = [...document.querySelectorAll('[data-subview]')];
const subPanels = [...document.querySelectorAll('[data-subpanel]')];
const loadedSubviews = new Set(['local']);
/** Which subview is showing, so a tab round-trip and a reload can restore it. */
let currentSubview = 'local';

/**
 * The subview lives in the hash alongside the tab, as "#models/runtimes".
 *
 * Without it a reload dropped you on the first tab, which is the other half of
 * "the page does not update": you change something, nothing appears to happen,
 * you reload to force it, and land somewhere else with the change hidden. The
 * two complaints are the same bug from opposite ends.
 */
function subviewFromHash() {
  const [tab, sub] = location.hash.replace(/^#/, '').split('/');
  return {
    tab: TABS.includes(tab) ? tab : 'keys',
    sub: SUBVIEWS.includes(sub) ? sub : null,
  };
}

function setSubview(name, push = true) {
  currentSubview = name;
  for (const b of subBtns) {
    const active = b.dataset.subview === name;
    b.setAttribute('aria-pressed', String(active));
  }
  for (const p of subPanels) {
    p.hidden = p.dataset.subpanel !== name;
  }
  if (name === 'browse' && !loadedSubviews.has('browse')) {
    loadedSubviews.add('browse');
    loadHfTrending();
  }
  // The catalog is static and ships with the server, so it is always safe to
  // refresh — no upstream call, nothing to re-fetch once it has rendered.
  if (push) {
    const want = `#models/${name}`;
    if (location.hash !== want) location.hash = want;
  }
  if (name === 'catalog') loadCatalog();
  if (name === 'local') {
    // The Ollama list is one fast local daemon call and depends on nothing
    // else on this panel; chained behind the runtime inventory below it sat
    // empty for as long as a cold probe took. Start it now, in parallel.
    void refreshLocalModels().catch(() => {});
    // Installed runtimes gate the Run buttons, so both facts are re-read.
    refreshCatalogRunState()
      .then(refreshLocalDownloads)
      .then(refreshHub)
      .then(renderAssignments);
  }
  if (name === 'downloads') {
    renderDownloads();
    void syncTransfers();
  }
  if (name === 'runtimes') {
    // Every visit re-reads the truth. Loading once meant the panel showed
    // whatever was installed at the moment you first opened it, which is
    // exactly the wrong answer after installing anything.
    const first = !loadedSubviews.has('runtimes');
    loadedSubviews.add('runtimes');
    loadRuntimes({ force: !first });
    // Read on entry rather than once: which backend is on the GPU changes when
    // a model is loaded or unloaded, and a panel frozen at first open is wrong
    // exactly when it is being looked at.
    refreshBackends();
  }
}

for (const b of subBtns) {
  b.addEventListener('click', () => setSubview(b.dataset.subview));
}
// The default panel is marked in the HTML, not by a click, so nothing had run
// its loader on arrival. Boot whichever one is pressed.
//
// Deferred to a microtask on purpose: running this inline only worked because
// the `const` declarations it can reach (TILE_GRADIENTS and friends) happened to
// sit earlier in the file. A module's top level evaluates fully before any
// microtask, so this can never touch a binding that is still in its TDZ.
queueMicrotask(() => {
  const initial = subBtns.find((b) => b.getAttribute('aria-pressed') === 'true');
  // push:false — this runs before the awaited startup restore further down, and
  // writing the hash here would overwrite "#models/runtimes" with the default
  // subview before anything had read it. The startup block owns the URL.
  if (initial) setSubview(initial.dataset.subview, false);
  // Always, on every load: a transfer started before the reload is still
  // running on the server, and the page should know about it without waiting
  // to be sent to the Downloads tab first.
  void syncTransfers();
});

const hfForm = document.getElementById('hf-search-form');
const hfQ = document.getElementById('hf-q');
/** No sort control: everything is shown together, grouped by task. */
const hfResults = document.getElementById('hf-results');
const hfModalBackdrop = document.getElementById('hf-modal-backdrop');
const hfModalTile = document.getElementById('hf-modal-tile');
const hfModalTitle = document.getElementById('hf-modal-title');
const hfModalStats = document.getElementById('hf-modal-stats');
const hfModalFiles = document.getElementById('hf-modal-files');
const hfModalClose = document.getElementById('hf-modal-close');
const hfRepoDl = document.getElementById('hf-repo-dl');
hfRepoDl?.addEventListener('click', () =>
  downloadWholeRepo(hfModalTitle.textContent, hfRepoDl, document.getElementById('hf-repo-summary')));
/**
 * Where the downloaded-file list renders. It used to have its own sub-view,
 * which meant two lists of the same files — one grouped for running, one flat
 * for deleting. They are the same data, so they are the same list now, with
 * delete on every row.
 */
const hfFiles = document.getElementById('local-downloads');

let hfSearching = false;
/** Set when a search arrives while one is already running. */
let hfPendingQuery = null;
/** Last page of hits, kept so the category chips filter without re-querying. */
let hfLastHits = [];
let hfActiveFilter = 'all';
let hfDownloading = false;
let hfModalReturnFocus = null;

function openHfModal() {
  hfModalReturnFocus = document.activeElement;
  hfModalBackdrop.hidden = false;
  document.body.classList.add('modal-open');
  hfModalClose.focus();
}

function closeHfModal() {
  hfModalBackdrop.hidden = true;
  document.body.classList.remove('modal-open');
  if (hfModalReturnFocus instanceof HTMLElement) hfModalReturnFocus.focus();
}

hfModalClose.addEventListener('click', closeHfModal);
hfModalBackdrop.addEventListener('click', (e) => {
  if (e.target === hfModalBackdrop) closeHfModal();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !hfModalBackdrop.hidden) closeHfModal();
});

function hfNote(text, cls = '') {
  const p = document.createElement('p');
  p.className = `section-sub ${cls}`.trim();
  p.textContent = text;
  return p;
}

/** Total bytes for a detail: the repo figure, else the sum of file sizes. */
function hfRepoBytes(detail) {
  if (typeof detail.usedStorage === 'number' && detail.usedStorage > 0) return detail.usedStorage;
  return detail.files.reduce((n, f) => n + (typeof f.size === 'number' ? f.size : 0), 0);
}

/**
 * Bucket a Hub hit into the five things this app can use it for. The Hub's own
 * `pipeline_tag` is the primary signal; tags fill the gaps, because plenty of
 * speech repos are published with no useful pipeline_tag at all.
 */
/** Hub pipeline_tag for each category chip, used when a filter has to fetch. */
const HF_PIPELINE = {
  'image-generation': 'text-to-image',
  'speech-to-text': 'automatic-speech-recognition',
  'text-to-speech': 'text-to-speech',
  text: 'text-generation',
};

const HF_CATEGORIES = [
  { id: 'text', label: 'Text' },
  { id: 'speech-to-text', label: 'Speech to text' },
  { id: 'text-to-speech', label: 'Text to speech' },
  { id: 'image-generation', label: 'Image generation' },
  { id: 'video-generation', label: 'Video generation' },
  { id: 'audio', label: 'Audio' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'reranking', label: 'Reranking' },
  { id: 'multimodal', label: 'Vision & multimodal' },
  { id: 'other', label: 'Other' },
];

function categorizeHit(hit) {
  const tag = String(hit.pipelineTag ?? '').toLowerCase();
  // The repo id is part of the evidence: plenty of speech repos are published
  // with a generic or missing pipeline_tag and say what they are only in their
  // own name (`...TTS-0.6B`, `Orpeus_Text_To_Speech`).
  const hay = [hit.id, hit.pipelineTag ?? '', hit.libraryName ?? '', ...(hit.tags ?? [])]
    .filter((t) => typeof t === 'string' && t)
    .join(' ')
    .toLowerCase();

  // Unambiguous task signals are checked first, so a repo that is plainly TTS
  // isn't swallowed by a generic `text-generation` tag on the same record.
  // `_` counts as a separator here, which is why these are spelled out by hand
  // rather than relying on \b (it would not break on `mms_tts`).
  if (
    /(^|[^a-z])(tts)([^a-z]|$)|text[-_]?to[-_]?speech|text[-_]?to[-_]?audio|texttospeech|speecht5|vocoder|kokoro|mms[-_]tts/.test(hay)
  ) {
    return 'text-to-speech';
  }
  if (/automatic[-_]speech[-_]recognition|speech[-_]to[-_]text|(^|[^a-z])(asr)([^a-z]|$)|whisper|wav2vec|speechrecognition/.test(hay)) {
    return 'speech-to-text';
  }
  if (/text[-_]to[-_]image|text2img|(^|[^a-z])sdxl([^a-z]|$)|stable[-_]diffusion|(^|[^a-z])diffusers([^a-z]|$)|controlnet|image[-_]generation/.test(hay)) {
    return 'image-generation';
  }
  if (/text[-_]?to[-_]?video|text2vid|image[-_]?to[-_]?video|video[-_]?generation|(^|[^a-z])(svd|ltx[-_]?video|wan[-_]?2)([^a-z]|$)/.test(hay)) {
    return 'video-generation';
  }
  if (
    tag.includes('audio-to-audio') ||
    tag.includes('audio-classification') ||
    /audio[-_]?generation|musicgen|music|sound[-_]?effect|audio[-_]?enhanc/.test(hay)
  ) {
    return 'audio';
  }
  if (
    tag.includes('sentence-similarity') ||
    tag.includes('feature-extraction') ||
    /embedding|(^|[^a-z])(bge|e5|gte|clip)([^a-z]|$)/.test(hay)
  ) {
    return 'embeddings';
  }
  if (/rerank|(^|[^a-z])reranker([^a-z]|$)/.test(hay)) {
    return 'reranking';
  }
  if (
    tag.includes('image-to-text') ||
    tag.includes('image-classification') ||
    tag.includes('visual-question-answering') ||
    tag.includes('video-classification') ||
    /(^|[^a-z])(vlm|llava|qwen[-_]?vl|blip|clip|vision)([^a-z]|$)/.test(hay)
  ) {
    return 'multimodal';
  }
  if (
    tag.includes('text-generation') ||
    tag.includes('text2text') ||
    tag.includes('fill-mask') ||
    tag.includes('summarization') ||
    tag.includes('question-answering') ||
    tag.includes('translation') ||
    /text[-_]generation|(^|[^a-z])(llama|qwen|mistral|bert|gpt|phi|falcon)([^a-z]|$)/.test(hay)
  ) {
    return 'text';
  }
  return 'other';
}

/** Same-origin avatar URL — the proxy caches the author's logo, CSP stays
 *  'self'. Keyed by author: every model by the same maker shares one image. */
function hfAvatarUrl(repo) {
  return `/api/hf/avatar/${encodeURIComponent(String(repo).split('/')[0])}`;
}

/** One model card: the Hub's own artwork, id, task, and stats. */
function renderHfHit(hit) {
  const card = document.createElement('button');
  card.type = 'button';
  card.className = 'hf-card';
  card.setAttribute('aria-label', `${hit.id} — model details`);

  const media = document.createElement('div');
  media.className = 'hf-card__media';
  const img = document.createElement('img');
  img.className = 'hf-card__img';
  img.alt = '';
  img.loading = 'lazy';
  img.decoding = 'async';
  img.addEventListener(
    'error',
    () => {
      // Not every author has an avatar. Swap in the generated tile so the grid
      // stays even instead of leaving a torn image box behind.
      media.replaceChildren(modelTile(hit.id, 'lg'));
    },
    { once: true },
  );
  img.src = hfAvatarUrl(hit.id);
  media.append(img);
  card.append(media);

  const body = document.createElement('div');
  body.className = 'hf-card__body';

  const name = document.createElement('div');
  name.className = 'hf-card__name';
  name.textContent = hit.id;
  name.title = hit.id;
  body.append(name);

  const role = document.createElement('div');
  role.className = 'hf-card__role';
  role.textContent = hit.pipelineTag ?? 'Model';
  role.title = hit.pipelineTag ?? '';
  body.append(role);

  const stats = document.createElement('div');
  stats.className = 'hf-card__stats';
  // Abbreviated: "1.6M" fits a 172px card, "1,582,728" does not and forced the
  // whole row to wrap onto four lines.
  const compact = (n) => {
    if (typeof n !== 'number') return { short: '0', full: '0' };
    if (n < 1000) return { short: String(n), full: n.toLocaleString() };
    if (n < 1_000_000) return { short: `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`, full: n.toLocaleString() };
    return { short: `${(n / 1_000_000).toFixed(n < 10_000_000 ? 1 : 0)}M`, full: n.toLocaleString() };
  };
  const likeN = compact(hit.likes);
  const likes = document.createElement('span');
  likes.className = 'hf-card__stat';
  likes.textContent = likeN.short;
  likes.title = `${likeN.full} likes`;
  const dlN = compact(hit.downloads);
  const dls = document.createElement('span');
  dls.className = 'hf-card__stat';
  dls.textContent = dlN.short;
  dls.title = `${dlN.full} downloads`;
  const size = document.createElement('span');
  size.className = 'hf-card__stat hf-card__size';
  size.textContent = '…';
  stats.append(likes, dls, size);

  const runtimes = document.createElement('div');
  runtimes.className = 'hf-card__runtimes';
  runtimes.textContent = 'checking…';
  stats.append(runtimes);

  body.append(stats);

  card.append(body);
  card.addEventListener('click', () => showHfDetail(hit.id));
  return card;
}

/**
 * Which runtimes can load a repo, derived from the weight files it actually
 * ships. This costs nothing: the card already fetches model detail for the file
 * count and byte size, so the same response answers "what can run this".
 */
function runtimesForFiles(files) {
  const exts = new Set();
  for (const f of files ?? []) {
    const m = String(f.name ?? '').match(/\.(gguf|safetensors|onnx|bin|pth|ckpt)$/i);
    if (m) exts.add(m[1].toLowerCase());
  }
  const out = [];
  if (exts.has('gguf')) out.push('llama.cpp');
  if (exts.has('safetensors') || exts.has('bin') || exts.has('pth')) out.push('transformers');
  if (exts.has('onnx')) out.push('onnxruntime');
  if (exts.has('ckpt') || exts.has('safetensors')) out.push('diffusers');
  return [...new Set(out)];
}

/** Paints the runtime chips onto a card once its detail call lands. */
function paintRuntimeChips(card, runtimes, fallback = 'weights unknown') {
  const host = card.querySelector('.hf-card__runtimes');
  if (!host) return;
  host.innerHTML = '';
  if (!runtimes.length) {
    host.textContent = fallback;
    return;
  }
  for (const r of runtimes) {
    const chip = document.createElement('span');
    chip.className = 'hf-runtime';
    chip.textContent = r;
    host.append(chip);
  }
}

/** Renders the current page of hits through the active category filter. */
function renderHfFiltered() {
  hfResults.innerHTML = '';
  const hits = hfActiveFilter === 'all'
    ? hfLastHits
    : hfLastHits.filter((h) => categorizeHit(h) === hfActiveFilter);

  if (!hits.length) {
    hfResults.append(
      hfNote(
        hfActiveFilter === 'all'
          ? 'Nothing to show.'
          : 'No models in this category on this page. Try All, or search for something specific.',
      ),
    );
    return;
  }

  const order = new Map(HF_CATEGORIES.map((c, i) => [c.id, i]));
  const grouped = new Map(HF_CATEGORIES.map((c) => [c.id, []]));
  for (const hit of hits) grouped.get(categorizeHit(hit)).push(hit);
  // Deliberately unsorted. Re-ranking by downloads inside each group put the
  // same handful of enormous repos at the top of every category, which is the
  // behaviour the sort control was removed to get rid of.

  const cards = [];
  for (const [catId, members] of [...grouped.entries()].sort(
    (a, b) => order.get(a[0]) - order.get(b[0]),
  )) {
    if (!members.length) continue;
    const label = HF_CATEGORIES[order.get(catId)].label;

    const section = document.createElement('section');
    section.className = 'hf-group';
    section.dataset.category = catId;

    const heading = document.createElement('h3');
    heading.className = 'hf-group__title';
    heading.textContent = label;
    const count = document.createElement('span');
    count.className = 'hf-group__count';
    count.textContent = String(members.length);
    heading.append(count);
    section.append(heading);

    const grid = document.createElement('div');
    grid.className = 'hf-grid';
    for (const hit of members) {
      const card = renderHfHit(hit);
      cards.push({ hit, card });
      grid.append(card);
    }
    section.append(grid);
    hfResults.append(section);
  }

  // Size and runtime come from one detail call per card. They are fetched
  // lazily, as each card scrolls into view: a page of 29 asks the Hub for the
  // handful actually on screen, instead of 29 requests for rows nobody can
  // read — which is both faster and far less likely to earn a rate limit.
  observeCardsForEnrichment(cards);
}

/** Starts enrichment as cards become visible, and never twice per card. */
let hfCardObserver = null;
function observeCardsForEnrichment(cards) {
  if (typeof IntersectionObserver !== 'function') {
    void Promise.all(cards.map(({ hit, card }) => enrichCard(hit, card)));
    return;
  }
  hfCardObserver?.disconnect();
  hfCardObserver = new IntersectionObserver(
    (entries, obs) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const card = entry.target;
        obs.unobserve(card);
        const hit = card._hfHit;
        if (hit) void enrichCard(hit, card);
      }
    },
    // Start a little before the card is on screen so it is usually resolved by
    // the time it is actually read.
    { rootMargin: '300px 0px' },
  );
  for (const { hit, card } of cards) {
    card._hfHit = hit;
    hfCardObserver.observe(card);
  }
}

/** Detail responses, keyed by repo. Filter clicks re-render every card, and
 *  without this the page re-asks the Hub about the same models each time —
 *  which is what earns a rate limit in the first place. */
const hfDetailCache = new Map();
/** The Hub throttles bursts hard, so detail calls go out a few at a time
 *  rather than all at once. */
let hfDetailInFlight = 0;
const HF_DETAIL_CONCURRENCY = 3;
const hfDetailQueue = [];

function hfDetailSlot(fn) {
  return new Promise((resolve) => {
    hfDetailQueue.push(async () => {
      hfDetailInFlight++;
      try {
        resolve(await fn());
      } finally {
        hfDetailInFlight--;
        // Pump on completion as well as on enqueue. Without this the first
        // batch drains and everything queued behind it waits forever, which
        // left nine of twelve cards stuck on "checking…".
        hfDetailPump();
      }
    });
    hfDetailPump();
  });
}

/** Starts queued work while under the concurrency limit. */
function hfDetailPump() {
  while (hfDetailInFlight < HF_DETAIL_CONCURRENCY && hfDetailQueue.length) {
    // The queued task's own promise is never awaited by anyone: its result is
    // delivered through the caller's promise instead. Without this catch a
    // failed lookup (a Hub 429, say) also surfaces as an unhandled rejection,
    // even though enrichCard already handled it properly.
    hfDetailQueue.shift()().catch(() => {});
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Fetched model detail, cached, throttled, with one retry after a 429. */
async function hfModelDetail(repo) {
  if (hfDetailCache.has(repo)) return hfDetailCache.get(repo);
  const work = hfDetailSlot(async () => {
    const url = `/api/hf/model?repo=${encodeURIComponent(repo)}`;
    for (let attempt = 0; attempt < 2; attempt++) {
      const res = await fetch(url);
      if (res.status === 429) {
        if (attempt === 1) throw new Error('Rate limited by the Hub — try again shortly.');
        await sleep(2500);
        continue;
      }
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      hfDetailCache.set(repo, data.model);
      return data.model;
    }
    throw new Error('Rate limited by the Hub — try again shortly.');
  });
  // Don't cache rejections: a rate limit is temporary and shouldn't stick.
  return work;
}

/** Fills one card's size and runtime chips from the detail call. */
async function enrichCard(hit, card) {
  const sizeEl = card.querySelector('.hf-card__size');
  if (!sizeEl) return;
  try {
    const model = await hfModelDetail(hit.id);
    // The card may have been re-rendered out from under this promise.
    if (!card.isConnected) return;
    const bytes = hfRepoBytes(model);
    const nFiles = model.files.length;
    const gguf = model.files.filter((f) => f.name.endsWith('.gguf')).length;
    sizeEl.textContent = bytes > 0 ? fmtSize(bytes) : 'size unknown';
    sizeEl.title = `${nFiles} file${nFiles === 1 ? '' : 's'}${gguf ? ` (${gguf} GGUF)` : ''}`;
    paintRuntimeChips(card, runtimesForFiles(model.files));
  } catch (err) {
    if (!card.isConnected) return;
    const message = err instanceof Error ? err.message : String(err);
    // Say *why* it's blank. A bare '?' reads like a broken card.
    const limited = /rate limit/i.test(message);
    sizeEl.textContent = limited ? 'throttled' : '?';
    sizeEl.title = message;
    paintRuntimeChips(card, [], limited ? 'Hub rate limit' : null);
  }
}

for (const chip of document.querySelectorAll('.hf-filter')) {
  chip.addEventListener('click', async () => {
    const task = chip.dataset.task;
    hfActiveFilter = task;
    for (const c of document.querySelectorAll('.hf-filter')) {
      const on = c === chip;
      c.classList.toggle('is-active', on);
      c.setAttribute('aria-pressed', String(on));
    }
    renderHfFiltered();

    // A quiet category on the current page isn't the same as an empty
    // category on the Hub, so ask for that pipeline before giving up.
    const haveSome = hfLastHits.some((h) => categorizeHit(h) === task);
    if (task === 'all' || haveSome) return;
    hfResults.append(hfNote(`Fetching ${chip.textContent.trim().toLowerCase()} models…`));
    const pipeline = HF_PIPELINE[task];
    if (!pipeline) return;
    try {
      const res = await fetch(
        `/api/hf/search?q=&pipeline=${encodeURIComponent(pipeline)}&sort=${encodeURIComponent(hfSort.value)}&limit=12`,
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      // Merge rather than replace, so All still shows everything afterwards.
      const seen = new Set(hfLastHits.map((h) => h.id));
      for (const m of data.models ?? []) {
        if (!seen.has(m.id)) {
          seen.add(m.id);
          hfLastHits.push(m);
        }
      }
      renderHfFiltered();
    } catch (err) {
      const note = hfNote(err instanceof Error ? err.message : String(err));
      note.classList.add('scan-note', 'warn');
      hfResults.append(note);
    }
  });
}

/** Runs a Hub search, or trending for an empty query, into the results list. */
async function runHfSearch(q) {
  // The browse panel loads trending models the moment it is first shown, and
  // a search typed in the next second used to be dropped on the floor with no
  // feedback at all. Remember the latest query and run it as soon as the
  // in-flight one finishes: what the user asked for last is what they want.
  if (hfSearching) {
    hfPendingQuery = q;
    return;
  }
  hfSearching = true;
  hfResults.innerHTML = '';
  hfResults.append(hfNote(q ? `Searching the Hub for "${q}"…` : 'Loading models from the Hub…'));
  try {
    const res = await fetch(
      `/api/hf/search?q=${encodeURIComponent(q)}&limit=48`,
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
    hfResults.innerHTML = '';
    if (!data.models?.length) {
      hfResults.append(hfNote(q ? `No models found for "${q}".` : 'The Hub returned nothing.'));
      return;
    }

    hfLastHits = data.models;
    renderHfFiltered();
  } catch (err) {
    hfResults.innerHTML = '';
    hfResults.append(hfNote(err instanceof Error ? err.message : String(err)));
  } finally {
    // Always released — an early `return` above used to leave this stuck true
    // and silently killed every later search.
    hfSearching = false;
    // A query that arrived while this one was in flight runs now rather than
    // being discarded without a word.
    if (hfPendingQuery !== null) {
      const next = hfPendingQuery;
      hfPendingQuery = null;
      void runHfSearch(next);
    }
  }
}

hfForm.addEventListener('submit', (e) => {
  e.preventDefault();
  runHfSearch(hfQ.value.trim());
});

/**
 * First visit to Browse. Trending on its own is almost entirely chat and
 * embedding models, which made the Image / Speech / TTS chips look empty. So
 * this seeds each quiet category with the Hub's top few for that pipeline and
 * merges them in; one-time cost, and the response cache makes filter clicks
 * free afterwards.
 */
async function loadHfTrending() {
  if (hfResults.firstChild) return;
  hfSearching = true;
  hfResults.innerHTML = '';
  hfResults.append(hfNote('Loading models across every task…'));

  // Every category, asked for at once, with no ranking applied afterwards.
  // Sorting by downloads was what made a handful of enormous chat repos fill the
  // page while image and speech models never appeared at all.
  const seeds = [
    { pipeline: undefined, limit: 48 },
    { pipeline: 'text-to-image', limit: 24 },
    { pipeline: 'image-to-image', limit: 12 },
    { pipeline: 'automatic-speech-recognition', limit: 18 },
    { pipeline: 'text-to-speech', limit: 18 },
    { pipeline: 'text-to-audio', limit: 12 },
    { pipeline: 'audio-to-audio', limit: 8 },
    { pipeline: 'fill-mask', limit: 10 },
    { pipeline: 'feature-extraction', limit: 10 },
  ];
  const seen = new Set();
  const merged = [];
  const fetched = await Promise.all(
    seeds.map(async (seed) => {
      const url =
        `/api/hf/search?q=&limit=${seed.limit}` +
        (seed.pipeline ? `&pipeline=${encodeURIComponent(seed.pipeline)}` : '');
      try {
        const res = await fetch(url);
        const data = await res.json().catch(() => ({}));
        return res.ok ? (data.models ?? []) : [];
      } catch {
        // One failed category must not empty the page.
        return [];
      }
    }),
  );
  for (const models of fetched) {
    for (const m of models) {
      if (seen.has(m.id)) continue;
      seen.add(m.id);
      merged.push(m);
    }
  }
  hfSearching = false;
  if (!merged.length) {
    hfResults.innerHTML = '';
    hfResults.append(hfNote('Could not reach the Hub. Check the network and try again.'));
    return;
  }
  hfLastHits = merged;
  renderHfFiltered();
}

/** Short uppercase extension chip text (.gguf → GGUF), or null for names
 *  without a usable extension. Capped so freak suffixes can't blow up the row;
 *  11 covers the longest real one (SAFETENSORS). */
function hfFileExt(name) {
  const base = name.split('/').pop() || '';
  const i = base.lastIndexOf('.');
  if (i <= 0) return null;
  const ext = base.slice(i + 1).toUpperCase();
  return ext.length > 0 && ext.length <= 11 ? ext : null;
}

async function showHfDetail(repo) {
  hfModalFiles.innerHTML = '';
  hfModalTile.replaceChildren(modelTile(repo, 'lg'));
  hfModalTitle.textContent = repo;
  hfModalStats.textContent = 'Loading model…';
  openHfModal();
  let detail;
  try {
    const res = await fetch(`/api/hf/model?repo=${encodeURIComponent(repo)}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
    detail = data.model;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    hfModalStats.textContent = message;
    hfModalFiles.append(hfNote(`Could not read this model: ${message}`));
    const summary = document.getElementById('hf-repo-summary');
    const btn = document.getElementById('hf-repo-dl');
    if (summary) summary.textContent = message;
    if (btn) btn.disabled = true;
    return;
  }

  const bytes = hfRepoBytes(detail);
  hfModalStats.textContent =
    `♥ ${(detail.likes ?? 0).toLocaleString()} · ` +
    `⬇ ${(detail.downloads ?? 0).toLocaleString()} · ` +
    (bytes > 0 ? fmtSize(bytes) : 'size unknown') +
    (detail.gated ? ' · gated' : '');

  if (!detail.files.length) {
    // "No files" and "the fetch failed" are different facts. A failed request
    // used to fall through to the empty-repo wording, which is a lie that sends
    // you looking for a model that has plenty of files.
    hfModalFiles.append(
      hfNote(
        detail.error
          ? `Could not read this model: ${detail.error}`
          : 'This repo lists no downloadable files.',
      ),
    );
    return;
  }
  for (const f of detail.files) {
    hfModalFiles.append(renderHfFile(detail.id, f));
  }

  // Offer the whole model as one action. The per-file rows stay, because
  // grabbing a single variant is legitimate — they are just no longer the only
  // way to get something that actually loads.
  void prepareRepoDownload(detail.id);
}

/**
 * Works out what the repository needs as a set, and enables "Download model".
 *
 * The filter is the server's, not the browser's: it knows that a diffusers repo
 * ships a single-file checkpoint nothing here opens, and that a CTranslate2
 * model is worthless without its tokenizer.
 */
async function prepareRepoDownload(repo) {
  const summary = document.getElementById('hf-repo-summary');
  const btn = document.getElementById('hf-repo-dl');
  if (!summary || !btn) return;
  btn.disabled = true;
  summary.textContent = 'Working out which files this model needs…';
  try {
    const res = await fetch(`/api/hf/repo-files?repo=${encodeURIComponent(repo)}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
    const count = data.count ?? 0;
    if (!count) {
      summary.textContent = 'This repository publishes no usable files.';
      btn.disabled = true;
      return;
    }
    const missing = data.missing ?? count;
    const have = data.present ?? 0;
    if (missing === 0) {
      // Everything is already on disk. Offering an enabled button here meant a
      // click that produced no visible response at all.
      summary.textContent =
        `Already downloaded — all ${count} file${count === 1 ? '' : 's'} ` +
        `(${fmtSize(data.total ?? 0)}) are here.`;
      btn.textContent = 'Downloaded';
      btn.dataset.restLabel = 'Download model';
      btn.disabled = true;
      return;
    }
    if (have > 0) {
      summary.textContent =
        `${missing} of ${count} file${count === 1 ? '' : 's'} still missing · ` +
        `${fmtSize(data.missingBytes ?? 0)} to go. The other ${have} ` +
        'are already downloaded and will be left alone.';
      btn.textContent = missing === count ? 'Download model' : 'Download the rest';
    } else {
      summary.textContent =
        `${count} file${count === 1 ? '' : 's'} · ${fmtSize(data.total ?? 0)} — ` +
        'weights, configs and tokenizers together';
      btn.textContent = 'Download model';
    }
    btn.dataset.restLabel = btn.textContent;
    btn.disabled = false;
  } catch (err) {
    summary.textContent = err instanceof Error ? err.message : String(err);
    btn.disabled = true;
  }
}

/** Fetches every file of a model, reporting each as it lands. */
async function downloadWholeRepo(repo, btn, summaryEl) {
  // The resting label was set by prepareRepoDownload ("Download model" or
  // "Download the rest"); progress overwrites the text, so it is stashed there.
  const original = btn.dataset.restLabel || 'Download model';
  btn.disabled = true;
  const res = await fetch('/api/hf/download-repo', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ repo }),
  });
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({}));
    notifyError(err.error ?? `HTTP ${res.status}`);
    btn.textContent = original;
    btn.disabled = false;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let failed = [];
  let total = 0;
  let saved = 0;
  let sawDone = false;
  // Byte accounting for the whole model, not just the file currently moving.
  // `needBytes` is what is left to fetch; `finishedBytes` is what has landed;
  // `curBytes` is the in-flight file. The file count is kept too, because
  // "3 of 4 files" answers a different question than "2.1 GB of 2.6 GB".
  let needBytes = 0;
  let finishedBytes = 0;
  let curBytes = 0;
  let curPath = '';
  let sized = false;

  /**
   * The one place the button text is set.
   *
   * Bytes and percentage when the Hub published sizes, because that is what
   * "how far along am I" actually means for a multi-gigabyte fetch. Without a
   * trustworthy denominator it falls back to the file count rather than showing
   * a number built on a zero.
   */
  function paintRepoProgress() {
    const files = `${saved} / ${total} file${total === 1 ? '' : 's'}`;
    if (!sized || !needBytes) {
      btn.textContent = `${saved} / ${total}`;
      btn.title = files;
      return;
    }
    const got = finishedBytes + curBytes;
    const pct = Math.min(100, (got / needBytes) * 100);
    btn.textContent = `${pct.toFixed(0)}% · ${fmtSize(got)} / ${fmtSize(needBytes)}`;
    btn.title = `${files} · ${fmtSize(got)} of ${fmtSize(needBytes)}`;
  }

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
      if (event === 'manifest') {
        // Counted here rather than read off each event: `file-done` carries the
        // path and byte count, not a position, so asking it for one produced
        // "NaN / 4" and then a counter frozen on "0 / 4".
        total = parsed.count ?? 0;
        saved = 0;
        needBytes = parsed.bytes ?? 0;
        sized = parsed.sized === true;
        finishedBytes = 0;
        curBytes = 0;
        curPath = '';
        paintRepoProgress();
        if (parsed.skipped) {
          notifySuccess(`${parsed.skipped} file${parsed.skipped === 1 ? '' : 's'} already on disk — fetching the rest.`);
        }
      } else if (event === 'progress') {
        // Bytes, which is what a download is actually made of. `file-done`
        // only arrives once a file lands, so with a single 2 GB weight the row
        // would sit at "0 / 1" for the whole fetch and then jump.
        if (parsed.path !== curPath) {
          curPath = parsed.path ?? '';
          curBytes = 0;
        }
        if (typeof parsed.received === 'number') curBytes = parsed.received;
        paintRepoProgress();
      } else if (event === 'file-done') {
        saved += 1;
        // Count this file's bytes once, at the end, rather than adding the
        // running total again on every progress tick of a file already counted.
        finishedBytes += parsed.bytes ?? curBytes;
        curBytes = 0;
        curPath = '';
        paintRepoProgress();
      } else if (event === 'file-failed') {
        failed.push(parsed.path);
      } else if (event === 'error') {
        notifyError(parsed.message);
      } else if (event === 'done') {
        sawDone = true;
        btn.dataset.restLabel = original;
        btn.textContent = original;
        btn.disabled = false;
        if (failed.length) {
          notifyWarning(
            `${parsed.files} of ${parsed.files + failed.length} files saved. ` +
            `Failed: ${failed.join(', ')}`,
          );
        } else if (parsed.alreadyPresent && !parsed.files) {
          notifySuccess(`${repo} is already complete under ./models/`);
        } else {
          notifySuccess(`${repo} saved under ./models/`);
        }
        if (summaryEl) summaryEl.textContent = 'Saved. Assign it to a job in Your models.';
        // Every derived view, not just this one page's: a finished download
        // changes what the hub groups and what "What each job uses" can offer.
        await syncDerived();
      }
    }
  }
  // Reaching the end of the stream is not the same as finishing. If the
  // connection dropped before a `done` arrived the server is still fetching,
  // so the button must not claim it is ready to go again.
  if (!sawDone) {
    btn.dataset.restLabel = original;
    btn.textContent = 'Check status';
    btn.disabled = true;
    summaryEl && (summaryEl.textContent =
      'Lost the connection to the server. The download may still be running — ' +
      'check the Downloads page.');
    await syncTransfersSoon();
    return;
  }
  btn.textContent = original;
  btn.disabled = false;
}

function renderHfFile(repo, f) {
  const row = document.createElement('div');
  row.className = 'row hf-file-row';

  const top = document.createElement('div');
  top.className = 'row-top';

  const name = document.createElement('span');
  name.className = 'model-name hf-file-name';
  name.textContent = f.name;
  name.title = f.name;

  const right = document.createElement('span');
  right.className = 'hf-file-right';
  const size = document.createElement('span');
  size.className = 'row-note';
  size.textContent = typeof f.size === 'number' ? fmtSize(f.size) : 'size unknown';
  const dl = document.createElement('button');
  dl.type = 'button';
  dl.className = 'btn-ghost btn-dl-file';
  dl.textContent = 'Download';
  dl.addEventListener('click', () => downloadHfFile(repo, f.name, dl, size));
  right.append(size, dl);
  const ext = hfFileExt(f.name);
  if (ext) {
    const b = document.createElement('span');
    b.className = `badge${ext === 'GGUF' ? ' prov' : ''}`;
    b.textContent = ext;
    right.prepend(b);
  }

  top.append(name, right);
  row.append(top);
  return row;
}

async function downloadHfFile(repo, file, btn, sizeEl, hooks = {}) {
  if (hfDownloading) {
    notifyWarning('Already downloading a file — wait a moment');
    return;
  }
  hfDownloading = true;
  btn.disabled = true;
  // `sizeEl` is the optional inline byte counter. When it is absent — which is
  // the catalog's whole-model download — the button itself is the readout, or
  // the bytes and percentage would be computed and thrown away and the row
  // would sit on "Downloading…" for the whole fetch.
  const prev = sizeEl ? sizeEl.textContent : '';
  const restore = (t) => {
    if (sizeEl) sizeEl.textContent = t;
    else if (btn && !hooks.onTick) btn.textContent = t;
  };
  const transfer = beginTransfer('download', file, { repo, file });
  try {
    const res = await fetch('/api/hf/download', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ repo, file }),
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error ?? `HTTP ${res.status}`);
    }
    // Not a stream: this exact file is already transferring server-side (a
    // second tab, or a click after a reload). The work is already running, so
    // follow it through the registry instead of starting a duplicate.
    const ctype = res.headers.get('content-type') ?? '';
    if (ctype.includes('application/json')) {
      const data = await res.json();
      if (data.attached) {
        transfer.update({ state: 'running', percent: 0, received: 0, total: null });
        void syncTransfersSoon();
        btn.disabled = false;
        restore(prev);
        return;
      }
      throw new Error(data.error ?? 'Download refused');
    }
    // Same manual SSE shape as the Ollama pull above.
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
        if (event === 'progress') {
          const pct =
            typeof parsed.total === 'number' && parsed.total > 0 && typeof parsed.received === 'number'
              ? Math.min(100, (parsed.received / parsed.total) * 100)
              : null;
          transfer.update({
            received: typeof parsed.received === 'number' ? parsed.received : 0,
            total: typeof parsed.total === 'number' && parsed.total > 0 ? parsed.total : null,
            percent: pct ?? 0,
          });
          void syncTransfersSoon();
          // A caller aggregating several files into one number (the catalog's
          // "download all") takes over the readout from here.
          if (hooks.onTick) {
            hooks.onTick(
              typeof parsed.received === 'number' ? parsed.received : 0,
              typeof parsed.total === 'number' && parsed.total > 0 ? parsed.total : null,
              file,
            );
          } else if (pct !== null) {
            restore(`${fmtSize(parsed.received)} / ${fmtSize(parsed.total)} (${pct.toFixed(1)}%)`);
          } else if (typeof parsed.received === 'number') {
            restore(fmtSize(parsed.received));
          }
        } else if (event === 'error') {
          throw new Error(parsed.message);
        } else if (event === 'done') {
          transfer.finish('done');
          restore(prev);
          notifySuccess(`Saved ${file}`);
          await refreshHfFiles();
          return;
        }
      }
    }
  } catch (err) {
    transfer.finish('error');
    transfer.error = err instanceof Error ? err.message : String(err);
    restore(prev);
    notifyError(err instanceof Error ? err.message : String(err));
  } finally {
    hfDownloading = false;
    btn.disabled = false;
  }
}

function renderHfDownload(d) {
  const row = document.createElement('div');
  row.className = 'row hf-file-row';
  const top = document.createElement('div');
  top.className = 'row-top';
  const name = document.createElement('span');
  name.className = 'model-name hf-file-name';
  // A bundle is one model. Naming the weights and counting the rest keeps the
  // row readable while still saying exactly what is on disk.
  const isBundle = Array.isArray(d.files) && d.files.length > 1;
  name.textContent = isBundle
    ? `${d.repo} · ${d.file} (+${d.files.length - 1} more file${d.files.length === 2 ? '' : 's'})`
    : `${d.repo} / ${d.file}`;
  name.title = `${d.repo}\n${(d.files ?? [d.file]).join('\n')}`;
  top.append(modelTile(d.repo));
  const right = document.createElement('span');
  right.className = 'hf-file-right';
  const size = document.createElement('span');
  size.className = 'row-note';
  size.textContent = fmtSize(d.size);
  const del = document.createElement('button');
  del.type = 'button';
  del.className = 'btn-ghost btn-del-file';
  del.textContent = 'Delete';
  del.addEventListener('click', async () => {
    if (!window.confirm(`Delete ${d.file} from this machine?`)) return;
    del.disabled = true;
    try {
      const res = await fetch('/api/hf/files/delete', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ repo: d.repo, file: d.file }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      await refreshHfFiles();
    } catch (err) {
      del.disabled = false;
      notifyError(err instanceof Error ? err.message : String(err));
    }
  });
  right.append(size, del);
  top.append(name, right);
  row.append(top);
  return row;
}


/* ---------------- Downloaded weights, with Run where possible ---------------- */

const localDownloadsEl = document.getElementById('local-downloads');

/** One downloaded file: repo, size, and a Run button when it can actually run. */
/**
 * @param resolved the runtime this row was filed under, so the row and its
 *   group header can never disagree. Recomputing from the extension here is
 *   what previously let a `.onnx` file sit in a "PyTorch" group while its own
 *   line claimed ONNX Runtime.
 */
/**
 * Deletes waiting for their second click, keyed by repo and file.
 *
 * This lives outside the DOM on purpose. A background refresh (a finishing
 * download, the transfer ticker, a tab re-entry) rebuilds these rows, and an
 * armed button stored on the element went with it: you click Delete, the page
 * repaints, and the second click armed it again instead of deleting — which is
 * exactly "the delete button doesn't work".
 */
const armedDeletes = new Set();

function deleteArmKey(repo, file) {
  return `${repo}::${file ?? ''}`;
}

/** Wraps a destructive button in the two-step confirm, whatever re-renders around it. */
function armThenDelete(btn, key, run, label = 'Delete') {
  const paint = () => {
    btn.textContent = armedDeletes.has(key) ? 'Sure?' : label;
    btn.dataset.armed = armedDeletes.has(key) ? '1' : '';
  };
  paint();
  btn.addEventListener('click', async () => {
    if (!armedDeletes.has(key)) {
      armedDeletes.add(key);
      paint();
      setTimeout(() => {
        armedDeletes.delete(key);
        if (btn.isConnected) paint();
      }, 4000);
      return;
    }
    armedDeletes.delete(key);
    btn.disabled = true;
    paint();
    try {
      await run();
    } finally {
      if (btn.isConnected) {
        btn.disabled = false;
        paint();
      }
    }
  });
}

function renderLocalDownload(d, resolved) {
  const row = document.createElement('div');
  row.className = 'row local-download-row';

  const top = document.createElement('div');
  top.className = 'row-top';

  const name = document.createElement('span');
  name.className = 'model-name';
  // A bundle is one model. Naming the weights and counting the rest keeps the
  // row readable while still saying exactly what is on disk.
  const members = Array.isArray(d.files) && d.files.length ? d.files : [d.file];
  name.textContent =
    members.length > 1
      ? `${d.file}  (+${members.length - 1} more file${members.length === 2 ? '' : 's'})`
      : d.file;
  name.title = `${d.repo}\n${members.join('\n')}`;
  top.append(name);

  const size = document.createElement('span');
  size.className = 'row-note';
  size.textContent = fmtSize(d.size);
  top.append(size);
  row.append(top);

  const runtime = runtimeForRow(d, resolved);

  // One line carries repo, format, runtime and state. This used to stack five
  // blocks with two rules between them, which made a folder of models read as a
  // wall rather than a list.
  const meta = document.createElement('div');
  meta.className = 'local-meta';

  const repoLine = document.createElement('span');
  repoLine.className = 'local-meta__repo';
  repoLine.textContent = d.repo;
  meta.append(repoLine);

  const chip = document.createElement('span');
  chip.className = 'fmt-chip';
  chip.textContent = formatChip(d.file);
  chip.title = d.file;
  meta.append(chip);

  if (d.location === 'cache') {
    // Fetched by another tool (huggingface-cli, a Python stack) into the
    // shared cache. It is a real download and can run from there; the badge
    // names where it lives, and Remove (title row) takes the whole repo out
    // of that shared cache.
    const badge = document.createElement('span');
    badge.className = 'local-meta__state';
    badge.textContent = 'HF cache';
    badge.title =
      'Downloaded outside the app, into the shared Hugging Face cache (~/.cache/huggingface), which other tools also read.';
    meta.append(badge);
  }

  if (d.missing?.length) {
    // Half a model is worse than none: it is present, it shows up, and it
    // fails inside the runtime loader with a message about a missing file.
    const warn = document.createElement('button');
    warn.type = 'button';
    warn.className = 'local-meta__state is-bad is-action';
    // Naming every missing file would overflow the row; the count plus the
    // hover list is enough to recognise the model, and the button carries the
    // exact URLs for the actual download.
    const n = d.missing.length;
    warn.textContent = `incomplete — missing ${n} file${n === 1 ? '' : 's'}`;
    warn.title = `${d.missing.join('\n')}\n\nClick to copy the download link${n === 1 ? '' : 's'} for these.`;
    warn.addEventListener('click', async () => {
      const links = d.missing.map((f) => hfFileUrl(d.repo, f)).join('\n');
      const ok = await copyToClipboard(links);
      if (ok) {
        notifySuccess(
          n === 1 ? 'Download link copied' : `${n} download links copied`,
        );
        warn.textContent = 'link copied';
        setTimeout(() => {
          warn.textContent = `incomplete — missing ${n} file${n === 1 ? '' : 's'}`;
        }, 1400);
      } else {
        notifyError('Could not copy to the clipboard');
      }
    });
    meta.append(warn);
  } else if (runtime) {
    const name = document.createElement('span');
    name.className = 'local-meta__runtime';
    name.textContent = runtime.label;
    name.title = runtime.blurb;
    meta.append(name);

    if (d.jobless) {
      // No job in the app loads this repo, so runtime states — "not
      // installed", install buttons — would read as steps toward using it.
      // They are noise for a row no picker offers; the tag is the truth, and
      // the runtime label above already says what format it is.
      const noJob = document.createElement('span');
      noJob.className = 'local-meta__state';
      noJob.textContent = 'no job uses this';
      noJob.title =
        "None of this app's jobs — speech to text, speech, image generation — loads this model. It is listed because it is on disk; use it with its own tooling.";
      meta.append(noJob);
    } else if (runtime.external) {
      // Nothing here installs llama.cpp; saying so is more useful than a button
      // that would do nothing.
      const note = document.createElement('span');
      note.className = 'local-meta__state';
      note.textContent = 'external tool';
      note.title = runtime.blurb;
      meta.append(note);
    } else {
      const ready = runtimeReady(runtime, catalogInstalled);
      // Same rule as the group header: no tag when the runtime is here. The
      // tag explains a gap; it is not decoration on every row.
      if (!ready) {
        const state = document.createElement('span');
        state.className = 'local-meta__state';
        state.textContent = 'not installed';
        meta.append(state);
      }

      const missing = runtime.requires.filter((g) => !catalogInstalled[g]);
      if (!ready && missing.length) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn-primary btn-tiny';
        btn.textContent = `Install ${missing.join(' + ')}`;
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          for (const g of missing) rtPicked.add(g);
          setSubview('runtimes');
          loadRuntimes();
        });
        // If the runtime cannot be installed here, do not advertise a button
        // that ends in a pip resolver error.
        if (typeof runtimeBlocked === 'function' && runtimeBlocked(missing)) {
          btn.disabled = true;
          btn.title = 'This runtime cannot be installed on this machine.';
          btn.replaceChildren(document.createTextNode('Unavailable here'));
        }
        meta.append(btn);
      }
    }
  } else {
    const note = document.createElement('span');
    note.className = 'local-meta__state';
    note.textContent = 'no local runtime can run this format';
    meta.append(note);
  }
  row.append(meta);

  // A downloaded GGUF chat model is one click from being used: the chat page's
  // llama.cpp provider runs it, and the server already worked out the id that
  // names it. Offered only once the runtime is installed — a button that lands
  // on a provider that cannot start is worse than the Install button beside it.
  const chatMarker = d.llama || d.transformers;
  const chatRuntime = d.llama ? LLAMA_CHAT_RUNTIME : (d.transformers ? TRANSFORMERS_CHAT_RUNTIME : null);
  const chatLink = chatMarker && chatRuntime && runtimeReady(chatRuntime, catalogInstalled) ? chatMarker : null;
  if (chatLink) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn-ghost btn-tiny btn-chat-model';
    btn.textContent = 'Chat';
    btn.title = `Start a conversation with ${chatLink.model}, run by llama.cpp on this machine`;
    // In place, not a new tab: "Chat" means go use this model now.
    btn.addEventListener('click', () => {
      window.location.href = `/?provider=${encodeURIComponent(chatLink.provider)}&model=${encodeURIComponent(chatLink.model)}`;
    });
    top.append(btn);
  }

  // Find the catalog entry for this repo, so the Run affordance and its gating
  // are the same rules the catalog uses — not a second guess written twice.
  const entry = catalogEntryForRepo(d.repo);
  const file = entry?.files?.find((f) => f.name === d.file);

  // Both actions take two clicks, so a stray one can't remove gigabytes of
  // weights. Delete removes the app's own files; Remove takes a whole repo
  // out of the shared Hugging Face cache — that directory is read by other
  // tools too, so it is a separate, explicit word rather than a second Delete.
  if (d.location === 'cache') {
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'btn-ghost btn-del-file btn-tiny';
    armThenDelete(rm, `cache::${d.repo}`, async () => {
      try {
        await api('/api/hf/cache/delete', {
          method: 'POST',
          body: JSON.stringify({ repo: d.repo }),
        });
        notifySuccess(`${d.repo} removed from the Hugging Face cache`);
        await syncDerived();
      } catch (err) {
        notifyError(err instanceof Error ? err.message : String(err));
      }
    }, 'Remove');
    top.append(rm);
  } else {
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'btn-ghost btn-del-file';
    armThenDelete(del, deleteArmKey(d.repo, null), async () => {
      try {
        // A row is one model: delete removes every file the row shows, not just
        // its anchor. Removing the anchor alone left sidecars behind as rows
        // with nothing left to run — the junk this page was accused of. One
        // request for the whole set: a fifty-eight-file model used to be as
        // many round trips, which read as a hang rather than a delete.
        const res = await api('/api/hf/files/delete-many', {
          method: 'POST',
          body: JSON.stringify({ repo: d.repo, files: members }),
        });
        const removed = res?.removed ?? 0;
        notifySuccess(removed > 1 ? `${removed} files deleted` : `${d.file} deleted`);
        await syncDerived();
      } catch (err) {
        notifyError(err instanceof Error ? err.message : String(err));
      }
    });
    // Delete rides on the title row rather than a bordered strip of its own;
    // the row is two lines now, and a rule between them doubled its height.
    del.classList.add('btn-tiny');
    top.append(del);
  }
  // There is deliberately no Run button on a downloaded row.
  //
  // It never ran anything from here: it navigated to the catalog's run panel,
  // so pressing it appeared to do nothing on this page and then moved you. And
  // transcription now belongs in the Studio. What a downloaded model is *for*
  // is decided once, under "What each job uses" at the top of this page.
  if (entry?.run) {
    // Not a dead end: say exactly what is missing and offer to fetch it,
    // carrying the needed groups across so the Runtimes tab is pre-ticked.
    // Inline on the meta line — a block of its own doubled the row height for
    // one sentence.
    const missing = entry.run.requires.filter((g) => !catalogInstalled[g]);
    if (missing.length) {
      const go = document.createElement('button');
      go.type = 'button';
      go.className = 'btn-primary btn-tiny';
      go.textContent = `Install ${missing.join(' + ')}`;
      go.addEventListener('click', () => {
        for (const g of missing) rtPicked.add(g);
        setSubview('runtimes');
        loadRuntimes();
      });
      meta.append(go);
    }
  }

  // Loose files collapsed under their repo: keep them hidden behind a toggle
  // so a folder of configs and sample clips is one row, not a wall. Each file
  // can still be removed on its own if only part of the set is wanted.
  if (d.loose && members.length > 1) {
    const details = document.createElement('details');
    details.className = 'local-loose';
    const summary = document.createElement('summary');
    summary.textContent = `${members.length} files in this directory`;
    details.append(summary);
    const ul = document.createElement('ul');
    ul.className = 'local-loose__files';
    for (const file of members) {
      const li = document.createElement('li');
      const fn = document.createElement('span');
      fn.textContent = file;
      li.append(fn);
      const rm = document.createElement('button');
      rm.type = 'button';
      rm.className = 'btn-ghost btn-tiny';
      armThenDelete(rm, deleteArmKey(d.repo, file), async () => {
        try {
          await api('/api/hf/files/delete-many', {
            method: 'POST',
            body: JSON.stringify({ repo: d.repo, files: [file] }),
          });
          notifySuccess(`${file} deleted`);
          await syncDerived();
        } catch (err) {
          notifyError(err instanceof Error ? err.message : String(err));
        }
      });
      li.append(rm);
      ul.append(li);
    }
    details.append(ul);
    row.append(details);
  }

  return row;
}

function catalogEntryForRepo(repo) {
  for (const section of catalogData) {
    for (const entry of section.entries) {
      if (entry.repo === repo) return entry;
    }
  }
  return null;
}

/** Downloads list, plus whatever the runner needs, so Run buttons are current. */
async function refreshLocalDownloads() {
  if (!localDownloadsEl) return;
  localDownloadsEl.innerHTML = '';
  let files = null;
  try {
    const res = await fetch('/api/hf/files');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    ({ files } = await res.json());
  } catch {
    files = null;
  }
  if (!files) {
    localDownloadsEl.append(hfNote('Could not list downloaded files.'));
    return;
  }
  if (!files.length) {
    localDownloadsEl.append(hfNote('Nothing downloaded yet. Grab weights from the Hugging Face tab.'));
    return;
  }
  // Group by repo: a repo usually contributes several related files, and the
  // flattened list reads as clutter.
  const byRepo = new Map();
  for (const f of files) {
    if (!byRepo.has(f.repo)) byRepo.set(f.repo, []);
    byRepo.get(f.repo).push(f);
  }
  for (const [repo, group] of byRepo) {
    const head = document.createElement('p');
    head.className = 'local-repo-head';
    head.textContent = `${repo} · ${group.length} file${group.length === 1 ? '' : 's'}`;
    localDownloadsEl.append(head);
    for (const f of group) localDownloadsEl.append(renderLocalDownload(f));
  }
}

async function refreshHfFiles() {
  if (!hfFiles) return;
  hfFiles.innerHTML = '';
  let files = null;
  try {
    const res = await fetch('/api/hf/files');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    ({ files } = await res.json());
  } catch {
    files = null;
  }
  if (!files) {
    hfFiles.append(hfNote('Could not list downloaded files.'));
    return;
  }
  if (!files.length) {
    hfFiles.append(hfNote('Nothing downloaded yet. Search above and grab a file.'));
    return;
  }
  for (const d of files) hfFiles.append(renderHfDownload(d));
}

await refresh();
await refreshCustom();
await refreshDictation();
// Restore the tab AND the subview. "#models/runtimes" has to land on
// Runtimes, not on Models with "Your models" showing — a reload that drops you
// on the first tab is the other half of "the page never updates".
{
  const { tab, sub } = subviewFromHash();
  await selectTab(tab, false);
  if (tab === 'models' && sub) setSubview(sub, false);
}
