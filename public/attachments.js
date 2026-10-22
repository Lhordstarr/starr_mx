/**
 * Attachments for a user turn: images, and text files folded into the message.
 *
 * Text files never reach the provider as anything but text — they are appended
 * to the message body inside a fenced block, which every provider already
 * understands. Only images need a structured payload, and only the three
 * non-OpenAI adapters translate them (see src/attachments.ts).
 *
 * Image bytes are deliberately not persisted. A single 5MB image is ~6.7MB of
 * base64, and localStorage caps out around 5MB for the whole origin, so saving
 * them would throw away every conversation on the first image.
 */

/** Mirrors LIMITS in src/attachments.ts; the server re-checks all of these. */
export const LIMITS = {
  imageBytes: 5 * 1024 * 1024,
  imagesPerMessage: 4,
  videoBytes: 12 * 1024 * 1024,
  videosPerMessage: 1,
  totalMediaBytes: 18 * 1024 * 1024,
  textFileBytes: 256 * 1024,
  textFileChars: 60_000,
};

const IMAGE_MIME = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif']);
const VIDEO_MIME = new Set(['video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska']);

const TEXT_EXT = new Set([
  'txt', 'md', 'markdown', 'csv', 'tsv', 'json', 'jsonl', 'ndjson', 'yaml', 'yml',
  'toml', 'xml', 'html', 'htm', 'log', 'ini', 'cfg', 'conf', 'env', 'properties',
  'sh', 'bash', 'zsh', 'fish', 'py', 'rb', 'pl', 'php', 'lua',
  'js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx', 'go', 'rs', 'java', 'kt', 'kts',
  'c', 'h', 'cc', 'cpp', 'hpp', 'cs', 'swift', 'scala', 'dart', 'sql',
  'dockerfile', 'makefile', 'gitignore', 'gradle', 'tf', 'tfvars',
]);

const extOf = (name) => name.split('.').pop()?.toLowerCase() ?? '';
const kb = (bytes) => `${Math.max(1, Math.round(bytes / 1024))}KB`;

export function createAttachments({ els, notifyWarning }) {
  /** Pending attachments: images carry base64, text files carry their content. */
  let items = [];
  /** Model ids the current provider reported as image-capable. */
  let visionModels = [];
  /** Whether the provider publishes capability info at all. */
  let visionKnown = false;
  /** The model Settings assigned to describe images and video, if any. */
  let assignedVision = '';
  /** False when the assignment points at a model that is no longer installed. */
  let assignedVisionReady = false;
  /** Audio is handled by the speech-to-text slot, not by the vision model. */
  let assignedStt = '';
  /** Active model, so the attach hint can be recomputed on change. */
  let activeModel = '';
  /** Active provider, so the capability probe only runs when it changes. */
  let activeProvider = '';
  let visionToken = 0;

  const hasImages = () => items.some((i) => i.kind === 'image');
  const hasVideo = () => items.some((i) => i.kind === 'video');
  const mediaBytes = () =>
    items
      .filter((i) => i.kind === 'image' || i.kind === 'video')
      .reduce((n, i) => n + (i.size ?? 0), 0);

  /* ---------------- Capability probe ---------------- */

  /**
   * Asks the server which models accept images. Probed on provider change only,
   * never per keystroke or per send, because Ollama's probe costs a request per
   * installed model.
   */
  async function probeVision(providerId) {
    const token = ++visionToken;
    try {
      const res = await fetch(`/api/vision?provider=${encodeURIComponent(providerId)}`);
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      // A slower provider must not overwrite a newer provider's answer.
      if (token !== visionToken) return;
      visionModels = Array.isArray(data.visionModels) ? data.visionModels : [];
      visionKnown = Boolean(data.known);
      assignedVision = typeof data.assigned?.vision === 'string' ? data.assigned.vision : '';
      assignedVisionReady = Boolean(data.assigned?.visionReady);
      assignedStt = typeof data.assigned?.stt === 'string' ? data.assigned.stt : '';
    } catch {
      if (token !== visionToken) return;
      // Probing is best-effort: an unreachable probe means "unknown", which the
      // UI shows as unverified rather than blocking the user.
      visionModels = [];
      visionKnown = false;
      assignedVision = '';
      assignedVisionReady = false;
      assignedStt = '';
    }
    refresh();
  }

  /** Whether media reaches a model that can see: assignment first, then native. */
  function visionState() {
    if (assignedVision) return assignedVisionReady ? 'assigned' : 'missing';
    if (!visionKnown) return 'unknown';
    return visionModels.includes(activeModel) ? 'yes' : 'no';
  }

  /* ---------------- Adding files ---------------- */

  async function addFiles(fileList) {
    const files = Array.from(fileList ?? []);
    if (!files.length) return;

    const rejected = [];
    for (const file of files) {
      const name = file.name || 'pasted image';
      const mime = (file.type || '').toLowerCase();

      if (IMAGE_MIME.has(mime)) {
        if (hasImages() && items.filter((i) => i.kind === 'image').length >= LIMITS.imagesPerMessage) {
          rejected.push(`${name} (max ${LIMITS.imagesPerMessage} images)`);
          continue;
        }
        if (file.size > LIMITS.imageBytes) {
          rejected.push(`${name} (over ${kb(LIMITS.imageBytes)})`);
          continue;
        }
        if (mediaBytes() + file.size > LIMITS.totalMediaBytes) {
          rejected.push(`${name} (media total over ${kb(LIMITS.totalMediaBytes)})`);
          continue;
        }
        try {
          const data = await toBase64(file);
          items.push({ kind: 'image', name, mimeType: mime, data, size: file.size, url: `data:${mime};base64,${data}` });
        } catch {
          rejected.push(`${name} (could not read)`);
        }
        continue;
      }

      if (VIDEO_MIME.has(mime)) {
        if (hasVideo() && items.filter((i) => i.kind === 'video').length >= LIMITS.videosPerMessage) {
          rejected.push(`${name} (max ${LIMITS.videosPerMessage} video)`);
          continue;
        }
        if (file.size > LIMITS.videoBytes) {
          rejected.push(`${name} (over ${kb(LIMITS.videoBytes)})`);
          continue;
        }
        if (mediaBytes() + file.size > LIMITS.totalMediaBytes) {
          rejected.push(`${name} (media total over ${kb(LIMITS.totalMediaBytes)})`);
          continue;
        }
        try {
          const data = await toBase64(file);
          items.push({ kind: 'video', name, mimeType: mime, data, size: file.size });
        } catch {
          rejected.push(`${name} (could not read)`);
        }
        continue;
      }

      if (TEXT_EXT.has(extOf(name)) || mime.startsWith('text/')) {
        if (file.size > LIMITS.textFileBytes) {
          rejected.push(`${name} (over ${kb(LIMITS.textFileBytes)})`);
          continue;
        }
        try {
          const raw = await file.text();
          const truncated = raw.length > LIMITS.textFileChars;
          items.push({
            kind: 'text',
            name,
            mimeType: mime || 'text/plain',
            size: file.size,
            text: truncated ? raw.slice(0, LIMITS.textFileChars) : raw,
            truncated,
          });
        } catch {
          rejected.push(`${name} (could not read)`);
        }
        continue;
      }

      rejected.push(`${name} (unsupported type)`);
    }

    if (rejected.length) notifyWarning(`Skipped ${rejected.join(', ')}`);
    refresh();
  }

  function toBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(reader.error ?? new Error('read failed'));
      // Strip the prefix: the wire format wants raw base64, and the data URL is
      // rebuilt where it's needed.
      reader.onload = () => {
        const result = String(reader.result);
        const comma = result.indexOf(',');
        resolve(comma === -1 ? result : result.slice(comma + 1));
      };
      reader.readAsDataURL(file);
    });
  }

  function remove(index) {
    items.splice(index, 1);
    refresh();
  }

  function clear() {
    items = [];
    refresh();
  }

  /* ---------------- Building the payload ---------------- */

  /**
   * Turns the composer text plus pending attachments into a message content
   * value. Returns a plain string when nothing is attached, so text-only turns
   * keep exactly the payload they had before attachments existed.
   */
  function buildContent(text) {
    const body = foldedText(text);
    const images = items.filter((i) => i.kind === 'image');
    const videos = items.filter((i) => i.kind === 'video');
    if (!images.length && !videos.length) return body;

    const parts = [];
    if (body.trim()) parts.push({ type: 'text', text: body });
    // Media goes last; the server appends a trailing text part where a provider
    // requires the turn to end with one.
    for (const img of images) parts.push({ type: 'image', data: img.data, mimeType: img.mimeType });
    for (const video of videos) parts.push({ type: 'video', data: video.data, mimeType: video.mimeType });
    return parts;
  }

  /**
   * The prompt text with any attached text files appended as fenced blocks.
   *
   * Shared by the payload and the rendered bubble, so what the user sees in the
   * transcript is exactly what the model is sent.
   */
  function foldedText(text) {
    const files = items.filter((i) => i.kind === 'text');
    const typed = text ?? '';
    if (!files.length) return typed;

    const blocks = files
      .map((f) => {
        // Fence with four backticks when the content itself contains a fence,
        // otherwise the file would close the block early.
        const longest = (f.text.match(/`{3,}/g) ?? []).reduce((n, m) => Math.max(n, m.length), 0);
        const fence = '`'.repeat(Math.max(3, longest + 1));
        const note = f.truncated ? ` (truncated to ${LIMITS.textFileChars} characters)` : '';
        return `--- ${f.name}${note} ---\n${fence}\n${f.text}\n${fence}`;
      })
      .join('\n\n');
    return typed.trim() ? `${typed}\n\n${blocks}` : blocks;
  }

  /**
   * Replaces image bytes with a short marker before a conversation is written to
   * localStorage. See the note at the top of this file for why.
   */
  function stripForStorage(message) {
    if (typeof message.content === 'string') return message;
    const images = message.content.filter((p) => p.type === 'image').length;
    const videos = message.content.filter((p) => p.type === 'video').length;
    const text = message.content
      .filter((p) => p.type === 'text')
      .map((p) => p.text)
      .join('\n')
      .trim();
    const what = [
      images ? `${images} image${images > 1 ? 's' : ''}` : '',
      videos ? `${videos} video${videos > 1 ? 's' : ''}` : '',
    ].filter(Boolean);
    const note = what.length ? `[${what.join(', ')} attached — not saved]` : '';
    return {
      ...message,
      content: [text, note].filter(Boolean).join('\n\n') || '[attachment]',
    };
  }

  /* ---------------- Rendering ---------------- */

  function refresh() {
    render();
    updateAttachHint();
  }

  function render() {
    const host = els.attachList;
    if (!host) return;
    host.replaceChildren();
    els.attachBox.hidden = items.length === 0;
    if (!items.length) return;

    items.forEach((item, index) => {
      const chip = document.createElement('div');
      chip.className = `attach-chip attach-${item.kind}`;

      if (item.kind === 'image') {
        const img = document.createElement('img');
        img.src = item.url;
        img.alt = item.name;
        img.className = 'attach-thumb';
        chip.appendChild(img);
      } else {
        const icon = document.createElement('span');
        icon.className = 'attach-icon';
        // Not an emoji: the video glyph has no colour of its own, so it follows
        // the same achromatic icon language as the rest of the composer.
        icon.textContent = item.kind === 'video' ? '▶' : '≡';
        chip.appendChild(icon);
      }

      const meta = document.createElement('span');
      meta.className = 'attach-meta';
      const name = document.createElement('span');
      name.className = 'attach-name';
      name.textContent = item.name;
      name.title = item.name;
      const size = document.createElement('span');
      size.className = 'attach-size';
      size.textContent = `${kb(item.size)}${item.truncated ? ' · truncated' : ''}`;
      meta.append(name, size);
      chip.appendChild(meta);

      // Named `removeBtn`, not `remove`: a local `remove` would shadow the
      // module-level remove() below, and the click handler would then invoke the
      // DOM element instead of removing the attachment.
      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'attach-remove';
      removeBtn.setAttribute('aria-label', `Remove ${item.name}`);
      removeBtn.textContent = '×';
      removeBtn.addEventListener('click', () => remove(index));
      chip.appendChild(removeBtn);

      host.appendChild(chip);
    });
  }

  function updateAttachHint() {
    if (!els.attach) return;
    const state = visionState();
    // Text files work on every model, so the button is never fully blocked: a
    // model that takes no images still takes a JSON or a log file. Only the
    // tooltip changes to say so.
    const blocked = false;
    els.attach.disabled = blocked;
    const audio = assignedStt ? ' Audio is transcribed by the Speech to text model.' : '';
    els.attach.title =
      state === 'assigned'
        ? `Attach images, video or text files — ${assignedVision} describes them first.${audio}`
        : state === 'missing'
          ? `The assigned vision model ${assignedVision} is not installed. Assign another in Settings.${audio}`
          : state === 'yes'
            ? `Attach images or text files${assignedStt ? '; audio is transcribed by the assigned speech model' : ''}`
            : state === 'no'
              ? 'This model does not accept images (text files still work)'
              : 'Attach images or text files — this provider does not advertise image support';
    els.attach.classList.toggle('unverified', state === 'unknown');
  }

  /* ---------------- Input wiring ---------------- */

  function init() {
    els.attach.addEventListener('click', () => els.attachInput.click());
    els.attachInput.addEventListener('change', async () => {
      await addFiles(els.attachInput.files);
      // Reset so re-picking the same file fires change again.
      els.attachInput.value = '';
    });

    // Paste: an image on the clipboard arrives as a file in clipboardData.
    els.input.addEventListener('paste', (event) => {
      const files = Array.from(event.clipboardData?.files ?? []);
      if (!files.length) return;
      const usable = files.filter(
        (f) =>
          IMAGE_MIME.has(f.type) ||
          VIDEO_MIME.has(f.type) ||
          TEXT_EXT.has(extOf(f.name)) ||
          f.type.startsWith('text/'),
      );
      if (!usable.length) return;
      // Stop the browser also inserting a filename as text.
      event.preventDefault();
      void addFiles(usable);
    });

    // Drag and drop onto the composer box.
    const box = els.composerBox;
    let dragDepth = 0;
    const over = (on) => box.classList.toggle('drag-over', on);

    box.addEventListener('dragenter', (e) => {
      if (!e.dataTransfer?.types?.includes('Files')) return;
      e.preventDefault();
      dragDepth += 1;
      over(true);
    });
    box.addEventListener('dragover', (e) => {
      if (!e.dataTransfer?.types?.includes('Files')) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    });
    box.addEventListener('dragleave', () => {
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) over(false);
    });
    box.addEventListener('drop', (e) => {
      if (!e.dataTransfer?.files?.length) return;
      e.preventDefault();
      dragDepth = 0;
      over(false);
      void addFiles(e.dataTransfer.files);
    });

    updateAttachHint();
  }

  return {
    init,
    addFiles,
    remove,
    clear,
    buildContent,
    /** The prompt with text files folded in, for rendering the sent turn. */
    foldedText,
    stripForStorage,
    hasImages,
    count: () => items.length,
    /**
     * Records the active provider and model.
     *
     * The capability probe runs only when the provider actually changes — it
     * costs a request per installed model on Ollama, so re-running it on every
     * model pick would be wasteful. Switching models within a provider just
     * re-evaluates the already-fetched answer.
     */
    setContext(providerId, nextModel) {
      activeModel = nextModel ?? '';
      if (providerId && providerId !== activeProvider) {
        activeProvider = providerId;
        void probeVision(providerId);
      } else if (!providerId) {
        activeProvider = '';
        visionModels = [];
        visionKnown = false;
        assignedVision = '';
        assignedVisionReady = false;
        assignedStt = '';
      }
      updateAttachHint();
    },
    /** Re-evaluates the hint without a network call. */
    refreshHint: updateAttachHint,
    items: () => items.slice(),
  };
}
