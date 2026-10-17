/* ==========================================================================
   viewer.js — shared image lightbox + downloader.

   Used by the chat page and the Image Studio page. Every image matching
   `SELECTOR` becomes click-to-enlarge, and each gets a small download button
   that reveals on hover (always visible on touch input). The lightbox carries
   its own download action plus previous/next when several images are present.

   A MutationObserver picks up images added after boot — streamed/sent turns on
   the chat page, generated images on Image Studio — so callers only need to
   call initImageViewer() once per page.
   ========================================================================== */

const SELECTOR = '.msg-images img';

/* ---------------- Filename + download ---------------- */

function extFromSrc(src) {
  const data = /^data:image\/([a-z0-9.+-]+)/i.exec(src || '');
  if (data) {
    const sub = data[1].toLowerCase();
    if (sub === 'jpeg') return 'jpg';
    if (sub === 'svg+xml') return 'svg';
    return sub;
  }
  const m = /\.(png|jpe?g|webp|gif|avif|svg)$/i.exec((src || '').split(/[?#]/)[0]);
  return m ? m[1].toLowerCase().replace('jpeg', 'jpg') : 'png';
}

function slug(text) {
  return (text || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64);
}

/** Prefers an explicit name (alt), falling back to a timestamped default. */
function filenameFor(img) {
  const clean = slug(img.dataset.name || img.alt || '');
  const ext = extFromSrc(img.currentSrc || img.src);
  return clean ? `${clean}.${ext}` : `image-${Date.now()}.${ext}`;
}

function triggerDownload(src, name) {
  const a = document.createElement('a');
  a.href = src;
  a.download = name;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* ---------------- Lightbox ---------------- */

const ICON = {
  download:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>',
  close:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  prev:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 5-7 7 7 7"/></svg>',
  next:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 5 7 7-7 7"/></svg>',
};

let lightbox = null;
let items = [];
let index = -1;
let lastFocused = null;

function buildLightbox() {
  const root = document.createElement('div');
  root.className = 'lightbox';
  root.hidden = true;
  root.setAttribute('role', 'dialog');
  root.setAttribute('aria-modal', 'true');
  root.setAttribute('aria-label', 'Image viewer');
  // Static markup only — no user text goes through innerHTML on this page.
  root.innerHTML = `
    <div class="lb-backdrop" data-close></div>
    <figure class="lb-figure">
      <img class="lb-image" alt="" />
      <figcaption class="lb-caption"></figcaption>
    </figure>
    <button type="button" class="icon-btn lb-prev" aria-label="Previous image" title="Previous">${ICON.prev}</button>
    <button type="button" class="icon-btn lb-next" aria-label="Next image" title="Next">${ICON.next}</button>
    <div class="lb-actions">
      <button type="button" class="icon-btn lb-download" aria-label="Download image" title="Download">${ICON.download}</button>
      <button type="button" class="icon-btn lb-close" aria-label="Close viewer" title="Close (Esc)">${ICON.close}</button>
    </div>`;

  const box = {
    root,
    image: root.querySelector('.lb-image'),
    caption: root.querySelector('.lb-caption'),
    prev: root.querySelector('.lb-prev'),
    next: root.querySelector('.lb-next'),
    download: root.querySelector('.lb-download'),
    close: root.querySelector('.lb-close'),
  };

  root.querySelector('[data-close]').addEventListener('click', () => close());
  box.close.addEventListener('click', () => close());
  box.download.addEventListener('click', () => downloadCurrent());
  box.prev.addEventListener('click', () => step(-1));
  box.next.addEventListener('click', () => step(1));

  document.body.appendChild(root);
  return box;
}

function show() {
  const img = items[index];
  if (!img) {
    close();
    return;
  }
  lightbox.image.src = img.currentSrc || img.src;
  lightbox.image.alt = img.alt || '';
  lightbox.caption.textContent = img.alt || '';
  const many = items.length > 1;
  lightbox.prev.hidden = !many;
  lightbox.next.hidden = !many;
}

function step(delta) {
  if (items.length < 2) return;
  index = (index + delta + items.length) % items.length;
  show();
}

function downloadCurrent() {
  const img = items[index];
  if (img) triggerDownload(img.currentSrc || img.src, filenameFor(img));
}

function open(img) {
  if (!lightbox) lightbox = buildLightbox();
  // Snapshot the current set so prev/next stays stable while the box is open.
  items = Array.from(document.querySelectorAll(SELECTOR));
  index = items.indexOf(img);
  if (index < 0) {
    items = [img];
    index = 0;
  }
  lastFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  document.documentElement.classList.add('lb-lock');
  lightbox.root.hidden = false;
  show();
  lightbox.close.focus();
}

function close() {
  if (!lightbox || lightbox.root.hidden) return;
  lightbox.root.hidden = true;
  lightbox.image.removeAttribute('src');
  document.documentElement.classList.remove('lb-lock');
  if (lastFocused && lastFocused.isConnected) lastFocused.focus();
  lastFocused = null;
}

function trapFocus(e) {
  const focusable = Array.from(
    lightbox.root.querySelectorAll('button:not([hidden]), a[href]'),
  );
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}

document.addEventListener('keydown', (e) => {
  if (!lightbox || lightbox.root.hidden) return;
  if (e.key === 'Escape') {
    e.preventDefault();
    close();
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    step(-1);
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    step(1);
  } else if (e.key === 'Tab') {
    trapFocus(e);
  }
});

/* ---------------- Enhancement ---------------- */

function enhance(img) {
  // Idempotent: the observer sees its own wrapping as an added node.
  if (img.closest('.viewer-item')) return;

  const wrap = document.createElement('span');
  wrap.className = 'viewer-item';
  img.replaceWith(wrap);
  wrap.appendChild(img);

  img.setAttribute('tabindex', '0');
  img.setAttribute('role', 'button');
  img.setAttribute('aria-label', `View ${img.alt || 'image'} full size`);
  img.addEventListener('click', () => open(img));
  img.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      open(img);
    }
  });

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'viewer-dl';
  btn.title = 'Download';
  btn.setAttribute('aria-label', `Download ${img.alt || 'image'}`);
  btn.innerHTML = ICON.download;
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    triggerDownload(img.currentSrc || img.src, filenameFor(img));
  });
  wrap.appendChild(btn);
}

function enhanceWithin(root) {
  if (root.nodeType !== 1) return;
  if (root.matches(SELECTOR)) enhance(root);
  root.querySelectorAll(SELECTOR).forEach(enhance);
}

/**
 * Wires the viewer onto a page. `root` is the subtree to watch for new images
 * (the whole body by default); it is also scanned once up front.
 */
export function initImageViewer({ root = document.body } = {}) {
  enhanceWithin(root);
  const observer = new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (node.nodeType === 1) enhanceWithin(node);
      }
    }
  });
  observer.observe(root, { childList: true, subtree: true });
  return { open, close, refresh: () => enhanceWithin(root) };
}
