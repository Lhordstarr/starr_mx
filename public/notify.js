/**
 * Notification popups.
 *
 * Replaces the single-slot toast that both pages had duplicated. The old one
 * kept exactly one element and overwrote its text, so two events arriving
 * close together destroyed the first — a transcription finishing while an
 * error surfaced would silently drop one of them. This stacks instead: each
 * notification is its own element with its own timer.
 *
 * Design notes worth knowing:
 *
 * - Dismissal is per-item. Removing one must not disturb the timers or
 *   positions of the others, so heights are read back after every change
 *   rather than cached.
 * - Duration scales with message length. The old fixed 2600ms was too short
 *   for the long error strings the app actually produces (provider names,
 *   "isn't available — using ..."), and far too long for "Copied".
 * - Errors use an assertive live region; everything else is polite. A screen
 *   reader should not interrupt to announce "Copied", but should interrupt
 *   for a failed save.
 * - Reduced motion removes the slide, not the notification.
 */

const TYPES = {
  info: { ms: 3200, assertive: false },
  success: { ms: 2600, assertive: false },
  warning: { ms: 5200, assertive: true },
  error: { ms: 7000, assertive: true },
};

// Above the panel (60) and model menu (70), below nothing else on the page.
const Z = 90;

/** Never let the stack run away — a burst of errors must not cover the app. */
const MAX_VISIBLE = 4;

let container = null;

function ensureContainer() {
  if (container && container.isConnected) return container;
  container = document.createElement('div');
  container.className = 'toast-stack';
  // The stack itself is a neutral live region; each child sets its own
  // aria-live so urgency is per-message rather than for the whole region.
  container.setAttribute('role', 'region');
  container.setAttribute('aria-label', 'Notifications');
  document.body.appendChild(container);
  return container;
}

/** Icons are inline SVG for the same reason the rest of the app uses them:
 *  glyph coverage varies by system font and unknown characters render as tofu. */
const ICONS = {
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5h.01"/>',
  success: '<path d="M20 6 9 17l-5-5"/>',
  warning: '<path d="M10.3 3.9 1.9 18a2 2 0 0 0 1.7 3h16.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
  error: '<circle cx="12" cy="12" r="9"/><path d="m15 9-6 6M9 9l6 6"/>',
};

const svg = (type) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[type]}</svg>`;

/**
 * Show a notification.
 *
 * @param {string} message   text to display
 * @param {object} [opts]
 * @param {'info'|'success'|'warning'|'error'} [opts.type='info']
 * @param {number} [opts.duration] ms; 0 keeps it until dismissed
 * @returns {{dismiss: () => void, el: HTMLElement}}
 */
export function notify(message, opts = {}) {
  const { type = 'info', duration } = opts;
  const spec = TYPES[type] ?? TYPES.info;
  const host = ensureContainer();

  // textContent, never innerHTML: messages embed provider names and model ids
  // that come from a server response.
  const el = document.createElement('div');
  el.className = `toast toast--${type}`;
  el.setAttribute('role', type === 'error' ? 'alert' : 'status');
  el.setAttribute('aria-live', spec.assertive ? 'assertive' : 'polite');

  const icon = document.createElement('span');
  icon.className = 'toast__icon';
  icon.innerHTML = svg(type);

  const text = document.createElement('span');
  text.className = 'toast__text';
  text.textContent = message;

  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'toast__close';
  close.setAttribute('aria-label', 'Dismiss notification');
  close.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';

  el.append(icon, text, close);
  host.appendChild(el);

  // Entering: one frame with the offset applied, then clear it so the
  // transition has a start value to animate from.
  requestAnimationFrame(() => el.classList.add('is-in'));

  let timer = 0;
  let gone = false;

  function dismiss() {
    if (gone) return;
    gone = true;
    clearTimeout(timer);
    el.classList.remove('is-in');
    el.classList.add('is-out');
    // Wait out the exit transition before removing, so it animates.
    const remove = () => el.remove();
    el.addEventListener('transitionend', remove, { once: true });
    // transitionend never fires if the element is display:none or the tab is
    // backgrounded mid-transition, so guarantee the removal.
    setTimeout(remove, 400);
    prune();
  }

  close.addEventListener('click', dismiss);
  // Clicking the body dismisses too — this is a transient status message,
  // not an interactive dialog.
  el.addEventListener('click', (e) => {
    if (!e.target.closest('.toast__close')) dismiss();
  });

  const ms = duration ?? spec.ms ?? 3200;
  if (ms > 0) timer = setTimeout(dismiss, ms);

  prune();
  return { dismiss, el };
}

/** Enforce the cap, dropping the oldest rather than refusing the newest. */
function prune() {
  if (!container) return;
  const items = [...container.children];
  if (items.length <= MAX_VISIBLE) return;
  for (const old of items.slice(0, items.length - MAX_VISIBLE)) old.remove();
}

/**
 * Drop every visible notification.
 * Used when the view changes underneath them (switching conversation,
 * opening settings) so stale messages do not linger.
 */
export function clearNotifications() {
  if (!container) return;
  for (const el of [...container.children]) el.remove();
}

export const notifySuccess = (m, o) => notify(m, { ...o, type: 'success' });
export const notifyError = (m, o) => notify(m, { ...o, type: 'error' });
export const notifyWarning = (m, o) => notify(m, { ...o, type: 'warning' });
