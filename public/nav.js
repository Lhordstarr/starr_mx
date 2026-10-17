/**
 * Primary navigation, rendered identically on chat, settings and studio.
 *
 * Markup lives here rather than in each page because the three documents are
 * maintained separately: a copy per page drifts within a week, and the drift is
 * invisible until someone notices Settings is missing from the chat rail.
 *
 * The collapse is CSS's job. A hidden checkbox holds the state, `:has()` reads
 * it, and this module only mirrors it to localStorage so the choice survives a
 * navigation. With scripting off the rail still collapses — it just forgets.
 */

/** Where each page's nav points, and which one is current. */
const PAGES = [
  {
    href: '/',
    id: 'chat',
    label: 'Chat',
    title: 'Chat',
    icon: 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z',
  },
  {
    href: '/studio',
    id: 'studio',
    label: 'Studio',
    title: 'Studio — image, transcription, speech',
    icon: 'M3 3h18v18H3zM8.5 8.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3M21 15l-5-5L5 21',
  },
  {
    href: '/settings',
    id: 'settings',
    label: 'Settings',
    title: 'Settings — keys, tools, models',
    icon:
      'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1',
  },
];

function svg(path) {
  return (
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ` +
    `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    `<path d="${path}"/></svg>`
  );
}

/** Which page we are on, matched on the path rather than a per-page constant. */
function currentPage() {
  const path = window.location.pathname.replace(/\/+$/, '') || '/';
  const hit = PAGES.find((p) => p.href === path);
  return hit ? hit.id : null;
}

function build(activeId) {
  const items = PAGES.map((p) => {
    const current = p.id === activeId;
    return (
      `<li><a class="nav-item" href="${p.href}" title="${p.title}"` +
      (current ? ' aria-current="page"' : '') +
      `>${svg(p.icon)}<span class="nav-label">${p.label}</span></a></li>`
    );
  }).join('');

  return `<nav class="nav" aria-label="Primary"><ul class="nav-list">${items}</ul></nav>`;
}

/**
 * Renders the rail and nothing else.
 *
 * It used to collapse, which needed a hidden checkbox for CSS to read, an
 * aria-pressed button, a localStorage key and a pre-paint restore in
 * theme-init.js. All of that was load-bearing for one behaviour nobody asked
 * for twice, and the collapsed state was a genuine hazard: a rail that is 60px
 * wide with no labels is a column of icons with nothing to tell them apart.
 */
export function mountNav() {
  const el = document.querySelector('[data-nav-mount]');
  if (!el || el.dataset.navMounted === '1') return;
  el.dataset.navMounted = '1';
  el.innerHTML = build(currentPage());
}