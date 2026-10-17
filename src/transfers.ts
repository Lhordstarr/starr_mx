/**
 * Server-side registry of running transfers.
 *
 * Downloads and runtime installs used to live entirely in the HTTP response:
 * the progress went out over SSE and the work was bound to that connection via
 * `res.on('close', abort)`. Reloading the page therefore *killed the download*,
 * abandoned the partial file and lost the progress — which is the opposite of
 * what someone expects from a background fetch.
 *
 * Here the work is detached from any connection and only its *reporting* is
 * attached. A client that connects gets the current state immediately, so a
 * reload re-attaches instead of restarting.
 */

export type TransferKind = 'download' | 'pull' | 'install';
export type TransferState = 'running' | 'done' | 'error';

export interface Transfer {
  id: string;
  kind: TransferKind;
  label: string;
  repo?: string;
  file?: string;
  state: TransferState;
  received: number;
  total: number | null;
  percent: number;
  error?: string;
  startedAt: number;
  endedAt?: number;
}

const active = new Map<string, Transfer>();
const recent: Transfer[] = [];

/** How long a finished transfer stays listed before it is forgotten. */
const RECENT_TTL_MS = 30 * 60_000;
const RECENT_MAX = 40;

export function startTransfer(t: Omit<Transfer, 'state' | 'received' | 'total' | 'percent' | 'startedAt'>): Transfer {
  const transfer: Transfer = {
    ...t,
    state: 'running',
    received: 0,
    total: null,
    percent: 0,
    startedAt: Date.now(),
  };
  active.set(transfer.id, transfer);
  return transfer;
}

export function updateTransfer(id: string, patch: Partial<Omit<Transfer, 'id'>>): void {
  const t = active.get(id);
  if (!t) return;
  Object.assign(t, patch);
  if (typeof t.total === 'number' && t.total > 0 && typeof t.received === 'number') {
    t.percent = Math.min(100, (t.received / t.total) * 100);
  }
}

export function finishTransfer(id: string, state: Exclude<TransferState, 'running'>, error?: string): void {
  const t = active.get(id);
  if (!t) return;
  t.state = state;
  t.endedAt = Date.now();
  if (error) t.error = error;
  active.delete(id);
  recent.unshift(t);
  while (recent.length > RECENT_MAX) recent.pop();
}

export function isRunning(id: string): boolean {
  return active.has(id);
}

/** Everything the UI should know about: what is running, and what just ran. */
export function listTransfers(): { active: Transfer[]; recent: Transfer[] } {
  const cutoff = Date.now() - RECENT_TTL_MS;
  while (recent.length && (recent[recent.length - 1].endedAt ?? 0) < cutoff) recent.pop();
  return {
    active: [...active.values()].sort((a, b) => a.startedAt - b.startedAt),
    recent: recent.filter((t) => t.state !== 'running'),
  };
}

/** Ids are derived from the work, so a repeat of the same thing re-attaches
 *  rather than piling up duplicates in the list. */
export function transferId(kind: TransferKind, key: string): string {
  return `${kind}:${key}`;
}