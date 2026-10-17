import type { Request, Response, NextFunction } from 'express';

/**
 * Abuse limits for an unauthenticated server.
 *
 * There is no login: anything that can reach the port can talk to the API. That
 * makes these limits the only thing between a runaway loop (or a bored person)
 * and your API credits, so they stay even though auth is gone.
 */

export function clientIp(req: Request): string {
  return req.ip ?? req.socket.remoteAddress ?? 'unknown';
}

/** Simple fixed-window limiter for expensive endpoints. */
const buckets = new Map<string, { count: number; resetAt: number }>();

/** Drops expired buckets so the map can't grow without bound. */
setInterval(() => {
  const now = Date.now();
  for (const [k, b] of buckets) if (b.resetAt < now) buckets.delete(k);
}, 60_000).unref();

/**
 * Fixed-window rate limit. Keeps one IP from burning through your API keys, or
 * pinning the CPU with concurrent local generations.
 */
/**
 * Paths the app's own UI polls, with their own ceiling.
 *
 * `/api/transfers` is how the Downloads page shows live progress, so it is hit
 * several times a minute by design. Under the general limit — 30 a minute — a
 * poll every 1.5s tripped the limiter and the progress bar froze at whatever
 * it had last drawn, which looks exactly like a stalled download.
 */
const POLL_LIMITS: Record<string, number> = {
  '/api/transfers': 240,
  // The resource box polls this from every open tab. Two tabs on the base
  // limit (30) is 40 requests a minute, so meters went stale on a coin flip.
  '/api/resources': 120,
  // Polled once a second while the process manager is open — 60 a minute on
  // its own, so it sat permanently over the base limit of 30. Thirty seconds
  // after opening the manager it started answering 429 and the list simply
  // stopped updating, which reads as the pop-out hanging. It is the same
  // endpoint /api/transfers was given room for.
  '/api/processes': 240,
};

/**
 * Read-only Hugging Face lookups are not rate limited here.
 *
 * The browse page fans out across nine pipelines on open, and every card the
 * user clicks costs a detail fetch and a file-manifest fetch — so ordinary
 * browsing blew through any per-minute budget a single user could plausibly
 * be given, and the result was a modal reading "Rate limit exceeded" for a
 * model that is perfectly public. These endpoints are cheap, they proxy a
 * third party that enforces its own limits, and a client looping on them
 * cannot do worse than make the Hub refuse us — which it will, on its terms.
 */
const HF_READ_PATHS = [/^\/api\/hf\/(search|model|repo-files|avatar)/];

export function rateLimit(req: Request, res: Response, next: NextFunction) {
  // The limiter is mounted at /api, so req.path is mount-relative ('/transfers',
  // not '/api/transfers'). Keys were matched against it anyway, so the poll
  // ceilings below and the HF exemption never fired; baseUrl + path restores
  // both without changing what the paths are.
  const route = `${req.baseUrl}${req.path}`;
  if (HF_READ_PATHS.some((re) => re.test(route))) return next();
  const base = Number(process.env.RATE_LIMIT_PER_MINUTE ?? 30);
  const perMinute = POLL_LIMITS[route] ?? base;
  const key = `${route}:${clientIp(req)}`;
  const now = Date.now();
  const bucket = buckets.get(key);

  if (!bucket || bucket.resetAt < now) {
    buckets.set(key, { count: 1, resetAt: now + 60_000 });
    return next();
  }
  if (bucket.count >= perMinute) {
    return res.status(429).json({ error: 'Rate limit exceeded. Slow down.' });
  }
  bucket.count += 1;
  next();
}

/** Caps simultaneous streams so one client can't pin the CPU or exhaust quota. */
let activeStreams = 0;
export function acquireStream(): boolean {
  if (activeStreams >= Number(process.env.MAX_CONCURRENT_STREAMS ?? 4)) return false;
  activeStreams += 1;
  return true;
}
export function releaseStream() {
  activeStreams = Math.max(0, activeStreams - 1);
}