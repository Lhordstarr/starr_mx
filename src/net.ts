import dns from 'node:dns';

/**
 * Prefer IPv4 when resolving hostnames.
 *
 * Hugging Face sits behind CloudFront and publishes AAAA records alongside its
 * A records. Node's default DNS result order is `verbatim`, so `fetch` tries
 * the IPv6 address first and waits out a connect timeout on any host without
 * real IPv6 connectivity — which is most home networks, including this one,
 * where `ip -6 route` shows only a link-local `fe80::/64`.
 *
 * The symptom was maddening and looked intermittent: the same download would
 * work, then fail seconds later with a bare `fetch failed` / `ETIMEDOUT`
 * before a single byte arrived. curl was unaffected because it implements
 * Happy Eyeballs (RFC 6555) and races the two families; `fetch` does not, so
 * it just sat on the dead IPv6 address.
 *
 * Forcing IPv4-first costs nothing on a dual-stack host — the A record is
 * reachable either way — and turns a hard failure into a working request.
 */
dns.setDefaultResultOrder('ipv4first');

export const DNS_ORDER = dns.getDefaultResultOrder();