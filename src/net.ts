import dns from 'node:dns';
import net from 'node:net';

/**
 * Network family policy for every Node connection in this process.
 *
 * Hosts like Wikipedia and Hugging Face publish AAAA records beside their A
 * records, and this machine (like many home and cloud networks) has no working
 * IPv6 route. Node 22 defaults to racing both families (`autoSelectFamily`),
 * which should fall back to IPv4 — but in practice the dual-family connect
 * never completed: a raw TCP connection to the IPv4 address worked, curl
 * worked, and `fetch` sat until ETIMEDOUT without a socket ever firing
 * `connect`. Passing `family: 4` or disabling the race both fixed it; the two
 * settings below are the process-wide equivalent.
 *
 * - `ipv4first` makes a host with an A record resolve to it first.
 * - disabling auto-selection makes the connection use that address instead of
 *   racing it against a family with no route. A host that publishes only AAAA
 *   still resolves to IPv6 and connects, so IPv6-only networks keep working.
 *
 * The symptom without this was a bare `fetch failed` in tool results: the
 * Wikipedia lookup, Hugging Face downloads, and any dual-stack API could all
 * fail while curl against the same URL succeeded. The previous version of this
 * file set only `ipv4first`, which was not enough — the lookup already
 * returned IPv4 first, and the connection still timed out.
 */
dns.setDefaultResultOrder('ipv4first');
net.setDefaultAutoSelectFamily(false);

export const DNS_ORDER = dns.getDefaultResultOrder();
export const AUTO_SELECT_FAMILY = net.getDefaultAutoSelectFamily();
