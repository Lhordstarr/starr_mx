#!/usr/bin/env node
/**
 * The card that says what is being read aloud.
 *
 * Speaking a reply used to be invisible: a button changed to "Stop" and sound
 * came out. Nothing said which reply was being read, which voice was reading
 * it, or how far along it was — and the slow half of the job (making the
 * voice) showed nothing at all beyond a disabled button.
 *
 * What is checked here is the card's contract with the page, so the browser is
 * driven through the real UI: a saved conversation is seeded, its Speak button
 * is clicked, and the card is watched from the moment synthesis starts to the
 * moment the clip ends. The two things a test cannot do for real are the voice
 * synthesis (seconds of model time, covered by test:tts against the real
 * runner) and audio playback (no sound device in here). Both are stood in for
 * — a held or on-demand answer from /api/infer/tts, and an audio element whose
 * clock the test can move — and everything else is the app's own code.
 *
 * Run with the app up (`npm run dev`):
 *   npm run test:speaker-ui
 */
import { spawn, spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000/';
const CDP_PORT = Number(process.env.CDP_PORT ?? 9382);

/** The reply that gets read. Plain prose: no markdown for the stripper to
 *  change, so the title the card shows is exactly this string's first 64. */
const REPLY =
  'The quick brown fox jumps over the lazy dog, and keeps running well past the point where anyone is still listening to it.';
const TITLE = REPLY.slice(0, 64);

const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
if (!browser) {
  console.log('SKIP — no chromium/chrome found on PATH');
  process.exit(0);
}

let ttsModel = null;
try {
  const res = await fetch(new URL('/api/runnable?task=tts', APP_URL), {
    signal: AbortSignal.timeout(15000),
  });
  ttsModel = ((await res.json())?.tts ?? []).find((m) => m.ready) ?? null;
} catch {
  console.log(`SKIP — app not reachable at ${APP_URL} (start it with npm run dev)`);
  process.exit(0);
}
if (!ttsModel) {
  console.log('SKIP — no runnable voice model on this machine (Settings → Models)');
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PROFILE = join(tmpdir(), `starr-speaker-ui-${process.pid}`);
rmSync(PROFILE, { recursive: true, force: true });

const chrome = spawn(
  browser,
  [
    '--headless=new',
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    `--remote-debugging-port=${CDP_PORT}`,
    `--user-data-dir=${PROFILE}`,
    'about:blank',
  ],
  { stdio: 'ignore' },
);

let passed = 0;
const failures = [];
function report(name, ok, extra = '') {
  if (ok) {
    passed += 1;
    console.log(`  PASS  ${name}`);
  } else {
    failures.push(name);
    console.log(`  FAIL  ${name}${extra ? ` — ${extra}` : ''}`);
  }
}

/** Painted means drawable: a filled shape, or a stroke. A path whose only
 *  presentation is `fill="none"` — an X drawn as two lines, say — has no area
 *  and comes out as a blank button, which is how the first version shipped. */
const PAINTS = `(() => {
  const svg = document.querySelector('.speaker__btn svg');
  if (!svg) return false;
  const fill = (svg.getAttribute('fill') ?? '').trim();
  const stroke = (svg.getAttribute('stroke') ?? '').trim();
  return (fill && fill !== 'none') || !!stroke;
})()`;

async function getWsUrl() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)).json();
      const page = list.find((t) => t.type === 'page');
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch {
      /* chromium still starting */
    }
    await sleep(200);
  }
  throw new Error('chromium never came up');
}

let chromeKilled = false;
async function cleanup() {
  if (!chromeKilled) {
    chromeKilled = true;
    chrome.kill('SIGKILL');
    await new Promise((resolve) => {
      if (chrome.exitCode !== null) return resolve();
      chrome.once('exit', resolve);
      setTimeout(resolve, 3000);
    });
  }
  await rm(PROFILE, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => {});
}

/**
 * Stands in for the two things a headless test cannot have.
 *
 * The audio element is a fake whose clock the test drives — `dispatch` fires
 * the same events a real one would, so the app's own listeners do the work.
 * `/api/infer/tts` is answered on demand or held, because synthesising a real
 * voice per check would make this suite minutes long for no extra fact: the
 * thing being watched is the card, and the runner is covered by test:tts.
 */
const INIT = `
(() => {
  const audios = [];
  class FakeAudio {
    constructor(src) {
      this.src = src;
      this.paused = true;
      this.currentTime = 0;
      this.duration = 0;
      this._l = {};
      audios.push(this);
    }
    addEventListener(type, fn) { (this._l[type] ||= []).push(fn); }
    removeEventListener(type, fn) {
      const a = this._l[type] || [];
      const i = a.indexOf(fn);
      if (i >= 0) a.splice(i, 1);
    }
    dispatch(type) { for (const fn of [...(this._l[type] || [])]) fn({ type }); }
    play() { this.paused = false; this.dispatch('play'); return Promise.resolve(); }
    pause() { this.paused = true; this.dispatch('pause'); }
    removeAttribute(name) { if (name === 'src') this.src = ''; }
  }
  window.__audios = audios;
  window.Audio = FakeAudio;

  const realFetch = window.fetch.bind(window);
  window.__tts = {
    hold: true,
    calls: 0,
    last: null,
    payload: { path: '/tmp/ui-selftest.wav', samples: 8000, seconds: 0.5, device: 'cpu' },
    release: null,
  };
  window.fetch = (input, opts = {}) => {
    const url = String(input && input.url ? input.url : input);
    if (!url.includes('/api/infer/tts')) return realFetch(input, opts);
    window.__tts.calls += 1;
    window.__tts.last = JSON.parse(opts.body || '{}');
    return new Promise((resolve, reject) => {
      const answer = () => resolve(
        new Response('event: done\\ndata: ' + JSON.stringify(window.__tts.payload) + '\\n\\n', {
          status: 200,
          headers: { 'content-type': 'text/event-stream' },
        }),
      );
      const abort = () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' }));
      if (opts.signal) {
        if (opts.signal.aborted) return abort();
        opts.signal.addEventListener('abort', abort);
      }
      if (window.__tts.hold) window.__tts.release = answer;
      else answer();
    });
  };
})();
`;

try {
  console.log('\n=== the card that says what is being read ===');
  console.log(`  voice model: ${ttsModel.label ?? ttsModel.repo} (${ttsModel.driver ?? '?'})`);

  const ws = new WebSocket(await getWsUrl());
  await new Promise((res, rej) => {
    ws.onopen = res;
    ws.onerror = rej;
  });

  let idc = 0;
  const pending = new Map();
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  };
  const cdp = (method, params = {}) => {
    const id = ++idc;
    ws.send(JSON.stringify({ id, method, params }));
    return new Promise((res) => pending.set(id, res));
  };
  const evalJs = async (expression) => {
    const res = await cdp('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (res.result?.exceptionDetails) throw new Error(`eval failed: ${res.result.exceptionDetails.text}`);
    return res.result?.result?.value;
  };
  const waitFor = async (expr, timeoutMs = 20000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      try {
        if (await evalJs(expr)) return true;
      } catch {
        /* context destroyed by a navigation */
      }
      await sleep(120);
    }
    return false;
  };

  await cdp('Page.enable');
  await cdp('Runtime.enable');
  await cdp('Page.addScriptToEvaluateOnNewDocument', { source: INIT });

  await cdp('Page.navigate', { url: APP_URL });
  await waitFor(`document.readyState === 'complete' && !!document.querySelector('.composer')`);

  // A saved conversation, restored by the app's own boot path — the Speak
  // button that the card hangs off only exists on a rendered reply.
  //
  // The store is seeded only once the app has written its own: boot saves an
  // empty state, and a seed that lands before that write is overwritten by it.
  const SEED = `
    localStorage.setItem('llm-chat-state', JSON.stringify({
      active: 'c1',
      chats: [{
        id: 'c1',
        title: 'Reading aloud',
        updatedAt: Date.now(),
        conversation: [
          { role: 'user', content: 'Say something.' },
          { role: 'assistant', content: ${JSON.stringify(REPLY)} },
        ],
      }],
    }))
  `;
  const speakBtn =
    `[...document.querySelectorAll('.message.assistant .msg-actions button')]
       .find(b => b.textContent === 'Speak')`;
  await waitFor(`localStorage.getItem('llm-chat-state') !== null`);
  let asked = false;
  for (let attempt = 0; attempt < 2 && !asked; attempt += 1) {
    await evalJs(SEED);
    await cdp('Page.reload');
    asked = await waitFor(`!!(${speakBtn})`, 10000);
  }
  report('the reply carries its Speak button', asked);
  if (asked) await evalJs(`window.__speakBtn = (${speakBtn})`);
  if (!asked) throw new Error('nothing to drive: no Speak button rendered');

  report('no card is in the page before anything is asked to speak', !(await evalJs(`!!document.querySelector('.speaker')`)));

  /* ---- while the voice is being made ---- */

  await evalJs(`(window.__speakBtn).click()`);
  const waiting = await waitFor(`!!document.querySelector('.speaker')`);
  report('the card goes up as soon as the voice is asked for, not when it plays', waiting);
  if (waiting) {
    const state = JSON.parse(
      await evalJs(`JSON.stringify({
        title: document.querySelector('.speaker__title')?.textContent ?? '',
        sub: document.querySelector('.speaker__sub')?.textContent ?? '',
        live: !!document.querySelector('.speaker__line.bar--live'),
        icon: document.querySelector('.speaker__btn')?.getAttribute('aria-label') ?? '',
      })`),
    );
    report(
      'it says which reply is being read',
      state.title === TITLE,
      `title was "${state.title}"`,
    );
    report(
      'it names the voice and says the voice is still being made',
      state.sub.includes(ttsModel.label ?? ttsModel.repo) && /making the voice/i.test(state.sub),
      `sub was "${state.sub}"`,
    );
    report(
      'the line is indeterminate — there is no figure yet',
      state.live,
      'the bar must not invent a position while the voice is being synthesised',
    );
    report(
      'the button offers to stop making it',
      state.icon === 'Stop making this voice',
      `label was "${state.icon}"`,
    );
    report('and its icon is one that paints, not an empty button', await evalJs(PAINTS));

    // The motion, stated as values rather than trusted as pixels: it drops in
    // from above and slightly small, rests unshifted, and leaves back the way
    // it came. Transitions are switched off for the read — otherwise both
    // reads return the value mid-flight and the check would pass on a card
    // that never moved at all.
    const motion = JSON.parse(
      await evalJs(`(() => {
        const card = document.querySelector('.speaker');
        card.style.transition = 'none';
        card.classList.remove('is-in');
        const start = getComputedStyle(card).transform;
        card.classList.add('is-in');
        const entered = getComputedStyle(card).transform;
        card.classList.add('is-out');
        const leaving = getComputedStyle(card).transform;
        card.classList.remove('is-out');
        card.classList.add('is-in');
        card.style.transition = '';
        const cs = getComputedStyle(card);
        return JSON.stringify({
          start, entered, leaving,
          props: cs.transitionProperty,
          durs: cs.transitionDuration,
        });
      })()`),
    );
    const rise = /matrix\(0\.9\d+/.test(motion.start) && /, -8\)$/.test(motion.start);
    const sink = /matrix\(0\.9\d+/.test(motion.leaving) && /, -6\)$/.test(motion.leaving);
    report(
      'it arrives by dropping in from the topbar edge, and leaves by retreating back up to it',
      motion.entered === 'none' && rise && sink,
      `entered ${motion.entered}, from ${motion.start}, leaving ${motion.leaving}`,
    );
    const durs = motion.durs.split(',').map((d) => parseFloat(d));
    report(
      'and both ends of that are animated rather than snapped',
      /transform/.test(motion.props) &&
        /opacity/.test(motion.props) &&
        durs.length > 0 &&
        durs.every((d) => d > 0.05),
      `transition: ${motion.props} / ${motion.durs}`,
    );
    report(
      'the reply own button says it is saying, and is disabled',
      await evalJs(
        `(() => { const b = window.__speakBtn; return b.disabled && b.textContent === 'Saying…'; })()`,
      ),
    );
  }

  // Stopping from the card is a decision, not a fault: no error toast. The
  // check needs the card to have been up — "gone" is not a pass on its own.
  const wasUp = (await evalJs(`!!document.querySelector('.speaker')`)) === true;
  await evalJs(`document.querySelector('.speaker__btn')?.click(), true`);
  const goneNow = await waitFor(`!document.querySelector('.speaker')`, 8000);
  report(
    'the card comes down when the synthesis is stopped from it',
    wasUp && goneNow,
    wasUp ? '' : 'the card was never up',
  );
  report(
    'and the reply is left as it was, with no error reported',
    (await evalJs(
      `(() => { const b = window.__speakBtn;
         return !b.disabled && b.textContent === 'Speak' && !document.querySelector('.toast--error'); })()`,
    )) === true,
  );

  /* ---- while the clip plays ---- */

  await evalJs(`window.__tts.hold = false; (window.__speakBtn).click()`);
  const playing = await waitFor(`!!document.querySelector('.speaker')`, 8000);
  report('the card goes up again for the clip that was made', playing);
  if (playing) {
    await waitFor(`!!document.querySelector('.speaker.is-in')`, 2000);
    const phase2 = JSON.parse(
      await evalJs(`JSON.stringify({
        title: document.querySelector('.speaker__title')?.textContent ?? '',
        sub: document.querySelector('.speaker__sub')?.textContent ?? '',
        playing: !!document.querySelector('.speaker.is-playing'),
        barState: getComputedStyle(document.querySelector('.speaker__eq b')).animationPlayState,
        live: !!document.querySelector('.speaker__line.bar--live'),
        clip: window.__audios.length ? String(window.__audios[window.__audios.length - 1].src) : '',
        paused: window.__audios.length ? window.__audios[window.__audios.length - 1].paused : null,
      })`),
    );
    report(
      'the same reply title stays on it — no second card, no empty one',
      phase2.title === TITLE,
      `title was "${phase2.title}"`,
    );
    report(
      'the sub reports the device the run landed on, not the one that was asked for',
      /on cpu\b/.test(phase2.sub) && !/auto/.test(phase2.sub),
      `sub was "${phase2.sub}"`,
    );
    report(
      'the line stops being indeterminate: a clip has a real position',
      !phase2.live,
    );
    report('the clip is playing', phase2.paused === false && /\.wav$/.test(phase2.clip), phase2.clip);
    report(
      'the equalizer runs while it plays',
      phase2.playing && phase2.barState === 'running',
      `is-playing=${phase2.playing}, animation-play-state=${phase2.barState}`,
    );
    report('the pause icon paints', await evalJs(PAINTS));

    // The line is the clip's own clock, not a decorative sweep.
    await evalJs(`(() => {
      const a = window.__audios[window.__audios.length - 1];
      a.duration = 2; a.currentTime = 0.5; a.dispatch('timeupdate');
    })()`);
    const width = await evalJs(
      `parseFloat(document.querySelector('.speaker__line > i')?.style.width ?? 'NaN')`,
    );
    report(
      'the line tracks the real position in the clip',
      Math.abs(width - 25) < 0.5,
      `width was "${width}%" of a 2 s clip at 0.5 s`,
    );

    // Pause and resume from the card, and send nothing while doing it: the
    // card is not in the composer form today, and it must never end up as a
    // default-type button in one — that would send the message instead.
    const messagesBefore = await evalJs(`document.querySelectorAll('.message').length`);
    await evalJs(`document.querySelector('.speaker__btn')?.click(), true`);
    const paused = JSON.parse(
      await evalJs(`JSON.stringify({
        paused: window.__audios[window.__audios.length - 1].paused,
        playing: !!document.querySelector('.speaker.is-playing'),
        barState: getComputedStyle(document.querySelector('.speaker__eq b')).animationPlayState,
        icon: document.querySelector('.speaker__btn')?.getAttribute('aria-label') ?? '',
      })`),
    );
    report(
      'pausing from the card stops the clip and freezes the bars',
      paused.paused === true && !paused.playing && paused.barState === 'paused',
      JSON.stringify(paused),
    );
    report('and the button then offers to resume', paused.icon === 'Resume reading', paused.icon);
    report('the play icon paints', await evalJs(PAINTS));
    report(
      'pressing it sent nothing — it is a button, never a submit',
      (await evalJs(`document.querySelectorAll('.message').length`)) === messagesBefore &&
        (await evalJs(`document.getElementById('input').value`)) === '',
    );

    await evalJs(`document.querySelector('.speaker__btn')?.click(), true`);
    report(
      'resuming brings the clip and the bars back',
      (await evalJs(
        `(() => { const a = window.__audios[window.__audios.length - 1];
           return !a.paused && !!document.querySelector('.speaker.is-playing')
             && document.querySelector('.speaker__btn').getAttribute('aria-label') === 'Pause reading'; })()`,
      )) === true,
    );

    // It belongs to the top of the window: under the topbar, above everything
    // in the conversation, and in the same place no matter what is below it.
    // Everything here is about where the bar *rests*, so let its entrance
    // finish first — mid-transition it is still travelling down, and a fast
    // run would measure it a few pixels high and call that a move.
    await waitFor(
      `getComputedStyle(document.querySelector('.speaker')).transform === 'none'`,
      2000,
    );
    const layout = JSON.parse(
      await evalJs(`JSON.stringify((() => {
        const card = document.querySelector('.speaker').getBoundingClientRect();
        const bar = document.querySelector('.topbar').getBoundingClientRect();
        const box = document.querySelector('.composer-box').getBoundingClientRect();
        return { cardTop: card.top, cardBottom: card.bottom, barBottom: bar.bottom, boxTop: box.top };
      })())`),
    );
    report(
      "it is pinned just under the topbar, out of the conversation's way",
      layout.cardTop >= Math.floor(layout.barBottom) - 1 &&
        layout.cardTop - layout.barBottom <= 40 &&
        layout.cardBottom <= layout.boxTop,
      `card ${Math.round(layout.cardTop)}–${Math.round(layout.cardBottom)}, topbar bottom ` +
        `${Math.round(layout.barBottom)}, composer top ${Math.round(layout.boxTop)}`,
    );

    // "Always above" also means it does not ride the composer: growing the
    // input's box must leave the bar exactly where it was.
    const before = JSON.parse(
      await evalJs(
        `(() => { const r = document.querySelector('.speaker').getBoundingClientRect();
           return JSON.stringify({ top: r.top, bottom: r.bottom }); })()`,
      ),
    );
    await evalJs(`(() => {
      const input = document.getElementById('input');
      input.value = Array.from({ length: 8 }, (_, i) => 'line ' + i).join('\\n');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    })()`);
    await sleep(300);
    const after = JSON.parse(
      await evalJs(
        `(() => { const r = document.querySelector('.speaker').getBoundingClientRect();
           return JSON.stringify({ top: r.top, bottom: r.bottom }); })()`,
      ),
    );
    report(
      'and the composer growing under it does not move it at all',
      Math.abs(after.top - before.top) < 0.5 && Math.abs(after.bottom - before.bottom) < 0.5,
      `moved from ${Math.round(before.top)} to ${Math.round(after.top)}`,
    );
    // Put the box back the way the test found it.
    await evalJs(`(() => {
      const input = document.getElementById('input');
      input.value = '';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    })()`);

    // And "on top" is literal: with a message scrolled underneath it, the
    // point still hit-tests to the card, not to the message.
    const covering = JSON.parse(
      await evalJs(`JSON.stringify((() => {
        document.querySelector('.messages').scrollTop = 0;
        const card = document.querySelector('.speaker');
        const c = card.getBoundingClientRect();
        const msg = [...document.querySelectorAll('.message')]
          .map((m) => m.getBoundingClientRect())
          .find((r) => r.bottom > c.top + 4 && r.top < c.bottom - 4);
        if (!msg) return { overlap: false };
        const x = Math.round(Math.max(c.left, msg.left) + 10);
        const y = Math.round(Math.max(c.top, msg.top) + 6);
        const hit = document.elementFromPoint(x, y);
        return { overlap: true, inside: !!hit && card.contains(hit), hit: hit ? String(hit.className) : null };
      })())`),
    );
    report(
      'nothing in the conversation is painted over it',
      covering.overlap === true && covering.inside === true,
      covering.overlap ? `the hit at that point was ${covering.hit}` : 'nothing overlapped it to test',
    );

    // The end of the clip is what takes it down, and it leaves by animating.
    await evalJs(`(() => {
      const a = window.__audios[window.__audios.length - 1];
      a.currentTime = 2; a.dispatch('timeupdate'); a.dispatch('ended');
    })()`);
    const leaving = await evalJs(
      `!!document.querySelector('.speaker')?.classList.contains('is-out')`,
    );
    const gone = await waitFor(`!document.querySelector('.speaker')`, 5000);
    report('the card leaves when the clip ends', gone);
    report('and it leaves by animating, not by vanishing', leaving === true);
    report(
      'the reply button is back to Speak',
      await evalJs(
        `(() => { const b = window.__speakBtn; return !b.disabled && b.textContent === 'Speak'; })()`,
      ),
    );

    // Playing it again needs no second synthesis.
    const callsBefore = await evalJs(`window.__tts.calls`);
    await evalJs(`(window.__speakBtn).click()`);
    const replay = await waitFor(`!!document.querySelector('.speaker')`, 5000);
    report(
      'playing the same reply again brings the card back without making the voice twice',
      replay && (await evalJs(`window.__tts.calls`)) === callsBefore,
      `synthesis calls went from ${callsBefore} to ${await evalJs(`window.__tts.calls`)}`,
    );
  }
} finally {
  await cleanup();
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);
