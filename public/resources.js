/**
 * The Resources box and the resource manager, shared by the chat page and the
 * Studio.
 *
 * Both pages answer the same two questions — how busy is this machine, and
 * what is it doing for me right now — and both have a rail on the left with
 * nothing at the bottom of it. Copying the meters and the manager into a
 * second script is how the two would start disagreeing: one would gain a row,
 * the other would not, and there would be no way to tell which was right.
 *
 * The markup lives here for the same reason it does in nav.js: the two
 * documents are maintained separately, so a hand-written copy per page drifts.
 * A page opts in with an empty `[data-resource-mount]` wherever the box belongs.
 */
import { notifySuccess, notifyError } from '/notify.js';

/**
 * How often to ask.
 *
 * One second, and the meter bars carry a transition nearly as long, so each
 * reading slides into the next instead of stepping — the machine looks like
 * what it is, a thing in constant motion. At three seconds the numbers were
 * readings separated by frozen gaps, which read as jumping.
 */
const RES_POLL_MS = 1000;

/** Percent to a severity word, so the bar colour means something consistent. */
function loadBand(pct) {
  if (pct >= 85) return 'high';
  if (pct >= 60) return 'warn';
  return 'ok';
}

function fmtMB(mb) {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
}

/** Bytes the way the rest of the page says it. */
function fmtBytesFor(n) {
  if (typeof n !== 'number' || !Number.isFinite(n) || n <= 0) return '0 MB';
  if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))}KB`;
  const mb = n / 1048576;
  if (mb < 1024) return `${Math.round(mb)} MB`;
  return `${(n / 1073741824).toFixed(1)} GB`;
}

/** The whole box. Built as a string because it is fixed and has no data in it. */
function buildMarkup() {
  return `
    <div class="resources__head">
      <span class="resources__title">Resources</span>
      <button type="button" id="open-procman" class="resources__manage"
              aria-haspopup="dialog" aria-expanded="false"
              title="Every model and job this app is running, and a way to stop them">Processes</button>
      <span class="resources__stamp" id="res-stamp"></span>
    </div>
    <div class="resources__list">
      <div class="resources__row" data-metric="cpu">
        <span class="resources__label">CPU</span>
        <span class="resources__bar"><i></i></span>
        <span class="resources__value">—</span>
      </div>
      <div class="resources__row" data-metric="ram">
        <span class="resources__label">RAM</span>
        <span class="resources__bar"><i></i></span>
        <span class="resources__value">—</span>
      </div>
      <div class="resources__row" data-metric="gpu">
        <span class="resources__label">GPU</span>
        <span class="resources__bar"><i></i></span>
        <span class="resources__value">—</span>
      </div>
      <!-- GPU load and the memory behind it are different questions and were
           sharing one row, which made the percentage carry a vendor prefix and
           a byte count at once.

           No bar here on purpose — at ~195px there is no room for a label, two
           byte counts and a bar, and the counts state the fraction already. -->
      <div class="resources__row resources__row--sub" data-metric="vram">
        <span class="resources__label">VRAM</span>
        <span class="resources__bar"><i></i></span>
        <span class="resources__value">—</span>
      </div>
    </div>

    <!-- Resource manager. Built with the box because that is where the
         question it answers gets asked: the meters say the machine is busy,
         and this says by what. mountResources hoists it onto <body> — the
         rail carries a backdrop-filter (a containing block for fixed
         children) and, on the Studio, scroll overflow, so left in place the
         manager landed offset and got clipped by it. -->
    <div class="procman" id="procman" role="dialog" aria-modal="false"
         aria-label="Running models and jobs" hidden>
      <div class="procman__head">
        <span class="popover__head">Running</span>
        <button type="button" class="icon-btn" id="procman-close" aria-label="Close the resource manager">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <path d="M6 6l12 12M18 6L6 18" />
          </svg>
        </button>
      </div>
      <div class="procman__body" id="procman-body">
        <p class="procman__empty">Reading what is running…</p>
      </div>
      <div class="procman__foot">
        <span class="procman__note" id="procman-note"></span>
        <button type="button" class="btn-ghost procman__all" id="procman-stop-all" hidden>
          Stop all jobs
        </button>
      </div>
    </div>`;
}

/**
 * Mounts the box, if the page asked for one.
 *
 * Returns the controls so a page can nudge the poll — the chat sidebar is a
 * drawer on narrow screens and wants a reading the moment it is opened, rather
 * than waiting out the interval with nothing on screen.
 */
export function mountResources() {
  const host = document.querySelector('[data-resource-mount]');
  if (!host) return null;

  host.innerHTML = buildMarkup();
  // The box is the styling hook for the meters; the manager pop-out moves to
  // <body> below.
  host.classList.add('resources');

  const $ = (sel) => host.querySelector(sel);
  const resStamp = $('#res-stamp');
  const procmanTrigger = $('#open-procman');
  const procman = $('#procman');
  // The pop-out lives on <body>, not in the rail. The rail carries a
  // backdrop-filter (glass), and a filtered ancestor is the containing block
  // for its fixed descendants — so viewport coordinates landed the manager 51px
  // down from where they said. On the Studio the rail also scrolls, which
  // clipped it to the rail's width on top of that. On <body> the measured
  // coordinates are the ones it actually lands at, and nothing can clip it.
  document.body.append(procman);
  const procmanBody = procman.querySelector('#procman-body');
  const procmanNote = procman.querySelector('#procman-note');
  const procmanClose = procman.querySelector('#procman-close');
  const procmanStopAll = procman.querySelector('#procman-stop-all');

  let resTimer = null;
  let resBusy = false;
  let procBusy = false;
  let procmanOpen = false;

  /* ---------------- Meters ---------------- */

  function setMeter(metric, text, pct, band) {
    const row = host.querySelector(`.resources__row[data-metric="${metric}"]`);
    if (!row) return;
    const value = row.querySelector('.resources__value');
    // Optional: the VRAM row has no bar, because in a ~195px row there is no
    // room for a label, two byte counts and a bar, and the counts say the
    // fraction already.
    const fill = row.querySelector('.resources__bar > i');
    if (value && value.textContent !== text) value.textContent = text;
    if (fill) {
      const width = pct === null ? '0%' : `${Math.min(100, Math.max(0, pct))}%`;
      if (fill.style.width !== width) fill.style.width = width;
    }
    // `none` greys the bar out entirely, for a metric this machine doesn't have.
    const next = pct === null ? 'none' : band;
    if (row.dataset.load !== next) row.dataset.load = next;
  }

  function renderResources(r) {
    // First CPU sample has no delta yet; show the load average rather than a lie.
    const cpuPct = r.cpu.percent;
    setMeter(
      'cpu',
      cpuPct === null ? `load ${r.cpu.loadAvg1}` : `${cpuPct}%`,
      cpuPct,
      cpuPct === null ? null : loadBand(cpuPct),
    );

    setMeter('ram', `${fmtMB(r.ram.usedMB)} / ${fmtMB(r.ram.totalMB)}`, r.ram.percent, loadBand(r.ram.percent));

    // Two rows, two questions. "How busy is the card" and "how much of it is
    // taken" were crammed into one line as `AMD 27% · 1.0 GB/2.0 GB`, which put
    // a vendor prefix in front of the percentage and buried the byte count
    // behind it. The vendor is still worth knowing, but as the row's tooltip,
    // where it is context rather than part of a number.
    const gpuRow = host.querySelector('.resources__row[data-metric="gpu"]');
    const vramRow = host.querySelector('.resources__row[data-metric="vram"]');
    if (r.gpu.available) {
      // A GPU we can see but not read is not the same as no GPU, so say which.
      // "no counters" covers a missing reader; a card that cannot be reached at
      // all deserves its own words, because the fix is completely different.
      let idleLabel = 'idle';
      if (r.gpu.reason) {
        idleLabel = /no drm node|not responding|firmware|registers/i.test(r.gpu.reason)
          ? 'unreachable'
          : 'no counters';
      }
      setMeter(
        'gpu',
        r.gpu.percent === null ? idleLabel : `${r.gpu.percent}%`,
        r.gpu.percent,
        r.gpu.percent === null ? null : loadBand(r.gpu.percent),
      );

      // Memory needs a total to be a fraction of anything. Both figures keep
      // their own unit rather than sharing one, because "364 / 2.0 GB" gives a
      // reader no way to know whether 364 is bytes, kilobytes or megabytes.
      const used = r.gpu.memUsedMB;
      const total = r.gpu.memTotalMB;
      const hasTotal = typeof used === 'number' && typeof total === 'number' && total > 0;
      const vramText = hasTotal ? `${fmtMB(used)} / ${fmtMB(total)}` : typeof used === 'number' ? fmtMB(used) : '—';
      const vramPct = hasTotal ? (used / total) * 100 : null;
      setMeter('vram', vramText, vramPct, vramPct === null ? null : loadBand(vramPct));

      const what = [
        r.gpu.vendor === 'amd' ? 'AMD' : r.gpu.vendor === 'nvidia' ? 'NVIDIA' : '',
        r.gpu.name,
        r.gpu.reason,
      ]
        .filter(Boolean)
        .join(' · ');
      if (gpuRow) gpuRow.title = what;
      if (vramRow) vramRow.title = what;
    } else {
      // Say why, rather than showing a permanently empty bar.
      setMeter('gpu', 'none', null, null);
      setMeter('vram', 'none', null, null);
      if (gpuRow) gpuRow.title = r.gpu.reason ?? 'No GPU detected';
      if (vramRow) vramRow.title = r.gpu.reason ?? 'No GPU detected';
    }

    // (Backend status chips used to sit here — an Ollama ping and a Python-env
    // stat. Removed: the daemon's state is said where it is managed, and the
    // chips flickering every second in a 195px box were noise, not a meter.)

    const when = new Date(r.at);
    resStamp.textContent = `${String(when.getHours()).padStart(2, '0')}:${String(when.getMinutes()).padStart(2, '0')}:${String(when.getSeconds()).padStart(2, '0')}`;
  }

  async function pollResources() {
    if (resBusy) return;
    resBusy = true;
    try {
      const res = await fetch('/api/resources');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderResources(data.resources);
    } catch {
      // A failed poll shouldn't spam the sidebar; mark the stamp and move on.
      resStamp.textContent = 'offline';
    } finally {
      resBusy = false;
    }
  }

  /* ---------------- Resource manager ---------------- */

  /** "6 min" / "48s" — how long a job has been going. */
  function fmtElapsed(ms) {
    const s = Math.max(0, Math.round(ms / 1000));
    if (s < 60) return `${s}s`;
    const m = Math.round(s / 60);
    if (m < 60) return `${m} min`;
    const h = Math.floor(m / 60);
    return `${h} h ${m % 60} min`;
  }

  /** What a job is doing, in the words the task is written in elsewhere. */
  const TASK_LABEL = { stt: 'Transcribing', tts: 'Speaking', image: 'Generating an image' };

  /**
   * Where it is running. The answer to "why is my machine busy".
   *
   * `cuda` and `rocm` are the runner's own words for what it resolved to and
   * are what a job is labelled with once it has said so; `nvidia` and `amd`
   * are requests. `auto` has no answer yet — the runner reports within
   * seconds — and until then "detecting…" is the only honest label. It used to
   * read "GPU", which was wrong on the exact machines most likely to ask a
   * question like this: ones whose torch cannot touch a card at all.
   */
  const DEVICE_LABEL = {
    cpu: 'CPU',
    cuda: 'GPU (CUDA)',
    rocm: 'GPU (ROCm)',
    nvidia: 'GPU (CUDA)',
    amd: 'GPU (ROCm)',
    auto: 'detecting…',
  };

  /**
   * Rows are made once and rewritten in place, keyed by what they describe.
   *
   * The list refreshes every second, and rebuilding rows from scratch would
   * replace the Stop button under the user's cursor mid-click: the mousedown
   * lands on one element, the mouseup on its replacement, and no click fires.
   * A row — especially its button — therefore outlives any number of polls,
   * and only the text inside it is rewritten.
   */
  const rowSlots = new Map();

  function rowSlot(key) {
    let slot = rowSlots.get(key);
    if (slot) return slot;

    const row = document.createElement('div');
    row.className = 'procman__row';

    const text = document.createElement('div');
    text.className = 'procman__text';

    const name = document.createElement('div');
    name.className = 'procman__name';

    const meta = document.createElement('div');
    meta.className = 'procman__meta';

    text.append(name, meta);
    row.append(text);
    slot = {
      row,
      text,
      name,
      meta,
      btn: null,
      action: null,
      extraSig: '',
      extraNodes: [],
      item: null,
    };
    rowSlots.set(key, slot);
    return slot;
  }

  /**
   * Brings one row up to date.
   *
   * The meta line is rewritten every poll because nearly all of it moves —
   * elapsed time, the memory figure, the device label. The button is only
   * touched when the action itself appears, disappears or is renamed.
   */
  function fillRow(slot, { item, name, metaParts, makeExtra = null, extraSig = '', action = null }) {
    slot.item = item;
    if (slot.name.textContent !== name) {
      slot.name.textContent = name;
      slot.name.title = name;
    }
    slot.meta.replaceChildren(metaLine(metaParts));

    // The split bar's numbers only move when a model is (un)loaded, so it is
    // keyed by them and rebuilt only when they change — a rebuild would
    // restart its width transitions from zero.
    if (extraSig !== slot.extraSig) {
      slot.extraSig = extraSig;
      for (const n of slot.extraNodes) n.remove();
      slot.extraNodes = [];
      if (makeExtra) {
        slot.text.append(makeExtra());
        // The appended nodes land after the name and the meta line.
        slot.extraNodes = [...slot.text.childNodes].slice(2);
      }
    }

    if (!action && slot.btn) {
      slot.btn.remove();
      slot.btn = null;
    }
    if (action) {
      if (!slot.btn) {
        slot.btn = document.createElement('button');
        slot.btn.type = 'button';
        slot.btn.className = 'procman__stop ' + (action.danger ? 'btn-danger' : 'btn-ghost');
        // Read through the slot: the action is remade every poll (it closes
        // over the job's current numbers), and the click must use the fresh
        // one, not the one that existed when the button was created.
        slot.btn.addEventListener('click', () => slot.action?.run());
        slot.row.append(slot.btn);
      }
      slot.action = action;
      if (slot.btn.textContent !== action.label) slot.btn.textContent = action.label;
      const title = action.title ?? action.label;
      if (slot.btn.title !== title) slot.btn.title = title;
    } else {
      slot.action = null;
    }
  }

  /**
   * One `a · b · c` line, with the numbers bolded.
   *
   * Returns a fragment rather than a Text node on purpose. `append` is a
   * `ParentNode` method and Text does not implement it — only Element, Document
   * and DocumentFragment do — so building this by appending into a Text node
   * throws and takes the whole render with it.
   */
  function metaLine(parts) {
    const frag = document.createDocumentFragment();
    parts.forEach((p, i) => {
      if (i) frag.append(document.createTextNode(' · '));
      if (typeof p === 'string') {
        frag.append(document.createTextNode(p));
        return;
      }
      if (p && typeof p.bar === 'number') {
        // A small inline bar for a job's share of system RAM. Width is the
        // fraction of total memory; the transition is on the element so it
        // slides rather than steps between polls.
        const wrap = document.createElement('span');
        wrap.className = 'procman__membar';
        const fill = document.createElement('i');
        fill.style.width = `${Math.min(100, Math.max(0, p.bar))}%`;
        wrap.append(fill);
        wrap.title = `${Math.round(p.bar)}% of system RAM`;
        frag.append(wrap);
        return;
      }
      const b = document.createElement('b');
      b.textContent = p.strong;
      frag.append(b, document.createTextNode(` ${p.rest}`));
    });
    return frag;
  }

  /**
   * How one loaded model's memory is divided between the card and system RAM.
   *
   * Ollama reports a total size and how much is resident in video memory; what
   * is left spilled into system RAM. When a model is bigger than the card it is
   * sitting on both at once, and that is the case the numbers have to explain.
   * Width carries the ratio, a stripe tells the halves apart when the numbers
   * are close, and an empty half is the plainest statement that nothing is being
   * spent there.
   */
  function splitBar(vramBytes, totalBytes) {
    const ram = Math.max(0, totalBytes - vramBytes);
    const total = Math.max(1, totalBytes);
    const pct = Math.max(0, Math.min(100, (vramBytes / total) * 100));

    const wrap = document.createDocumentFragment();

    const bar = document.createElement('div');
    bar.className = 'procman__split';
    bar.setAttribute('role', 'img');
    bar.setAttribute(
      'aria-label',
      `${fmtBytesFor(vramBytes)} of ${fmtBytesFor(totalBytes)} on the GPU, ${fmtBytesFor(ram)} in RAM`,
    );
    const gpuSeg = document.createElement('i');
    gpuSeg.dataset.part = 'vram';
    gpuSeg.style.width = `${pct}%`;
    const ramSeg = document.createElement('i');
    ramSeg.dataset.part = 'ram';
    ramSeg.style.width = `${100 - pct}%`;
    bar.append(gpuSeg, ramSeg);

    const legend = document.createElement('div');
    legend.className = 'procman__legend';
    for (const [part, label, bytes] of [
      ['vram', 'GPU', vramBytes],
      ['ram', 'RAM', ram],
    ]) {
      const cell = document.createElement('span');
      cell.className = 'procman__legend-item';
      const dot = document.createElement('i');
      dot.dataset.part = part;
      const text = document.createElement('b');
      text.textContent = `${fmtBytesFor(bytes)} ${label}`;
      cell.append(dot, text);
      legend.append(cell);
    }

    wrap.append(bar, legend);
    return wrap;
  }

  /* Group headers and the empty message live as long as the manager does;
     only their text changes. Rows come and go around them. */
  const ollamaGroup = document.createElement('div');
  ollamaGroup.className = 'procman__group';
  const jobsGroup = document.createElement('div');
  jobsGroup.className = 'procman__group';
  const emptyMsg = document.createElement('p');
  emptyMsg.className = 'procman__empty';

  /**
   * Puts exactly `nodes` into the body, in that order, by moving rather than
   * remaking them — identity is what keeps the buttons clickable — and drops
   * whatever is left over.
   */
  function syncOrder(nodes) {
    let cursor = procmanBody.firstChild;
    for (const node of nodes) {
      if (cursor === node) {
        cursor = node.nextSibling;
        continue;
      }
      procmanBody.insertBefore(node, cursor);
    }
    while (cursor) {
      const next = cursor.nextSibling;
      cursor.remove();
      cursor = next;
    }
  }

  /**
   * Paints the list.
   *
   * Two groups, kept apart because they are not the same kind of thing and have
   * different actions: Ollama models are held by a separate daemon on a
   * keep-alive timer and are released with Unload, while local jobs are real
   * children of this page's server and are signalled with Stop. Neither is a pid
   * the browser holds, so there is nothing here that could be pointed at an
   * arbitrary process.
   */
  function renderProcesses(data) {
    const now = Date.now();
    const ramTotal = data.ramTotalBytes ?? 0;
    const models = data.ollama?.models ?? [];
    const jobs = data.active ?? [];
    const running = jobs.filter((j) => j.state !== 'stopping');
    const desired = [];
    const live = new Set();

    if (models.length) {
      ollamaGroup.textContent = `Ollama · ${models.length} loaded`;
      desired.push(ollamaGroup);
      for (const m of models) {
        const key = `ollama:${m.name}`;
        live.add(key);
        const ram = Math.max(0, m.sizeBytes - m.vramBytes);
        const where =
          m.device === 'gpu'
            ? 'entirely on the GPU'
            : m.device === 'cpu'
              ? 'none of it fits in VRAM'
              : `split — ${fmtBytesFor(ram)} spilled into RAM`;
        // "never" is a real answer from Ollama — a keep-alive of 0 means the
        // model stays until something releases it — but "releases in never"
        // reads as a bug, so it is stated as the instruction it actually is.
        const when =
          m.expires === 'never' ? 'stays until you unload it' : `releases in ${m.expires}`;
        const slot = rowSlot(key);
        fillRow(slot, {
          item: m,
          name: m.name,
          metaParts: [{ strong: fmtBytesFor(m.sizeBytes), rest: 'loaded' }, where, when],
          extraSig: `${m.vramBytes}/${m.sizeBytes}`,
          makeExtra: () => splitBar(m.vramBytes, m.sizeBytes),
          action: {
            label: 'Unload',
            title: 'Let go of this model and free the memory it is holding',
            run: () => unloadOllamaModel(slot.item.name),
          },
        });
        desired.push(slot.row);
      }
    }

    if (jobs.length) {
      jobsGroup.textContent = `Local jobs · ${jobs.length}`;
      desired.push(jobsGroup);
      for (const j of jobs) {
        const key = `job:${j.id}`;
        live.add(key);
        const slot = rowSlot(key);
        const rts = (j.runtimes ?? []).join(' + ') || 'python';
        const isStopping = j.state === 'stopping';
        const parts = [{ strong: fmtElapsed(now - j.startedAt), rest: 'running' }];
        // What it is holding right now, which for these jobs is mostly model
        // weights. Left out for the beat where a read fails before the row
        // goes, rather than showing a zero that was never true.
        if (typeof j.memoryBytes === 'number' && j.memoryBytes > 0) {
          parts.push({ strong: fmtBytesFor(j.memoryBytes), rest: 'in memory' });
          if (ramTotal > 0) parts.push({ bar: (j.memoryBytes / ramTotal) * 100 });
        }
        // The runner's own answer where it has one: the request ("auto") is
        // not the truth on a machine that cannot run on a GPU at all.
        const device = j.resolvedDevice ?? j.device;
        parts.push(DEVICE_LABEL[device] ?? device, `via ${rts}`, `pid ${j.pid}`);
        fillRow(slot, {
          item: j,
          name: `${TASK_LABEL[j.kind] ?? 'Working'} · ${j.repo ?? j.script}`,
          metaParts: parts,
          action: isStopping
            ? null
            : {
                label: 'Stop',
                danger: true,
                title: `Stop this job (pid ${j.pid})`,
                run: () => stopJob(slot.item),
              },
        });
        desired.push(slot.row);
      }
    }

    // Rows for things that have gone leave the DOM and the keyed map both.
    for (const [key, slot] of rowSlots) {
      if (!live.has(key)) {
        rowSlots.delete(key);
        slot.row.remove();
      }
    }

    if (!desired.length) {
      emptyMsg.textContent =
        data.ollama && !data.ollama.ok
          ? `Nothing running. ${data.ollama.detail}`
          : 'Nothing running. Ollama keeps a model in memory for a few minutes after a reply, and local jobs appear here while they work.';
      desired.push(emptyMsg);
      procmanNote.textContent = '';
    } else if (jobs.some((j) => j.state === 'stopping')) {
      procmanNote.textContent = 'Stopping — it gets a moment to clean up first.';
    } else {
      // "0 models · 1 job" read as a contradiction: the count is only ever
      // *Ollama-resident* models, and a local job loads its own. Say which is
      // which instead, and say when the Ollama half could not be read at all.
      const bits = [];
      if (jobs.length) bits.push(`${jobs.length} job${jobs.length === 1 ? '' : 's'} running`);
      if (models.length) bits.push(`${models.length} Ollama model${models.length === 1 ? '' : 's'} loaded`);
      else if (data.ollama && !data.ollama.ok) bits.push('Ollama busy or not reachable');
      procmanNote.textContent = bits.join(' · ');
    }

    syncOrder(desired);
    procmanStopAll.hidden = running.length === 0;
  }

  async function pollProcesses() {
    if (procBusy) return;
    procBusy = true;
    try {
      // Bounded on purpose. Without a client-side deadline a single hung
      // request would leave procBusy set and the panel frozen on whatever it
      // last said — the "it just goes off until you close and reopen" report.
      const res = await fetch('/api/processes', { signal: AbortSignal.timeout(8000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      renderProcesses(await res.json());
    } catch (err) {
      // Keep the last list on screen: a hiccup reading the machine must not
      // wipe what it just showed, and closing/reopening is not a fix. The
      // footer says what happened; the next tick retries on its own.
      if (!(err instanceof Error && err.name === 'TimeoutError')) {
        procmanNote.textContent = "Can't read what's running — retrying…";
      } else {
        procmanNote.textContent = "Can't read what's running — the server is busy; retrying…";
      }
    } finally {
      procBusy = false;
    }
  }

  async function stopJob(job) {
    try {
      const res = await fetch(`/api/processes/${encodeURIComponent(job.id)}/stop`, { method: 'POST' });
      if (!res.ok) {
        const err = (await res.json().catch(() => null))?.error ?? `HTTP ${res.status}`;
        notifyError(err);
      } else {
        notifySuccess(`Stopped ${TASK_LABEL[job.kind] ?? 'the job'}.`);
      }
    } catch {
      notifyError('Could not reach the server to stop that job.');
    }
    void pollProcesses();
  }

  async function unloadOllamaModel(name) {
    try {
      const res = await fetch('/api/ollama/unload', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) {
        const err = (await res.json().catch(() => null))?.error ?? `HTTP ${res.status}`;
        notifyError(err);
      } else {
        notifySuccess(`Released ${name}.`);
      }
    } catch {
      notifyError('Could not reach the server to unload that model.');
    }
    void pollProcesses();
  }

  function setProcman(open) {
    procmanOpen = open;
    procmanTrigger.setAttribute('aria-expanded', String(open));
    if (open) {
      // Position fixed so the pop-out escapes any scrolling rail and can never
      // be clipped by it. It is pinned by its bottom edge, not its top: the
      // body fills in asynchronously, and a top-anchored box grows downward —
      // straight over the button it hangs from. Bottom-pinned, it grows up and
      // away, which is the motion the CSS animation already implies.
      procman.hidden = false;
      procman.style.visibility = 'hidden';
      procman.style.position = 'fixed';
      procman.style.top = '';
      procman.style.left = '';
      procman.style.bottom = '';
      procman.style.width = '';
      const rect = procmanTrigger.getBoundingClientRect();
      procman.style.bottom = `${Math.max(8, innerHeight - rect.top + 8)}px`;
      procman.style.width = `${Math.min(336, innerWidth - 16)}px`;
      const left = Math.min(Math.max(8, rect.left), innerWidth - procman.offsetWidth - 8);
      procman.style.left = `${left}px`;
      procman.style.visibility = '';
      void pollProcesses();
    } else {
      procman.hidden = true;
    }
  }

  procmanTrigger.addEventListener('click', () => setProcman(!procmanOpen));
  procmanClose?.addEventListener('click', () => setProcman(false));
  procmanStopAll?.addEventListener('click', async () => {
    // Two jobs is the ceiling this app ever runs, so there is no scarier case
    // than "both at once" — and the button only appears when there is one.
    const n = await fetch('/api/processes/stop-all', { method: 'POST' })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
    notifySuccess(n ? `Stopped ${n.stopped} job${n.stopped === 1 ? '' : 's'}.` : 'Nothing to stop.');
    void pollProcesses();
  });
  // A click anywhere else closes it. The pop-out has no backdrop, so without
  // this it would sit open over the conversation indefinitely.
  document.addEventListener('click', (ev) => {
    if (!procmanOpen) return;
    if (procman.contains(ev.target) || procmanTrigger.contains(ev.target)) return;
    setProcman(false);
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && procmanOpen) setProcman(false);
  });

  /* ---------------- Wiring ---------------- */

  function start() {
    if (resTimer) return;
    // One poll to get the first reading on screen. This used to be written as
    // `if (!sidebarIsDrawer()) poll(); poll();` — the guard was meant to skip
    // the first call on a drawer, and the unconditional one was meant to always
    // have one, but together they fired two polls on every desktop load.
    void pollResources();
    resTimer = setInterval(() => {
      // Polling a hidden tab is work for nobody.
      if (!document.hidden) void pollResources();
      // The manager only refreshes while it is open. It is a list of things the
      // user is about to act on, so a stale closed pop-out is worth nothing.
      if (procmanOpen && !document.hidden) void pollProcesses();
    }, RES_POLL_MS);
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) return;
    // Catch up immediately on return rather than waiting out the interval.
    void pollResources();
  });

  start();
  return { refresh: pollResources, start };
}
