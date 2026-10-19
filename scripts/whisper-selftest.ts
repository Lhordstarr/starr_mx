#!/usr/bin/env tsx
/**
 * whisper.cpp checks.
 *
 * Most of these exist because of a specific way this went wrong the first
 * time, which is worth recording because none of it is visible from the code:
 *
 *  - whisper.cpp's loader reads the GGML magic and rejects GGUF with
 *    "invalid model data (bad magic)". The catalog had two entries labelled
 *    `runtime: 'whisper.cpp'` pointing at GGUF Whisper files, which are good
 *    models for *transcribe.cpp* and unreadable here. So the format is asserted
 *    against the catalog, not just in the classifier.
 *  - whisper-cli reads wav/flac/mp3/ogg and nothing else, while every browser
 *    recording is webm/opus or m4a. That gap is closed in the browser, and the
 *    browser half is checked in a real page with a real opus recording rather
 *    than by re-implementing the decode.
 *  - upstream publishes binaries only on rolling `bNNNN` releases; the `vX.Y.Z`
 *    tags are source-only. Pinning a tag would install nothing.
 */
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { CATALOG } from '../src/model-catalog.js';
import { RUNTIME_GROUPS } from '../src/runtimes.js';
import {
  WHISPER_BUILD,
  WHISPER_REPO,
  parseWhisperJson,
  whisperAsset,
} from '../src/whispercpp.js';
import { SCRIPT_RUNTIMES } from '../src/processes.js';

let passed = 0;
const failures: string[] = [];
function report(name: string, ok: boolean, detail = '') {
  if (ok) {
    passed += 1;
    console.log(`  PASS  ${name}`);
  } else {
    failures.push(name);
    console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

console.log('The pinned build');
report(
  'it is a rolling build number, not a release tag',
  /^b\d+$/.test(WHISPER_BUILD),
  `${WHISPER_BUILD} — the vX.Y.Z tags carry no binaries`,
);
report(
  'the pinned build actually publishes a Linux archive',
  Boolean(whisperAsset()) || process.platform !== 'linux',
  String(whisperAsset()?.name),
);
report(
  'the chosen asset is named for this machine',
  (() => {
    const a = whisperAsset();
    if (!a) return process.platform !== 'linux';
    return process.platform === 'linux'
      ? a.name.startsWith('whisper-bin-ubuntu-') && a.name.endsWith('.tar.gz')
      : a.name.endsWith('.zip');
  })(),
  String(whisperAsset()?.name),
);

console.log('\nThe model catalogue');
{
  const entries = CATALOG.flatMap((s) => s.entries).filter((e) => e.run?.requires?.includes('whisper-cpp'));
  report('there are entries for it', entries.length > 0, `${entries.length}`);
  report(
    'every one names a file whisper.cpp can open',
    entries.every((e) => (e.files ?? []).every((f) => f.name.endsWith('.bin') && f.name.startsWith('ggml-'))),
    entries.flatMap((e) => (e.files ?? []).map((f) => f.name)).filter((n) => !n.endsWith('.bin')).join(', '),
  );
  report(
    'and they all come from the repository whisper.cpp loads from',
    entries.every((e) => e.repo === WHISPER_REPO),
    entries.map((e) => e.repo).join(', '),
  );
  report(
    'no entry anywhere still claims a GGUF is a whisper.cpp model',
    !CATALOG.flatMap((s) => s.entries).some(
      (e) => e.runtime === 'whisper.cpp' && (e.files ?? []).some((f) => f.name.toLowerCase().endsWith('.gguf')),
    ),
    'a GGUF Whisper is a transcribe.cpp model',
  );
  report(
    'every entry has a run block, or it is invisible to every picker',
    entries.every((e) => e.run?.kind === 'stt'),
    entries.filter((e) => e.run?.kind !== 'stt').map((e) => e.repo).join(', '),
  );
  report(
    'and each entry lists exactly one file, because every listed file is required',
    entries.every((e) => (e.files ?? []).length === 1),
    entries.filter((e) => (e.files ?? []).length !== 1).map((e) => e.repo).join(', '),
  );
  report(
    'the sizes are real, not placeholders',
    entries.every((e) => (e.files ?? []).every((f) => typeof f.bytes === 'number' && f.bytes > 1024)),
  );
}

console.log('\nThe runtime group');
{
  const g = RUNTIME_GROUPS.find((x) => x.id === 'whisper-cpp');
  report('it exists', Boolean(g));
  report('it installs a pinned binary rather than a package', g?.installer === 'whisper-cpp', String(g?.installer));
  report('it brings no pip specs with it', (g?.pkgs ?? []).length === 0);
  report('it needs no torch', g?.needsTorch === false);
  report('it unlocks speech-to-text', g?.unlocks?.includes('speech-to-text'));
  // usesPython is private, so the property it guarantees is asserted from the
  // outside: a selection of only this group must not create a Python venv.
  // That is the whole point of an installer-backed group.
  report(
    'selecting only this group asks for no Python environment',
    !RUNTIME_GROUPS.filter(
      (x) => x.installer === 'whisper-cpp' || x.installer === 'llama-cpp' || x.installer === 'ollama',
    ).some((x) => x.pkgs.length || x.needsTorch),
  );
  report('the process map knows what the CLI is', SCRIPT_RUNTIMES['whisper-cli']?.includes('whisper-cpp') === true);
}

console.log('\nReading whisper.cpp output');
{
  // A real file, produced by whisper-cli 1.9.5 against ggml-tiny.en.
  const real = `{"systeminfo":"WHISPER : VITISAI = 0 | COREML = 0 | OPENVINO = 0 | CPU : SSE3 = 1","model":{"type":"tiny","multilingual":false},"params":{"model":"ggml-tiny.en.bin","language":"en"},"result":{"language":"en"},"transcription":[{"timestamps":{"from":"00:00:00,000","to":"00:00:07,940"},"offsets":{"from":0,"to":7940},"text":" And so my fellow Americans ask not what your country can do for you"},{"timestamps":{"from":"00:00:07,940","to":"00:00:10,760"},"offsets":{"from":7940,"to":10760},"text":" ask what you can do for your country."}]}`;
  const parsed = parseWhisperJson(real);
  report(
    'the transcript comes back as one line',
    parsed.text === 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country.',
    parsed.text,
  );
  report('the language is carried through', parsed.language === 'en', String(parsed.language));
  report('the duration is the last offset, in seconds', parsed.duration === 10.76, String(parsed.duration));
  report('offsets become seconds, not milliseconds', parsed.segments[0]?.end === 7.94, String(parsed.segments[0]?.end));
  report('both key names the browser reads are present', parsed.chunks.length === parsed.segments.length);
  report(
    'a segment without offsets does not produce NaN',
    parseWhisperJson('{"transcription":[{"text":"hi"}]}').segments[0]?.end === 0,
  );
  report('empty output is an empty transcript, not a crash', parseWhisperJson('{"transcription":[]}').text === '');
  let threw = false;
  try {
    parseWhisperJson('not json at all');
  } catch {
    threw = true;
  }
  report('unreadable output says so instead of returning nothing', threw);
  threw = false;
  try {
    parseWhisperJson('{}');
  } catch {
    threw = true;
  }
  report('a document with no transcription key is still a result, not a crash', !threw && parseWhisperJson('{}').text === '');
}

console.log('\nThe browser half: webm/opus to 16 kHz WAV');
{
  // The part that cannot be checked from Node: that a real recording a browser
  // produces survives the conversion and arrives as something whisper-cli opens.
  const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
  const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
  if (!browser) {
    console.log('  SKIP  no chromium');
  } else {
    const PROFILE = mkdtempSync(join(tmpdir(), 'whisper-audio-'));
    const PORT = Number(process.env.WHISPER_TEST_CDP_PORT ?? 9491);
    const chrome = spawn(
      browser,
      [
        '--headless=new',
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--autoplay-policy=no-user-gesture-required',
        '--window-size=900,700',
        `--remote-debugging-port=${PORT}`,
        `--user-data-dir=${PROFILE}`,
        'about:blank',
      ],
      { stdio: 'ignore' },
    );
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
    try {
      let ws = '';
      for (let i = 0; i < 60 && !ws; i += 1) {
        try {
          const l = (await (
            await fetch(`http://127.0.0.1:${PORT}/json/list`)
          ).json()) as Array<{ type: string; webSocketDebuggerUrl?: string }>;
          ws = l.find((t) => t.type === 'page')?.webSocketDebuggerUrl ?? '';
        } catch {
          await sleep(250);
        }
      }
      if (!ws) throw new Error('chromium never came up');
      const sock = new WebSocket(ws);
      await new Promise<void>((resolve, reject) => {
        sock.addEventListener('open', () => resolve());
        sock.addEventListener('error', () => reject(new Error('cdp socket failed')));
      });
      let cid = 0;
      const wait = new Map<number, (m: any) => void>();
      sock.addEventListener('message', (ev) => {
        const m = JSON.parse(String((ev as MessageEvent).data));
        if (m.id && wait.has(m.id)) {
          wait.get(m.id)!(m);
          wait.delete(m.id);
        }
      });
      const call = (method: string, params: Record<string, unknown> = {}) =>
        new Promise<any>((resolve) => {
          const n = ++cid;
          wait.set(n, resolve);
          sock.send(JSON.stringify({ id: n, method, params }));
        });
      const run = async (expression: string) => {
        const r = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
        if (r.result?.exceptionDetails) {
          throw new Error(String(r.result.exceptionDetails.exception?.description ?? 'eval failed').slice(0, 200));
        }
        return r.result?.result?.value;
      };

      await call('Page.enable');
      await call('Runtime.enable');
      // Served from the repo so the page is same-origin with /audio.js.
      await call('Page.navigate', { url: 'about:blank' });
      await sleep(300);
      await call('Page.navigate', { url: 'http://localhost:3000/' });
      await sleep(2500);

      const out = JSON.parse(
        String(
          await run(`(async () => {
            const { audioForUpload, needsDecoding } = await import('/audio.js');
            const results = {};

            // A recording made the way the app makes one: an oscillator captured
            // through MediaRecorder, so the blob really is webm/opus.
            const ctx = new AudioContext();
            results.ctxState0 = ctx.state;
            // A context created without a gesture starts suspended in headless,
            // and a suspended graph records nothing — an empty blob that looks
            // like a decoder failure and is not one.
            await ctx.resume().catch(() => undefined);
            results.ctxState1 = ctx.state;
            const osc = ctx.createOscillator();
            const dest = ctx.createMediaStreamDestination();
            osc.connect(dest);
            osc.start();
            const rec = new MediaRecorder(dest.stream, { mimeType: 'audio/webm;codecs=opus' });
            const chunks = [];
            rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
            const stopped = new Promise((r) => { rec.onstop = r; });
            rec.start();
            await new Promise((r) => setTimeout(r, 700));
            rec.stop();
            await stopped;
            osc.stop();
            await ctx.close();
            const recorded = new Blob(chunks, { type: 'audio/webm;codecs=opus' });

            results.recordedType = recorded.type;
            results.recordedBytes = recorded.size;
            results.wantedDecode = needsDecoding(recorded, 'whisper-cpp');
            results.wantedSkip = needsDecoding(recorded, 'faster-whisper');

            // A control, not the thing under test: decodeAudioData is the
            // browser's own and is expected to work. It is measured so that a
            // failure below can be attributed to the conversion rather than to
            // the browser quietly refusing to decode anything.
            const probeCtx = new AudioContext({ sampleRate: 16000 });
            const probe = await probeCtx.decodeAudioData(await recorded.arrayBuffer());
            results.probeRate = probe.sampleRate;
            results.probeFrames = probe.length;
            results.probeSeconds = Math.round((probe.duration || 0) * 1000);
            await probeCtx.close();

            const wav = await audioForUpload(recorded, 'whisper-cpp');
            results.wavType = wav.type;
            const bytes = new Uint8Array(await wav.arrayBuffer());
            const dv = new DataView(bytes.buffer);
            const tag = (o, n) => String.fromCharCode(...bytes.slice(o, o + n));
            results.riff = tag(0, 4);
            results.wave = tag(8, 4);
            results.channels = dv.getUint16(22, true);
            results.rate = dv.getUint32(24, true);
            results.bits = dv.getUint16(34, true);
            results.dataBytes = dv.getUint32(40, true);
            results.pcmBytes = bytes.length - 44;
            results.frameMs = Math.round((results.dataBytes / 2 / results.rate) * 1000);

            // The two Python runners must be handed the original bytes.
            const passthrough = await audioForUpload(recorded, 'faster-whisper');
            results.passthroughSameSize = passthrough.size === recorded.size;
            results.passthroughType = passthrough.type;

            // A format whisper-cli already reads must not be re-encoded.
            const already = new Blob([new Uint8Array(64)], { type: 'audio/wav' });
            const kept = await audioForUpload(already, 'whisper-cpp');
            results.wavUntouched = kept.size === already.size && kept.type === 'audio/wav';
            return JSON.stringify(results);
          })()`),
        ),
      );

      report('a MediaRecorder blob really is webm/opus', out.recordedType.includes('webm'), out.recordedType);
      report('and it has content', out.recordedBytes > 1000, String(out.recordedBytes));
      report('whisper.cpp asks for it to be decoded', out.wantedDecode === true);
      report('the Python runners do not', out.wantedSkip === false);
      report('the result is a WAV', out.wavType === 'audio/wav' && out.riff === 'RIFF' && out.wave === 'WAVE', out.wavType);
      report('at the rate whisper.cpp resamples to', out.rate === 16000, String(out.rate));
      report('mono', out.channels === 1, String(out.channels));
      report('16-bit', out.bits === 16, String(out.bits));
      report('the data chunk length matches the bytes', out.dataBytes === out.pcmBytes, `${out.dataBytes} vs ${out.pcmBytes}`);
      report(
        'and the length is the length that was recorded',
        Math.abs(out.frameMs - out.probeSeconds) < 100,
        `${out.frameMs}ms against a decoded ${out.probeSeconds}ms`,
      );
      report('the Python runners receive the original bytes untouched', out.passthroughSameSize === true && out.passthroughType === out.recordedType);
      report('a wav is not re-encoded', out.wavUntouched === true);

      sock.close();
    } catch (err) {
      report('the browser conversion ran', false, (err as Error).message);
    } finally {
      try {
        chrome.kill('SIGKILL');
      } catch {
        /* already gone */
      }
      rmSync(PROFILE, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    }
  }
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);