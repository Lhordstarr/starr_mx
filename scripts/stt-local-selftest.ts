/**
 * The local transformers speech-to-text runner, against a real CTC model.
 *
 * Whisper is not the only speech-to-text architecture, and the runner is not a
 * Whisper wrapper — it is `pipeline("automatic-speech-recognition")`, which
 * loads wav2vec2, HuBERT, WavLM, MMS and the rest. The runner nevertheless
 * asked every model for Whisper-style timestamps:
 *
 *   ValueError: CTC can either predict character level timestamps, or word
 *   level timestamps. Set `return_timestamps='char'` or
 *   `return_timestamps='word'` as required.
 *
 * That is a hard failure before a single frame is read, so a perfectly good
 * wav2vec2 model could never transcribe. The checks below are split: the shape
 * of the request (which catches the regression without a 377 MB download) and
 * the real run (which proves the shape works against the model that used to
 * fail).
 *
 * Run with: npm run test:stt-local
 */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import * as path from 'node:path';

let passed = 0;
const failures: string[] = [];
function report(name: string, ok: boolean, detail = ''): void {
  if (ok) {
    passed += 1;
    console.log(`  PASS  ${name}`);
  } else {
    failures.push(name);
    console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

const script = path.resolve('scripts/py/transcribe.py');
const source = readFileSync(script, 'utf8');

console.log('The runner must ask each architecture for the timestamps it accepts');
report(
  'CTC models are asked for word timestamps, not Whisper segments',
  /ForCTC/.test(source) && /return_timestamps\s*=\s*["']word["']/.test(source),
  'the CTC branch is missing, so wav2vec2/HuBERT/MMS will raise',
);
report(
  'Whisper still gets its segment timestamps',
  /return_timestamps\s*=\s*True/.test(source),
);
report(
  'a tokenizer that cannot word-align falls back to a plain transcript',
  /except\s+ValueError/.test(source) && /timestamp/i.test(source),
  'losing word times is acceptable; losing the transcript is not',
);

console.log('\nAnd the model that used to fail must actually transcribe');
{
  const py = path.resolve('.runtimes/venv/bin/python');
  const modelDir = path.resolve('models/facebook--wav2vec2-base-960h');
  const config = path.join(modelDir, 'config.json');

  if (!existsSync(py) || !existsSync(config) || !existsSync(path.join(modelDir, 'model.safetensors'))) {
    console.log('  SKIP  no shared venv or wav2vec2 weights on this machine');
  } else {
    // The model has to be CTC for this to be the failing case rather than a
    // second Whisper test. If the fixture ever changes family, say so instead of
    // passing for the wrong reason.
    const parsed = JSON.parse(readFileSync(config, 'utf8')) as { architectures?: string[] };
    const arch = (parsed.architectures ?? []).join(', ');
    report('the fixture really is a CTC model', /ForCTC/i.test(arch), arch || '(no architectures)');

    // One second of a 440 Hz tone. The transcript will be empty — a tone has no
    // words — and that is fine. What is being checked is that the runner gets as
    // far as transcribing instead of dying on the request's shape.
    const outDir = path.resolve('.runtimes/out');
    mkdirSync(outDir, { recursive: true });
    const wav = path.join(outDir, 'stt-selftest-tone.wav');
    const rate = 16_000;
    const samples = rate;
    const body = Buffer.alloc(samples * 2);
    for (let i = 0; i < samples; i += 1) {
      const v = Math.round(Math.sin((2 * Math.PI * 440 * i) / rate) * 8000);
      body.writeInt16LE(v, i * 2);
    }
    const head = Buffer.alloc(44);
    head.write('RIFF', 0);
    head.writeUInt32LE(36 + body.length, 4);
    head.write('WAVEfmt ', 8);
    head.writeUInt32LE(16, 16);
    head.writeUInt16LE(1, 20);
    head.writeUInt16LE(1, 22);
    head.writeUInt32LE(rate, 24);
    head.writeUInt32LE(rate * 2, 28);
    head.writeUInt16LE(2, 32);
    head.writeUInt16LE(16, 34);
    head.write('data', 36);
    head.writeUInt32LE(body.length, 40);
    writeFileSync(wav, Buffer.concat([head, body]));

    const run = spawnSync(py, [script, modelDir, wav, 'cpu'], {
      encoding: 'utf8',
      timeout: 10 * 60_000,
    });
    const lines = String(run.stdout ?? '')
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    let result: { text?: unknown; error?: unknown } | null = null;
    for (const line of lines) {
      try {
        const doc = JSON.parse(line) as { stage?: unknown; text?: unknown; error?: unknown };
        if ('text' in doc || 'error' in doc) result = doc;
      } catch {
        /* progress */
      }
    }
    report('the runner exits cleanly', run.status === 0, `exit ${run.status}`);
    report(
      'no CTC timestamps error is printed',
      !/CTC can either predict/i.test(String(run.stdout) + String(run.stderr)),
      String(run.stdout).slice(-300),
    );
    report(
      'it returns a transcript field rather than an error',
      result !== null && typeof result.text === 'string' && !('error' in result!),
      JSON.stringify(result),
    );
  }
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);
