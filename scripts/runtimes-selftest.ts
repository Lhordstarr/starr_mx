#!/usr/bin/env tsx
/**
 * Runtime install checks: the venv failure text, Ollama's per-OS install, the
 * on-demand password prompt, and the browser half of the password contract.
 *
 * Three things are being asserted here, and they are the same three that broke
 * on a fresh machine:
 *
 *  - a Python without the venv module says which package to install, instead of
 *    relaying "ensurepip is not available";
 *  - Ollama has an install command on every supported OS rather than a
 *    Linux-only dead end, and only the one that needs root says so;
 *  - the password box appears when an installer asks for one, and not before.
 *
 * The server needs to be up for the HTTP and browser sections
 * (`npm run dev`); those skip rather than fail without it. The unit checks
 * always run.
 */
import { spawn, spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { nativeBackendReports } from '../src/backends.ts';
import {
  describeVenvFailure,
  downloadTool,
  ensureVenv,
  installRuntimes,
  invalidateRuntimeProbes,
  missingInstallerPrerequisite,
  MISSING_PIP_MESSAGE,
  MissingVenvPackage,
  needsPrivilege,
  ollamaCommand,
  ollamaRecipe,
  llamaCppUpgradeKind,
  ollamaVulkanEnabled,
  parseComputeCapability,
  computeCapabilitySupportsLlamaCuda,
  permissionMessage,
  RUNTIME_GROUPS,
  runtimeStatus,
  unwritableAncestor,
  venvHasPip,
  venvPython,
} from '../src/runtimes.ts';

/**
 * The runtime status, read with a retry.
 *
 * The page under test asks for this endpoint on every tab switch, and a test
 * that reads it as well can find the endpoint answering "slow down" — which
 * is the limiter doing its job and says nothing about the page.
 */
async function readStatus(): Promise<any> {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const res = await fetch(`${APP_URL}/api/runtimes`, { signal: AbortSignal.timeout(60000) });
    if (res.ok) return await res.json();
    await new Promise((r) => setTimeout(r, 1500));
  }
  throw new Error('the runtime status could not be read');
}

const APP_URL = process.env.APP_URL ?? 'http://localhost:3000';
const PORT = Number(process.env.CDP_PORT ?? 9381);

let passed = 0;
const failures: string[] = [];
function report(name: string, ok: boolean, extra = '') {
  if (ok) {
    passed += 1;
    console.log(`  PASS  ${name}`);
  } else {
    failures.push(name);
    console.log(`  FAIL  ${name}${extra ? ` — ${extra}` : ''}`);
  }
}
function section(title: string) {
  console.log(`\n${title}`);
}

/* ---- 1. The venv failure names the package, not ensurepip ---- */
section('Python environment errors');
{
  // What a fresh Ubuntu actually prints.
  const ubuntu = describeVenvFailure(
    'The virtual environment was not created successfully because the ensurepip module is not available.',
  );
  report(
    'an ensurepip failure names the package to install',
    /python3-venv/.test(ubuntu) && !/ensurepip/.test(ubuntu),
    ubuntu,
  );
  report('the Ubuntu hint says apt', /sudo apt install python3-venv/.test(ubuntu), ubuntu);

  // The bug this log exposed: `python3-venv` is a metapackage that only covers
  // the distro's *default* Python. When the interpreter in play is not that one,
  // following the metapackage advice installs cleanly and fails identically
  // again, which reads as the advice being wrong.
  const versioned = describeVenvFailure(
    'The virtual environment was not created successfully because the ensurepip module is not available.',
    { version: '3.12' },
  );
  report(
    'the package matches the interpreter that was used',
    /sudo apt install python3\.12-venv/.test(versioned),
    versioned,
  );
  report(
    'and it says why the metapackage is not enough',
    /does not cover this interpreter/.test(versioned) && /3\.12/.test(versioned),
    versioned,
  );
  report(
    'a versioned hint does not leave the placeholder in it',
    !/python3\.<version>/.test(versioned),
    versioned,
  );
  report(
    'the version is not invented when it is unknown',
    /python3\.<version>-venv/.test(describeVenvFailure('ensurepip is not available')),
  );

  const noVenv = describeVenvFailure('No module named venv');
  report('a missing venv module also names the package', /python3-venv/.test(noVenv), noVenv);

  // The same cause as a bare traceback rather than prose.
  const traceback = describeVenvFailure("ModuleNotFoundError: No module named 'ensurepip'");
  report(
    'a traceback-shaped ensurepip failure is recognised too',
    /python3-venv/.test(traceback) && !/ModuleNotFoundError/.test(traceback),
    traceback,
  );

  // Anything unrecognised keeps the interpreter's own words: a guess here
  // would be worse than the truth. A full disk used to be the example here and
  // is now diagnosed, so this is something no branch claims to understand.
  const odd = describeVenvFailure(
    "ValueError: Refusing to create a venv in '/x' because it contains the PATH separator ':'.",
  );
  report(
    'an unrecognised failure keeps the raw message',
    odd.includes('PATH separator'),
    odd,
  );
  report(
    'but it is no longer the end of the sentence',
    /RUNTIMES_DIR/.test(odd) && /RUNTIMES_PYTHON/.test(odd),
    odd,
  );
  report(
    'a full disk is diagnosed as a full disk',
    /free disk space/i.test(describeVenvFailure('OSError: [Errno 28] No space left on device')),
    describeVenvFailure('OSError: [Errno 28] No space left on device'),
  );
  report(
    'PEP 668 is named as PEP 668',
    /PEP 668/.test(describeVenvFailure('error: externally-managed-environment')),
    describeVenvFailure('error: externally-managed-environment'),
  );
  const symlink = describeVenvFailure(
    "OSError: [Errno 18] Invalid cross-device link: '/x/bin/python3' -> '/y/bin/python'",
  );
  report(
    'a filesystem that cannot symlink says so, and names the fix',
    /symbolic links/i.test(symlink) && /RUNTIMES_DIR/.test(symlink),
    symlink,
  );
  report('an empty failure does not become an empty error', describeVenvFailure('').length > 0);

  // "Does this need sudo?" — the answer is almost always no, and the fix for a
  // root-owned folder is to hand it back, never to build the next one as root.
  const rootOwned = permissionMessage('/home/u/app/.runtimes', 0);
  report(
    'a root-owned folder is named as root-owned',
    /owned by root/.test(rootOwned),
    rootOwned,
  );
  report(
    'it says the environment does not need root',
    /does not need root/i.test(rootOwned),
    rootOwned,
  );
  report(
    'the command given gives the folder back rather than building as root',
    /sudo chown -R/.test(rootOwned) && !/sudo (?!chown)/.test(rootOwned.replace(/sudo chown -R [^\s]+\s*/, '')),
    rootOwned,
  );
  report(
    'it warns that running as root repeats the failure',
    /again/i.test(rootOwned),
    rootOwned,
  );
  const otherOwned = permissionMessage('/srv/shared', 1001);
  report(
    'a folder owned by another user is not called root',
    /owned by uid 1001/.test(otherOwned) && !/owned by root/.test(otherOwned),
    otherOwned,
  );

  // The walk-up, which is the half that matters: the leaf usually does not
  // exist yet, so the unwritable folder is its parent.
  const locked = mkdtempSync(join(tmpdir(), 'starr-perm-'));
  report('a writable folder reports no problem', unwritableAncestor(locked) === null);
  chmodSync(locked, 0o500);
  report(
    'an unwritable folder is found',
    unwritableAncestor(locked)?.path === locked,
    JSON.stringify(unwritableAncestor(locked)),
  );
  report(
    'a missing leaf inside it resolves to the unwritable parent',
    unwritableAncestor(join(locked, '.runtimes'))?.path === locked,
    JSON.stringify(unwritableAncestor(join(locked, '.runtimes'))),
  );
  report(
    'the owner of an unwritable folder is reported',
    typeof unwritableAncestor(locked)?.uid === 'number',
    JSON.stringify(unwritableAncestor(locked)?.uid),
  );
  chmodSync(locked, 0o700);
  rmSync(locked, { recursive: true, force: true });

  // This is the exact failure a real Ubuntu box produced: the interpreter exited
  // non-zero and wrote nothing at all, so the message was the advice tail with no
  // reason in front of it.
  const mute = describeVenvFailure('');
  report(
    'a silent interpreter is reported as silent, not as a mystery',
    /without saying why/i.test(mute) && !/^python -m venv failed: $/.test(mute),
    mute,
  );
  report(
    'a silent interpreter is told how to see the real error',
    /python3 -m venv/.test(mute),
    mute,
  );
  // A process killed by a signal reports code === null, which used to be
  // indistinguishable from a clean non-zero exit.
  const killed = describeVenvFailure('', {
    killed: 'The interpreter was killed by SIGKILL before it could report anything.',
  });
  report(
    'a killed interpreter is named as killed',
    /SIGKILL/.test(killed) && /out-of-memory/i.test(killed),
    killed,
  );
  report(
    'a killed interpreter is not blamed on the filesystem',
    !/RUNTIMES_DIR/.test(killed),
    killed,
  );
  const term = describeVenvFailure('', {
    killed: 'The interpreter was killed by SIGTERM before it could report anything.',
  });
  report('a SIGTERM death is not called the out-of-memory killer', !/out-of-memory/i.test(term), term);
}

/* ---- 1c. The one failure with a scripted fix ---- */
section('Installing the missing venv package');
{
  const pkg = new MissingVenvPackage('3.12');
  report(
    'the command names the package for that interpreter',
    pkg.command === 'sudo apt-get install -y python3.12-venv',
    pkg.command,
  );
  report('the command asks for no confirmation twice', /install -y/.test(pkg.command), pkg.command);

  /* End to end against a stand-in interpreter: the venv module present, the
     ensurepip module absent. This is what a fresh Ubuntu without python3-venv
     looks like from here, and it is the failure a real box produced. */
  const home = mkdtempSync(join(tmpdir(), 'starr-missing-'));
  const fake = join(home, 'python3');
  writeFileSync(
    fake,
    [
      '#!/bin/sh',
      'case "$*" in',
      '  *"-c import sys;print"*) echo "3.12"; exit 0 ;;',
      '  *"-m venv"*)',
      '    echo "The virtual environment was not created successfully because the ensurepip module is not available." >&2',
      '    exit 1 ;;',
      'esac',
      'exit 0',
      '',
    ].join('\n'),
  );
  chmodSync(fake, 0o755);

  const saved = { dir: process.env.RUNTIMES_DIR, py: process.env.RUNTIMES_PYTHON };
  process.env.RUNTIMES_PYTHON = fake;
  process.env.RUNTIMES_DIR = join(home, '.runtimes');

  const asked: string[] = [];
  let caught = '';
  try {
    await installRuntimes({
      groups: [{ id: 'onnxruntime', pkgs: ['onnxruntime'] }] as never,
      device: 'cpu',
      onProgress: () => {},
      onNeedsPrivilege: (c: string) => asked.push(c),
    });
  } catch (err) {
    caught = (err as Error).message;
  }
  report(
    'a missing venv module asks for the package to be installed',
    asked.length === 1 && asked[0] === 'sudo apt-get install -y python3.12-venv',
    JSON.stringify(asked),
  );
  report(
    'and the failure names the same command',
    caught.includes('sudo apt-get install -y python3.12-venv'),
    caught,
  );
  report(
    'it does not claim the environment was built',
    !/installed\.$/i.test(caught) && /missing the venv module/.test(caught),
    caught,
  );
  report(
    'the versioned package is used, not the metapackage',
    !/install -y python3-venv\b/.test(caught),
    caught,
  );

  // Restored before the fixtures are torn down, so a failure above cannot leave
  // the real environment pointed at a throwaway interpreter.
  if (saved.dir === undefined) delete process.env.RUNTIMES_DIR;
  else process.env.RUNTIMES_DIR = saved.dir;
  if (saved.py === undefined) delete process.env.RUNTIMES_PYTHON;
  else process.env.RUNTIMES_PYTHON = saved.py;
  rmSync(home, { recursive: true, force: true });
  report('the real environment is untouched by the fixture', process.env.RUNTIMES_PYTHON === saved.py);
}

/* ---- 1d. Installing Ollama must not need Python at all ---- */
section('A vendor installer never builds a Python environment');
{
  // A machine with no python3-venv, and a group that installs a vendored binary.
  // The shared environment used to be created before the selection was looked
  // at, so this died in a Python error with no Ollama anywhere in it.
  const home = mkdtempSync(join(tmpdir(), 'starr-novenv-'));
  const badPy = join(home, 'python3');
  writeFileSync(
    badPy,
    [
      '#!/bin/sh',
      'case "$*" in',
      '  *"-c import sys;print"*) echo "3.13"; exit 0 ;;',
      '  *"-m venv"*) echo "No module named venv" >&2; exit 1 ;;',
      '  *) exit 0 ;;',
      'esac',
      '',
    ].join('\n'),
  );
  chmodSync(badPy, 0o755);

  const saved = { dir: process.env.RUNTIMES_DIR, py: process.env.RUNTIMES_PYTHON };
  process.env.RUNTIMES_PYTHON = badPy;
  process.env.RUNTIMES_DIR = join(home, '.runtimes');

  const said: string[] = [];
  const asked: string[] = [];
  let caught = '';
  try {
    await installRuntimes({
      groups: [{ id: 'ollama', installer: 'ollama', pkgs: [] }] as never,
      device: 'cpu',
      onProgress: (l: string) => said.push(l),
      onNeedsPrivilege: (c: string) => asked.push(c),
    });
  } catch (err) {
    caught = (err as Error).message;
  }
  report(
    'installing Ollama never builds a Python environment',
    !said.some((l) => /Creating the Python environment/.test(l)),
    said.join(' | '),
  );
  report(
    'so a Python without the venv module cannot stop it',
    !/venv/i.test(caught),
    caught,
  );
  report(
    'and it never asks for a password for a package it did not need',
    asked.every((c) => !/python3/.test(c)),
    JSON.stringify(asked),
  );
  report('it actually reaches the Ollama installer', said.some((l) => /Ollama/i.test(l)), said.join(' | '));

  // The other direction: a group that genuinely goes through pip must still get
  // its environment, or this would just be a quieter way of breaking installs.
  const pipLines: string[] = [];
  try {
    await installRuntimes({
      groups: [{ id: 'onnxruntime', pkgs: ['onnxruntime'] }] as never,
      device: 'cpu',
      onProgress: (l: string) => pipLines.push(l),
    });
  } catch {
    /* it is meant to fail here — this Python has no venv module */
  }
  report(
    'a pip group still builds its environment',
    pipLines.some((l) => /Creating the Python environment/.test(l)),
    pipLines.join(' | '),
  );

  if (saved.dir === undefined) delete process.env.RUNTIMES_DIR;
  else process.env.RUNTIMES_DIR = saved.dir;
  if (saved.py === undefined) delete process.env.RUNTIMES_PYTHON;
  else process.env.RUNTIMES_PYTHON = saved.py;
  rmSync(home, { recursive: true, force: true });
}

/* ---- 1e. No Python environment is not the same as nothing installed ---- */
section('A vendored runtime is reported without a Python environment');
{
  const home = mkdtempSync(join(tmpdir(), 'starr-novenv2-'));
  const rt = join(home, '.runtimes');
  const badPy = join(home, 'python3');
  writeFileSync(badPy, '#!/bin/sh\nexit 1\n');
  chmodSync(badPy, 0o755);

  // A llama.cpp install, laid out the way its own installer leaves it.
  mkdirSync(join(rt, 'llama-b99999', 'bin'), { recursive: true });
  writeFileSync(join(rt, 'llama-b99999', 'bin', 'llama-server'), 'x');
  mkdirSync(join(rt, 'llama.cpp'), { recursive: true });
  writeFileSync(join(rt, 'llama.cpp', 'build.txt'), 'llama-b99999/bin');

  const saved = { dir: process.env.RUNTIMES_DIR, py: process.env.RUNTIMES_PYTHON };
  process.env.RUNTIMES_PYTHON = badPy;
  process.env.RUNTIMES_DIR = rt;

  const status = await runtimeStatus(false);
  report('it admits there is no Python environment', status.venv === false && status.python === null);
  report(
    'but still reports the vendored runtime as installed',
    status.installed['llama-cpp'] === 'installed',
    JSON.stringify(status.installed),
  );
  report(
    'and still reports the pip runtimes as absent',
    status.installed.onnxruntime === null && status.installed.torch === null,
    JSON.stringify(status.installed),
  );
  report(
    'every group is present in the map, not just the ones with a venv',
    Object.keys(status.installed).length === RUNTIME_GROUPS.length,
    `${Object.keys(status.installed).length} of ${RUNTIME_GROUPS.length}`,
  );

  if (saved.dir === undefined) delete process.env.RUNTIMES_DIR;
  else process.env.RUNTIMES_DIR = saved.dir;
  if (saved.py === undefined) delete process.env.RUNTIMES_PYTHON;
  else process.env.RUNTIMES_PYTHON = saved.py;
  rmSync(home, { recursive: true, force: true });
}

/* ---- 1f. An installed package that cannot import is not installed ---- */
section('ONNX Runtime is installed only when it can import');
{
  const home = mkdtempSync(join(tmpdir(), 'starr-onnx-probe-'));
  const py = join(home, '.runtimes', 'venv', 'bin', 'python');
  mkdirSync(join(home, '.runtimes', 'venv', 'bin'), { recursive: true });
  const saved = { dir: process.env.RUNTIMES_DIR, py: process.env.RUNTIMES_PYTHON };
  process.env.RUNTIMES_DIR = join(home, '.runtimes');
  delete process.env.RUNTIMES_PYTHON;

  const writeProbe = (providers: string[]) => {
    writeFileSync(
      py,
      [
        '#!/bin/sh',
        'if [ "$1" = "-c" ]; then',
        `  echo '${JSON.stringify({
          packages: { onnxruntime: true },
          torch: null,
          onnx_providers: providers,
          ctranslate2_cuda: 0,
        })}'`,
        '  exit 0',
        'fi',
        'exit 1',
        '',
      ].join('\n'),
    );
    chmodSync(py, 0o755);
    invalidateRuntimeProbes();
  };

  // This is the state a GPU install left behind: `find_spec` found the
  // namespace directory, but importing it produced no execution providers at
  // all. The old code called that installed; the model then failed to load.
  writeProbe([]);
  const broken = await runtimeStatus(false);
  report(
    'an ONNX package with no execution providers is not installed',
    broken.installed.onnxruntime === null,
    JSON.stringify({ installed: broken.installed.onnxruntime, onnx: broken.onnx }),
  );

  writeProbe(['CPUExecutionProvider']);
  const working = await runtimeStatus(false);
  report(
    'an ONNX package that imports with a CPU provider is installed',
    working.installed.onnxruntime === 'installed',
    JSON.stringify({ installed: working.installed.onnxruntime, onnx: working.onnx }),
  );
  report(
    'and the provider is reported, not just the package name',
    working.onnx.some((o) => o.venv === 'venv' && o.providers.includes('CPUExecutionProvider')),
    JSON.stringify(working.onnx),
  );

  if (saved.dir === undefined) delete process.env.RUNTIMES_DIR;
  else process.env.RUNTIMES_DIR = saved.dir;
  if (saved.py === undefined) delete process.env.RUNTIMES_PYTHON;
  else process.env.RUNTIMES_PYTHON = saved.py;
  invalidateRuntimeProbes();
  rmSync(home, { recursive: true, force: true });
}

/* ---- 1g. Native runners are reported even when idle ---- */
section('Every native backend reports its own state');
{
  const rows = nativeBackendReports({
    onnx: [
      { venv: 'venv', label: 'shared', providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'] },
      { venv: 'venv-kokoro', label: 'Kokoro', providers: ['CPUExecutionProvider'] },
    ],
    llama: { installed: true, running: false, model: null, pid: null, kind: 'cuda', cudaCompute: 12, cudaUsable: true },
    transformers: {
      installed: true,
      running: false,
      model: null,
      pid: null,
      cuda: true,
      deviceName: 'Tesla T4',
    },
    sd: { installed: true, kind: 'cpu', activeDevice: null, activePid: null },
    whisper: { installed: true, device: 'cpu' },
    fasterWhisper: { installed: true, cudaDevices: 1 },
  });
  const byId = new Map(rows.map((r) => [r.id, r]));
  report('an idle native runtime still gets a row', rows.length === 7, JSON.stringify([...byId.keys()]));
  report(
    'ONNX reports the provider it actually loaded',
    byId.get('onnxruntime')?.device === 'gpu' && /CUDA/.test(byId.get('onnxruntime')?.detail ?? ''),
    JSON.stringify(byId.get('onnxruntime')),
  );
  report(
    'the voice environment is reported separately from the shared one',
    byId.get('kokoro-onnx')?.device === 'cpu',
    JSON.stringify(byId.get('kokoro-onnx')),
  );
  report(
    'an idle GPU llama.cpp build says why there is nothing to unload',
    byId.get('llamacpp')?.device === 'none' && /idle/.test(byId.get('llamacpp')?.detail ?? ''),
    JSON.stringify(byId.get('llamacpp')),
  );
  report(
    'Stable Diffusion on the CPU is not hidden behind the Python row',
    byId.get('stable-diffusion-cpp')?.device === 'cpu' &&
      /no Linux CUDA archive/.test(byId.get('stable-diffusion-cpp')?.readonlyReason ?? ''),
    JSON.stringify(byId.get('stable-diffusion-cpp')),
  );
  report(
    'CTranslate2 CUDA is reported independently of torch',
    byId.get('faster-whisper')?.device === 'gpu',
    JSON.stringify(byId.get('faster-whisper')),
  );

  const broken = nativeBackendReports({
    onnx: [],
    llama: { installed: true, running: false, model: null, pid: null, kind: 'cuda', cudaCompute: 7.5, cudaUsable: false },
    transformers: { installed: false, running: false, model: null, pid: null, cuda: false, deviceName: null },
    sd: { installed: false, kind: null, activeDevice: null, activePid: null },
    whisper: { installed: false, device: 'cpu' },
    fasterWhisper: { installed: false, cudaDevices: null },
  }).find((r) => r.id === 'llamacpp');
  report(
    'an installed CUDA build with no kernels is not called GPU',
    broken?.device === 'cpu' && /Blackwell/.test(broken.readonlyReason ?? ''),
    JSON.stringify(broken),
  );

  const fallback = nativeBackendReports({
    onnx: [],
    llama: { installed: true, running: false, model: null, pid: null, kind: 'cpu', cudaCompute: 7.5, cudaUsable: false },
    transformers: { installed: false, running: false, model: null, pid: null, cuda: false, deviceName: null },
    sd: { installed: false, kind: null, activeDevice: null, activePid: null },
    whisper: { installed: false, device: 'cpu' },
    fasterWhisper: { installed: false, cudaDevices: null },
  }).find((r) => r.id === 'llamacpp');
  report(
    'the CPU fallback does not tell a T4 owner to reinstall for a GPU that cannot run',
    /source build/.test(fallback?.readonlyReason ?? '') && !/Reinstall .*Compute set to GPU/.test(fallback?.readonlyReason ?? ''),
    JSON.stringify(fallback),
  );

  const active = nativeBackendReports({
    onnx: [],
    llama: { installed: false, running: false, model: null, pid: null, kind: null, cudaCompute: null, cudaUsable: false },
    transformers: { installed: false, running: false, model: null, pid: null, cuda: false, deviceName: null },
    sd: { installed: true, kind: 'vulkan', activeDevice: 'Vulkan0', activePid: 4321 },
    whisper: { installed: false, device: 'cpu' },
    fasterWhisper: { installed: false, cudaDevices: null },
  }).find((r) => r.id === 'stable-diffusion-cpp');
  report(
    'an active SD job is attributed to its device and pid',
    active?.device === 'gpu' && /Vulkan0/.test(active.detail) && /4321/.test(active.evidence ?? ''),
    JSON.stringify(active),
  );
}

/* ---- 1h. A CUDA archive must match the card ---- */
section('The pinned CUDA build is not offered to a card it cannot run');
{
  report('a T4\'s compute capability is parsed', parseComputeCapability('7.5\n') === 7.5);
  report('a trimmed capability is accepted', parseComputeCapability(' 8.6 ') === 8.6);
  report('an unreadable capability is not guessed', parseComputeCapability('') === null);
  report('a non-numeric capability is not guessed', parseComputeCapability('N/A') === null);
  report(
    'the Blackwell-only archive is rejected on compute 7.5',
    !computeCapabilitySupportsLlamaCuda(7.5),
  );
  report(
    'the Blackwell-only archive is rejected on compute 8.6',
    !computeCapabilitySupportsLlamaCuda(8.6),
  );
  report('the Blackwell archive is accepted on compute 12.0', computeCapabilitySupportsLlamaCuda(12));
  report('an unreadable card falls back rather than guessing CUDA', !computeCapabilitySupportsLlamaCuda(null));
}

/* ---- 1i. A CUDA build the card cannot run must be swapped out ---- */
section('An installed CUDA build is replaced when the card cannot run it');
{
  const home = mkdtempSync(join(tmpdir(), 'starr-t4-'));
  const rt = join(home, '.runtimes');
  const bin = join(home, 'bin');
  mkdirSync(join(rt, 'llama-b11706'), { recursive: true });
  mkdirSync(join(rt, 'llama.cpp'), { recursive: true });
  mkdirSync(bin, { recursive: true });
  writeFileSync(join(rt, 'llama-b11706', 'llama-server'), '');
  writeFileSync(join(rt, 'llama-b11706', 'libggml-cuda.so'), '');
  writeFileSync(join(rt, 'llama.cpp', 'build.txt'), 'llama-b11706');
  // A T4, as nvidia-smi describes one. No vulkaninfo is on this PATH, so the
  // only GPU build that remains is the CPU one.
  writeFileSync(
    join(bin, 'nvidia-smi'),
    [
      '#!/bin/sh',
      'case "$*" in',
      '  *compute_cap*) echo "7.5" ;;',
      '  *-L*) echo "GPU 0: Tesla T4 (UUID: GPU-test)" ;;',
      'esac',
      'exit 0',
      '',
    ].join('\n'),
  );
  chmodSync(join(bin, 'nvidia-smi'), 0o755);

  const savedDir = process.env.RUNTIMES_DIR;
  const savedPath = process.env.PATH;
  process.env.RUNTIMES_DIR = rt;
  process.env.PATH = bin;
  try {
    const kind = await llamaCppUpgradeKind();
    report(
      'a CUDA build on a T4 is replaced by the CPU build',
      kind === 'cpu',
      String(kind),
    );
  } finally {
    if (savedDir === undefined) delete process.env.RUNTIMES_DIR;
    else process.env.RUNTIMES_DIR = savedDir;
    if (savedPath === undefined) delete process.env.PATH;
    else process.env.PATH = savedPath;
    rmSync(home, { recursive: true, force: true });
  }
}

/* ---- 1b. An environment with no pip in it ---- */
section('An environment without pip');
{
  // The failure from the field: a venv whose interpreter exists, so every
  // check passed and it was reported as reusable — and then every install
  // failed with "No module named pip" and no advice.
  report(
    'a good environment reports that it has pip',
    await venvHasPip('venv'),
  );
  report(
    'the message names the package to install',
    /python3-venv/.test(MISSING_PIP_MESSAGE) && /no pip/i.test(MISSING_PIP_MESSAGE),
    MISSING_PIP_MESSAGE,
  );

  /* Both halves are exercised on throwaway environments, so nothing here can
     touch the one the app is actually using. */
  const saved = { dir: process.env.RUNTIMES_DIR, py: process.env.RUNTIMES_PYTHON };
  const real = venvPython();

  // a. Repairable: a real venv built without pip. ensurepip is still importable
  //    from the interpreter, so pip comes back in place and the directory — and
  //    anything in it — survives.
  const repairable = mkdtempSync(join(tmpdir(), 'starr-nopip-'));
  const built = spawnSync(real, ['-m', 'venv', '--without-pip', join(repairable, 'venv')], {
    stdio: 'ignore',
  });
  report(
    'a pip-less environment can be built to test against',
    built.status === 0 && existsSync(join(repairable, 'venv', 'bin', 'python')),
    `exit ${built.status}`,
  );
  process.env.RUNTIMES_DIR = repairable;
  process.env.RUNTIMES_PYTHON = real;
  report('the test environment really has no pip', (await venvHasPip('venv')) === false);
  const said: string[] = [];
  let repairError = '';
  try {
    await ensureVenv('venv', undefined, (line) => said.push(line));
  } catch (err) {
    repairError = (err as Error).message;
  }
  report('an environment with no pip is repaired instead of reused', repairError === '', repairError);
  report('the repaired environment can run pip', await venvHasPip('venv'));
  report(
    'the repair never claims the environment was reusable',
    !said.includes('Reusing the existing environment.'),
    said.join(' | '),
  );
  report(
    'the repair explains itself',
    said.some((line) => /no pip/i.test(line)) && said.some((line) => /pip is back/i.test(line)),
    said.join(' | '),
  );

  // b. Unrepairable: an interpreter with neither pip nor ensurepip, which is
  //    what a stripped python3 looks like. No amount of rebuilding helps, so
  //    the only honest answer is the one that names the package to install.
  const doomed = mkdtempSync(join(tmpdir(), 'starr-nopip2-'));
  const inner = join(doomed, 'venv-bin-python');
  writeFileSync(
    inner,
    [
      '#!/bin/sh',
      'case "$1 $2" in',
      '  "-m pip") echo "No module named pip" >&2; exit 1 ;;',
      '  "-m ensurepip") echo "No module named ensurepip" >&2; exit 1 ;;',
      'esac',
      'exit 1',
      '',
    ].join('\n'),
  );
  chmodSync(inner, 0o755);
  const fake = join(doomed, 'python3');
  writeFileSync(
    fake,
    [
      '#!/bin/sh',
      '# Answers the version probe, and builds a venv whose interpreter has',
      '# nothing: no pip, no ensurepip.',
      'if [ "$1" = "-c" ]; then echo "3.12"; exit 0; fi',
      'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then',
      '  mkdir -p "$3/bin"',
      `  cp ${JSON.stringify(inner)} "$3/bin/python"`,
      '  chmod +x "$3/bin/python"',
      '  exit 0',
      'fi',
      'exit 1',
      '',
    ].join('\n'),
  );
  chmodSync(fake, 0o755);
  process.env.RUNTIMES_DIR = join(doomed, 'runtimes');
  process.env.RUNTIMES_PYTHON = fake;
  let doomedMessage = '';
  try {
    await ensureVenv('venv', undefined, () => undefined);
  } catch (err) {
    doomedMessage = (err as Error).message;
  }
  report(
    'an environment that cannot be repaired says which package fixes it',
    /python3-venv/.test(doomedMessage) && /no pip/i.test(doomedMessage),
    doomedMessage || '(no error was thrown)',
  );

  // c. The exact field sequence. A Python without python3-venv builds most of
  //    a venv and then fails on ensurepip — leaving an interpreter that passes
  //    every check the old code made. Every later attempt "reused" it and died
  //    with "No module named pip", which is why the original error was never
  //    seen again. The wreckage must not survive the failure.
  const partial = mkdtempSync(join(tmpdir(), 'starr-partial-'));
  const partialFake = join(partial, 'python3');
  writeFileSync(
    partialFake,
    [
      '#!/bin/sh',
      'if [ "$1" = "-c" ]; then echo "3.12"; exit 0; fi',
      'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then',
      '  mkdir -p "$3/bin"',
      '  : > "$3/bin/python"',
      '  echo "The virtual environment was not created successfully because the ensurepip module is not available." >&2',
      '  exit 1',
      'fi',
      'exit 1',
      '',
    ].join('\n'),
  );
  chmodSync(partialFake, 0o755);
  process.env.RUNTIMES_DIR = join(partial, 'runtimes');
  process.env.RUNTIMES_PYTHON = partialFake;
  const partialSaid: string[] = [];
  let partialMessage = '';
  try {
    await ensureVenv('venv', undefined, (line) => partialSaid.push(line));
  } catch (err) {
    partialMessage = (err as Error).message;
  }
  report(
    'a venv that dies on ensurepip is reported with the package to install',
    /python3-venv/.test(partialMessage),
    partialMessage || '(no error was thrown)',
  );
  report(
    'the half-built environment it left behind is discarded',
    !existsSync(join(partial, 'runtimes', 'venv', 'bin', 'python')),
  );
  let partialAgain = '';
  try {
    await ensureVenv('venv', undefined, () => undefined);
  } catch (err) {
    partialAgain = (err as Error).message;
  }
  report(
    'the next attempt fails the same way instead of reusing the wreckage',
    partialAgain !== '' && partialAgain === partialMessage,
    (partialAgain || '(no error was thrown)').slice(0, 90),
  );
  rmSync(partial, { recursive: true, force: true });

  if (saved.dir === undefined) delete process.env.RUNTIMES_DIR;
  else process.env.RUNTIMES_DIR = saved.dir;
  if (saved.py === undefined) delete process.env.RUNTIMES_PYTHON;
  else process.env.RUNTIMES_PYTHON = saved.py;
  rmSync(repairable, { recursive: true, force: true });
  rmSync(doomed, { recursive: true, force: true });
  report('the real environment is untouched by the fixtures', await venvHasPip('venv'));
}

/* ---- 2. Ollama has a command on every OS ---- */
section('Ollama installer per OS');
{
  const linux = ollamaRecipe('linux');
  report('Linux uses the vendor script', /install\.sh/.test(linux?.shown ?? ''), linux?.shown);
  report('Linux is the one that needs root', linux?.needsRoot === true);
  report(
    'the Linux command keeps `sh -c` (dropping it installs nothing)',
    linux?.argv[0] === 'sh' && linux?.argv[1] === '-c' && linux?.argv.length === 3,
    linux?.argv.join(' '),
  );

  const mac = ollamaRecipe('darwin');
  report('macOS uses Homebrew', /brew install ollama/.test(mac?.shown ?? ''), mac?.shown);
  report(
    'macOS is not escalated (brew refuses to run as root)',
    mac?.needsRoot === false,
  );

  const win = ollamaRecipe('win32');
  report('Windows uses the vendor PowerShell script', /install\.ps1/.test(win?.shown ?? ''), win?.shown);
  report('Windows installs per-user, so no elevation', win?.needsRoot === false);
  report(
    'the Windows command is spawned, not piped through a shell',
    win?.argv.includes('powershell.exe') && win?.argv.includes('-NoProfile'),
    win?.argv.join(' '),
  );

  report('an unsupported OS admits it rather than guessing', ollamaRecipe('aix' as NodeJS.Platform) === null);

  /* A minimal or server Ubuntu has no curl, and the vendor command is
     `curl … | sh`, so it died with "curl: command not found" — which reads as
     the app being broken rather than the machine being minimal. */
  const wgetLinux = ollamaRecipe('linux', 'wget');
  report(
    'Linux can fetch the installer with wget when there is no curl',
    /wget -qO-\s+https:\/\/ollama\.com\/install\.sh \| sh/.test(wgetLinux?.shown ?? ''),
    wgetLinux?.shown,
  );
  report(
    'the wget form keeps the same `sh -c` shape',
    wgetLinux?.argv[0] === 'sh' && wgetLinux?.argv[1] === '-c' && wgetLinux?.argv.length === 3,
    wgetLinux?.argv.join(' '),
  );
  report(
    'the wget form still needs root, like the curl one',
    wgetLinux?.needsRoot === true,
  );
  report(
    'curl stays the default when nothing says otherwise',
    ollamaRecipe('linux')?.shown === ollamaRecipe('linux', 'curl')?.shown,
    ollamaRecipe('linux')?.shown,
  );
  report(
    'a fetcher choice does not leak onto the other platforms',
    /brew/.test(ollamaRecipe('darwin', 'wget')?.shown ?? '') &&
      /install\.ps1/.test(ollamaRecipe('win32', 'wget')?.shown ?? ''),
  );
  const fetcher = await downloadTool();
  report(
    'the downloader probe names the one this machine actually has',
    fetcher === null || fetcher === 'curl' || fetcher === 'wget',
    String(fetcher),
  );

  /* The exact failure a lean image produces: Ollama's tarball is zstd, and a
     machine without it stops the installer dead. */
  const zstdReport = [
    'ERROR: This version requires zstd for extraction. Please install zstd and try again:',
    '- Debian/Ubuntu: sudo apt-get install zstd',
    '- RHEL/CentOS/Fedora: sudo dnf install zstd',
  ].join('\n');
  report(
    'the installer naming zstd is recognised',
    missingInstallerPrerequisite(zstdReport) === 'zstd',
    zstdReport,
  );
  report(
    'a reworded version of the same message still works',
    missingInstallerPrerequisite('This version requires ZSTD for decompression.') === 'zstd',
  );
  report(
    'the please-install phrasing works too',
    missingInstallerPrerequisite('Please install curl and try again') === 'curl',
  );
  report(
    'an unrelated installer failure names nothing to install',
    missingInstallerPrerequisite('curl: (23) Failure writing output to destination') === null,
  );
  /* This text comes from a script fetched over the network. Letting it choose
     the apt argument would make "install this" an instruction. */
  report(
    'a package outside the fixed list is not acted on',
    missingInstallerPrerequisite('This version requires netcat for extraction.') === null,
  );
  report(
    'a chained command is not read as a package name',
    missingInstallerPrerequisite('Please install rm -rf / --force') === null,
  );
  report(
    'an apt flag is not read as a package name',
    missingInstallerPrerequisite('requires --allow-downgrades for extraction') === null,
  );
  // The status field is how the Runtimes page answers "is it on the GPU?"
  // without a terminal, so it has to exist and have to be a boolean.
  const vkState = await ollamaVulkanEnabled();
  report('the Ollama Vulkan setting is readable as a boolean', typeof vkState === 'boolean', String(vkState));

  // The escalated and bare forms are the same installer with a wrapper, so the
  // two argument vectors have to agree apart from that wrapper.
  const bare = ollamaCommand(linux!, false);
  const sudoed = ollamaCommand(linux!, true);
  report('bare form is the recipe itself', bare.cmd === 'sh' && bare.args.join(' ') === linux!.argv.slice(1).join(' '), `${bare.cmd} ${bare.args.join(' ')}`);
  report(
    'escalated form is sudo in front of the same recipe',
    sudoed.cmd === 'sudo' &&
      sudoed.args.join(' ') === ['-S', '-p', '', ...linux!.argv].join(' '),
    `${sudoed.cmd} ${sudoed.args.join(' ')}`,
  );

  /* ---- which NVIDIA kernel module actually landed ----
   *
   * `nvidia-driver` is a 12 kB metapackage; apt picks the rest, and on current
   * Ubuntu that is the *open* GPU kernel module, not the proprietary one. The
   * script used to announce "installing the proprietary driver" either way,
   * which is a claim about a choice it never made — and on Maxwell, Pascal or
   * Volta the open module does not drive the card at all, so the mislabel hides
   * the one case where it matters.
   *
   * The function is lifted out of the script and run against a stubbed
   * dpkg-query, so this checks the logic that ships rather than a copy of it.
   */
  const driverScript = readFileSync(join(process.cwd(), 'scripts/install-gpu-drivers.sh'), 'utf8');
  const variantFn = driverScript.match(/^nvidia_module_variant\(\) \{[\s\S]*?^\}/m)?.[0];
  report('the driver script can name the module it installed', Boolean(variantFn), variantFn ? '' : 'function not found');
  report(
    'and no longer claims "proprietary" before it knows',
    !/installing the proprietary driver/.test(driverScript),
    'banner still asserts a variant',
  );
  report(
    'a dry run does not report a variant for an install it did not do',
    /if \[ "\$ASSUME_YES" -eq 1 \]; then\s*\n\s*printf '  installed: %s/.test(driverScript),
    'the variant line is not guarded by --yes',
  );
  if (variantFn) {
    const probeVariant = (installed: string): string => {
      const dir = mkdtempSync(join(tmpdir(), 'starr-dpkg-'));
      try {
        const stub = join(dir, 'dpkg-query');
        // Answers for whichever package the function asks about; `installed` is
        // the one that reports "ok installed".
        writeFileSync(
          stub,
          `#!/bin/sh\nfor a in "$@"; do\n  case "$a" in\n` +
            `    *nvidia-dkms-open) if [ "${installed}" = "nvidia-dkms-open" ]; then echo "install ok installed"; else echo "install ok not-installed"; fi ;;\n` +
            `    *nvidia-dkms) if [ "${installed}" = "nvidia-dkms" ]; then echo "install ok installed"; else echo "install ok not-installed"; fi ;;\n` +
            `  esac\ndone\n`,
        );
        chmodSync(stub, 0o755);
        const out = spawnSync('bash', ['-c', `set -euo pipefail\n${variantFn}\nnvidia_module_variant`], {
          env: { ...process.env, PATH: `${dir}:${process.env.PATH ?? ''}` },
          encoding: 'utf8',
        });
        return (out.stdout ?? '').trim() || `(${out.stderr ?? ''}).trim()`.slice(0, 80);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    };
    const asOpen = probeVariant('nvidia-dkms-open');
    const asProprietary = probeVariant('nvidia-dkms');
    const asNeither = probeVariant('');
    report('the open module is named as open', /open GPU kernel module/.test(asOpen), asOpen);
    report('the proprietary module is named as proprietary', /^proprietary/.test(asProprietary), asProprietary);
    report(
      'and it admits when it cannot tell, rather than guessing',
      /does not recognise/.test(asNeither),
      asNeither,
    );
    report(
      'neither name is reported for the other package',
      !/open/.test(asProprietary) && !/proprietary/.test(asOpen),
      `${asOpen} / ${asProprietary}`,
    );
  }

  /* And it actually runs. A fake `curl` earlier on PATH answers the real
     command line without touching the network, which is the only way to catch
     a mangled argv: `sh "curl … | sh"` fails with "cannot open …: No such
     file", which is exactly what the app used to do on Linux. */
  const runDir = join(tmpdir(), `starr-argv-${process.pid}`);
  try {
    mkdirSync(runDir, { recursive: true });
    const fakeCurl = join(runDir, 'curl');
    // The real thing is a shell script fetched and piped into `sh`, so the
    // fake has to output shell — not arbitrary text, which `sh` would rightly
    // try to run as a command.
    writeFileSync(fakeCurl, '#!/bin/sh\necho "echo ARGV_OK"\n', { mode: 0o755 });
    const { code, out } = await new Promise<{ code: number; out: string }>((resolve) => {
      const child = spawn(bare.cmd, bare.args, {
        env: { ...process.env, PATH: `${runDir}:${process.env.PATH ?? ''}` },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let seen = '';
      child.stdout.on('data', (d) => (seen += String(d)));
      child.stderr.on('data', (d) => (seen += String(d)));
      child.on('error', (err) => resolve({ code: 127, out: err.message }));
      child.on('close', (c) => resolve({ code: c ?? 0, out: seen }));
    });
    report(
      'the Linux command actually runs the script',
      code === 0 && out.includes('ARGV_OK'),
      `exit ${code}: ${out.trim().slice(0, 120)}`,
    );
  } finally {
    rmSync(runDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  }
}

/* ---- 3. "Needs a password" is read from the output, not the exit code ---- */
section('Privilege detection');
{
  report('sudo without a tty is recognised', needsPrivilege('sudo: a terminal is required to read the password'));
  report('the newer sudo wording is recognised', needsPrivilege('sudo: no tty present and no askpass program specified'));
  report('a permission denial is recognised', needsPrivilege('cp: /usr/local/bin/ollama: Permission denied'));
  report(
    'an ordinary install failure is not read as a password problem',
    !needsPrivilege('curl: (7) Failed to connect to ollama.com port 443: Connection timed out'),
  );
  report(
    'an empty log is not a privilege problem',
    !needsPrivilege(''),
  );
}

/* ---- 4. The server refuses a nonsense install and answers a real one ---- */
section('Server contract');
let serverUp = false;
// The root page is static and answers immediately. /api/runtimes is not: it
// probes Python, and a cold cache on a loaded machine takes longer than any
// sensible timeout — which used to read as "the server is down" and skip the
// checks that matter.
for (let i = 0; i < 3 && !serverUp; i += 1) {
  try {
    const res = await fetch(APP_URL, { signal: AbortSignal.timeout(5000) });
    serverUp = res.ok;
  } catch {
    await new Promise((r) => setTimeout(r, 1000));
  }
}
if (!serverUp) {
  console.log(`  SKIP  server not reachable at ${APP_URL} (start it with npm run dev)`);
} else {
  // A labelled block so the rate-limit case can bail out of the whole section
  // rather than fall through into checks that would all fail for the same
  // reason.
  serverSection: {
    const statusRes = await fetch(`${APP_URL}/api/runtimes`);
  const statusText = await statusRes.text();
  let status: any = null;
  try {
    status = JSON.parse(statusText);
  } catch {
    status = null;
  }
  // The server rate-limits /api to 30 a minute per route. A check suite run
  // several times in a row earns that, and every later check then fails for a
  // reason that has nothing to do with the code — so say what happened and stop
  // instead of reporting a wall of failures.
  if (statusRes.status === 429) {
    console.log('  SKIP  the server is rate-limiting this IP; run this on its own');
    serverUp = false;
    break serverSection;
  } else {
    report(
      'runtimes status still lists every group',
      Boolean(status?.runtimes?.installed && Array.isArray(status?.groups)),
      `HTTP ${statusRes.status}: ${statusText.slice(0, 160)}`,
    );
  }
  // Ollama is detected, not assumed: the old code answered "installed" on any
  // OS without looking, which is what made a missing install unrecoverable.
  // Read with one retry, because the server answers this from a probe that can
  // time out on a busy machine and a timeout is not what is under test here.
  let ollamaState: unknown;
  for (let i = 0; i < 2; i += 1) {
    try {
      ollamaState = (await fetch(`${APP_URL}/api/runtimes`, { signal: AbortSignal.timeout(40000) }).then(
        (r) => r.json(),
      ))?.runtimes?.installed?.ollama;
    } catch {
      /* the probe timed out; one more try */
    }
    if (ollamaState !== undefined) break;
    await new Promise((r) => setTimeout(r, 1500));
  }
  report(
    'ollama is reported from the machine, not assumed',
    typeof ollamaState === 'string' || ollamaState === null,
    String(ollamaState),
  );

  const bad = await fetch(`${APP_URL}/api/runtimes/install`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ groups: ['definitely-not-a-runtime'] }),
  });
  report('an unknown runtime id is refused', bad.status === 400, String(bad.status));

  const empty = await fetch(`${APP_URL}/api/runtimes/install`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ groups: [] }),
  });
  report('an empty selection is refused', empty.status === 400, String(empty.status));

  /* The button label comes from a status read the client makes the moment it
     sees `done`, so the lock has to be free by then. It used to be released
     after, and that one ordering was enough to leave the button reading
     "Installing…" with nothing running.

     Checked against the real server with a real (already-installed, so it
     resolves offline) install. This asserts the invariant rather than the race:
     on localhost the release wins the photo finish, and what the browser
     actually lost to was its own follow-up work between `done` and the read.
     The check that pins the symptom is the browser one below — a status read
     answered "busy" a moment too late. */
  try {
    const res = await fetch(`${APP_URL}/api/runtimes/install`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ groups: ['onnxruntime'], device: 'cpu' }),
      signal: AbortSignal.timeout(180_000),
    });
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let sawDone = false;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      if (buf.includes('event: done')) {
        sawDone = true;
        break;
      }
    }
    await reader.cancel().catch(() => undefined);
    report('a real install reports completion', sawDone);
    if (sawDone) {
      const after = await fetch(`${APP_URL}/api/runtimes`).then((r) => r.json());
      report(
        'the lock is already free when completion is announced',
        after?.runtimes?.busy === false,
        `busy=${String(after?.runtimes?.busy)}`,
      );
    }
  } catch (err) {
    report('the lock-ordering check ran', false, (err as Error).message);
  }

  /* The stale-busy bug: `busy` is what disables the install button, and it
     used to be served from a 30-second cache. A click that landed while an
     install was already running was therefore allowed by the button, refused
     by the server, and answered with "Already running on the server" — for an
     install the user had only started once.

     Proved here on a throwaway server whose Python stalls, so an install is
     deterministically in flight with no network and nothing to install. */
  const stallDir = join(tmpdir(), `starr-stall-${process.pid}`);
  const stallPort = 3199;
  let stallServer: ReturnType<typeof spawn> | null = null;
  try {
    mkdirSync(stallDir, { recursive: true });
    const fakePython = join(stallDir, 'python3');
    // Answers the version probe like a real interpreter, then takes its time
    // about creating an environment — which is exactly the window in question.
    writeFileSync(
      fakePython,
      ['#!/bin/sh', 'case "$1 $2" in', '  "-c import"*) echo "3.12"; exit 0 ;;', 'esac', 'sleep 25', 'exit 1', ''].join(
        '\n',
      ),
    );
    chmodSync(fakePython, 0o755);
    stallServer = spawn(
      // The local binary, not `npx`: npx adds a wrapper process, and killing
      // the wrapper leaves the server listening — which then serves the *old*
      // build to the next run and turns this check into a coin toss.
      join(process.cwd(), 'node_modules', '.bin', 'tsx'),
      ['src/server.ts'],
      {
        cwd: process.cwd(),
        detached: true,
        env: {
          ...process.env,
          PORT: String(stallPort),
          RUNTIMES_PYTHON: fakePython,
          RUNTIMES_DIR: join(stallDir, 'runtimes'),
        },
        stdio: 'ignore',
      },
    );
    let ready = false;
    for (let i = 0; i < 60 && !ready; i += 1) {
      await new Promise((r) => setTimeout(r, 500));
      try {
        await fetch(`http://localhost:${stallPort}/api/runtimes`, { signal: AbortSignal.timeout(1500) });
        ready = true;
      } catch {
        /* still booting */
      }
    }
    if (!ready) throw new Error('the throwaway server never came up');

    // Warm the cache while nothing is installing.
    const idle = await fetch(`http://localhost:${stallPort}/api/runtimes`).then((r) => r.json());
    report('the throwaway server starts out not busy', idle?.runtimes?.busy === false);

    // Start an install and leave it hanging.
    void fetch(`http://localhost:${stallPort}/api/runtimes/install`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ groups: ['kokoro'], device: 'cpu' }),
    }).catch(() => undefined);

    let sawBusy = false;
    for (let i = 0; i < 20 && !sawBusy; i += 1) {
      await new Promise((r) => setTimeout(r, 500));
      const now = await fetch(`http://localhost:${stallPort}/api/runtimes`).then((r) => r.json());
      if (now?.runtimes?.busy === true) sawBusy = true;
    }
    report(
      'a cached read still reports an install that is running',
      sawBusy,
      'busy stayed false for 10s while an install held the lock',
    );
  } catch (err) {
    report('the stale-busy check ran', false, (err as Error).message);
  } finally {
    // The whole process group: tsx spawns the server it runs, and killing only
    // the parent leaves that child holding the port for the next run.
    if (stallServer?.pid) {
      try {
        process.kill(-stallServer.pid, 'SIGKILL');
      } catch {
        try {
          stallServer.kill('SIGKILL');
        } catch {
          /* already gone */
        }
      }
    }
    // Wait for the port to actually come back, so the next run cannot attach to
    // a server that is still shutting down.
    for (let i = 0; i < 20; i += 1) {
      try {
        await fetch(`http://localhost:${stallPort}/api/runtimes`, { signal: AbortSignal.timeout(500) });
      } catch {
        break;
      }
      await new Promise((r) => setTimeout(r, 250));
    }
    rmSync(stallDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  }
  }
}

/* ---- 4b. The field failure, on a real server ---- */
section('A server whose Python cannot install anything');
{
  /* The log that came back from the machine this is about:

       Reusing the existing environment.
       .../venv/bin/python: No module named pip

     An environment was reported reusable and every install into it failed.
     Both halves are checked above; this runs the whole path — HTTP in, SSE out —
     against a server whose interpreter has neither pip nor ensurepip, which is
     the one case where there is nothing to repair and the honest answer is the
     package to install. Nothing here touches the network: the failure happens
     before the first download. */
  const noPipDir = join(tmpdir(), `starr-nopip-server-${process.pid}`);
  const noPipPort = 3198;
  let noPipServer: ReturnType<typeof spawn> | null = null;
  try {
    mkdirSync(noPipDir, { recursive: true });
    const inner = join(noPipDir, 'venv-bin-python');
    writeFileSync(
      inner,
      [
        '#!/bin/sh',
        'case "$1 $2" in',
        '  "-m pip") echo "No module named pip" >&2; exit 1 ;;',
        '  "-m ensurepip") echo "No module named ensurepip" >&2; exit 1 ;;',
        'esac',
        'exit 1',
        '',
      ].join('\n'),
    );
    chmodSync(inner, 0o755);
    const noPipPython = join(noPipDir, 'python3');
    writeFileSync(
      noPipPython,
      [
        '#!/bin/sh',
        'if [ "$1" = "-c" ]; then echo "3.12"; exit 0; fi',
        'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then',
        '  mkdir -p "$3/bin"',
        `  cp ${JSON.stringify(inner)} "$3/bin/python"`,
        '  chmod +x "$3/bin/python"',
        '  exit 0',
        'fi',
        'exit 1',
        '',
      ].join('\n'),
    );
    chmodSync(noPipPython, 0o755);
    noPipServer = spawn(join(process.cwd(), 'node_modules', '.bin', 'tsx'), ['src/server.ts'], {
      cwd: process.cwd(),
      detached: true,
      env: {
        ...process.env,
        PORT: String(noPipPort),
        RUNTIMES_PYTHON: noPipPython,
        RUNTIMES_DIR: join(noPipDir, 'runtimes'),
      },
      stdio: 'ignore',
    });
    let ready = false;
    for (let i = 0; i < 60 && !ready; i += 1) {
      await new Promise((r) => setTimeout(r, 500));
      try {
        await fetch(`http://localhost:${noPipPort}/api/runtimes`, { signal: AbortSignal.timeout(1500) });
        ready = true;
      } catch {
        /* still booting */
      }
    }
    if (!ready) throw new Error('the no-pip server never came up');

    const res = await fetch(`http://localhost:${noPipPort}/api/runtimes/install`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ groups: ['onnxruntime'], device: 'cpu' }),
      signal: AbortSignal.timeout(120_000),
    });
    const stream = await res.text();
    report(
      'the failure reaches the browser, naming the package to install',
      /event: error/.test(stream) && /python3-venv/.test(stream),
      stream.split('\n').filter(Boolean).slice(-2).join(' '),
    );
    report(
      'the log says why, in the same words as the error',
      /no pip/.test(stream) && /cannot install packages/.test(stream),
      stream.split('\n').filter((l) => /pip/.test(l)).slice(0, 3).join(' | '),
    );
    const status = await fetch(`http://localhost:${noPipPort}/api/runtimes`).then((r) => r.json());
    report(
      'a failed install leaves nothing busy behind',
      status?.runtimes?.busy === false,
      `busy=${String(status?.runtimes?.busy)}`,
    );
  } catch (err) {
    report('the no-pip install check ran', false, (err as Error).message);
  } finally {
    if (noPipServer?.pid) {
      try {
        process.kill(-noPipServer.pid, 'SIGKILL');
      } catch {
        try {
          noPipServer.kill('SIGKILL');
        } catch {
          /* already gone */
        }
      }
    }
    for (let i = 0; i < 20; i += 1) {
      try {
        await fetch(`http://localhost:${noPipPort}/api/runtimes`, { signal: AbortSignal.timeout(500) });
      } catch {
        break;
      }
      await new Promise((r) => setTimeout(r, 250));
    }
    rmSync(noPipDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  }
}

/* ---- 5. The browser: no password box unless one is asked for ---- */
const BROWSERS = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'];
const browser = BROWSERS.find((b) => spawnSync('which', [b], { stdio: 'ignore' }).status === 0);
if (!browser) {
  console.log('\n  SKIP  no chromium/chrome found on PATH');
} else if (!serverUp) {
  console.log('\n  SKIP  browser checks need the server up');
} else {
  console.log('\nBrowser: the password prompt is on demand');
  const PROFILE = join(tmpdir(), `starr-runtimes-${process.pid}`);
  rmSync(PROFILE, { recursive: true, force: true });
  // The real status with onnxruntime removed, so the check below sees a machine
  // that does not have it — the situation the pre-ticked box was actually seen
  // in. On a machine where it *is* installed the row is disabled and unchecked
  // already, so the check would pass without ever reaching the bug.
  const realStatus = await readStatus();
  const doctoredRuntimes = Buffer.from(
    JSON.stringify(
      (() => {
        const copy = JSON.parse(JSON.stringify(realStatus));
        delete copy.runtimes?.installed?.onnxruntime;
        return copy;
      })(),
    ),
  ).toString('base64');

  const chrome = spawn(
    browser,
    [
      '--headless=new',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--window-size=1280,1400',
      `--remote-debugging-port=${PORT}`,
      `--user-data-dir=${PROFILE}`,
      'about:blank',
    ],
    { stdio: 'ignore' },
  );
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  const cleanup = () => {
    try {
      chrome.kill('SIGKILL');
    } catch {
      /* already gone */
    }
    // Chromium writes its profile out as it dies, so this can lose a race with
    // it. A leftover temp directory is not worth failing a check over.
    try {
      rmSync(PROFILE, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    } catch {
      /* leaving it behind */
    }
  };

  try {
    let wsUrl = '';
    for (let i = 0; i < 60; i += 1) {
      try {
        const list = (await (
          await fetch(`http://127.0.0.1:${PORT}/json/list`)
        ).json()) as Array<{ type: string; webSocketDebuggerUrl?: string }>;
        const page = list.find((t) => t.type === 'page');
        if (page?.webSocketDebuggerUrl) {
          wsUrl = page.webSocketDebuggerUrl;
          break;
        }
      } catch {
        /* not up yet */
      }
      await sleep(250);
    }
    if (!wsUrl) throw new Error('chromium never came up');

    const ws = new WebSocket(wsUrl);
    await new Promise<void>((resolve, reject) => {
      ws.addEventListener('open', () => resolve());
      ws.addEventListener('error', () => reject(new Error('cdp socket failed')));
    });
    let id = 0;
    const pending = new Map<number, (m: any) => void>();
    const pageErrors: string[] = [];
    ws.addEventListener('message', (ev) => {
      const m = JSON.parse(String((ev as MessageEvent).data));
      if (m.id && pending.has(m.id)) {
        pending.get(m.id)!(m);
        pending.delete(m.id);
      }
      if (m.method === 'Runtime.exceptionThrown') {
        pageErrors.push(
          String(m.params?.exceptionDetails?.exception?.description ?? 'unknown page error').split(
            '\n',
          )[0],
        );
      }
      /* Only ever armed for the check below, and only for this one endpoint:
         fulfilling every /api/* answer with the runtimes body breaks the page. */
      if (m.method === 'Fetch.requestPaused') {
        if (m.params?.request?.url?.includes('/api/runtimes')) {
          pending.set(
            ++id,
            () => {},
          );
          ws.send(
            JSON.stringify({
              id: id,
              method: 'Fetch.fulfillRequest',
              params: {
                requestId: m.params.requestId,
                responseCode: 200,
                responseHeaders: [{ name: 'content-type', value: 'application/json' }],
                body: doctoredRuntimes,
              },
            }),
          );
        } else {
          ws.send(
            JSON.stringify({
              id: ++id,
              method: 'Fetch.continueRequest',
              params: { requestId: m.params.requestId },
            }),
          );
        }
      }
    });
    const cdp = (method: string, params: Record<string, unknown> = {}) =>
      new Promise<any>((resolve) => {
        const n = ++id;
        pending.set(n, resolve);
        ws.send(JSON.stringify({ id: n, method, params }));
      });
    const evalJs = async (expression: string) => {
      const r = await cdp('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      if (r.result?.exceptionDetails) {
        throw new Error(String(r.result.exceptionDetails.exception?.description ?? 'eval failed'));
      }
      return r.result?.result?.value;
    };
    /**
     * Waits for a condition instead of sleeping a fixed time.
     *
     * These assertions are about what the page does after an async stream, and
     * a fixed wait makes them fail on a loaded machine while passing on an idle
     * one — which is how a check stops being a check.
     */
    const waitFor = async (expression: string, timeoutMs = 10000) => {
      const until = Date.now() + timeoutMs;
      for (;;) {
        if (await evalJs(`Boolean(${expression})`)) return true;
        if (Date.now() > until) return false;
        await sleep(150);
      }
    };

    await cdp('Page.enable');
    await cdp('Runtime.enable');
    await cdp('Page.navigate', { url: `${APP_URL}/settings` });
    // The subview buttons are in the static HTML, so they exist long before the
    // module finishes evaluating and wires their click handlers. Waiting on the
    // button's presence and clicking once is a race that a third of runs lost —
    // the click simply went to a button nobody was listening on yet. So: wait
    // for the page's own boot to show a panel (only the module does that), then
    // click, and click again if the panel has not switched.
    await waitFor(`document.querySelector('[data-subpanel]:not([hidden])')`, 30000);
    let opened = false;
    for (let i = 0; i < 10 && !opened; i += 1) {
      await evalJs(
        `document.querySelector('[data-subview="runtimes"]').click(); true`,
      );
      await sleep(250);
      opened = (await evalJs(`document.querySelector('[data-subpanel="runtimes"]').hidden`)) === false;
    }
    if (!opened) throw new Error('the Runtimes subview would not open');
    await waitFor(`document.querySelectorAll('.rt-row').length > 0`, 30000);

    // The ONNX Runtime row has to say which build it has. The CPU wheel and
    // the GPU wheel import under the same name, so "Installed" cannot say —
    // and "is my ONNX model on the card?" is the question a person asks when
    // the answer turns out to be "no".
    {
      const row = await evalJs(
        `(() => {
          const r = [...document.querySelectorAll('.rt-row')].find((r) => /ONNX Runtime/.test(r.textContent));
          return r ? r.textContent : '';
        })()`,
      );
      report(
        'the ONNX Runtime row names the build it has',
        /GPU build|CPU build/.test(String(row)),
        String(row) || '(no ONNX Runtime row)',
      );
    }

    // Stubs the install endpoint and records what the client sent. The real
    // install is not what is under test here; the prompt contract is.
    const stub = `(() => {
      window.__origFetch = window.fetch;
      window.__sent = [];
      window.__stubNext = null;
      const sse = (events) => {
        const enc = new TextEncoder();
        const body = events.map((e) => 'event: ' + e.event + '\\ndata: ' + JSON.stringify(e.data) + '\\n\\n').join('');
        return new Response(new ReadableStream({
          start(c) { c.enqueue(enc.encode(body)); c.close(); },
        }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
      };
      window.fetch = async (url, opts = {}) => {
        const u = String(url);
        if (u.includes('/api/runtimes/install')) {
          window.__sent.push(JSON.parse(opts.body ?? '{}'));
          return sse(window.__stubNext ?? [{ event: 'done', data: { runtimes: { installed: {} } } }]);
        }
        return window.__origFetch(url, opts);
      };
      // Ticking a group is the only precondition the install button has, and a
      // re-render un-ticks anything already installed — so this is a helper,
      // called again before each attempt rather than done once.
      window.__pick = () => {
        const boxes = [...document.querySelectorAll('.rt-row input[type=checkbox]')];
        const free = boxes.find((b) => !b.disabled) ?? boxes[0];
        if (!free) return false;
        free.disabled = false;
        if (!free.checked) { free.checked = true; free.dispatchEvent(new Event('change')); }
        document.getElementById('rt-install').disabled = false;
        return true;
      };
      return window.__pick();
    })()`;
    const staged = await evalJs(stub);
    report('a runtime group can be selected', staged === true);

    // A. An install that never asks for a password never shows the box.
    await evalJs(`window.__stubNext = [{ event: 'progress', data: 'Collecting kokoro\\n' }, { event: 'done', data: { runtimes: { installed: {} } } }]; true`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    await waitFor(`window.__sent.length === 1`);
    report('the first install actually ran', (await evalJs(`window.__sent.length`)) === 1);
    report(
      'an install with no password request shows no password box',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === true,
    );
    report(
      'the first request carried no password field',
      (await evalJs(`typeof (window.__sent[0] ?? {}).password === 'undefined'`)) === true,
    );
    // The label comes from a status read, so it has to end up right even when
    // that read is answered a moment late.
    const labelA = await waitFor(
      `document.getElementById('rt-install').textContent.trim() === 'Install selected'`,
      8000,
    );
    report('the button returns to its resting label after the install', labelA, String(await evalJs(`document.getElementById('rt-install').textContent.trim()`)));

    // B. An installer that asks opens the box, and the retry carries it.
    await evalJs(`window.__stubNext = [
      { event: 'progress', data: 'Installing Ollama on Linux\\n' },
      { event: 'privilege', data: { command: 'sudo curl -fsSL https://ollama.com/install.sh | sh' } },
      { event: 'error', data: { message: 'The Ollama installer needs root and could not elevate.' } },
    ]; true`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    await waitFor(`document.getElementById('rt-pass').hidden === false`);
    report(
      'a privilege request opens the password box',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === false,
    );
    report(
      'the log says which command needs the password',
      /needs your password/i.test(
        String(await evalJs(`document.getElementById('rt-log-body').textContent`)),
      ),
    );

    await evalJs(`(() => {
      const i = document.getElementById('rt-pass-input');
      i.value = 'correct horse battery';
      document.getElementById('rt-pass-go').click();
      return true;
    })()`);
    await waitFor(`window.__sent.length === 3`);
    report(
      'the retry sends the password in the request body',
      (await evalJs(`window.__sent[window.__sent.length - 1]?.password`)) === 'correct horse battery',
    );
    report('the password box closes once used', (await evalJs(`document.getElementById('rt-pass').hidden`)) === true);
    report(
      'the password is cleared from the box after use',
      (await evalJs(`document.getElementById('rt-pass-input').value`)) === '',
    );
    report(
      'the password never lands in the install log',
      !String(await evalJs(`document.getElementById('rt-log-body').textContent`)).includes(
        'correct horse battery',
      ),
    );

    // C. A second request with a password in hand does not re-open the box.
    await evalJs(`window.__stubNext = [
      { event: 'privilege', data: { command: 'sudo curl -fsSL https://ollama.com/install.sh | sh' } },
      { event: 'error', data: { message: 'That password was not accepted, so nothing was changed.' } },
    ]; true`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    await sleep(2000);
    await evalJs(`(() => {
      const i = document.getElementById('rt-pass-input');
      i.value = 'wrong one';
      document.getElementById('rt-pass-go').click();
      return true;
    })()`);
    await waitFor(`window.__sent.length === 5 && document.getElementById('rt-pass').hidden === true`);
    report(
      'a refused password does not open the box again',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === true,
    );

    /* D. The GPU driver button, which used to prompt before it knew. */
    await evalJs(`(() => {
      window.__sent = [];
      window.fetch = async (url, opts = {}) => {
        const u = String(url);
        if (u.includes('/api/drivers/update')) {
          window.__sent.push(JSON.parse(opts.body ?? '{}'));
          const enc = new TextEncoder();
          const frames = [
            { event: 'log', data: { line: 'Reading what the machine has…' } },
            ...(window.__sent.length === 1
              ? [{ event: 'privilege', data: { command: 'sudo scripts/install-gpu-drivers.sh --yes   # needs your password' } },
                 { event: 'log', data: { line: 'The server could not elevate.' } }]
              : [{ event: 'done', data: { apply: true } }]),
          ];
          const body = frames.map((e) => 'event: ' + e.event + '\\ndata: ' + JSON.stringify(e.data) + '\\n\\n').join('');
          return new Response(new ReadableStream({
            start(c) { c.enqueue(enc.encode(body)); c.close(); },
          }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
        }
        return window.__origFetch(url, opts);
      };
      document.getElementById('rt-pass').hidden = true;
      return true;
    })()`);

    // First press arms, second presses starts it. Neither may prompt.
    await evalJs(`document.getElementById('rt-update-drivers').click(); true`);
    await waitFor(`document.getElementById('rt-driver-log').textContent.includes('Press "Update GPU drivers" again')`);
    report(
      'arming the driver update shows no password box',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === true,
    );
    await evalJs(`document.getElementById('rt-update-drivers').click(); true`);
    await waitFor(`window.__sent.length === 1`);
    report(
      'the armed press starts the install rather than the prompt',
      (await evalJs(`window.__sent.length`)) === 1,
    );
    report(
      'that first request asked for no password',
      (await evalJs(`typeof (window.__sent[0] ?? {}).password === 'undefined'`)) === true,
    );
    await waitFor(`document.getElementById('rt-pass').hidden === false`);
    report(
      'the password box opens only once the installer says it needs one',
      (await evalJs(`document.getElementById('rt-pass').hidden`)) === false,
    );

    await evalJs(`(() => {
      document.getElementById('rt-pass-input').value = 'driver secret';
      document.getElementById('rt-pass-go').click();
      return true;
    })()`);
    await waitFor(`window.__sent.length === 2`);
    report(
      'the driver retry sends the password',
      (await evalJs(`window.__sent[window.__sent.length - 1]?.password`)) === 'driver secret',
    );
    report(
      'the driver password never lands in its log',
      !String(await evalJs(`document.getElementById('rt-driver-log').textContent`)).includes(
        'driver secret',
      ),
    );

    /* E. A retry that lands while the last attempt still holds the lock. */
    await evalJs(`(() => {
      window.__sent = [];
      let locked = false;
      const enc = new TextEncoder();
      const sse = (frames) => {
        const body = frames.map((e) => 'event: ' + e.event + '\\ndata: ' + JSON.stringify(e.data) + '\\n\\n').join('');
        return new Response(new ReadableStream({
          start(c) { c.enqueue(enc.encode(body)); c.close(); },
        }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
      };
      window.fetch = async (url, opts = {}) => {
        const u = String(url);
        if (u.includes('/api/runtimes/install')) {
          const sent = JSON.parse(opts.body ?? '{}');
          window.__sent.push(sent);
          if (!sent.password) {
            // The first attempt asks for a password, as the real Ollama
            // install does when it cannot elevate.
            return sse([
              { event: 'privilege', data: { command: 'sudo curl -fsSL https://ollama.com/install.sh | sh' } },
              { event: 'error', data: { message: 'The Ollama installer needs root and could not elevate.' } },
            ]);
          }
          if (window.__sent.filter((s) => s.password).length === 1) {
            // It answers the retry with the lock still held by the attempt
            // that just asked for the password, then frees it.
            locked = true;
            setTimeout(() => { locked = false; }, 900);
            return new Response(JSON.stringify({ error: 'An install is already running.' }), {
              status: 409, headers: { 'content-type': 'application/json' },
            });
          }
          return sse([{ event: 'done', data: { runtimes: { installed: {} } } }]);
        }
        if (u.endsWith('/api/runtimes')) {
          return new Response(JSON.stringify({ runtimes: { busy: locked, installed: {} } }), {
            status: 200, headers: { 'content-type': 'application/json' },
          });
        }
        return window.__origFetch(url, opts);
      };
      document.getElementById('rt-pass').hidden = true;
      return true;
    })()`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    await waitFor(`document.getElementById('rt-pass').hidden === false`);
    await evalJs(`(() => {
      const i = document.getElementById('rt-pass-input');
      i.value = 'lock retry';
      document.getElementById('rt-pass-go').click();
      return true;
    })()`);
    await waitFor(`window.__sent.length === 3`, 15000);
    report(
      'an install refused for the lock is retried once the lock clears',
      (await evalJs(`window.__sent.length`)) === 3,
      String(await evalJs(`window.__sent.length`)),
    );
    report(
      'the password is still on the retried request',
      (await evalJs(`window.__sent[window.__sent.length - 1]?.password`)) === 'lock retry',
    );

    /* F. A status read that is answered a moment too late.
       This is the shape that left the button stuck on "Installing…" with
       nothing running: the read that repaints the label is issued while the
       install still holds the lock, and lands after it has finished. The read
       is truthful about the moment it was made and wrong by the time it
       paints, so the only defence is to check it. */
    await evalJs(`(() => {
      window.__staleReads = 2;
      window.fetch = async (url, opts = {}) => {
        const u = String(url);
        if (u.includes('/api/runtimes/install')) {
          const enc = new TextEncoder();
          const body = 'event: done\\ndata: {"runtimes":{"installed":{}}}\\n\\n';
          return new Response(new ReadableStream({
            start(c) { c.enqueue(enc.encode(body)); c.close(); },
          }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
        }
        if (u.includes('/api/runtimes')) {
          // The first reads after the install answer "busy" — exactly what the
          // server said at the instant the client asked, and exactly what left
          // the label lying.
          const stale = window.__staleReads > 0;
          if (stale) window.__staleReads -= 1;
          return new Response(JSON.stringify({ runtimes: { busy: stale, installed: {} }, groups: [], blocked: [] }), {
            status: 200, headers: { 'content-type': 'application/json' },
          });
        }
        return window.__origFetch(url, opts);
      };
      document.getElementById('rt-pass').hidden = true;
      return true;
    })()`);
    await evalJs(`window.__pick(); document.getElementById('rt-install').click(); true`);
    // The stale read paints "Installing…" — the check is that it is not the
    // last word.
    const recovered = await waitFor(
      `document.getElementById('rt-install').textContent.trim() === 'Install selected' && window.__staleReads === 0`,
      15000,
    );
    report(
      'a late "busy" read does not leave the button stuck on Installing…',
      recovered,
      String(await evalJs(`document.getElementById('rt-install').textContent.trim()`)),
    );

    await evalJs(`window.fetch = window.__origFetch; true`);

    /* F. Nothing is pre-ticked. ONNX Runtime used to be seeded into the
     * selection on every fresh page, so it read "always on" and put an 18 MB
     * download in front of anyone who had not asked for one — for a runtime
     * whose own blurb says none of the bundled runners use it.
     *
     * The status is doctored first, because on a machine where it *is*
     * installed the row is disabled and unchecked already, and the check would
     * pass without ever exercising the bug. */
    /* The trailing star matters: the page asks for /api/runtimes?device=cpu, and a
       pattern without a wildcard does not match that. */
    await cdp('Fetch.enable', { patterns: [{ urlPattern: `${APP_URL}/api/runtimes*` }] });
    /* Blank first. Page.navigate resolves as the new document *commits*, while
       the old one — which still has the rows the earlier checks left on it — is
       still there to be polled. Waiting on that stale copy returns immediately,
       and the assertions then read the fresh document before it has rendered
       anything. Clearing first makes the wait mean what it says. */
    await cdp('Page.navigate', { url: 'about:blank' });
    await sleep(400);
    await cdp('Page.navigate', { url: `${APP_URL}/settings#models/runtimes` });
    await waitFor(`document.querySelectorAll('.rt-row').length > 0`, 30000);
    await sleep(600);
    const readRows = async () =>
      JSON.parse(
        String(
          await evalJs(`JSON.stringify([...document.querySelectorAll('.rt-row')].map(r => ({
            label: r.querySelector('.rt-row__title')?.textContent,
            checked: r.querySelector('input[type=checkbox]')?.checked,
            disabled: r.querySelector('input[type=checkbox]')?.disabled,
          })))`),
        ),
      );
    let allRows = await readRows();
    const onnxRow = allRows.find((r: any) => r.label === 'ONNX Runtime');
    report(
      'onnxruntime is offered as installable when it is not installed',
      onnxRow?.disabled === false,
      JSON.stringify(onnxRow),
    );
    report(
      'onnxruntime is not ticked before the user asks for it',
      onnxRow?.checked === false,
      JSON.stringify(onnxRow),
    );
    report(
      'no runtime at all is pre-ticked',
      allRows.every((r: any) => !r.checked),
      JSON.stringify(allRows.filter((r: any) => r.checked)),
    );

    // And the tick still works, which is what removing the seed could have cost.
    await evalJs(`(() => {
      const row = [...document.querySelectorAll('.rt-row')]
        .find(r => r.querySelector('.rt-row__title').textContent === 'ONNX Runtime');
      const box = row.querySelector('input');
      box.checked = true;
      box.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    })()`);
    await sleep(300);
    allRows = await readRows();
    report(
      'it can still be ticked',
      allRows.find((r: any) => r.label === 'ONNX Runtime')?.checked === true,
      String(await evalJs(`document.getElementById('rt-total')?.textContent ?? ''`)),
    );
    await evalJs(`(() => {
      const row = [...document.querySelectorAll('.rt-row')]
        .find(r => r.querySelector('.rt-row__title').textContent === 'ONNX Runtime');
      const box = row.querySelector('input');
      box.checked = false;
      box.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    })()`);
    await sleep(300);
    allRows = await readRows();
    report(
      'and it can be unticked again',
      allRows.find((r: any) => r.label === 'ONNX Runtime')?.checked === false,
    );
    await cdp('Fetch.disable');

    report(
      'the settings page ran without a single uncaught error',
      pageErrors.length === 0,
      pageErrors.join(' | '),
    );
    ws.close();
  } catch (err) {
    failures.push('browser checks');
    console.log(`  FAIL  browser checks — ${(err as Error).message}`);
  } finally {
    cleanup();
  }
}

/* ---- the install log follows its own tail ----
 *
 * Last, because it navigates and intercepts. Driven through a stubbed install
 * rather than by calling the helper: settings.js is a module, so its functions
 * are not reachable from the page, and a check that re-implements the scroll
 * logic proves nothing about it.
 */
if (browser && serverUp) {
  const PORT2 = PORT + 1;
  const PROFILE2 = join(tmpdir(), `starr-logtail-${process.pid}`);
  rmSync(PROFILE2, { recursive: true, force: true });
  const real = await (await fetch(`${APP_URL}/api/runtimes`, { signal: AbortSignal.timeout(60000) })).json();
  const statusBody = Buffer.from(
    JSON.stringify({
      ...real,
      runtimes: { ...real.runtimes, installed: { ...real.runtimes.installed, onnxruntime: null } },
    }),
  ).toString('base64');
  const LINES = 400;
  const frames: string[] = [];
  for (let i = 1; i <= LINES; i++) {
    frames.push(`event: progress\ndata: ${JSON.stringify(`installing line ${i} — pip output`)}\n\n`);
  }
  frames.push(`event: done\ndata: ${JSON.stringify({ runtimes: { installed: {} } })}\n\n`);
  const installBody = Buffer.from(frames.join('')).toString('base64');

  const chrome2 = spawn(
    browser,
    [
      '--headless=new',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--window-size=1280,1400',
      `--remote-debugging-port=${PORT2}`,
      `--user-data-dir=${PROFILE2}`,
      'about:blank',
    ],
    { stdio: 'ignore' },
  );
  const sleep2 = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const done2 = () => {
    try {
      chrome2.kill('SIGKILL');
    } catch {
      /* already gone */
    }
    try {
      rmSync(PROFILE2, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    } catch {
      /* leaving it behind */
    }
  };
  try {
    let ws2 = '';
    for (let i = 0; i < 60 && !ws2; i += 1) {
      try {
        const l = (await (
          await fetch(`http://127.0.0.1:${PORT2}/json/list`)
        ).json()) as Array<{ type: string; webSocketDebuggerUrl?: string }>;
        ws2 = l.find((t) => t.type === 'page')?.webSocketDebuggerUrl ?? '';
      } catch {
        await sleep2(250);
      }
    }
    if (!ws2) throw new Error('chromium never came up');

    const sock = new WebSocket(ws2);
    await new Promise<void>((resolve, reject) => {
      sock.addEventListener('open', () => resolve());
      sock.addEventListener('error', () => reject(new Error('cdp socket failed')));
    });
    let cid = 0;
    const cwait = new Map<number, (m: any) => void>();
    sock.addEventListener('message', (ev) => {
      const m = JSON.parse(String((ev as MessageEvent).data));
      if (m.id && cwait.has(m.id)) {
        cwait.get(m.id)!(m);
        cwait.delete(m.id);
      }
      if (m.method === 'Fetch.requestPaused') {
        const url = String(m.params?.request?.url ?? '');
        const post = (method: string, params: Record<string, unknown>) => {
          const n = ++cid;
          cwait.delete(n);
          sock.send(JSON.stringify({ id: n, method, params }));
        };
        const body = url.includes('/api/runtimes/install') ? installBody : statusBody;
        post('Fetch.fulfillRequest', {
          requestId: m.params.requestId,
          responseCode: 200,
          responseHeaders: [{ name: 'content-type', value: url.includes('/install') ? 'text/event-stream' : 'application/json' }],
          body,
        });
      }
    });
    const call = (method: string, params: Record<string, unknown> = {}) =>
      new Promise<any>((resolve) => {
        const n = ++cid;
        cwait.set(n, resolve);
        sock.send(JSON.stringify({ id: n, method, params }));
      });
    const run = async (expression: string) => {
      const r = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      if (r.result?.exceptionDetails) {
        throw new Error(String(r.result.exceptionDetails.exception?.description ?? 'eval failed'));
      }
      return r.result?.result?.value;
    };
    const until = async (expression: string, timeoutMs = 20000) => {
      const end = Date.now() + timeoutMs;
      for (;;) {
        if (await run(`Boolean(${expression})`)) return true;
        if (Date.now() > end) return false;
        await sleep2(150);
      }
    };

    await call('Page.enable');
    await call('Runtime.enable');
    await call('Fetch.enable', { patterns: [{ urlPattern: `${APP_URL}/api/runtimes*` }] });
    // Blank first: Page.navigate resolves as the new document commits, and the
    // previous one is still pollable.
    await call('Page.navigate', { url: 'about:blank' });
    await sleep2(400);
    await call('Page.navigate', { url: `${APP_URL}/settings#models/runtimes` });
    if (!(await until(`document.querySelectorAll('.rt-row').length > 0`))) {
      throw new Error('the runtime list did not render');
    }
    await run(`(() => {
      const row = [...document.querySelectorAll('.rt-row')]
        .find(r => r.querySelector('.rt-row__title')?.textContent === 'ONNX Runtime');
      const box = row.querySelector('input');
      box.checked = true; box.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    })()`);
    await sleep2(300);
    await run(`document.getElementById('rt-install').click(); true`);
    const got = await until(`document.getElementById('rt-log-body').textContent.includes('installing line ${LINES}')`, 30000);
    report('a streamed install fills the install log', got);

    const m = JSON.parse(
      String(
        await run(`JSON.stringify((() => {
          const el = document.getElementById('rt-log-body');
          return { scrollTop: el.scrollTop, scrollHeight: el.scrollHeight, clientHeight: el.clientHeight };
        })())`),
      ),
    );
    report('the log outgrew its box', m.scrollHeight > m.clientHeight, JSON.stringify(m));
    report(
      'and scrolled itself to the newest line',
      Math.abs(m.scrollTop + m.clientHeight - m.scrollHeight) <= 2,
      JSON.stringify(m),
    );

    const held = JSON.parse(
      String(
        await run(`JSON.stringify((() => {
          const el = document.getElementById('rt-log-body');
          el.scrollTop = 0;
          const before = el.scrollTop;
          el.textContent += 'a line that arrives after the reader scrolled up\\n';
          return { before, after: el.scrollTop };
        })())`),
      ),
    );
    report(
      'a reader who scrolled up is left where they are',
      held.after === held.before,
      JSON.stringify(held),
    );

    await call('Fetch.disable');
    sock.close();
  } catch (err) {
    failures.push('the install log tail check');
    console.log(`  FAIL  the install log tail check — ${(err as Error).message}`);
  } finally {
    done2();
  }
}

/* ---- every meter is the same bar ----
 *
 * Last, because it drives a third browser and intercepts /api/processes.
 *
 * Two things are load-bearing here and both are the kind that pass a naive
 * check while the page is wrong:
 *
 *  - `hidden` loses to an element's own `display`, so an indicator that was
 *    only ever hidden by attribute stayed on screen. These assert the computed
 *    display, not the property;
 *  - a figure that is *absent* is a different thing from a figure that is
 *    *zero*. Zero gets a bar with an empty half; absent gets no bar at all,
 *    because an empty track beside a blank legend reads as a failed render.
 */
if (browser && serverUp) {
  const PORT3 = PORT + 2;
  const PROFILE3 = join(tmpdir(), `starr-bars-${process.pid}`);
  rmSync(PROFILE3, { recursive: true, force: true });
  const GIB = 1024 ** 3;
  const procBody = Buffer.from(
    JSON.stringify({
      active: [
        { id: 'j1', kind: 'image', repo: 'stabilityai/sd-turbo', pid: 4242, startedAt: Date.now() - 90_000, memoryBytes: 3.2 * GIB, device: 'cuda', resolvedDevice: 'cuda', runtimes: ['diffusers'] },
        { id: 'j2', kind: 'stt', script: 'transcribe', pid: 4243, startedAt: Date.now() - 8_000, memoryBytes: 0.9 * GIB, device: 'cpu', runtimes: [] },
        { id: 'j3', kind: 'tts', script: 'speak', pid: 4244, startedAt: Date.now() - 4_000, memoryBytes: null, device: 'cpu', runtimes: [] },
      ],
      recent: [],
      ramTotalBytes: 16 * GIB,
      ollama: {
        ok: true,
        models: [
          { name: 'qwen2.5:7b', sizeBytes: 6 * GIB, vramBytes: 4.5 * GIB, expires: '5m', device: 'gpu' },
          { name: 'tiny:1b', sizeBytes: 1 * GIB, vramBytes: 0, expires: 'never', device: 'cpu' },
        ],
      },
    }),
  ).toString('base64');

  const chrome3 = spawn(
    browser,
    [
      '--headless=new',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--window-size=1280,1400',
      `--remote-debugging-port=${PORT3}`,
      `--user-data-dir=${PROFILE3}`,
      'about:blank',
    ],
    { stdio: 'ignore' },
  );
  const sleep3 = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const done3 = () => {
    try {
      chrome3.kill('SIGKILL');
    } catch {
      /* already gone */
    }
    try {
      rmSync(PROFILE3, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    } catch {
      /* leaving it behind */
    }
  };
  try {
    let ws3 = '';
    for (let i = 0; i < 60 && !ws3; i += 1) {
      try {
        const l = (await (
          await fetch(`http://127.0.0.1:${PORT3}/json/list`)
        ).json()) as Array<{ type: string; webSocketDebuggerUrl?: string }>;
        ws3 = l.find((t) => t.type === 'page')?.webSocketDebuggerUrl ?? '';
      } catch {
        await sleep3(250);
      }
    }
    if (!ws3) throw new Error('chromium never came up');

    const sock = new WebSocket(ws3);
    await new Promise<void>((resolve, reject) => {
      sock.addEventListener('open', () => resolve());
      sock.addEventListener('error', () => reject(new Error('cdp socket failed')));
    });
    let cid = 0;
    const cwait = new Map<number, (m: any) => void>();
    sock.addEventListener('message', (ev) => {
      const m = JSON.parse(String((ev as MessageEvent).data));
      if (m.id && cwait.has(m.id)) {
        cwait.get(m.id)!(m);
        cwait.delete(m.id);
      }
      if (m.method === 'Fetch.requestPaused') {
        const url = String(m.params?.request?.url ?? '');
        const post = (method: string, params: Record<string, unknown>) => {
          const n = ++cid;
          cwait.delete(n);
          sock.send(JSON.stringify({ id: n, method, params }));
        };
        if (url.includes('/api/processes')) {
          post('Fetch.fulfillRequest', {
            requestId: m.params.requestId,
            responseCode: 200,
            responseHeaders: [{ name: 'content-type', value: 'application/json' }],
            body: procBody,
          });
        } else {
          post('Fetch.continueRequest', { requestId: m.params.requestId });
        }
      }
    });
    const call = (method: string, params: Record<string, unknown> = {}) =>
      new Promise<any>((resolve) => {
        const n = ++cid;
        cwait.set(n, resolve);
        sock.send(JSON.stringify({ id: n, method, params }));
      });
    const run = async (expression: string) => {
      const r = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      if (r.result?.exceptionDetails) {
        throw new Error(String(r.result.exceptionDetails.exception?.description ?? 'eval failed'));
      }
      return r.result?.result?.value;
    };
    const until = async (expression: string, timeoutMs = 20000) => {
      const end = Date.now() + timeoutMs;
      for (;;) {
        if (await run(`Boolean(${expression})`)) return true;
        if (Date.now() > end) return false;
        await sleep3(150);
      }
    };

    await call('Page.enable');
    await call('Runtime.enable');
    await call('Fetch.enable', { patterns: [{ urlPattern: `${APP_URL}/api/processes*` }] });
    await call('Page.navigate', { url: 'about:blank' });
    await sleep3(400);
    await call('Page.navigate', { url: `${APP_URL}/` });
    // Wait for the button rather than for a fixed time: on a loaded machine
    // the page can take longer than three seconds to be ready, and a check
    // that fires into a half-loaded document fails for a reason that has
    // nothing to do with what it is checking.
    if (!(await until(`!!document.getElementById('open-procman')`, 25000))) {
      throw new Error('the chat page did not load');
    }
    await run(`document.getElementById('open-procman').click(); true`);
    if (!(await until(`document.querySelectorAll('.procman__row').length >= 5`, 25000))) {
      throw new Error('the process rows did not render');
    }
    await sleep3(800);

    const bars = JSON.parse(
      String(
        await run(`JSON.stringify([...document.querySelectorAll('.procman__row')].map(r => {
          const split = r.querySelector('.procman__split');
          return {
            name: (r.querySelector('.procman__name')?.textContent ?? r.textContent),
            parts: [...r.querySelectorAll('.procman__split i')].map(i => i.dataset.part + ':' + i.style.width),
            legend: [...r.querySelectorAll('.procman__legend-item b')].map(b => b.textContent),
            aria: split ? split.getAttribute('aria-label') : null,
            visible: split ? getComputedStyle(split).display !== 'none' : false,
            height: split ? getComputedStyle(split).height : null,
            radius: split ? getComputedStyle(split).borderTopLeftRadius : null,
          };
        }))`),
      ),
    );
    const named = (frag: string) => bars.find((b: any) => b.name.includes(frag));
    const jobs = bars.filter((b: any) => b.visible && b.parts.some((p: string) => p.startsWith('job:')));
    const models = bars.filter((b: any) => b.visible && b.parts.some((p: string) => p.startsWith('vram:')));

    report('a local job gets the same bar an Ollama model gets', jobs.length === 2, JSON.stringify(bars.map((b: any) => b.parts)));
    report(
      'and it is a share of the machine, not a four-pixel sliver',
      jobs.some((j: any) => j.legend.join(' ').includes('this job') && j.parts.some((p: string) => p.startsWith('rest:'))),
      JSON.stringify(jobs.map((j: any) => j.legend)),
    );
    const big = named('sd-turbo');
    const pct = Number(
      String(big?.parts.find((p: string) => p.startsWith('job:')) ?? 'job:0%').split(':')[1].replace('%', ''),
    );
    report("the width is the job's share of memory", Math.abs(pct - 20) < 1.5, `${pct}%`);
    report(
      'both Ollama models keep their VRAM/RAM split',
      models.length === 2 && models.every((m: any) => m.parts.some((p: string) => p.startsWith('ram:'))),
      JSON.stringify(models.map((m: any) => m.parts)),
    );
    report(
      'the bar is one rounded pill everywhere',
      [...jobs, ...models].every((b: any) => b.radius === '99px' && b.height === '8px'),
      JSON.stringify([...jobs, ...models].map((b: any) => [b.height, b.radius])),
    );
    report(
      'and it carries the same sentence for a screen reader',
      [...jobs, ...models].every((b: any) => typeof b.aria === 'string' && b.aria.length > 4),
      JSON.stringify([...jobs, ...models].map((b: any) => b.aria)),
    );

    // Zero is an answer; absent is not.
    const zero = named('tiny:1b');
    report(
      'a model with zero VRAM still gets a painted bar',
      zero?.visible === true && zero.parts[0] === 'vram:0%' && zero.parts[1] === 'ram:100%',
      JSON.stringify(zero),
    );
    report(
      'with a legend that says so rather than sitting blank',
      zero?.legend.length === 2 && zero.legend.every((l: string) => l.trim().length > 0),
      JSON.stringify(zero?.legend),
    );
    const blind = named('speak');
    report(
      'a job whose memory could not be read shows no bar at all',
      blind?.visible === false && blind.aria === null,
      JSON.stringify(blind),
    );

    // The number moves every poll. Node identity is what separates a repaint
    // from a rebuild, and a rebuild would restart the width transition from
    // zero once a second, so the bar would never appear to move.
    const repaint = JSON.parse(
      String(
        await run(`(async () => {
          const before = document.querySelector('.procman__row .procman__split');
          await new Promise(r => setTimeout(r, 2500));
          return JSON.stringify({ sameNode: before === document.querySelector('.procman__row .procman__split') });
        })()`),
      ),
    );
    report('a bar whose number moves is repainted, not rebuilt', repaint.sameNode === true, JSON.stringify(repaint));

    // The size knob must not be the top bar's height. `--bar-h` is set on
    // :root for the page header, so a meter that reached for it came out 34px
    // tall and every use site had to override it.
    const meter = JSON.parse(
      String(
        await run(`JSON.stringify((() => {
          const probe = document.createElement('span');
          probe.className = 'bar';
          probe.style.position = 'absolute';
          probe.style.left = '-9999px';
          document.body.append(probe);
          const h = getComputedStyle(probe).height;
          probe.remove();
          return { height: h, rootBarH: getComputedStyle(document.documentElement).getPropertyValue('--bar-h').trim() };
        })())`),
      ),
    );
    report(
      'a bare bar is bar-sized, not sized from the page header',
      meter.height === '8px' && meter.rootBarH !== '8px',
      JSON.stringify(meter),
    );

    await call('Fetch.disable');
    sock.close();
  } catch (err) {
    failures.push('the meter bar checks');
    console.log(`  FAIL  the meter bar checks — ${(err as Error).message}`);
  } finally {
    done3();
  }
}

/* ---- the indicator holds still, and a download row fits its box ----
 *
 * Three defects, all of them "it looked fine until you watched it for a
 * second":
 *
 *  - the progress bar sat beside a label whose width followed its text, so a
 *    new pip line walked the bar sideways. A bar you cannot read progress off
 *    because it keeps moving is worse than no bar;
 *  - following the log's tail meant reading `scrollHeight` per line, and that
 *    forces a layout — of a log growing to a thousand lines, a thousand times.
 *    It locked the tab;
 *  - a download row's status was `flex-shrink: 0`, and it is built out of a
 *    file count, two byte counts, a percentage and a rate, so it pushed
 *    straight through the right edge of its card.
 */
if (browser && serverUp) {
  const PORT4 = PORT + 3;
  const PROFILE4 = join(tmpdir(), `starr-anchor-${process.pid}`);
  rmSync(PROFILE4, { recursive: true, force: true });
  const GIB4 = 1024 ** 3;
  const real4 = await readStatus();
  /* The status the page is given. Mutable, because the models-page check
     below needs a different one: a machine with nothing installed, so
     several runtime groups are missing at once. */
  let status4 = Buffer.from(
    JSON.stringify({
      ...real4,
      runtimes: { ...real4.runtimes, installed: { ...real4.runtimes.installed, onnxruntime: null } },
    }),
  ).toString('base64');
  const nothingInstalled4 = {
    ...real4,
    runtimes: {
      ...real4.runtimes,
      installed: Object.fromEntries(
        Object.keys(real4.runtimes?.installed ?? {}).map((k) => [k, null]),
      ),
    },
  };
  const LINES4 = [
    'Collecting',
    'Using cached pip wheel for onnxruntime on this platform',
    'Created wheel for onnxruntime: filename=onnxruntime-1.23.0-cp313-cp313-linux_x86_64.whl size=17123456',
  ];
  const install4 = Buffer.from(
    Array.from({ length: 1500 }, (_, i) => `event: progress\ndata: ${JSON.stringify(LINES4[i % LINES4.length])}\n\n`).join(''),
  ).toString('base64');
  const transfers4 = Buffer.from(
    JSON.stringify({
      active: [
        { id: 'a1', kind: 'pull', label: 'Qwen/Qwen2.5-Coder-32B-Instruct-GGUF', repo: 'Qwen/Qwen2.5-Coder-32B-Instruct-GGUF', state: 'running', received: 1.23 * GIB4, total: 20.3 * GIB4, startedAt: Date.now() - 60000 },
        { id: 'a2', kind: 'pull', label: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B-GGUF', repo: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B-GGUF', state: 'error', error: 'the remote host closed the connection while reading the LFS pointer for safetensors (HTTP 502 from the CDN edge)' },
      ],
      recent: [],
    }),
  ).toString('base64');

  const chrome4 = spawn(
    browser,
    [
      '--headless=new',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--window-size=1000,1000',
      `--remote-debugging-port=${PORT4}`,
      `--user-data-dir=${PROFILE4}`,
      'about:blank',
    ],
    { stdio: 'ignore' },
  );
  const sleep4 = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const done4 = () => {
    try {
      chrome4.kill('SIGKILL');
    } catch {
      /* already gone */
    }
    try {
      rmSync(PROFILE4, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    } catch {
      /* leaving it behind */
    }
  };
  try {
    let ws4 = '';
    for (let i = 0; i < 60 && !ws4; i += 1) {
      try {
        const l = (await (
          await fetch(`http://127.0.0.1:${PORT4}/json/list`)
        ).json()) as Array<{ type: string; webSocketDebuggerUrl?: string }>;
        ws4 = l.find((t) => t.type === 'page')?.webSocketDebuggerUrl ?? '';
      } catch {
        await sleep4(250);
      }
    }
    if (!ws4) throw new Error('chromium never came up');

    const sock = new WebSocket(ws4);
    await new Promise<void>((resolve, reject) => {
      sock.addEventListener('open', () => resolve());
      sock.addEventListener('error', () => reject(new Error('cdp socket failed')));
    });
    let cid = 0;
    const cwait = new Map<number, (m: any) => void>();
    sock.addEventListener('message', (ev) => {
      const m = JSON.parse(String((ev as MessageEvent).data));
      if (m.id && cwait.has(m.id)) {
        cwait.get(m.id)!(m);
        cwait.delete(m.id);
      }
      if (m.method === 'Fetch.requestPaused') {
        const url = String(m.params?.request?.url ?? '');
        const post = (method: string, params: Record<string, unknown>) => {
          const n = ++cid;
          cwait.delete(n);
          sock.send(JSON.stringify({ id: n, method, params }));
        };
        const body = url.includes('/api/transfers')
          ? transfers4
          : url.includes('/api/runtimes/install')
            ? install4
            : url.includes('/api/runtimes')
              ? status4
              : null;
        if (body === null) {
          post('Fetch.continueRequest', { requestId: m.params.requestId });
        } else {
          post('Fetch.fulfillRequest', {
            requestId: m.params.requestId,
            responseCode: 200,
            responseHeaders: [
              { name: 'content-type', value: url.includes('install') ? 'text/event-stream' : 'application/json' },
            ],
            body,
          });
        }
      }
    });
    const call = (method: string, params: Record<string, unknown> = {}) =>
      new Promise<any>((resolve) => {
        const n = ++cid;
        cwait.set(n, resolve);
        sock.send(JSON.stringify({ id: n, method, params }));
      });
    const run = async (expression: string) => {
      const r = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      if (r.result?.exceptionDetails) {
        throw new Error(String(r.result.exceptionDetails.exception?.description ?? 'eval failed'));
      }
      return r.result?.result?.value;
    };
    const until = async (expression: string, timeoutMs = 20000) => {
      const end = Date.now() + timeoutMs;
      for (;;) {
        if (await run(`Boolean(${expression})`)) return true;
        if (Date.now() > end) return false;
        await sleep4(150);
      }
    };

    await call('Page.enable');
    await call('Runtime.enable');
    await call('Fetch.enable', {
      patterns: [
        { urlPattern: `${APP_URL}/api/runtimes*` },
        { urlPattern: `${APP_URL}/api/transfers*` },
      ],
    });
    await call('Emulation.setDeviceMetricsOverride', { width: 1000, height: 1000, deviceScaleFactor: 1, mobile: false });
    await call('Page.navigate', { url: 'about:blank' });
    await sleep4(400);
    await call('Page.navigate', { url: `${APP_URL}/settings#models/runtimes` });
    if (!(await until(`document.querySelectorAll('.rt-row').length > 0`))) {
      throw new Error('the runtime list did not render');
    }
    await run(`(() => {
      const row = [...document.querySelectorAll('.rt-row')]
        .find(r => r.querySelector('.rt-row__title')?.textContent === 'ONNX Runtime');
      const box = row.querySelector('input');
      box.checked = true; box.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    })()`);
    await sleep4(300);

    // A timer, not requestAnimationFrame: headless throttles rAF so hard that a
    // sampler built on it saw five frames in six seconds, and every check built
    // on it passed for the wrong reason.
    await run(`(() => {
      window.__samples = [];
      const tick = () => {
        const state = document.getElementById('rt-log-state');
        const bar = state.querySelector('.bar');
        const vis = getComputedStyle(state).display !== 'none';
        const head = document.querySelector('.rt-log__head');
        window.__samples.push({
          vis,
          gap: vis ? Math.round(head.getBoundingClientRect().right - bar.getBoundingClientRect().right) : -1,
        });
        setTimeout(tick, 16);
      };
      tick();
      return true;
    })()`);
    await run(`document.getElementById('rt-install').click(); true`);
    await sleep4(6000);
    const live = JSON.parse(
      String(
        await run(`JSON.stringify((() => {
          const v = window.__samples.filter(s => s.vis);
          return { visible: v.length, ticks: window.__samples.length,
                   endsHidden: getComputedStyle(document.getElementById('rt-log-state')).display === 'none',
                   gap: v.length ? Math.max(...v.map(s => Math.abs(s.gap))) : null };
        })())`),
      ),
    );
    report('the indicator comes up during an install and goes down after', live.visible > 0 && live.endsHidden, JSON.stringify(live));
    // A log that re-lays itself out once per line locks the tab. Measured side
    // by side: 3 ticks in 6s with the per-line layout read, 91 without, and
    // 32 on a busy machine where the runtimes panel is also re-rendering.
    // The threshold sits between the broken number and the busy one with room
    // on each side — this is here to catch the regression, not to benchmark
    // headless timers on a machine that is already busy.
    report('the page is not blocked while a long install streams', live.ticks > 15, `${live.ticks} ticks in 6s`);
    report('and the bar sits against the panel edge throughout', live.gap !== null && live.gap <= 1, JSON.stringify(live));

    // Whether the bar moves is a layout question about the heading, not about
    // the install — `logWorking` only sets the label's textContent — so it is
    // measured directly instead of racing a stream that lands in one burst.
    const runs = JSON.parse(
      String(
        await run(`JSON.stringify((() => {
          const state = document.getElementById('rt-log-state');
          const label = document.getElementById('rt-log-statelabel');
          const bar = state.querySelector('.bar');
          state.hidden = false;
          const measure = (text) => {
            label.textContent = text;
            const b = bar.getBoundingClientRect();
            const l = label.getBoundingClientRect();
            return { left: Math.round(b.left), right: Math.round(b.right),
                     labelW: Math.round(l.width), labelLeft: Math.round(l.left),
                     cut: label.scrollWidth > label.clientWidth + 1 };
          };
          const out = [
            'Collecting',
            'Building wheel for onnxruntime (cp313)',
            'Created wheel for onnxruntime: filename=onnxruntime-1.23.0-cp313-cp313-linux_x86_64.whl size=17123456',
            'Installing collected packages: onnxruntime',
          ].map(measure);
          state.hidden = true;
          return out;
        })())`),
      ),
    );
    report(
      'the bar does not move for any label length',
      new Set(runs.map((r: any) => r.left)).size === 1 && new Set(runs.map((r: any) => r.right)).size === 1,
      JSON.stringify(runs),
    );
    report(
      'nor does the label change the width of its own track',
      new Set(runs.map((r: any) => r.labelW)).size === 1 && runs[0].labelW >= 60,
      JSON.stringify(runs),
    );
    report(
      'nor do the words jump sideways under it',
      new Set(runs.map((r: any) => r.labelLeft)).size === 1,
      JSON.stringify(runs.map((r: any) => r.labelLeft)),
    );
    report('and a long line is cut rather than running past the panel', runs[3].cut === true, JSON.stringify(runs[3]));

    await call('Page.navigate', { url: 'about:blank' });
    await sleep4(400);
    await call('Page.navigate', { url: `${APP_URL}/settings#models/downloads` });
    await sleep4(2500);
    if (!(await until(`document.querySelectorAll('.dl-row').length >= 2`))) {
      throw new Error('the transfer rows did not render');
    }
    const rows = JSON.parse(
      String(
        await run(`JSON.stringify([...document.querySelectorAll('.dl-row')].map(r => {
          const top = r.querySelector('.dl-row__top');
          const status = r.querySelector('.dl-row__status');
          return {
            over: Math.round(status.getBoundingClientRect().right - top.getBoundingClientRect().right),
            scrollOver: Math.round(r.scrollWidth - r.clientWidth),
            titled: Boolean(status.title && status.title.length),
          };
        }))`),
      ),
    );
    report('no download row spills its text past the card', rows.every((r: any) => r.over <= 1), JSON.stringify(rows));
    report('and nothing scrolls sideways inside one', rows.every((r: any) => r.scrollOver <= 0), JSON.stringify(rows));
    report('while the full text stays reachable on hover', rows.every((r: any) => r.titled), JSON.stringify(rows));

    /* What the buttons say. A finished download has nothing to resume, so it
       offers Delete and no Resume; a paused one offers Resume; a failed one
       offers Retry. Stubbed in the page rather than through the CDP Fetch
       domain: the ticker polls /api/transfers on its own, and the stub has to
       answer every poll. The three states are the server's own words —
       "paused" is a deliberate stop that kept the partial file, anything else
       with an error is a failure — so the rows have to read the same. */
    await run(`(() => {
      const rows = [
        { id: 'dl-finished', kind: 'download', repo: 'acme/finished', file: 'a.bin',
          state: 'done', received: 1024, total: 1024, startedAt: 1 },
        { id: 'dl-paused', kind: 'download', repo: 'acme/paused', file: 'b.bin',
          state: 'error', error: 'download paused', received: 512, total: 1024, startedAt: 2 },
        { id: 'dl-failed', kind: 'download', repo: 'acme/failed', file: 'c.bin',
          state: 'error', error: 'HTTP 404', received: 0, total: 1024, startedAt: 3 },
      ];
      const original = window.fetch;
      window.__origTransfersFetch = original;
      window.fetch = (url, opts) =>
        String(url).includes('/api/transfers')
          ? Promise.resolve(new Response(JSON.stringify({ active: [], recent: rows }), {
              status: 200,
              headers: { 'content-type': 'application/json' },
            }))
          : original(url, opts);
      document.querySelector('[data-subview="downloads"]').click();
      return true;
    })()`);
    if (!(await until(`document.querySelectorAll('.dl-row').length === 3`, 15000))) {
      throw new Error('the stubbed transfer rows did not render');
    }
    const labelRows = JSON.parse(
      String(
        await run(`JSON.stringify([...document.querySelectorAll('.dl-row')].map(r => ({
          repo: r.querySelector('.dl-row__name')?.textContent ?? '',
          status: r.querySelector('.dl-row__status')?.textContent ?? '',
          cls: r.className,
          buttons: [...r.querySelectorAll('button')].map(b => b.textContent.trim()),
        })))`),
      ),
    );
    const byRepo = (frag: string) => labelRows.find((r: any) => r.repo.includes(frag));
    const finishedRow = byRepo('finished');
    report(
      'a finished download offers Delete and no Resume',
      Boolean(finishedRow) &&
        finishedRow.buttons.includes('Delete') &&
        !finishedRow.buttons.some((b: string) => /Resume|Retry/.test(b)),
      JSON.stringify(finishedRow),
    );
    const pausedRow = byRepo('paused');
    report(
      'a paused download says so and offers Resume',
      Boolean(pausedRow) && pausedRow.status === 'paused' && pausedRow.buttons.includes('Resume'),
      JSON.stringify(pausedRow),
    );
    const failedRow = byRepo('failed');
    report(
      'a failed download offers Retry, not Resume',
      Boolean(failedRow) && failedRow.buttons.includes('Retry') && !failedRow.buttons.includes('Resume'),
      JSON.stringify(failedRow),
    );
    await run(`window.fetch = window.__origTransfersFetch; true`);

    /* The models page: one button for the runtimes the models are waiting on,
       not one per group. Three models waiting on three different runtimes
       used to put three buttons on the page, and all three did the same
       thing — they opened the Runtimes tab. */
    console.log('\nBrowser: one button for the runtimes the models are waiting on');
    {
      await call('Page.navigate', { url: 'about:blank' });
      await sleep4(400);
      await call('Page.navigate', { url: `${APP_URL}/settings#models/local` });
      if (!(await until(`!!document.getElementById('hub-runtimes')`, 30000))) {
        throw new Error('the settings page did not load');
      }
      /* The status the page is given: nothing installed, so several runtime
         groups are missing at once. Stubbed inside the page rather than
         through the CDP Fetch domain — that pauses every matching request,
         and the hub's own file list came back empty on the runs where it was
         armed for the whole API. */
      await run(`(() => {
        const body = ${JSON.stringify(JSON.stringify(nothingInstalled4))};
        const orig = window.fetch;
        window.fetch = (...args) => {
          if (String(args[0]).includes('/api/runtimes')) {
            return Promise.resolve(new Response(body, { status: 200, headers: { 'content-type': 'application/json' } }));
          }
          return orig(...args);
        };
        return true;
      })()`);
      /* Re-open the subview, so the hub is read again with the stub in place.
         The click is what re-reads it — the panel is a view, and a view is
         re-read when it is shown. */
      await run(`(() => { const b = [...document.querySelectorAll('[data-subview]')].find(x => x.dataset.subview === 'local'); b?.click(); return true; })()`);
      /* Wait for the hub itself, not for the first paint of it. The hub's
         first render can land before the file list has arrived, and that
         render says "no downloaded weights yet" — which is a true thing to
         say at that moment and not what this check is about. */
      const rendered = await until(
        `document.querySelectorAll('#hub-runtimes .hub-group').length > 0 || document.querySelectorAll('#hub-runtimes .hub-group__actions button').length > 0`,
        45000,
      );
      if (!rendered) {
        const where = String(
          await run(`JSON.stringify({
            url: location.href,
            panels: [...document.querySelectorAll('[data-subpanel]')].map(p => p.dataset.subpanel + ':' + (p.hidden ? 'hidden' : 'shown')),
            hub: (document.getElementById('hub-runtimes')?.textContent ?? '').slice(0, 200),
          })`),
        );
        throw new Error(`the hub did not render — ${where}`);
      }
      await sleep4(600);
      const hub = JSON.parse(
        String(
          await run(`JSON.stringify({
            groups: document.querySelectorAll('#hub-runtimes .hub-group').length,
            missing: document.querySelectorAll('#hub-runtimes .hub-group.is-missing').length,
            installButtons: document.querySelectorAll('#hub-runtimes .hub-group__actions button').length,
            notes: [...document.querySelectorAll('#hub-runtimes .hub-group__note')].map(n => n.textContent),
          })`),
        ),
      );
      report(
        'the models page shows one button, not one per runtime group',
        hub.installButtons === 1,
        JSON.stringify(hub),
      );
      report('even when several runtimes are missing at once', hub.missing >= 2, JSON.stringify(hub));
      report(
        'and each group still says what it is waiting on',
        hub.notes.filter((n: string) => /waiting on/.test(n)).length >= 2,
        JSON.stringify(hub.notes),
      );
    }

    await call('Fetch.disable');
    sock.close();
  } catch (err) {
    failures.push('the anchored-bar and download-row checks');
    console.log(`  FAIL  the anchored-bar and download-row checks — ${(err as Error).message}`);
  } finally {
    done4();
  }
}

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);
