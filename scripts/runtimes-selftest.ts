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
import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  describeVenvFailure,
  downloadTool,
  ensureVenv,
  installRuntimes,
  missingInstallerPrerequisite,
  MISSING_PIP_MESSAGE,
  MissingVenvPackage,
  needsPrivilege,
  ollamaCommand,
  ollamaRecipe,
  ollamaVulkanEnabled,
  permissionMessage,
  RUNTIME_GROUPS,
  runtimeStatus,
  unwritableAncestor,
  venvHasPip,
  venvPython,
} from '../src/runtimes.ts';

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
  const realStatus = await (
    await fetch(`${APP_URL}/api/runtimes`, { signal: AbortSignal.timeout(60000) })
  ).json();
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

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) console.log(failures.map((f) => `  - ${f}`).join('\n'));
process.exit(failures.length ? 1 : 0);
