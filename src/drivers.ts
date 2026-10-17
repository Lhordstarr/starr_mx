/**
 * Runs the GPU driver installer.
 *
 * The script already exists and is distro-aware; this is a thin, honest wrapper
 * rather than a reimplementation. The one thing it adds is refusing to pretend:
 * `sudo` inside a server process has no terminal to prompt on, so the install
 * cannot silently half-happen. When privilege is required the wrapper reports
 * it and hands back the exact command, instead of a stream of coloured output
 * that ends in "sudo: a terminal is required".
 */

import { spawn } from 'node:child_process';
import path from 'node:path';

export interface DriverRunOptions {
  /** false = dry run, which is all the script does without --yes. */
  apply: boolean;
  onLog: (line: string) => void;
  /** Called when the script needs root it cannot get. */
  onNeedsPrivilege: (command: string) => void;
  /**
   * The user's password, used once to satisfy `sudo` and never kept.
   *
   * Written to the child's stdin, never to its argument vector — argv is world
   * readable through `ps`, so a password there would be visible to every process
   * on the machine for as long as the install ran.
   */
  password?: string;
}

const SCRIPT = path.resolve(process.cwd(), 'scripts/install-gpu-drivers.sh');

/** Patterns that mean "this stopped because it wanted a password". */
/**
 * Strips CSI/OSC colour codes. The script colours its output for a terminal;
 * in a browser those bytes arrive as literal "^[1m" text on screen.
 */
const ANSI = /\u001b\][^\u0007]*\u0007|\u001b\[[0-9;]*[A-Za-z]|\u001b[@-Z\\-_]/g;

const PRIVILEGE_HINTS = [
  /a terminal is required to read the password/i,
  /sudo:\s*no tty present/i,
  /ask for a password/i,
  /permission denied/i,
];

export async function runDriverInstaller(opts: DriverRunOptions): Promise<void> {
  const args = opts.apply ? ['--yes'] : [];
  const base = `scripts/install-gpu-drivers.sh${args.length ? ` ${args.join(' ')}` : ''}`;
  // With a password the invocation goes through `sudo -S`, which reads it from
  // stdin. Without one the script runs exactly as before and reports that it
  // needs a terminal — the honest answer when there is nobody to type a
  // password into.
  const shown = opts.password ? `sudo ${base}` : base;
  const password = opts.password ?? '';
  // Last line of defence for the secret: if any tool ever echoes it back, it
  // never reaches the browser, the transcript or a console line.
  const scrub = (text: string) =>
    password.length >= 3 ? text.split(password).join('[redacted]') : text;

  // Declared out here, not inside the executor: it is read after the promise
  // settles to decide whether the failure was a missing password, and a
  // variable scoped to the executor is gone by then.
  let tail = '';

  const code = await new Promise<number>((resolve) => {
    let child: ReturnType<typeof spawn>;
    try {
      // `sudo -S` reads the password from stdin. `-p ''` suppresses sudo's own
      // prompt so it cannot sit waiting for a second one. `-k` is deliberately
      // NOT passed: if the user authenticated in a terminal a moment ago,
      // sudo's own cache is what stops this asking twice.
      const [cmd, cmdArgs] = password
        ? ['sudo', ['-S', '-p', '', 'bash', SCRIPT, ...args]]
        : ['bash', [SCRIPT, ...args]];
      child = spawn(cmd, cmdArgs, {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, DEBIAN_FRONTEND: 'noninteractive' },
      });
    } catch (err) {
      opts.onLog(`Could not start the installer: ${(err as Error).message}`);
      resolve(127);
      return;
    }

    // Feed the password, then close: sudo retries on stdin when it is wrong,
    // and a pipe left open would leave the install hanging on a second prompt.
    if (password) {
      child.stdin?.on('error', () => {
        /* sudo exited early — a wrong password does exactly that */
      });
      child.stdin?.end(`${password}\n`);
    }

    // One buffer per stream. Sharing one across stdout and stderr interleaves
    // them into nonsense: whichever handler runs second keeps half a line the
    // other already split, and the two get glued together.
    const bufs: Record<'out' | 'err', string> = { out: '', err: '' };
    const emit = (raw: string) => {
      const text = scrub(raw.replace(ANSI, '').replace(/\s+$/, ''));
      if (!text) return;
      opts.onLog(text);
      // Only a window of recent output is kept: a long apt transcript would
      // otherwise match some unrelated "permission denied" from an unrelated
      // package and report a privilege problem that never happened.
      tail = `${tail}\n${text}`.slice(-4000);
    };
    const pump = (stream: NodeJS.ReadableStream | null, key: 'out' | 'err') => {
      stream?.setEncoding('utf8');
      stream?.on('data', (c: string) => {
        bufs[key] += c;
        const lines = bufs[key].split('\n');
        bufs[key] = lines.pop() ?? '';
        for (const line of lines) emit(line);
      });
    };
    pump(child.stdout, 'out');
    pump(child.stderr, 'err');

    child.on('error', (err) => {
      opts.onLog(`Could not start the installer: ${err.message}`);
      resolve(127);
    });
    child.on('close', (c) => {
      // A short script can end without a trailing newline, which leaves the
      // last line sitting in its buffer forever. The dry run is exactly that
      // short, and it is the whole point of the dry run.
      for (const key of ['out', 'err'] as const) {
        emit(bufs[key]);
        bufs[key] = '';
      }
      resolve(c ?? 0);
    });
  });

  // A refused password is not a broken script, and saying so is what stops the
  // user going looking for one.
  if (password && /incorrect password|N Sorry, try again|authentication failure/i.test(tail)) {
    opts.onLog('That password was not accepted, so nothing was changed.');
    return;
  }

  if (opts.apply && code !== 0 && PRIVILEGE_HINTS.some((re) => re.test(tail))) {
    // Not a failure of the script — a failure of the context. Saying "the
    // installer failed" here would send the user looking for a broken script
    // instead of a missing password.
    opts.onNeedsPrivilege(`${shown}   # needs your password`);
    opts.onLog(
      'The server could not elevate. Run the command above in a terminal, where ' +
        'it can ask for your password directly.',
    );
    return;
  }

  opts.onLog(
    opts.apply
      ? `Installer finished with status ${code}.`
      : `Dry run finished with status ${code}. Nothing was changed.`,
  );
}