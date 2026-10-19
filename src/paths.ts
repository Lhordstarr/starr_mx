/**
 * Where the app keeps the things it installs.
 *
 * Its own module because two independent installers need it: `runtimes.ts`
 * owns the Python environments and Ollama, and `whispercpp.ts` owns a pinned
 * binary build. If the second imported the path from the first, every module
 * that loads either would be in a cycle with every module that loads the other
 * — and the function it wanted is one line.
 *
 * Single source of truth is the point. Two copies of this would drift the
 * moment one gained a second override, and the failure mode is an installer
 * that writes to a directory nothing else reads.
 */
import * as path from 'node:path';

/** Root of everything the app installs. `RUNTIMES_DIR` moves it for tests. */
export const runtimesDir = (): string => path.resolve(process.env.RUNTIMES_DIR ?? './.runtimes');