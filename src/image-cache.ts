/**
 * Generated images kept on disk, so they outlive the tab that made them.
 *
 * Until now a cloud generation came back as base64 and the session stored it
 * as a data URL. That was the shortest path to two failures: browsers cap
 * localStorage around 5 MB, so a single 1024×1024 image (1–2 MB of base64) put
 * every saved session at risk, and `persist()` had to strip the bytes to stay
 * under the cap — which is exactly why a generated image came back as
 * "[image — not saved]" after a reload. Writing the file here and handing the
 * page a URL fixes both at once: sessions store a few bytes of path, and the
 * image is still there after a restart.
 *
 * Files are named `img-<ms>-<random>.<ext>` and served by name only
 * (`/api/image/file/:name`), which keeps the route from being a directory
 * browser. The directory is a cache, not an archive: it is swept oldest-first
 * once it grows past its cap, so it cannot fill the disk over time.
 */
import { randomBytes } from 'node:crypto';
import { mkdir, readdir, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

/** Where the images live. `IMAGE_CACHE_DIR` moves it for tests. */
export const imageCacheDir = () => path.resolve(process.env.IMAGE_CACHE_DIR ?? './.cache/images');

/** What the sweep aims to stay under, in bytes. */
const CACHE_CAP_BYTES = (Number(process.env.IMAGE_CACHE_CAP_MB ?? 1024) || 1024) * 1024 * 1024;

const EXT: Record<string, string> = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/webp': 'webp',
  'image/gif': 'gif',
};

/**
 * The only file names this cache hands out, and the only ones the serving
 * route will read. Fixed extensions and an own-naming scheme: no name that
 * arrives over HTTP can contain a separator or a dot-dot.
 */
const NAME = /^img-\d{13}-[0-9a-f]{6}\.(png|jpg|webp|gif)$/;

export function cacheNameError(name: string): string | null {
  return NAME.test(name) ? null : 'Bad image name';
}

export function imageCachePath(name: string): string {
  return path.join(imageCacheDir(), name);
}

/**
 * Stores one base64 image and returns its file name, or null when the mime
 * type is not something this serves (the caller then keeps the bytes in
 * memory, which still works for the session it is in — it just will not
 * survive a reload).
 */
export async function cacheImage(base64: string, mimeType: string): Promise<string | null> {
  const ext = EXT[mimeType];
  if (!ext) return null;
  const bytes = Buffer.from(base64, 'base64');
  if (!bytes.length) return null;
  const dir = imageCacheDir();
  await mkdir(dir, { recursive: true });
  const name = `img-${Date.now()}-${randomBytes(3).toString('hex')}.${ext}`;
  await writeFile(path.join(dir, name), bytes);
  void sweep();
  return name;
}

/**
 * Deletes the oldest files once the directory passes its cap.
 *
 * Fire-and-forget from `cacheImage` (the write must not wait on a directory
 * scan) and best-effort throughout: a cache that cannot be swept is still a
 * working cache, and an error here must never fail a generation. Targets 90%
 * of the cap so the sweep runs on file boundaries rather than every write.
 */
async function sweep(): Promise<void> {
  try {
    const dir = imageCacheDir();
    const entries = await readdir(dir, { withFileTypes: true });
    const files = await Promise.all(
      entries
        .filter((e) => e.isFile() && NAME.test(e.name))
        .map(async (e) => {
          const info = await stat(path.join(dir, e.name));
          return { name: e.name, size: info.size, mtime: info.mtimeMs };
        }),
    );
    let total = files.reduce((sum, f) => sum + f.size, 0);
    if (total <= CACHE_CAP_BYTES) return;
    const target = CACHE_CAP_BYTES * 0.9;
    for (const file of files.sort((a, b) => a.mtime - b.mtime)) {
      if (total <= target) break;
      try {
        await rm(path.join(dir, file.name), { force: true });
        total -= file.size;
      } catch {
        /* raced or locked; it counts against the budget for one more round */
      }
    }
  } catch {
    /* no cache yet, or the directory vanished under us */
  }
}

/** Total bytes and file count, for the Studio's "cached images" note. */
export async function cacheUsage(): Promise<{ files: number; bytes: number }> {
  try {
    const dir = imageCacheDir();
    const entries = await readdir(dir, { withFileTypes: true });
    const files = entries.filter((e) => e.isFile() && NAME.test(e.name));
    const sizes = await Promise.all(
      files.map(async (f) => (await stat(path.join(dir, f.name))).size),
    );
    return { files: files.length, bytes: sizes.reduce((a, b) => a + b, 0) };
  } catch {
    return { files: 0, bytes: 0 };
  }
}