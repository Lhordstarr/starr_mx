/**
 * The source repositories behind one Downloads row.
 *
 * A model card can span repositories: Stable Diffusion 3.5's quants come from
 * second-state while its VAE comes from an un-gated mirror. Downloads still
 * shows one row for the card, because two rows for one model is not a useful
 * answer to "where is my model up to".
 *
 * Actions, however, must name the repository that owns each file. Pause aborts
 * a server controller keyed by source repo, Resume re-fetches from it, and
 * Delete must remove the VAE as well as the weights. This maps a displayed
 * group back to those sources and the exact files each one supplied.
 *
 * @param {{repo?: string, files?: Array<{repo?: string, sourceRepo?: string, file?: string}>}} group
 * @returns {Map<string, Set<string>>} source repo -> file paths
 */
export function transferSources(group) {
  const sources = new Map();
  for (const t of group.files ?? []) {
    const repo = t.sourceRepo ?? t.repo ?? group.repo;
    if (!repo) continue;
    if (!sources.has(repo)) sources.set(repo, new Set());
    if (t.file) sources.get(repo).add(t.file);
  }
  return sources;
}

/**
 * Every filename that can satisfy one catalog file slot.
 *
 * Kokoro's q4 weights have a second, smaller export (`q4f16`) that the runner
 * accepts. The catalog still prefers q4 for a fresh download, but a Browse
 * download or an older card can legitimately have only the alternative. A
 * slot is present when any of these names is on disk.
 *
 * @param {{name: string, oneOf?: string[]}} file
 * @returns {string[]}
 */
export function fileCandidates(file) {
  return [file.name, ...(file.oneOf ?? [])].filter(Boolean);
}

/**
 * The catalog files that are not on disk, counting `oneOf` alternatives as
 * present.
 *
 * @param {Array<{name: string, oneOf?: string[]}>} files
 * @param {string[]} present
 * @returns {string[]}
 */
export function missingCatalogFiles(files, present) {
  const have = new Set(present);
  return (files ?? [])
    .filter((file) => !fileCandidates(file).some((name) => have.has(name)))
    .map((file) => file.name);
}

/**
 * Groups a model row's files by the repository that owns each one.
 *
 * The hung-model row knows one repo for the card, but a bundled card's files
 * can have their own `repo` (SD 3.5's VAE). Every delete call has to name the
 * source that actually owns the file, or the VAE stays on disk after the row
 * disappears.
 *
 * @param {string} defaultRepo
 * @param {string[]} files
 * @param {Array<{name: string, repo?: string}>} specs
 * @returns {Map<string, string[]>}
 */
export function groupFilesBySource(defaultRepo, files, specs = []) {
  const owner = new Map(specs.map((f) => [f.name, f.repo ?? defaultRepo]));
  const bySource = new Map();
  for (const file of files) {
    const repo = owner.get(file) ?? defaultRepo;
    if (!bySource.has(repo)) bySource.set(repo, []);
    bySource.get(repo).push(file);
  }
  return bySource;
}
