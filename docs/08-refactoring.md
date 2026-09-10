# Is a refactor worth it?

**Short answer: no big rewrite. Yes to four targeted, mechanical changes.**

This is research code whose output has been used. A rewrite risks changing published
numbers in ways nobody would notice, and the only defence against that is the test suite —
which covers the *behaviour* of individual functions, not a full-pipeline golden output.
So: change things the tests can verify, and leave the rest alone.

## Do these

### 1. Delete the dead plotting code (safe, immediate)

Four functions in `code/summarize_and_create_plots.py` are never called:

| function | lines |
|---|---|
| `plot_biomass` | ~50 |
| `plot_normalized_biomass_for_sawmill_categories` | ~110 |
| `plot_normalized_biomass_for_sawmill_categories_and_altitues` | ~105 |
| `plot_normalized_biomass_for_sawmill_categories_and_altitues_old` | ~155 |

Together with `rolling_stats` (used only by those) and `calculate_biomass_not_for_sawmills`
(replaced by a subtraction), that is roughly **450 of 1 270 lines**. The two live plotting
functions are `plot_percentages_of_wood` and `plot_biomass_by_diameter_class`.

The commented-out block at the end of `process_combination` was already removed in the
working tree — same category.

Git remembers all of it; nothing is lost by deleting.

### 2. Stop duplicating `plot_only.py` — it has already drifted

`code/plot_only.py` is 305 lines, of which **256 are near-copies** of functions in
`summarize_and_create_plots.py`:

| function | lines | similarity |
|---|---|---|
| `plot_biomass_by_diameter_class` | 65 | 99% |
| `plot_percentages_of_wood` | 73 | 99% |
| `plot_percentages_of_wood_quality` | 67 | 98% |
| `process_combination` | 51 | 50% |

"99% similar" rather than identical is the problem: the copies have already diverged, and
so has the behaviour around them — `plot_only.py` still lists `valid_management_scenarios`
without `HYBRID`, and filters `simtype == 1` as an integer where the summariser uses the
string `'1'`. Both are individually correct in their own context, but nobody can tell that
by looking.

The fix is small and safe: move the plotting functions into `code/plots.py`, and have both
scripts import them. No logic changes, and the figures are byte-comparable before and
after.

### 3. Split the 1 270-line module along the seams it already has

After (1) and (2), `summarize_and_create_plots.py` naturally falls into:

```
code/naming.py     file-name rules                      (done)
code/plots.py      the two live plotting functions      (~140 lines)
code/summaries.py  load -> preprocess -> augment -> classify   (~350 lines)
code/summarize_and_create_plots.py   the CLI and the orchestration  (~120 lines)
```

This is worth doing because you are about to add a region and a cohort, i.e. more callers.
It is *not* worth doing as a general tidy-up — the value is that `summaries.py` becomes
importable from a notebook, which is what `quick_check.ipynb` was reaching for.

### 4. Take the performance changes in [07-performance.md](07-performance.md)

Those are refactoring too, and they are the ones with a measured payoff: 16.7× on the
read, 227× on the sawmill split, 26× on memory. Do these before any cosmetic restructuring
— they change the shape of the problem (no more `sample_size`, no more 20 GB job), and
that changes what the surrounding code needs to look like.

## Do not do these

| | Why not |
|---|---|
| Rewrite the pipeline as a package with a config file | The two CLIs are used by two `sbatch` scripts. Nothing else consumes them. The cost is real and the benefit is aesthetic |
| Restructure `preprocess_data`'s weight arithmetic | It is fragile-looking but correct ([known issue 6](06-known-issues.md)). Any change here changes published volumes. The tests pin the behaviour; leave the code |
| Unify `convert_data.py` and `convert_data_from_intermediate.py` | They now share `naming.py`, which was the part that actually drifted. The remaining ~100 lines of duplication are a `Pool.starmap` and an argument parser |
| Replace `multiprocessing` with something else | After the stage-2 read fix, stage 2 barely needs parallelism at all. Measure before rearchitecting |
| Touch `minimal/sorsim/` | Vendored third-party code from WSL |
| Rename the columns to drop the German/umlaut names | They match SorSim's own output and every existing summary file. A migration, not a cleanup |

## Suggested sequence

1. Commit what is currently uncommitted (see [04-cleanup-proposal.md](04-cleanup-proposal.md) §D).
2. Delete the dead plotting code — one commit, tests still green.
3. Apply performance changes 1–4 from [07-performance.md](07-performance.md), one commit each, tests green after each.
4. Extract `plots.py`, then `summaries.py`.
5. Only then add the new region — on a codebase that runs in minutes and where a mistake fails a test instead of a job.
