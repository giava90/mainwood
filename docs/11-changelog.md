# Changelog — September 2026

What changed, and why. The README describes the pipeline as it is now; this file is the
history, so nothing here is needed to run anything.

## The problem behind most of it

Configuration lived inside tracked source. The Euler paths were commented-out lines at
the bottom of `convert_data.py`; region, scenario and cohort were variables at the top of
the SLURM wrappers. Switching between the laptop and the cluster therefore meant editing
files git tracks, so every `git pull` on Euler conflicted. That produced a stale `euler`
branch, and on the Euler checkout, 23 commits of which 20 were `Merge branch 'main'` from
repeated pulls over local edits.

Fixing that surfaced a family of related defects: **a path taken from an argument in one
place and hardcoded in another**, so the two agreed on a laptop and disagreed on Euler —
the only machine where it mattered.

## Configuration

- `code/paths.py` resolves every path from templates expanding `{case_study}`,
  `{scenario}` and `{cohort}`, read from the environment first and from a git-ignored
  `code/local.env` second. Defaults are the old repository-relative paths, so an
  unconfigured machine is unchanged.
- `code/load_env.sh` gives the shell wrappers the same precedence. Sourcing `local.env`
  with `set -a` had assigned unconditionally, so `MAINWOOD_INPUT_TEMPLATE=... ./run_conversion.sh`
  was silently ignored while `paths.py` honoured it. Writing a test that parsed one file
  with both readers then found a second drift: Python accepted keys the shell cannot
  assign.
- The wrappers take region, scenario and cohort as arguments, validate them before
  loading modules, name their jobs, and write identifiable `slurm-*.out` logs.
- `code/regions.py` is the one list of regions and scenarios. It had been copy-pasted
  into five entry points which had already diverged — `plot_only` silently dropped
  `HYBRID` while the others kept it.

## Defects found and fixed

| what | consequence |
|---|---|
| `data/Vaud/stand.details.csv` had **no `area_ha` column** — an older ForClim delivery | every Vaud volume from a fresh clone wrong or failing; the correct file existed but was never committed |
| stage 2 wrote the deliverable to a hardcoded `../data/summaries_for_plots/` | on Euler it read assortments from scratch and wrote summaries onto the home quota |
| `preflight` resolved `stand.details.csv` from `MAINWOOD_DATA_ROOT`, stage 2 from `../data/` | a green preflight proved nothing about the file stage 2 opens |
| `convert_data_plantations.py` — a 200-line copy of the converter differing in two path lines | forked before the `.csv` suffix fix, so uncompressed input produced an empty output folder silently; and it hardcoded `inputs/WOOD/` while taking a scenario argument, writing WOOD inputs into other scenarios' output folders |
| `pyarrow` absent from the Euler module stack | Parquet is the default format and is imported only at the final write, so stage 2 aborted after all the work |
| `data/` git-ignored while five files under it were tracked | `git add` warns on an ignored path and a chained command skips silently — how the Vaud file stayed uncommitted |

## Behaviour changes worth knowing

- **Excluded stands.** Stage 1 now skips stands with `area_ha <= 0` or absent from
  `stand.details.csv`, before SorSim runs, and reports them to
  `excluded_stands_<region>_<scenario>_<cohort>.csv`. Previously the first produced a
  confident `0` and the second produced `NaN`. Entlebuch has three of the first kind:
  2501, 2625, 3471.
- **Parquet falls back to CSV** if `pyarrow` is missing, loudly, rather than discarding a
  finished stage 2.
- **Folders are created on demand.** Stage 1 makes its own `intermediate/` and
  `outputs/`; `setup_data_tree.py` replaces the manual `mkdir` skeleton.
- **Jurapark** was onboarded. Its delivery spells the area column `Area_ha`; every other
  region writes `area_ha`. `import_stand_details.py` normalises that and records the
  source md5. It has no `Above1000m` column, so the altitude-split figures do not work
  for it — nothing was guessed to fill the gap.

## Corrections to earlier claims in this repository

- The warning that Surselva summaries "will not fit a home quota" was an absolute claim
  made without knowing the quota, and derived from a pre-Parquet measurement. Measured
  extrapolation: ~1.0 GB per scenario as Parquet, ~5.5 GB as CSV. The advice is
  unchanged — they belong on scratch — but the certainty was misplaced.
- `data/Entlebuch/outputs/WOOD/*.csv` and `data/summaries_for_plots/Entlebuch_BAU.csv`
  were untracked, as [04-cleanup-proposal.md](04-cleanup-proposal.md) §B had recommended.
  Five reference files remain tracked: four `stand.details.csv` and
  `fraction_quality.xlsx`.
- `README.txt` and `bash_code_to_create_folder_structure_for_data.sh` are superseded by
  [03-runbook.md](03-runbook.md) §0–1.2 and `setup_data_tree.py`.

## Tests

0 → 171, all passing on Euler. The suite is pure Python and runs in about 12 seconds.
Its job is to fail in seconds on the things that otherwise fail hours into a job: naming,
planting weights, area rescaling, path resolution, the region list, config precedence,
the exclusion rules, and the summary format round trip.
