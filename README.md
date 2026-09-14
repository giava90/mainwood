# mainwood — ForClim → SorSim

Turns **ForClim** forest-dynamics simulations into **wood assortments** (which log grades,
in what volume, per year and per stand), then into regional summaries and figures.

Two stages, both submitted to SLURM on ETH Euler:

```
ForClim cohort tables                       MAINWOOD_INPUT_TEMPLATE
   │  code/convert_data.py        (run_conversion.sh)   ← needs Java
   ▼
<output root>/<region>/outputs/<scenario>/sorsim_output*.csv
   │  code/summarize_and_create_plots.py  (run_analysis.sh)
   ▼
<summary dir>/<region>_<scenario>.parquet   ← the deliverable
figures/*.png
```

Every path in that diagram is configured per machine, never edited in the source.

## Quick start

Once per machine. `code/local.env` is git-ignored, so `git pull` on Euler never
conflicts and no tracked file is edited to change machine, region or scenario:

```bash
cd code
cp local.env.example local.env      # uncomment the Euler block
python setup_data_tree.py           # build the folder skeleton (idempotent)
```

Then every run:

```bash
cd code
module load stack/2024-06 python/3.12.8
module load stack/2024-06 openjdk/21.0.3_9   # stage 1 only

git pull
python -m pytest ../                              # 255 tests, ~15 s
python preflight.py Jurapark WOOD dead            # exits 1 if the run would fail
./run_conversion.sh WOOD Jurapark dead            # stage 1  → assortments
./run_analysis.sh  Jurapark WOOD dead             # stage 2  → summaries + figures
```

Region, scenario and cohort are arguments; sbatch resources are environment variables
(`N_CORES=8 CONVERT_WALLTIME=24:00:00 ./run_conversion.sh ALL All`). Neither wrapper is
ever edited. Full details in [docs/03-runbook.md](docs/03-runbook.md).

Interactively, without SLURM:

```bash
python convert_data.py WOOD True 4 False Jurapark          # 50-file sample, dead
python convert_data.py WOOD True 4 False Jurapark alive    # alive cohort
python summarize_and_create_plots.py Jurapark WOOD ../data 4 False
python summary_to_csv.py ../data/summaries_for_plots/ --gzip   # CSV for collaborators
```

## Configuration

All in `code/local.env` (template: `code/local.env.example`). The environment beats the
file, so a one-off run can override any of these without editing it. Templates expand
`{case_study}`, `{scenario}` and `{cohort}`.

| variable | what it names |
|---|---|
| `MAINWOOD_INPUT_TEMPLATE` | where stage 1 reads ForClim output |
| `MAINWOOD_INPUT_TEMPLATE_ALIVE` / `_DEAD` | per-cohort override — the cohorts are on different filesystems |
| `MAINWOOD_INTERMEDIATE_TEMPLATE` | where saved SorSim tree lists are read back |
| `MAINWOOD_OUTPUT_TEMPLATE` | region root holding `intermediate/` and `outputs/` |
| `MAINWOOD_DATA_ROOT` | stage 2's `folder_data` default |
| `MAINWOOD_STAND_DETAILS` | the `stand.details.csv` for a region |
| `MAINWOOD_SUMMARY_DIR` | where the deliverable is written |
| `MAINWOOD_SAMPLE_SIZE` | files used when `use_sample=True` |

sbatch defaults: `N_CORES`, `CONVERT_CORES`, `CONVERT_MEM_PER_CPU`, `CONVERT_WALLTIME`,
`ANALYSIS_CORES`, `ANALYSIS_MEM_PER_CPU`, `ANALYSIS_WALLTIME`, `MAIL_TYPE`.

## Cohorts

| cohort | ForClim column | filter | meaning | SorSim output name |
|---|---|---|---|---|
| `dead` (default) | `dtrees` | `type == 2` | harvested trees | `sorsim_output<stand>_<simtype>.csv` |
| `alive` | `trees` | none | standing stock | `sorsim_alive_output<stand>_<simtype>.csv` |

All naming rules live in one place: [`code/naming.py`](code/naming.py). Regions and
scenarios live in [`code/regions.py`](code/regions.py) — one line to add a region.

## Excluded stands

Stage 1 skips stands it cannot compute, before SorSim runs, and writes
`excluded_stands_<region>_<scenario>_<cohort>.csv` to the region root naming each one
and why: `area_ha <= 0` or absent from `stand.details.csv`. That file is the list to
hand back to the ForClim side. See [docs/03-runbook.md](docs/03-runbook.md) §2.5.

## Layout

```
code/                       the pipeline
  convert_data.py           stage 1 — ForClim → SorSim assortments
  summarize_and_create_plots.py   stage 2 — summaries + figures
  preflight.py              check a run before submitting it
  setup_data_tree.py        build the folder skeleton
  import_stand_details.py   import a ForClim stand.details delivery
  exclusions.py             which stands cannot be computed, and why
  make_paper_figures.py     the SZF paper figures, from the summaries
  plotting_tools_for_paper.py   their figure functions, verbatim from the paper code
  scaling_benchmark.sh      submit a cores x samples grid for stage 1
  scaling_run.py            one measured grid point
  plot_scaling.py           the scaling table and plot
  paths.py                  per-machine paths, from local.env
  regions.py                the one list of regions and scenarios
  naming.py                 the one set of file-name rules
  summary_io.py             the Parquet/CSV deliverable
  run_*.sh                  SLURM wrappers (arguments, never edited)
  local.env                 per-machine config — git-ignored
minimal/     the ForClim→SorSim converter and the vendored SorSim jar + Java sources
data/        reference data, tracked; simulation data, ignored (see the inventory)
figures/     generated PNGs — git-ignored, regenerated by stage 2
tests/       255 pytest tests, pure Python, ~15 s
docs/        the documents below
```

Requirements: `requirements.txt`. Java is needed for stage 1 only. On Euler `pyarrow`
is not in the module stack — `pip install --user pyarrow` (preflight checks it).

## Documentation

| | |
|---|---|
| [docs/01-pipeline.md](docs/01-pipeline.md) | How the pipeline works, and the conventions you have to know |
| [docs/02-data-inventory.md](docs/02-data-inventory.md) | What every folder and file is, and where it came from |
| [docs/03-runbook.md](docs/03-runbook.md) | **Start here to run something** — including onboarding a new region |
| [docs/05-testing.md](docs/05-testing.md) | The test suite and what it protects |
| [docs/06-known-issues.md](docs/06-known-issues.md) | Bugs found and fixed, and the fragile spots left in place |
| [docs/07-performance.md](docs/07-performance.md) | Where the time and the memory go, with measured speed-ups |
| [docs/09-summary-format.md](docs/09-summary-format.md) | Parquet, how to get CSV back, and how to read them from R |
| [docs/10-environments.md](docs/10-environments.md) | The Python and R installs on the workstation |
| [docs/11-changelog.md](docs/11-changelog.md) | What changed in September 2026, and why |

Background, superseded proposals and restructuring notes:
[docs/04-cleanup-proposal.md](docs/04-cleanup-proposal.md),
[docs/08-refactoring.md](docs/08-refactoring.md).
