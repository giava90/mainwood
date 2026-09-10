# mainwood — ForClim → SorSim

Turns **ForClim** forest-dynamics simulations into **wood assortments** (which log grades,
in what volume, per year and per stand), then into regional summaries and figures.

Two stages, both submitted to SLURM on ETH Euler:

```
ForClim cohort tables
   │  code/convert_data.py            (run_conversion.sh)  ← needs Java
   ▼
data/<region>/outputs/<scenario>/sorsim_output*.csv
   │  code/summarize_and_create_plots.py  (run_analysis.sh)
   ▼
data/summaries_for_plots/<region>_<scenario>.csv   ← the deliverable
figures/*.png
```

## Documentation

| | |
|---|---|
| [docs/01-pipeline.md](docs/01-pipeline.md) | How the pipeline works, and the conventions you have to know (cohorts, `simtype`, planting weights, file naming) |
| [docs/02-data-inventory.md](docs/02-data-inventory.md) | What every folder and file in here is, and where it came from |
| [docs/03-runbook.md](docs/03-runbook.md) | **Start here to run something** — including onboarding a new case study region |
| [docs/04-cleanup-proposal.md](docs/04-cleanup-proposal.md) | What to keep and what to delete (proposal only — nothing has been deleted) |
| [docs/05-testing.md](docs/05-testing.md) | The test suite and what it protects |
| [docs/06-known-issues.md](docs/06-known-issues.md) | Bugs found and fixed, and the fragile spots left in place |
| [docs/07-performance.md](docs/07-performance.md) | Where the time and the 20 GB actually go, with measured speed-ups |
| [docs/08-refactoring.md](docs/08-refactoring.md) | What is worth restructuring, and what is not |

## Quick start

```bash
cd code
module load stack/2024-06 python/3.12.8
module load stack/2024-06 openjdk/21.0.3_9

# stage 1 — dead cohort (harvested trees), 50-file sample
python convert_data.py WOOD True 4 False Entlebuch

# stage 1 — alive cohort (standing stock)
python convert_data.py WOOD True 4 False Entlebuch alive

# stage 2 — summaries + figures
python summarize_and_create_plots.py Entlebuch WOOD ../data 4 100
```

Run tests from the repository root:

```bash
python -m pytest
```

## Cohorts

| cohort | ForClim column | filter | meaning | SorSim output name |
|---|---|---|---|---|
| `dead` (default) | `dtrees` | `type == 2` | harvested trees | `sorsim_output<stand>_<simtype>.csv` |
| `alive` | `trees` | none | standing stock | `sorsim_alive_output<stand>_<simtype>.csv` |

Dead-cohort names are unchanged from every earlier run. All naming rules live in one
place: [`code/naming.py`](code/naming.py).

## Layout

```
code/        the pipeline (stage 1, stage 2, plotting, SLURM wrappers)
minimal/     the ForClim→SorSim converter and the vendored SorSim jar + Java sources
data/        reference data (kept) and simulation data (regenerable — see the inventory)
figures/     generated PNGs — git-ignored, reproducible via code/plot_only.py
tests/       pytest suite, pure Python, ~5 s
docs/        the documents listed above
```

Requirements: `requirements.txt`. Java is needed for stage 1 only.
