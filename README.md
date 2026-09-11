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
| [docs/09-summary-format.md](docs/09-summary-format.md) | The summaries are Parquet; how to get CSV back, and how to read them from R |
| [docs/10-environments.md](docs/10-environments.md) | The Python and R installs on this machine, and how the broken `base` env was fixed |

## Quick start

Once per machine — paths live in `code/local.env`, which is git-ignored, so `git pull`
on Euler never conflicts and no tracked file is edited to switch machine or region:

```bash
cd code
cp local.env.example local.env      # uncomment the Euler block
```

Then every run is the same four commands:

```bash
cd code
module load stack/2024-06 python/3.12.8
module load stack/2024-06 openjdk/21.0.3_9   # stage 1 only

git pull
python preflight.py Entlebuch WOOD dead      # ~10 s; exits 1 if the run would fail
./run_conversion.sh WOOD Entlebuch dead      # stage 1  → assortments
./run_analysis.sh  Entlebuch WOOD dead       # stage 2  → summaries + figures
```

Region, scenario and cohort are arguments; sbatch resources are environment variables
(`N_CORES=8 CONVERT_WALLTIME=24:00:00 ./run_conversion.sh ALL All`). Neither wrapper
is ever edited. Full details in [docs/03-runbook.md](docs/03-runbook.md).

Interactively, without SLURM:

```bash
python convert_data.py WOOD True 4 False Entlebuch          # 50-file sample, dead
python convert_data.py WOOD True 4 False Entlebuch alive    # alive cohort
python summarize_and_create_plots.py Entlebuch WOOD ../data 4 100
python summary_to_csv.py ../data/summaries_for_plots/ --gzip   # CSV for collaborators
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
code/        the pipeline (stage 1, stage 2, plotting, SLURM wrappers, preflight)
code/local.env   per-machine paths — git-ignored, template in local.env.example
minimal/     the ForClim→SorSim converter and the vendored SorSim jar + Java sources
data/        reference data (kept) and simulation data (regenerable — see the inventory)
figures/     generated PNGs — git-ignored, reproducible via code/plot_only.py
tests/       119 pytest tests, pure Python, ~10 s
docs/        the documents listed above
```

Requirements: `requirements.txt`. Java is needed for stage 1 only.
