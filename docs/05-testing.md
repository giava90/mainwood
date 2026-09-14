# Tests

```bash
python -m pytest          # from the repository root, ~15 s, 248 tests
```

Pure Python — no Java, no SLURM, no data beyond what is in the repository. The suite runs
on a laptop and on Euler.

## Why these tests exist

Every job in this pipeline is long (8 h walltime) and most failures are silent: a
mis-parsed file name produces an *empty output folder*, a wrong weight produces plausible
numbers that are off by a constant factor. The tests target exactly those failure modes.

| File | Covers | The failure it prevents |
|---|---|---|
| `test_naming.py` | `code/naming.py` | A stand/simtype pair that stage 1 writes and stage 2 cannot read → empty summary |
| `test_summary_filename_parsing.py` | round trip stage 1 name → stage 2 parser | The two parsers drifting apart (they have before) |
| `test_output_input_converter.py` | `minimal/functions/tools.py` | Wrong cohort column, missing tree expansion, species dropped unnoticed |
| `test_preprocess_weights.py` | `preprocess_data` planting weights | Every volume wrong by a constant factor |
| `test_stand_and_quality.py` | area rescaling, diameter classes, quality split | A new region with a different `stand.details.csv` producing `NaN` volumes |
| `test_summary_end_to_end.py` | the whole read chain on a real SorSim file | Marker detection and the replacement-character round trip |
| `test_optimised_paths.py` | the vectorised replacements | A fast path that quietly disagrees with the per-row version it replaced |
| `test_summary_io.py` | the Parquet/CSV deliverable and the converter | Handing collaborators a CSV that differs from the one their R pipeline expects |
| `test_regions.py` | `code/regions.py` | A region registered in some entry points but not others — the run then aborts on an argument check hours after you thought you were done |
| `test_exclusions.py` | `code/exclusions.py` | A stand with no area, or none in `stand.details.csv`, reaching the summary as a confident zero or a `NaN` |
| `test_scaling.py` | `code/scaling_run.py`, `code/plot_scaling.py` | Grid points sharing an output tree or a result file, so the benchmark measures the collision rather than the code |
| `test_alive_cohort.py` | the alive delivery end to end | Its different file names, its 2015-only snapshot, and its weights — each of which silently emptied or rescaled the summary |
| `test_setup_data_tree.py` | `code/setup_data_tree.py` | A per-scenario alive folder (four copies of one snapshot), or directories created inside a filesystem that is not ours |
| `test_no_shared_state.py` | stage 1 failure reporting | The Manager-list deadlock that froze a 48-core run at 0% CPU for five hours while SLURM called it RUNNING |
| `test_paper_figures.py` | `make_paper_figures.py`, `plotting_tools_for_paper.py` | A simtype compared as an integer, which matches nothing and yields silently empty figures |
| `test_paths.py` | `code/paths.py` per-machine path templates | A laptop-vs-Euler path edit leaking into a tracked file, or a typo'd placeholder silently reading an empty folder |

## The R side

`code/read_summaries.R` is exercised by a separate R script that is **not** part of the
pytest suite (it needs R, arrow and a real Parquet summary). It was run against a
591 360-row summary and checks 26 behaviours, including that R's totals equal Python's to
the last digit. Re-run it after changing the R helper:

```bash
"$LOCALAPPDATA/Programs/R/R-4.6.1/bin/x64/Rscript.exe" <the test script>
```

The environment it needs is documented in [10-environments.md](10-environments.md).

## What is deliberately not tested

- **SorSim itself.** The jar is vendored, third-party and unchanged; testing it would need
  Java in the test environment. `minimal/README.txt` describes the manual GUI-diff check.
- **The plotting functions.** They produce PNGs; asserting on pixels is brittle and the
  figures are cheap to regenerate and easy to eyeball.
- **`multiprocessing` orchestration.** `process_files` is a thin `Pool.starmap` wrapper;
  the logic it distributes is covered directly.

## Fixtures

`tests/conftest.py` puts `code/` and `minimal/` on `sys.path` (the scripts are not a
package) and sets `MPLBACKEND=Agg` so matplotlib never tries to open a window.

The end-to-end test copies `minimal/testdata/outputs/deadCohorts_sample.csv` — a genuine
2 300-line SorSim result — into a temp folder under a pipeline-shaped name. That file is
the only real fixture; everything else is built inline so the expected numbers are
visible next to the assertion.

## Adding the new region to the tests

When the new case study lands, two things are worth pinning immediately:

1. **Its file names.** Add the real ForClim name to the parametrisation in
   `test_naming.py::test_compression_does_not_change_the_parsed_simtype` (and the alive
   variant). If ForClim uses a token other than `alive`, that test will tell you before
   the cluster does.
2. **Its `stand.details.csv`.** A test asserting `{"fsID", "area_ha"} <= set(columns)`
   and `area_ha > 0` for the new region catches the most common onboarding failure — the
   one that otherwise shows up as `NaN` volumes after an 8-hour job.

## Running them without a full environment

If the interpreter you use lacks `pytest`, install it beside the environment rather than
into it:

```bash
python -m pip install --target /some/dir pytest
PYTHONPATH=/some/dir python -m pytest
```

On Euler, `module load stack/2024-06 python/3.12.8` already provides everything in
`requirements.txt`.
