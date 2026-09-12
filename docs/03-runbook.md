# Runbook

Everything below is run from `code/`, on Euler unless stated otherwise.

## 0. First time on a machine

Paths differ between the laptop and Euler. They live in **`code/local.env`**, which is
git-ignored — so `git pull` on Euler never conflicts, and no tracked file is ever edited
to change machine, region or scenario. Do this once per machine:

```bash
cd code
cp local.env.example local.env
# uncomment the Euler block in local.env, then check the cohort sub-folder name:
ls /cluster/work/climate/amauri/Entlebuch/Results/mgmt_BAU/
```

The four settings are path templates expanding `{case_study}`, `{scenario}` and
`{cohort}`; anything left unset falls back to the repository-relative default, which is
what you want on a laptop. See [`code/paths.py`](../code/paths.py).

| variable | Euler value |
|---|---|
| `MAINWOOD_INPUT_TEMPLATE` | `/cluster/work/climate/amauri/{case_study}/Results/mgmt_{scenario}/{cohort}.trees/` |
| `MAINWOOD_INTERMEDIATE_TEMPLATE` | `/cluster/scratch/giacomov/mainwood/{case_study}/intermediate/{scenario}/` |
| `MAINWOOD_OUTPUT_TEMPLATE` | `/cluster/scratch/giacomov/mainwood/{case_study}/` |
| `MAINWOOD_DATA_ROOT` | `/cluster/scratch/giacomov/mainwood/` |

## 0.1 Once per session

```bash
cd code
module load stack/2024-06 python/3.12.8
module load stack/2024-06 openjdk/21.0.3_9   # stage 1 only: SorSim is a Java jar
chmod u+x run_conversion.sh run_analysis.sh
```

`code/*.py` resolve relative paths **relative to `code/`**, so always `cd code` first.

## 0.2 Python packages on Euler

The module stack does not carry everything. `pyarrow` in particular is needed for the
**default** summary format, and it is imported only at the final write — so a missing
one aborts stage 2 after all the work is done, not at the start.

```bash
python -c "import pyarrow, openpyxl; print('ok')"
```

If that fails, either install into your user site-packages:

```bash
pip install --user pyarrow openpyxl
```

or skip Parquet for that run — `run_analysis.sh` takes the format as its fifth argument:

```bash
./run_analysis.sh <Region> WOOD dead False csv
```

`preflight.py` checks both and refuses the run if either is missing. A quick way to tell
whether your environment is complete: `python -m pytest ../` reporting a skip usually
means a package is absent rather than a test being broken — `-rs` prints the reason.

## 0.3 The whole loop

```bash
cd code
git pull                                    # never conflicts: local.env is ignored
python -m pytest ../                        # ~10 s
python preflight.py Entlebuch WOOD dead     # ~10 s; exits 1 if the run would fail
./run_conversion.sh WOOD Entlebuch dead     # stage 1
# ...wait for the mail...
./run_analysis.sh Entlebuch WOOD dead       # stage 2
```

`preflight.py` checks the things that otherwise fail eight hours in: the input folder
exists and holds files matching the cohort convention, `stand.details.csv` covers every
stand on disk, the output tree exists (it creates it), `java` is on `PATH`, and the
packages stage 2 needs are importable. Chain it so a failed check blocks the submission:

```bash
python preflight.py Entlebuch WOOD dead && ./run_conversion.sh WOOD Entlebuch dead
```

Neither wrapper needs editing any more — region, scenario and cohort are arguments, and
sbatch resources are environment variables:

```bash
./run_conversion.sh WOOD Entlebuch                  # dead cohort, all files
./run_conversion.sh WOOD Entlebuch alive            # alive cohort
./run_conversion.sh WOOD Entlebuch dead True        # 50-file smoke test
N_CORES=8 CONVERT_WALLTIME=24:00:00 ./run_conversion.sh ALL All

./run_analysis.sh Entlebuch WOOD                    # dead, all files, parquet
./run_analysis.sh Entlebuch WOOD alive
ANALYSIS_MEM_PER_CPU=40000 ./run_analysis.sh Surselva ALL
```

Both write `slurm-<stage>-<region>-<scenario>-<cohort>-<jobid>.out` next to the wrappers
(git-ignored), so a failed job is identifiable without opening it.

**`/cluster/scratch` is purged.** Euler deletes scratch files older than 15 days. The
summaries are the deliverable — copy them off scratch as soon as stage 2 finishes:

```bash
cp /cluster/scratch/giacomov/mainwood/summaries_for_plots/*.parquet ~/mainwood-summaries/
```

## 1. Onboarding a new case study region

### 1.1 Register the region

One line in [`code/regions.py`](../code/regions.py) — it is the single list every entry
point reads:

```python
CASE_STUDIES = ("Entlebuch", "Vaud", "Surselva", "Misox", "Jurapark")
```

Spell it exactly as it appears in the ForClim file names; `naming.forclim_pattern`
matches against this string.

> Before 2026-09 this list was copy-pasted into five entry points, and they had already
> drifted — `plot_only` silently dropped `HYBRID` while the others kept it. If you need a
> script to support a narrower set, pass `exclude=` to `regions.valid_scenarios()` so the
> exception stays visible instead of becoming a sixth divergent copy.

### 1.2 Folder skeleton

A fresh (or freshly purged) `/cluster/scratch/<user>/mainwood/` is built from the same
templates the pipeline reads, so it matches whatever `code/local.env` says:

```bash
cd code
python setup_data_tree.py --dry-run     # show, create nothing
python setup_data_tree.py               # every region, every scenario
python setup_data_tree.py Jurapark      # just one region
python setup_data_tree.py --inputs      # also make inputs/, if you copy files onto scratch
```

It is idempotent — re-run it after a scratch purge and it rebuilds only what is gone.

Strictly speaking nothing here is required: stage 1 creates `intermediate/` and
`outputs/` on demand and stage 2 creates the summary folder. The reason to run it is to
see the layout, and to catch the configuration error it warns about:

```
WARNING: assortments are on scratch but the summaries are not
```

That combination used to be the default. Stage 2 took its input root as an argument but
hardcoded `../data/summaries_for_plots/` for the output, so on Euler it read the
assortments from scratch and wrote the summaries back onto the home quota.

How much that matters depends on the format. Extrapolating from the measured table in
[09-summary-format.md](09-summary-format.md), a Surselva-sized region (~40 M rows) is
about **1.0 GB per scenario as Parquet and 5.5 GB as CSV** — so all four scenarios are
roughly 4 GB of Parquet, or 22 GB of CSV. Home quotas on Euler are far smaller than
scratch; check yours with `lquota`. `MAINWOOD_SUMMARY_DIR` fixes it, and
`setup_data_tree.py` tells you if you forgot.

### 1.3 Provide `stand.details.csv`

This file is the join key and the area column behind **every volume in the summary**,
so it is the one input worth checking before a long run.

It arrives from the ForClim side as a one-off delivery (`manag4giacomo`) and is **not**
hosted on their cluster folder, so we host it. The canonical copy is the one tracked in
this repository at `data/<Region>/stand.details.csv`: small, not regenerable, and it
scales every number in the output — so it is versioned and travels with the code.

> The Vaud copy sat uncommitted for months while the repository carried an older
> delivery that had no `area_ha` column at all. `data/` being git-ignored wholesale was
> the reason a plain `git add` warned and got skipped. `.gitignore` now ignores only the
> regenerable sub-folders, so these files commit normally.

**When a new delivery arrives** use `import_stand_details.py` rather than copying by
hand. The deliveries are not uniform: the file is `manag_areas_all.csv`,
`manag_areas_all7.csv` or `manag_areas_all12.csv` depending on the region, the region
folder is lower-case, the column sets differ entirely, and **Jurapark spells the area
column `Area_ha`** while every other region writes `area_ha`. The importer normalises the
case, checks `fsID` is present and unique, and prints the provenance to paste into the
commit message.

```bash
# look at it first -- --check writes nothing
python import_stand_details.py <Region> ../data/manag4giacomo/manag4giacomo/<region>/<file>.csv --check

# then write and commit
python import_stand_details.py <Region> ../data/manag4giacomo/manag4giacomo/<region>/<file>.csv
python preflight.py <Region> WOOD dead
git add ../data/<Region>/stand.details.csv   # no -f needed any more
git commit -m "stand.details.csv for <Region>, delivery <md5>"
```

**`--check` first, always.** The Surselva copy is a curated 14-column subset of a
23-column delivery plus an `elev` column from elsewhere; every shared value matches, but
a blind re-import would discard that curation.

Required columns:

| column | required | used for |
|---|---|---|
| `fsID` | **yes** | join key against the `stand` parsed from file names |
| `area_ha` | **yes** | rescaling patch volumes to the real stand area |
| `Above1000m` | no | the altitude split (only if you want those figures) |

Preflight verifies the columns, flags missing or zero areas, confirms every stand on disk
joins, and prints the area total:

```
[ok  ] 2687 stands listed, 15,105.0 ha total, all 1 stands on disk join
```

That total is what every volume gets scaled by. Check it against the figure the ForClim
side quotes; if it disagrees, you have the wrong delivery. `MAINWOOD_STAND_DETAILS`
exists to point at a delivery elsewhere for testing, before you commit it.

### 1.4 Point at the ForClim results

Set `MAINWOOD_INPUT_TEMPLATE` in `code/local.env` (section 0) — do **not** edit
`convert_data.py`, and do not rename folders on disk. If a region lays its ForClim
results out differently, the template is the place to say so.

**Check the file names first.** Stage 1 only picks up files matching
`dataSim.<cohort><stand>_<simtype>`; anything else is skipped with a message. If ForClim
names the alive cohort with a different token than `alive`, change `COHORT_TOKEN` in
[`code/naming.py`](../code/naming.py) — one line, one place.

```bash
ls /cluster/work/climate/amauri/<Region>/Results/mgmt_BAU/*/ | head
python preflight.py <Region> BAU dead    # confirms the template resolves to real files
```

## 2. Stage 1 — assortments

Test on a handful of files interactively before submitting:

```bash
python convert_data.py WOOD True 4 False <Region>          # dead cohort, 50 files
python convert_data.py WOOD True 4 False <Region> alive    # alive cohort, 50 files
```

Arguments: `<scenario> <use_sample> <n_cores> <save_intermediate> <case_study> [cohort]`

- `use_sample` — `True` processes the first 50 files only
- `save_intermediate` — `True` keeps the SorSim tree list as a `.zip` (useful when you
  expect to re-run SorSim; see step 4)
- `cohort` — `dead` (default) or `alive`

Then submit. The wrapper takes arguments, so it is never edited:

```bash
./run_conversion.sh <scenario> <Region> [cohort] [use_sample] [save_intermediate]
./run_conversion.sh WOOD Entlebuch dead
```

Run the two cohorts as **two separate jobs**. They write different intermediate and
output names, so they can also run concurrently.

Expect in `<MAINWOOD_OUTPUT_TEMPLATE>/outputs/<scenario>/`:

```
sorsim_output<stand>_<simtype>.csv           # dead
sorsim_alive_output<stand>_<simtype>.csv     # alive
```

Sanity check afterwards:

```bash
root=/cluster/scratch/giacomov/mainwood/<Region>     # = MAINWOOD_OUTPUT_TEMPLATE
ls /cluster/work/climate/amauri/<Region>/Results/mgmt_BAU/dead.trees/ | wc -l  # match...
ls "$root"/outputs/BAU | wc -l                                                 # ...this
ls "$root"/intermediate/BAU | wc -l            # should be 0 (unless save_intermediate)
```

A non-empty `intermediate/` means the run was interrupted — use step 4 rather than
redoing the conversion.

## 2.4 The alive cohort comes from somewhere else

The alive data is produced by a different pipeline and does not look like the dead
cohort in any respect that matters:

| | dead | alive |
|---|---|---|
| input tree | `/cluster/work/climate/amauri/<Region>/Results/mgmt_<scenario>/dead.trees/` | `/nfs/.../raw/<Region>/alive.data/<region>/` |
| scenario in the path | yes | **no** — see below |
| file name | `dataSim.dead207_1_planted_06.csv.gz` | `dataSim_4810_scen7.csv` |
| years | the full simulation, floored at 2020 | **2015 only** — a first-year snapshot |
| simtypes | several | **7 only** — climate has not diverged yet |
| planting | several variants per stand | **none**, one file per stand |

The two cohorts are on **different filesystems**, so a single template with a
`{cohort}` placeholder cannot reach both — pointed at the dead tree it resolves an
alive run to a non-existent `.../mgmt_BAU/alive.trees/`. Set the cohort-specific
template in `code/local.env` instead; it wins for alive runs only, and the shared
template keeps serving the dead cohort:

```bash
MAINWOOD_INPUT_TEMPLATE='/cluster/work/climate/amauri/{case_study}/Results/mgmt_{scenario}/{cohort}.trees/'
MAINWOOD_INPUT_TEMPLATE_ALIVE='/nfs/ites-formdata.ethz.ch/mnt/formdata/wood_valuation/Price/data/raw/{case_study}/alive.data/{case_study_lower}/'
```

With both set, `./run_conversion.sh BAU Misox alive` and `preflight.py Misox BAU alive`
resolve the right folder with no per-run override.

`{case_study_lower}` exists for this layout — the region appears capitalised once and
lower-case once in the same path.

**What the pipeline does differently for `alive`,** all automatic:

- the `dataSim_<stand>_scen<n>` name is parsed alongside the historical form;
- the 2020 year floor is skipped, because it would discard the entire 2015 snapshot;
- every row is weighted 1 — one simulation per stand, no planting, so the planting
  arithmetic does not apply;
- figures are not filtered to `simtype == '1'`, which would leave nothing to plot.

**What preflight checks,** because the weighting depends on it:

```
[ok  ] 2113 alive files, one per stand
[ok  ] one simtype (7), as expected
[ok  ] year is 2015 in the 3 file(s) sampled
```

**The alive path carries no scenario.** It is a snapshot of 2015, before management
diverges — which is also why there is only simtype 7 and no planting. If that is right,
the alive result is identical for BAU, WOOD, BIO and HYBRID, and running all four
produces four copies of the same numbers. Confirm with the ForClim side before queueing
more than one.

A stand appearing in **two** alive files is a blocking failure, not a warning: every
alive row is weighted 1 on the assumption of one simulation per stand, so a repeat
would be double-counted rather than merely odd. More than one simtype, or a year
other than 2015, warns.

## 2.5 Stands that were excluded

Stage 1 drops stands it cannot compute **before** SorSim runs, so no cluster time
goes into a result that would be discarded. Two rules, both about the area used to
rescale patch volumes:

| reason | what it was doing before |
|---|---|
| `area_ha <= 0` (or `NaN`) | produced a confident `0`, indistinguishable from a stand that genuinely harvested nothing, and dragged down every per-stand mean |
| `not in stand.details.csv` | produced `NaN` volumes after a single printed warning |

Each run writes the list to the region root — **not** into `outputs/<scenario>/`,
which stage 2 reads in full:

```
<MAINWOOD_OUTPUT_TEMPLATE>/excluded_stands_<Region>_<scenario>_<cohort>.csv
```

```
stand  reason                    area_ha  n_files  example_file                       ...
2501   area_ha <= 0              0.0      2        dataSim.dead2501_1_planted_00.csv
999    not in stand.details.csv           2        dataSim.dead999_1_planted_00.csv
```

That file is the list to hand back to the ForClim side, and the list to re-run once
corrected data arrives — delete nothing, just re-run stage 1 for the region and the
report shrinks. If **nothing** is excluded the file is removed rather than left
behind, so a stale report is never mistaken for a current one.

### Seeing the list before you submit

`preflight.py` reports the same exclusions without running anything, and writes the full
list (the printed message names only five) to the region root:

```
preflight_excluded_<Region>_<scenario>_<cohort>.csv
```

It is deliberately a different name from the report stage 1 writes: this is what *would*
be excluded, not the record of a run that happened.

Stands absent from `stand.details.csv` are a **warning**, not a blocker — stage 1
excludes them and carries on, so refusing the run would be refusing one that succeeds.
What still blocks: a missing or unreadable `stand.details.csv`, a missing input folder,
input files that match no stand, and absent Python packages.

The job log carries the same thing in short form:

```
Excluded 2 stand(s) covering 4 file(s); 2 of 6 files will be processed.
  area_ha <= 0: 1 stand(s) -- 2501
  not in stand.details.csv: 1 stand(s) -- 999
```

`convert_data_from_intermediate.py` applies the same rule, per cohort, so re-running
SorSim over tree lists written before the rule existed does not put the excluded
stands back.

**If `stand.details.csv` cannot be read at all** — missing, or no `area_ha` column,
as the old Vaud delivery had — nothing is excluded and the run says so. Dropping
every stand in that case would look like a successful empty run.

## 3. Stage 2 — summaries and figures

```bash
python summarize_and_create_plots.py <Region> BAU ../data 4 100          # dead, 100-file sample
python summarize_and_create_plots.py <Region> BAU ../data 4 False alive  # alive, everything
```

Arguments: `<case_study> <scenario> <folder_data> <n_cores> <sample_size> [cohort] [format]`

- `folder_data` — root that holds `<region>/outputs/<scenario>/`. `run_analysis.sh`
  passes `MAINWOOD_DATA_ROOT` from `code/local.env`; pass it explicitly only when
  calling the script by hand
- `sample_size` — `False` for all files, or an integer (random sample, seed 42)
- `cohort` — `dead` (default) or `alive`
- `format` — `parquet` (default), `csv` or `csv.gz`; see [09-summary-format.md](09-summary-format.md)

Then submit. `folder_data` comes from `MAINWOOD_DATA_ROOT`, so the wrapper only needs
the region and scenario:

```bash
./run_analysis.sh <Region> <scenario> [cohort] [sample_size] [format]
./run_analysis.sh Entlebuch WOOD dead
```

Writes:

```
../data/summaries_for_plots/<Region>_<scenario>.parquet          # dead — historical name
../data/summaries_for_plots/<Region>_<scenario>_alive.parquet    # alive
../figures/*_8_5_*.png          (dead)
../figures/*_8_5_alive_*.png    (alive)
```

To hand these to someone whose pipeline reads CSV:

```bash
python summary_to_csv.py ../data/summaries_for_plots/ --gzip
```

**Memory.** `run_analysis.sh` defaults to 20 GB on 1 core (`ANALYSIS_MEM_PER_CPU`). Since the 2026-09 optimisation
that is generous: measured at 125 MB per 591 k summary rows, a Surselva-sized run
(~40 M rows) needs roughly 8 GB. Keep the 20 GB for the first run of a new region, then
scale it down from the reported frame size. If it still does not fit, run scenario by
scenario rather than `ALL`. See [07-performance.md](07-performance.md).

**Sampling.** `sample_size` was a workaround for the old read path. 2 GB of SorSim output
now summarises in about 12 seconds, so prefer `False` (all files) over `100`.

## 3.5 Several jobs at once, against different input folders

To split stage 1 across more SLURM jobs, or to read one run from a different ForClim
location, override the template for that submission only. The environment beats
`local.env`, so nothing else changes and no file is edited:

```bash
MAINWOOD_INPUT_TEMPLATE='/cluster/scratch/giacomov/mainwood/{case_study}/inputs/WOOD/{cohort}.trees/'   ./run_conversion.sh WOOD Surselva

MAINWOOD_INPUT_TEMPLATE='/cluster/work/climate/amauri/{case_study}/Results/mgmt_{scenario}/{cohort}.trees/'   ./run_conversion.sh WOOD Entlebuch
```

Both run concurrently and write to their own `outputs/<scenario>/`.

> This replaces `convert_data_plantations.py` on the Euler checkout — a copy of the
> whole converter that differed from it only in two path lines. Because it was forked
> from a 2025 base it never received the `.csv` suffix fix, so any uncompressed input
> would have produced an empty output folder silently, and it hardcoded
> `inputs/WOOD/` while still accepting a `scenario` argument, so a non-WOOD run read
> WOOD inputs and wrote them into the other scenario's output folder. Delete it.

## 3.7 How stage 1 scales

Nothing in this repository measures stage 1 throughput, so before committing a long
walltime to a region nobody has converted before, measure it:

```bash
./scaling_benchmark.sh BAU Jurapark          # 3 core counts x 3 sample sizes = 9 jobs
squeue -u $USER
python plot_scaling.py                       # when they have finished
```

Defaults are `{5, 10, 20}` cores by `{40, 80, 200}` files, 8 h walltime each. Override
either axis:

```bash
SCALING_CORES="5 20 48" SCALING_SAMPLES="100 500" ./scaling_benchmark.sh BAU Jurapark
```

Each point runs in **its own output tree**. Nine concurrent stage 1 jobs writing into one
`outputs/<scenario>/` would overwrite each other's files and time each other's I/O, which
measures the collision rather than the code. Each point also writes its own result file,
so nine jobs finishing at once cannot interleave a line.

`plot_scaling.py` prints the table and writes `<root>/scaling.png` with two panels: wall
time, and throughput against an ideal-linear reference. **Throughput is the one to read** —
wall time always falls with more cores, but throughput flattening is what tells you the
extra cores stopped paying, and that is the number that decides `N_CORES` for the real run.

The benchmark tree is disposable. Delete it when you have the plot:

```bash
rm -rf "$MAINWOOD_DATA_ROOT/scaling_Jurapark_BAU_dead"
```

## 4. Re-running SorSim without re-converting

Only if the tree lists are still in `intermediate/` (i.e. `save_intermediate=True`, or
the job died between the two pool steps):

```bash
python convert_data_from_intermediate.py WOOD False 4 False <Region>
```

It reads `MAINWOOD_INTERMEDIATE_TEMPLATE` and writes to `MAINWOOD_OUTPUT_TEMPLATE`, the
same configuration stage 1 used.

The cohort is read back from each tree list name, so a folder containing both
`deadCohorts*` and `aliveCohorts*` is handled in one pass.

## 5. Re-plotting only

Figures are cheap; the summaries are not. To change a plot, do not re-run stage 2:

```bash
python plot_only.py <Region> BAU ../data 1 False
```

It reads `../data/summaries_for_plots/<Region>_<scenario>.{parquet,csv}` — whichever
exists. Both it and stage 2 now select RCP 8.5 by comparing `simtype` as text, so the
result no longer depends on which format the summary was stored in.

## 6. Tests

```bash
python -m pytest            # from the repository root
```

See [05-testing.md](05-testing.md). Run these before submitting a long job after any
change to naming, weights, or the converter — they take about 5 seconds and cover
exactly the failures that otherwise surface 8 hours in.
