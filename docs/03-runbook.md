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

## 0.2 The whole loop

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
stand on disk, the output tree exists (it creates it), and `java` is on `PATH`. Chain it
so a failed check blocks the submission:

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

Nothing to do: stage 1 creates `intermediate/<scenario>/` and `outputs/<scenario>/` under
`MAINWOOD_OUTPUT_TEMPLATE` itself, and `preflight.py` creates them early so you can see
where the run will write before submitting. (`bash_code_to_create_folder_structure_for_data.sh`
is kept only for the `inputs/` side of a laptop copy.)

### 1.3 Provide `stand.details.csv`

`../data/<Region>/stand.details.csv` must have:

| column | required | used for |
|---|---|---|
| `fsID` | **yes** | join key against the `stand` parsed from file names |
| `area_ha` | **yes** | rescaling patch volumes to the real stand area |
| `Above1000m` | no | the altitude split (only if you want those figures) |

Check the join before launching an 8-hour job — a stand that is missing only prints a
warning and silently produces `NaN` volumes:

```bash
python - <<'EOF'
import pandas as pd
s = pd.read_csv("../data/<Region>/stand.details.csv")
assert {"fsID", "area_ha"} <= set(s.columns), sorted(s.columns)
assert s["area_ha"].notna().all() and (s["area_ha"] > 0).all()
print(len(s), "stands, area_ha total:", s["area_ha"].sum())
EOF
```

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
