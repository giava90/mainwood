# Runbook

Everything below is run from `code/`, on Euler unless stated otherwise.

## 0. Once per session

```bash
module load stack/2024-06 python/3.12.8
module load stack/2024-06 openjdk/21.0.3_9   # stage 1 only: SorSim is a Java jar
chmod u+x run_conversion.sh run_analysis.sh
```

`code/*.py` resolve their data paths **relative to `code/`** (`../data/...`), so always
`cd code` first.

## 1. Onboarding a new case study region

### 1.1 Register the region

`Jurapark` (or whatever the region is called) must be added to `valid_case_studies` in
**both** entry points, and it must be spelled exactly as in the ForClim file names:

- `code/convert_data.py` (`__main__`)
- `code/summarize_and_create_plots.py` (`__main__`)
- `code/plot_only.py` (`__main__`) if you will re-plot

### 1.2 Create the folder skeleton

```bash
mkdir -p ../data/<Region>
cd ../data/<Region> && bash ../../code/bash_code_to_create_folder_structure_for_data.sh
```

That creates `inputs/`, `intermediate/`, `outputs/`, each with `BAU BIO WOOD HYBRID`.
Do the same under `/cluster/scratch/giacomov/mainwood/<Region>/` for the real run.

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

Either copy/symlink the ForClim files into `../data/<Region>/inputs/<scenario>/`, or
uncomment the Euler paths at the bottom of `convert_data.py`:

```python
input_folder_path  = f"/cluster/work/climate/amauri/{cs}/Results/mgmt_{ms}/dead.trees/"
output_folder_path = f"/cluster/scratch/giacomov/mainwood/{cs}/"
```

**Check the file names first.** Stage 1 only picks up files matching
`dataSim.<cohort><stand>_<simtype>...`; anything else is skipped with a message. If
ForClim names the alive cohort with a different token than `alive`, change
`COHORT_TOKEN` in [`code/naming.py`](../code/naming.py) — one line, one place.

```bash
ls /cluster/work/climate/amauri/<Region>/Results/mgmt_BAU/*/ | head
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

Then submit, editing the variables at the top of `run_conversion.sh`:

```bash
./run_conversion.sh
```

Run the two cohorts as **two separate jobs**. They write different intermediate and
output names, so they can also run concurrently.

Expect in `../data/<Region>/outputs/<scenario>/`:

```
sorsim_output<stand>_<simtype>.csv           # dead
sorsim_alive_output<stand>_<simtype>.csv     # alive
```

Sanity check afterwards:

```bash
ls ../data/<Region>/inputs/BAU | wc -l        # should match...
ls ../data/<Region>/outputs/BAU | wc -l       # ...this
ls ../data/<Region>/intermediate/BAU | wc -l  # should be 0 (unless save_intermediate)
```

A non-empty `intermediate/` means the run was interrupted — use step 4 rather than
redoing the conversion.

## 3. Stage 2 — summaries and figures

```bash
python summarize_and_create_plots.py <Region> BAU ../data 4 100          # dead, 100-file sample
python summarize_and_create_plots.py <Region> BAU ../data 4 False alive  # alive, everything
```

Arguments: `<case_study> <scenario> <folder_data> <n_cores> <sample_size> [cohort]`

- `folder_data` — root that holds `<region>/outputs/<scenario>/`; on Euler
  `/cluster/scratch/giacomov/mainwood/`
- `sample_size` — `False` for all files, or an integer (random sample, seed 42)
- `cohort` — `dead` (default) or `alive`

Then submit with `./run_analysis.sh`.

Writes:

```
../data/summaries_for_plots/<Region>_<scenario>.csv          # dead — historical name
../data/summaries_for_plots/<Region>_<scenario>_alive.csv    # alive
../figures/*_8_5_*.png          (dead)
../figures/*_8_5_alive_*.png    (alive)
```

**Memory.** `run_analysis.sh` asks for 20 GB on 1 core. Surselva WOOD produced a 6.6 GB
CSV, and the frame is held in memory in full before writing. For a large new region,
start with `sample_size=100` to size the job, then scale `mem_per_cpu` from the observed
row count. If it does not fit, run scenario by scenario rather than `ALL`.

## 4. Re-running SorSim without re-converting

Only if the tree lists are still in `intermediate/` (i.e. `save_intermediate=True`, or
the job died between the two pool steps):

```bash
python convert_data_from_intermediate.py WOOD False 4 False <Region>
```

The cohort is read back from each tree list name, so a folder containing both
`deadCohorts*` and `aliveCohorts*` is handled in one pass.

## 5. Re-plotting only

Figures are cheap; the summaries are not. To change a plot, do not re-run stage 2:

```bash
python plot_only.py <Region> BAU ../data 1 False
```

It reads `../data/summaries_for_plots/<Region>_<scenario>.csv` directly. Note it filters
`simtype == 1` as an **integer** (correct when reading back from CSV), whereas stage 2
filters the string `'1'` (correct in-memory). Both mean RCP 8.5.

## 6. Tests

```bash
python -m pytest            # from the repository root
```

See [05-testing.md](05-testing.md). Run these before submitting a long job after any
change to naming, weights, or the converter — they take about 5 seconds and cover
exactly the failures that otherwise surface 8 hours in.
