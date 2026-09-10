# Performance: measured, applied, verified

Four optimisations were applied to stage 2 on 2026-09-10 (branch `perf-optimisation`,
baseline tagged `v1.0-documented`). **Stage 2 is 4.6–4.9× faster and the summary frame is
3.2× smaller, with every output value unchanged.**

## Result

Benchmark: real Entlebuch BAU SorSim outputs replicated to a local corpus (OneDrive files
are cloud placeholders and stall the reader — see the note at the end).

| | 340 files / 692 MB | 1020 files / 2.0 GB |
|---|---|---|
| before | 23.00 s | 54.75 s |
| after | 4.67 s | 11.87 s |
| | **4.9×** | **4.6×** |

Per phase, at 1020 files (591 360 summary rows):

| phase | before | after | |
|---|---|---|---|
| `load_data_parallel` | 44.06 s | 5.23 s | **8.4×** |
| `preprocess_data` | 1.52 s | 0.96 s | 1.6× |
| `compact_dtypes` (new) | — | 0.70 s | |
| `augment_with_stand_data` | 0.11 s | 0.21 s | — |
| `add_sawmill_diameter_info` | 0.16 s | 0.02 s | 8× |
| `split_by_soft_hard` | 0.23 s | 0.01 s | 23× |
| `map_species_for_quality` | 0.10 s | 0.03 s | 3× |
| sawmill split | 4.11 s | 0.03 s | **137×** |
| `to_csv` | 4.45 s | 4.68 s | — |
| **total** | **54.75 s** | **11.87 s** | **4.6×** |

| memory | before | after | |
|---|---|---|---|
| summary frame | 402 MB | 125 MB | 3.2× |
| text columns alone | 272 MB | 26 MB | 10× |
| peak process RSS | 572 MB | 366 MB | 1.6× |

### Values are unchanged

The benchmark checksums the result. All nine match exactly, at both corpus sizes:

```
rows 591360 | vol_or 8047460.19 | vol_ir 8966548.09 | wert 547203450.4
sawmill 4295094.98 | not_sawmill 3752365.21 | n_soft 342720 | n_hard 246960
by diameter class: {'20-40cm': 2871594.70, '<20cm': 839220.43, '>40cm': 6336645.05}
```

The 93-test suite passes before and after.

## What changed

### 1. Read only the block that is used — 8.4×

`process_file` parsed the whole SorSim output and then sliced off the first 97.5%. It now
finds the `#Gruppierungsmerkmal` marker with a raw byte scan and hands only the tail to
the parser:

```python
with open(path, "rb") as handle:
    blob = handle.read()
position = blob.find(b"\n#Gruppierungsmerkmal")
df = pd.read_csv(io.BytesIO(blob[position + 1:]), sep=";",
                 encoding="utf-8", encoding_errors="replace")
```

Two things come free: the columns now parse as `float64` instead of `object` (the
per-tree block above the marker is what forced everything to strings), and the encoding
is explicit rather than depending on pandas choosing to replace rather than raise —
see [known issue 7](06-known-issues.md), which also corrects an earlier wrong
recommendation to use `cp1252`.

### 2. Vectorise the sawmill split — 137×

`summaries.apply(..., axis=1)` built a `Series` per row. Replaced by
`calculate_sawmill_split`, which maps the fraction over the distinct species and uses
`np.where`. `calculate_biomass_for_sawmills` is kept so the test suite can check the fast
path against it row by row.

### 3. Compact dtypes — 10× on the text columns

`compact_dtypes` stores the seven repeated text columns as categories and `year` as
`int16`, after the concat and after `preprocess_data`.

`diameter_class` is deliberately left as a plain column: the plots group by it, and a
categorical group key makes pandas emit the full product of categories rather than the
observed rows.

### 4. Vectorise the remaining `.apply(lambda)` calls — 3–23×

`add_sawmill_diameter_info`, `split_by_soft_hard`, `map_species_for_quality` and the
umlaut fix-ups in `preprocess_data` now use `.map`, `.isin` and `.str.replace`. The
unknown-`Staerkenklasse` check is still fatal but now reports every offending value at
once instead of dying on the first row.

## What this means for the real jobs

- **`sample_size` is much less necessary.** 2 GB of SorSim output now summarises in
  12 seconds. Prefer `False` (all files) over `100`.
- **`mem_per_cpu=20000` is far more than needed.** Extrapolating the measured 125 MB per
  591 k rows, a Surselva-sized run (~40 M rows) needs roughly 8 GB rather than 20 GB, and
  peak RSS is set by `preprocess_data`, before compaction.
- **`to_csv` was the remaining bottleneck** — 4.68 s of 11.87 s, ~39%. Writing Parquet
  instead takes 0.51 s for the same table (7× faster, 5.4× smaller). That change landed
  separately; see [09-summary-format.md](09-summary-format.md).

## Not done

### Stage 1 launches one JVM per file

`run_sorsim` calls `JavaGateway.launch_gateway(...)` once per file — a fresh JVM start and
classpath load for each of thousands of stands. Not measured here (it needs Java, which is
not installed on this machine). To measure it on Euler:

```bash
time python ../minimal/run_sorsim.py ../minimal/sorsim/SorSim4Python.jar \
     <one tree list> /tmp/out.csv 6 True
```

If startup dominates, launch one gateway and loop over files inside it. That would remove
most of the reason stage 1 needs a process pool at all.

### Peak memory is set before compaction

`compact_dtypes` runs after `preprocess_data`, so the peak (366 MB of 366 MB at 1020
files) is reached while the frame is still object-dtype. Compacting per-file would need
`union_categoricals` at concat time — worth doing only if a real region runs out of memory.

---

## A note on the benchmark: OneDrive placeholders

The first benchmark attempt appeared to hang. The files in
`data/Entlebuch/outputs/` carry the `ReparsePoint` attribute — they are OneDrive
Files-On-Demand placeholders, not local data, so reading them triggers a download.
(The stall that actually blocked that first run was a bug in the benchmark harness, not
OneDrive; but the placeholders are real and will slow any local run.)

This reinforces [cleanup §G](04-cleanup-proposal.md): the working data should not live in
a synced folder. Any timing measured inside OneDrive measures the network.

The benchmark harness itself is not part of the repository — it lives in the scratch
directory and is reproduced in this document rather than committed, since it depends on a
generated corpus.
