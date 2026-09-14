# The summaries format, and how collaborators read it

The summaries are the product of the pipeline and go to researchers whose own
pipelines are written in R and read CSV. **They are now stored as Parquet, and CSV is
one command away and byte-identical to what those pipelines received before.**

## Why change at all

Measured on a real 591 360-row summary (the Entlebuch BAU corpus):

| format | size | vs CSV | write | read |
|---|---|---|---|---|
| CSV (what we wrote before) | 82.9 MB | 1.0× | 3.55 s | 0.71 s |
| CSV gzipped | 15.9 MB | 5.2× | 7.52 s | 0.91 s |
| **Parquet (snappy, default)** | **15.3 MB** | **5.4×** | **0.51 s** | **0.34 s** |
| Parquet (zstd) | 12.7 MB | 6.5× | 0.29 s | 0.08 s |

Extrapolated to the real archive: **`Surselva_WOOD` goes from 6.6 GB to 1.2 GB**, and the
whole `summaries_for_plots` folder from ~13 GB to ~2.4 GB.

Three things beyond size:

- **Parquet is bit-exact for floats; CSV is not.** `to_csv` writes about 16 significant
  digits, so a CSV round trip perturbs the volumes by ~1e-16 relative. Scientifically
  irrelevant, but it means "the CSV" and "the numbers" were never quite the same thing.
- **Columns are typed.** `year` comes back as `int16`, `Baumart` as a category. After a
  CSV round trip everything is re-guessed, which is how `plot_only.py` ended up comparing
  `simtype == 1` as an integer while the summariser compared `'1'` as a string.
- **Column selection is free.** Reading only `year` and `Volumen OR [m3]` out of a 1.2 GB
  Parquet never touches the other columns on disk.

snappy is the default rather than the smaller zstd because every Parquet reader
understands it, including older R `arrow` builds — these files are handed to other people.

## For the collaborators: three options

### 1. Ask for CSV — nothing changes for them

```bash
python code/summary_to_csv.py data/summaries_for_plots/            # every file
python code/summary_to_csv.py data/summaries_for_plots/ --gzip     # 5x smaller, R reads it directly
python code/summary_to_csv.py data/summaries_for_plots/Surselva_WOOD.parquet -o /tmp/csv
```

The output is **byte-identical** to a direct `to_csv()` of the same table: same columns in
the same order, the same unnamed leading row-number column, `True`/`False` spelled the
Python way. This is asserted by
`tests/test_summary_io.py::test_converted_csv_is_byte_identical_to_the_old_output`.

The conversion streams row group by row group, so a 6 GB summary converts in a few GB of
memory, not 6.

The pipeline can also just write CSV directly — see the `format` argument below.

### 2. Read the Parquet from R

`code/read_summaries.R` wraps it:

```r
install.packages("arrow")        # or "nanoparquet" — smaller, read-only
source("code/read_summaries.R")

s <- read_summary("data/summaries_for_plots/Vaud_BAU.parquet")

# only what you need; the rest is never read from disk
s <- read_summary("data/summaries_for_plots/Surselva_WOOD.parquet",
                  columns = c("year", "Baumart", "Volumen OR [m3]"))

# too big for memory? query it lazily
library(dplyr)
open_summary("data/summaries_for_plots/Surselva_WOOD.parquet") |>
  filter(simtype == "1") |>
  group_by(year, diameter_class) |>
  summarise(volume = sum(`Volumen OR [m3]`), .groups = "drop") |>
  collect()
```

`read_summary()` also reads the old CSVs, so it works whichever format a given region is
in. `list_summaries()` shows what is available; `summary_columns()` documents what each
column means.

**Tested** against a real 591 360-row summary with R 4.6.1 and arrow 25.0.1 — 26 checks,
all passing: path resolution, Parquet via arrow, the column subset, CSV and `.csv.gz`,
the lazy `open_summary()` dplyr query, the nanoparquet fallback, and the helpers. The
totals agree with Python to the last digit (`10047460.187642`, and `3068124.598854` for
the RCP 8.5 group-by).

Two defects were found and fixed on the way: `col_select = all_of(NULL)` returned a
zero-column table for the ordinary no-columns call (found by review), and `readr` printed
a `New names:` message on every CSV read (found by running it). See
[10-environments.md](10-environments.md) for how R was installed.

### 3. Keep writing CSV from the pipeline

```bash
python summarize_and_create_plots.py <Region> BAU ../data 4 False dead csv
```

The 7th argument is the format: `parquet` (default), `csv`, or `csv.gz`. `csv` reproduces
exactly what the pipeline wrote before.

## What changed in the code

| file | role |
|---|---|
| `code/summary_io.py` | `write_summary` / `read_summary`, with format auto-detection |
| `code/summary_to_csv.py` | Standalone streaming Parquet → CSV converter |
| `code/read_summaries.R` | R-side reader and column documentation |

`read_summary` takes whichever format exists. `plot_only.py`'s `simtype` filter was comparing
against the integer `1`; that matched nothing when the summary came from Parquet (where
`simtype` is text), so it now compares as text and raises if the selection is empty
instead of silently plotting nothing.

## The existing 13 GB

Nothing was converted — the current CSVs are untouched. To convert them once you are
happy:

```bash
# one-off: CSV -> Parquet for the files already on disk
python - <<'EOF'
import glob, os, pandas as pd
for path in glob.glob("data/summaries_for_plots/*.csv"):
    base = path[:-4]
    print(path, "->", base + ".parquet")
    pd.read_csv(path, index_col=0, low_memory=False).to_parquet(
        base + ".parquet", index=False, compression="snappy")
EOF
```

Read the 6.6 GB one in chunks if memory is tight, or simply re-run stage 2 — which now
takes minutes rather than hours.
