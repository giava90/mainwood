# Performance: measurements and proposals

Measured 2026-09-10 on the real `data/Entlebuch/outputs/BAU` files (17 files, 35 MB) and
on synthetic frames sized like Surselva. Nothing here has been implemented yet.

**Headline: the parallelism is compensating for work that should not be happening.**
Stage 2 spends 92% of its time in `read_csv`, and ~97.5% of what it parses is thrown away
one line later.

---

## 1. Stage 2 reads every file ~40× larger than it needs to — **16.7×**

A SorSim output has a long per-tree block followed by a short aggregated block. Only the
aggregated block is used:

| file | lines parsed | lines kept |
|---|---|---|
| `sorsim_output108_1_planted_02.csv` | 23 473 | 560 |
| `sorsim_output21_7_planted_01.csv` | 31 223 | 757 |
| `sorsim_output47_7_planted_999.csv` | 29 942 | 831 |

`process_file` parses all 23 473 lines into a DataFrame, then slices off the first 22 913.

```python
df = pd.read_csv(..., sep=";", low_memory=False)     # parses everything
cut = df[df["#ID"] == "#Gruppierungsmerkmal"]
df = df[cut.index[0]:]                                # keeps 2.5%
```

Finding the marker with a raw byte scan and parsing only the tail:

```python
def process_file(args):
    file_path, file_name, management_scenario = args
    with open(os.path.join(file_path, file_name), "rb") as fh:
        blob = fh.read()
    pos = blob.find(b"\n#Gruppierungsmerkmal")
    if pos == -1:
        return None
    df = pd.read_csv(io.BytesIO(blob[pos + 1:]), sep=";", encoding="cp1252")
    ...
```

| | 17 files |
|---|---|
| current | 668 ms |
| byte-scan + parse tail | 40 ms |
| | **16.7×** |

Two things come free with this:

- **Correct dtypes.** Today the per-tree block forces every column to `object`
  (`Volumen OR [m3]` arrives as strings); parsing only the aggregated block yields
  `float64` directly. This is the same problem as [known issue 4](06-known-issues.md).
- **`encoding="cp1252"`** becomes safe to add, since the marker is found on raw bytes —
  which also removes [known issue 7](06-known-issues.md).

## 2. The sawmill split runs a Python function per row — **227×**

```python
summaries["Volumen OR [m3]_for_sawmills"] = summaries.apply(
    lambda x: calculate_biomass_for_sawmills(x, baumart2fraction), axis=1
)
```

`apply(axis=1)` builds a `Series` object for **every row**. On 2 M rows that is 13.6 s; on
a Surselva-sized frame (~40 M rows) roughly **4.5 minutes**, twice (the second call was
already replaced by a subtraction).

```python
frac = summaries["baumart_for_quality"].map(baumart2fraction).fillna(0.0).to_numpy()
summaries["Volumen OR [m3]_for_sawmills"] = np.where(
    summaries["is_for_sawmills_diameter"].to_numpy(),
    summaries["Volumen OR [m3]"].to_numpy() * frac,
    0.0,
)
```

| 2 M rows | |
|---|---|
| `apply(axis=1)` | 13.63 s |
| vectorised | 0.06 s |
| | **227×** — bit-identical output (verified) |

## 3. The other `.apply(lambda)` calls — **3–4.5×**

Same pattern, smaller stakes. On 2 M rows:

| function | current | vectorised | gain |
|---|---|---|---|
| `add_sawmill_diameter_info` | 0.57 s | 0.19 s (`.map` on a `category`) | 3.0× |
| `split_by_soft_hard` | 0.86 s | 0.19 s (`.isin`) | 4.5× |
| `map_species_for_quality` | 0.64 s | 0.49 s (`.map().fillna()`) | 1.3× |
| `preprocess_data` umlaut fixups | — | `.str.replace` | small |

## 4. Memory: 20 GB → under 1 GB — **26×**

This is why `run_analysis.sh` asks for 20 GB on one core. Ten of the columns are
low-cardinality strings held as Python objects:

| 2 M rows | |
|---|---|
| object/str dtypes (today) | 1 001 MB |
| categorical + `int16` year | 39 MB |

Extrapolated to Surselva WOOD (~40 M rows): **20.0 GB today vs 0.8 GB** with

```python
for col in ["Baumart", "Laengenklasse", "Staerkenklasse", "stand", "simtype",
            "planted_species", "diameter_class", "cohort"]:
    summaries[col] = summaries[col].astype("category")
summaries["year"] = summaries["year"].astype("int16")
```

Apply this right after `preprocess_data`. `groupby` on categoricals is also faster. One
caveat: `pd.concat` of frames with different categories falls back to object — set the
dtypes **after** the concat, not in `process_file`.

## 5. Writing the summary

`to_csv` on 500 k rows takes 1.5 s → about **2 minutes** for a 6.6 GB file. That is not a
bottleneck and needs no change. (For reference only, since you want plain CSV: the same
data is ~10× smaller gzipped and ~15–20× smaller as Parquet, and Parquet round-trips in
seconds rather than minutes.)

## 6. Stage 1: one JVM per file

`run_sorsim` calls `JavaGateway.launch_gateway(...)` **once per file** — a fresh JVM start,
classpath load and shutdown for each of thousands of stands. This is the reason stage 1
needs cores rather than the conversion itself.

I have not measured it (it needs Java, which is not on this machine), but it is worth
timing on Euler:

```bash
time python ../minimal/run_sorsim.py ../minimal/sorsim/SorSim4Python.jar \
     <one tree list> /tmp/out.csv 6 True
```

If JVM startup is a meaningful share of that, the fix is to launch **one** gateway and
loop over files inside it — `tools.run_sorsim` would take a list of (in, out) pairs, or
grow a `gateway=None` argument so a caller can reuse one. A single JVM processing files
sequentially may well beat 4 JVMs each restarting per file, and it would remove most of
the reason for the multiprocessing pool in stage 1.

---

## Suggested order

| # | Change | Gain | Risk | Effort |
|---|---|---|---|---|
| 1 | Byte-scan to the marker in `process_file` | 16.7× on the dominant cost | low — same rows, better dtypes | ~15 lines |
| 2 | Vectorise the sawmill split | 227× on that step | low — verified identical | ~6 lines |
| 3 | Categorical dtypes after concat | 26× memory | low | ~5 lines |
| 4 | Vectorise the other `.apply`s | 3–4.5× | low | ~10 lines |
| 5 | Measure and reuse the JVM in stage 1 | unknown, possibly large | medium — touches the SorSim call | half a day |

1–4 together should turn stage 2 from an 8-hour 20 GB single-core job into something that
runs in minutes inside a couple of GB — at which point the `sample_size` argument stops
being necessary and you can summarise every file rather than 100 of them.

The existing tests cover every one of these changes: they assert on the values that come
out of `process_file`, `preprocess_data`, the classification steps and the sawmill split,
so a vectorisation that changes a number will fail rather than quietly ship.
