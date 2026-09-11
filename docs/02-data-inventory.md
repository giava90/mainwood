# What is in this folder

Sizes measured 2026-09-10 on the local (Windows/OneDrive) copy. **13 GB of the ~14 GB
total is `data/summaries_for_plots/`.**

## Code

| Path | What it is | Status |
|---|---|---|
| `code/naming.py` | Single source of truth for every file name in the pipeline | new (2026-09) |
| `code/convert_data.py` | Stage 1 driver: ForClim → SorSim tree list → assortments | active |
| `code/convert_data_from_intermediate.py` | Same, but starts from tree lists already on disk (resume after a failed SorSim step) | active, rarely used |
| `code/summarize_and_create_plots.py` | Stage 2: assortments → `summaries_for_plots` + figures | active, 1270 lines |
| `code/plot_only.py` | Re-plots from an existing summary CSV — no SorSim reading | active |
| `code/run_conversion.sh` | `sbatch` wrapper for stage 1 | active |
| `code/run_analysis.sh` | `sbatch` wrapper for stage 2 | active |
| `code/bash_code_to_create_folder_structure_for_data.sh` | `mkdir` skeleton for a new region | **superseded** by `code/setup_data_tree.py`, which reads the configured paths |
| `code/SortimentsVorgabenListe.csv` | Copy of the SorSim assortment specification | duplicate of `minimal/SortimentsVorgabenListe.csv` and `minimal/sorsim/data/…`; not read by any script |
| `minimal/functions/tools.py` | The real conversion logic + the py4j SorSim call | active — the scientific core |
| `minimal/output_input_converter.py` | CLI wrapper around `tools.output_input_converter` | active |
| `minimal/run_sorsim.py` | CLI wrapper around `tools.run_sorsim` | active |
| `minimal/sorsim/` | Vendored SorSim (WSL): Java source, `SorSim4Python.jar`, test data | keep — the jar is what actually runs |
| `minimal/testdata/` | Small real input/intermediate/output triplet | keep — now used by the tests |
| `quick_check.ipynb` | 588 KB scratch notebook with embedded outputs, untracked | see [cleanup](04-cleanup-proposal.md) |

`code/summarize_and_create_plots.py` also carries four plotting functions that are no
longer called: `plot_biomass`, `plot_normalized_biomass_for_sawmill_categories`,
`plot_normalized_biomass_for_sawmill_categories_and_altitues[_old]`,
`plot_percentages_of_wood_quality`. The live ones are `plot_percentages_of_wood` and
`plot_biomass_by_diameter_class`.

## Reference data (small, keep, tracked)

| Path | Rows | What it is |
|---|---|---|
| `data/<region>/stand.details.csv` | Entlebuch 9 300, Surselva 7 444, Misox 4 940, Vaud 2 688 | Per-stand attributes. **`fsID` and `area_ha` are mandatory**; `Above1000m` is optional (only Entlebuch and Misox have it) |
| `data/fraction_quality.xlsx` | — | Share of each species' stem wood in quality classes A/B/C/D. Only `Wertklasse == 2` rows are used; A+B+C becomes "For Sawmills" |
| `minimal/templateSpec_v2.txt` | 31 | ForClim `speciesID` → Latin name; drives the species mapping |
| `data/manag4giacomo/` + `.zip` | — | Management-area tables received from a colleague (entlebuch, jurapark, misox, surselva, vaud). The zip and the unpacked folder are the same content |
| `data/<region>/manag_areas_all*.csv` | — | Copies of the above, per region. Not read by any script in `code/` |

`stand.details.csv` column sets differ per region — Misox has 32 columns, Vaud has 8.
Only `fsID`, `area_ha` and (optionally) `Above1000m` are consumed.

## Simulation data (large, local copies are partial leftovers)

Local state, `data/<region>/`:

| Region | inputs | intermediate | outputs | total |
|---|---|---|---|---|
| Entlebuch | BAU 1, BIO 103, WOOD 4, HYBRID 96 files (178 MB) | BAU 1, WOOD 8 files (13 MB) | BAU 17, BIO 19, WOOD 15, HYBRID 16 files (548 MB) | 739 MB |
| Vaud | BAU 6, WOOD 3, HYBRID 2 files (4.7 MB) | BAU 6, HYBRID 4 files (404 KB) | BAU 6, HYBRID 2 files (22 MB) | 27 MB |
| Surselva | empty | empty | empty | 1.5 MB (stand details only) |
| Misox | empty | empty | empty | 856 KB |

These counts do not line up with each other (17 BAU outputs from 1 BAU input), which is
the signature of ad-hoc local testing rather than a complete run. **The complete runs
live on Euler scratch** (`/cluster/scratch/giacomov/mainwood/`).

Leftover files in `intermediate/` are a reliable sign of an interrupted run: a
successful `run_sorsim` deletes its tree list.

## The product — `data/summaries_for_plots/` (13 GB)

| File | Size | Written |
|---|---|---|
| `Surselva_WOOD.csv` | 6.6 GB | 2025-11-05 |
| `Surselva_BAU.csv` | 5.0 GB | 2025-11-05 |
| `Surselva_BIO.csv` | 551 MB | 2025-11-05 |
| `Entlebuch_BAU.csv` | 15 MB | 2025-10-30 |
| `Vaud_BAU.csv` | 519 KB | 2026-05-07 |
| `Vaud_WOOD.csv` | 73 KB | 2026-05-08 |
| `Vaud_HYBRID.csv` | 1.9 KB | 2026-05-12 |

The three small Vaud files are test-sized, not real runs. The Surselva and Entlebuch
files are the real output.

### Columns

```
<unnamed index>, Baumart, Laengenklasse, Staerkenklasse,
Volumen OR [m3], Volumen IR [m3], Wert [CHF], Anzahl,
simtype, stand, planted_species, plantation, cohort, year,
sim_area (m2), area, diameter_class, is_soft, is_hard,
Volumen OR [m3]_for_sawmills, Volumen OR [m3]_not_for_sawmills
```

- `Volumen OR/IR` are **already weighted** by the planting share and **already rescaled**
  to the real stand area in m³ (not per hectare).
- `area` is the stand area in ha; `sim_area (m2)` is always 62 500.
- `cohort` is new (`dead` / `alive`); files written before 2026-09 do not have it —
  treat a missing column as `dead`.
- The unnamed first column is a pandas index artefact (`to_csv` without `index=False`).

Why they are so large: one row per (year × species × length class × diameter class ×
stand × planting variant × climate). Surselva has 7 444 stands.

## Figures

`figures/` holds 32 current PNGs plus `figures/all_figs/` (39 PNGs, 25 MB) and
`figures/all_figs.tar.gz` (48 MB — a *larger* archive of the same 39 files, made before
the folder was unpacked). Everything here is reproducible from the summaries with
`code/plot_only.py`. `figures/` is in `.gitignore`.

## Repository hygiene

`.gitignore` contains only `figures/` and `data/`, but 24 data files were committed
before that: 10 Entlebuch WOOD outputs (now deleted in the working tree), the four
`stand.details.csv`, `fraction_quality.xlsx`, and `data/summaries_for_plots/Entlebuch_BAU.csv`
(7.3 MB — the largest blob in the history). The repository is still only 7.9 MB, so no
history rewriting is needed.

`minimal/functions/__pycache__/tools.cpython-311.pyc` is tracked and shows up as
modified on every run.
