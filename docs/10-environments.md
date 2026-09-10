# Environments

What is installed on this machine, what was broken, and how it was fixed
(2026-09-10). Euler is unaffected by any of this — there the module stack provides
everything (`module load stack/2024-06 python/3.12.8`).

## Python: the conda `base` environment

### What was wrong

`import pandas` and `import matplotlib` both failed:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
            Expected 96 from C header, got 88 from PyObject
ImportError: numpy.core.multiarray failed to import
```

`numpy 2.4.6` had been installed **with pip** (`pypi_0` in `conda list`), replacing the
conda numpy, while pandas 2.0.3, matplotlib 3.7.2, scipy 1.11.1, scikit-learn 1.3.0 and
contourpy 1.0.5 were still the conda builds **compiled against numpy 1.x**. numpy 2
changed the `PyArray_Descr` layout and removed `numpy.core`, so every one of those
extension modules failed at import.

### The fix: newer wheels, not a recompile, and numpy untouched

No local compilation was needed — and none was possible, there is no MSVC compiler on
this machine. Since numpy 2.0, the projects ship Windows wheels **already built against
the numpy 2 headers** (and ABI-compatible with numpy 1.x as well), so installing a
current wheel *is* "matplotlib built against the newer numpy":

```bash
printf 'numpy==2.4.6\n' > pin.txt          # constraint: numpy must not move
python -m pip install --upgrade -c pin.txt \
    "matplotlib>=3.9" "pandas>=2.2.3,<3" "scipy>=1.13" "scikit-learn>=1.5" "contourpy>=1.2.1"
python -m pip install --upgrade -c pin.txt numexpr bottleneck
python -m pip install --upgrade -c pin.txt numba shapely geopandas
```

The `-c pin.txt` constraint is the important part: it lets pip upgrade everything else
while making a numpy downgrade impossible.

`pandas` was deliberately capped at `<3`. Left unpinned, pip wanted **pandas 3.0.5**,
which is a major release with breaking changes (copy-on-write, string dtypes) that would
have quietly altered behaviour across every project using this environment. 2.3.3 is
numpy-2-compatible and behaviourally close to the 2.0.3 that was there.

### Result

| package | before | after |
|---|---|---|
| numpy | 2.4.6 (broken deps) | **2.4.6 — unchanged** |
| matplotlib | 3.7.2 ✗ | 3.11.1 ✓ |
| pandas | 2.0.3 ✗ | 2.3.3 ✓ |
| scipy | 1.11.1 ✗ | 1.17.1 ✓ |
| scikit-learn | 1.3.0 ✗ | 1.9.1 ✓ |
| numexpr / bottleneck | numpy-1 builds (warned loudly) | 2.14.2 / 1.6.0 ✓ |
| numba / shapely / geopandas | ✗ | 0.67.0 / 2.1.2 / 1.1.4 ✓ |

matplotlib renders, pandas computes, and the import is silent. The full test suite of
this repository passes in `base` (111 tests).

### Still broken in `base`, and why

`darts 0.31.0` declares `numpy<2.0.0`. That is the package's own constraint, not an ABI
problem — it cannot be fixed without either downgrading numpy or upgrading darts. If you
need darts, give it its own environment rather than moving `base` back to numpy 1.

If you truly want a source build of matplotlib against your numpy, that needs MSVC Build
Tools plus freetype/qhull; the wheel route above achieves the same ABI result.

## R and RStudio

### What was wrong

The conda env `rstudio` contained an **R from October 2019** that no longer starts
(`R.exe` exits with code 53 and no output). It was not repaired.

### What was installed

```bash
winget install --id RProject.R        # R 4.6.1
winget install --id Posit.RStudio     # RStudio 2026.08.2+200
```

| | path |
|---|---|
| R | `C:\Users\giacomov\AppData\Local\Programs\R\R-4.6.1` |
| `Rscript.exe` | `...\R-4.6.1\bin\x64\Rscript.exe` |
| RStudio | `C:\Users\giacomov\AppData\Local\Programs\RStudio\rstudio.exe` |
| user library | `C:\Users\giacomov\AppData\Local\R\win-library\4.6` |

Packages installed there: **arrow 25.0.1**, readr 2.2.0, dplyr 1.2.1, nanoparquet 0.5.1.

RStudio finds R 4.6.1 by itself; nothing further is needed to open it and
`source("code/read_summaries.R")`.

### Two leftovers to decide about

- **`rstudio` conda env** — the broken 2019 R. Nothing uses it now.
  `conda env remove -n rstudio` if you agree.
- **`r-mainwood` conda env (2.4 GB)** — created as a fallback R+arrow while the winget
  installs were running. Redundant now that the CRAN R works.
  `conda env remove -n r-mainwood` if you agree.

Neither was removed; that is your call.

## Which interpreter for what

| task | use |
|---|---|
| the pipeline, the tests | `base` (now healthy) or the Euler module stack |
| reading summaries in R | `...\R-4.6.1\bin\x64\Rscript.exe`, or RStudio |
| anything needing numpy < 2 (darts) | a separate env |

The repository's dependencies are in `requirements.txt`; the R side needs only `arrow`
(or `nanoparquet`), plus `readr`/`dplyr` for the convenience paths.
