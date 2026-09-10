# Cleanup proposal

**Nothing in this document has been executed.** Every command is here for you to run
(or not) yourself. Sizes are from 2026-09-10.

Summary: ~830 MB of local leftovers can go, `figures/all_figs.tar.gz` (48 MB) is
redundant, and the 13 GB in `data/summaries_for_plots/` should be **kept but moved off
OneDrive**.

---

## A. Keep — do not touch

| Path | Why |
|---|---|
| `data/summaries_for_plots/Surselva_{WOOD,BAU,BIO}.csv` (12 GB) | The product. Reproducing them means re-running SorSim over 7 444 stands |
| `data/summaries_for_plots/Entlebuch_BAU.csv` (15 MB) | Real run |
| `data/*/stand.details.csv`, `data/fraction_quality.xlsx` | Small, irreplaceable reference data |
| `minimal/` (all of it, including `sorsim/` and `testdata/`) | The jar is what runs; `testdata/` is now a test fixture |
| `code/` | Active |
| `data/manag4giacomo/` | Received from a colleague; keep the folder |

## B. Delete — local test leftovers

Partial runs from local debugging. The complete versions are on Euler scratch, and
anything here is one job away from being regenerated.

```bash
# 768 MB — partial inputs/outputs from ad-hoc local testing
rm -rf data/Entlebuch/inputs/*   data/Entlebuch/outputs/*   data/Entlebuch/intermediate/*
rm -rf data/Vaud/inputs/*        data/Vaud/outputs/*        data/Vaud/intermediate/*
# keep the empty scenario folders, or recreate them with:
#   cd data/<Region> && bash ../../code/bash_code_to_create_folder_structure_for_data.sh
```

The `intermediate/` folders in particular (Entlebuch 9 files, Vaud 10) are debris from
interrupted runs — a successful `run_sorsim` deletes its own tree list.

Also:

```bash
rm -f  output.png                                   # 48 KB, untitled scratch plot at the repo root
rm -rf minimal/functions/__pycache__                # tracked .pyc, see section D
rm -f  data/manag4giacomo.zip                       # 4.3 MB, identical to the unpacked folder
```

## C. Decide — small test-sized summaries

```
data/summaries_for_plots/Vaud_BAU.csv      519 KB
data/summaries_for_plots/Vaud_WOOD.csv      73 KB
data/summaries_for_plots/Vaud_HYBRID.csv   1.9 KB
```

These came from 6, 3 and 2 input files respectively — they are test artefacts, not
results, and a 1.9 KB "regional summary" will mislead whoever finds it next. Either
delete them or rename them `*_TEST.csv`.

## D. Repository hygiene

`.gitignore` lists `data/` and `figures/`, but 24 data files were committed before that
existed. They are still tracked, so they keep showing up in `git status`.

```bash
# 1. Commit the code work already in the tree (this is real work worth keeping):
git add code/ tests/ docs/ requirements.txt pytest.ini README.md
git commit -m "Add alive-cohort support, shared naming module, tests and docs"

# 2. Stop tracking generated data and byte-code (leaves the files on disk):
git rm --cached -r data/Entlebuch/outputs
git rm --cached data/summaries_for_plots/Entlebuch_BAU.csv
git rm --cached minimal/functions/__pycache__/tools.cpython-311.pyc
git commit -m "Untrack generated data and byte-code"
```

Keep tracking `data/*/stand.details.csv` and `data/fraction_quality.xlsx` — they are
inputs, not outputs, and `.gitignore` does not affect already-tracked files.

Extend `.gitignore`:

```gitignore
figures/
data/
__pycache__/
*.pyc
.pytest_cache/
output.png
```

The repository is 7.9 MB in total, so there is **no need to rewrite history**.

### Uncommitted code changes worth committing

`git diff` currently holds four real fixes that would be lost if the working tree were
reset: the `compress_file` signature fix, the `output_folder_path` argument, re-enabling
`HYBRID` in `valid_management_scenarios`, and the `README.txt` path correction. Commit
before doing anything else.

## E. Redundant figures

```bash
rm -f figures/all_figs.tar.gz     # 48 MB archive of the 25 MB folder next to it
```

`figures/` is git-ignored and every PNG is reproducible with `code/plot_only.py`. If you
want a snapshot for a paper, keep the folder and drop the tarball.

## F. Documentation duplicates

- `README.md` was one line (`# mainwood`) and is now the entry point.
- `README.txt` predates it and only covers the folder-creation step, which is now in
  [03-runbook.md](03-runbook.md) §1.2. Delete it once you are happy with the new docs.
- `code/SortimentsVorgabenListe.csv` is a third copy of a file that also lives in
  `minimal/` and `minimal/sorsim/data/`, and no script reads the `code/` copy. Delete it
  or document which copy SorSim actually loads.
- `quick_check.ipynb` (588 KB, untracked, embedded outputs) — either clear the outputs
  and commit it as a documented exploration, or delete it. As it stands it is the only
  record of some manual checks and nobody can tell which.

## G. The 13 GB, and where it should live

`data/summaries_for_plots/` sits inside a **OneDrive-synced folder**. 13 GB of
continuously-rewritten CSV is a poor fit for that: sync is slow, and a partially-synced
6.6 GB file is indistinguishable from a complete one.

Options, in the order I would consider them:

1. **Move the folder out of OneDrive** to a local `~/data/mainwood/` and point the
   scripts at it (`folder_data` is already an argument for stage 2; the summary write
   path is hard-coded in `process_combination` and would need one edit).
2. **Keep only what you plot.** The summaries hold every `(year × species × length class
   × diameter class × stand × planting variant × climate)` row, but the figures aggregate
   over stands and species immediately. A pre-aggregated file would be a few MB.
3. Keep the archive on Euler scratch (note: **scratch is purged**, so this is not
   long-term storage) or on the group's project space.

**Update (2026-09-10):** the pipeline now writes Parquet by default, which takes the
folder from ~13 GB to ~2.4 GB, with a converter that reproduces the old CSVs
byte-for-byte for collaborators. See [09-summary-format.md](09-summary-format.md). The
existing CSVs have not been touched.
