# Known issues

## Fixed on 2026-09-10

### 1. Uncompressed ForClim inputs were parsed wrongly — *silent data loss*

`convert_data.parse_filename` removed the file suffix by character count **after** the
regex matched:

```python
if simtype[-3:] == ".gz":
    simtype = simtype[1:-7]     # strips ".csv.gz" — correct
else:
    simtype = simtype[1:-2]     # strips two characters — wrong for ".csv"
```

`dataSim.dead207_1_planted_06.csv` therefore yielded `simtype = "1_planted_06.c"`, and the
pipeline went looking for a tree list that never existed. Because almost every ForClim
file was delivered as `.csv.gz`, this stayed hidden — but `data/Vaud/inputs/WOOD/` holds a
plain `.csv` and `data/Vaud/outputs/WOOD/` is empty, which is this bug.

Now the suffix is stripped before matching (`naming.strip_data_suffix`), and `.csv`,
`.csv.gz`, `.gz` and `.zip` all parse identically. Pinned by
`test_naming.py::test_compression_does_not_change_the_parsed_simtype`.

**This matters for the new region**: if its ForClim files arrive uncompressed, the old
code would have produced an empty output folder with no error.

### 2. The intermediate file name was hard-coded as `deadCohorts…` in three places

`convert_forclim` wrote it, `run_sorsim` read it back, and `compress_file` deleted it —
each with its own copy of the f-string, and all three saying "dead" regardless of what was
being processed. Now all names come from `code/naming.py`; adding the alive cohort touched
one module.

### 3. `load_data` wrote metadata onto the wrong frame

```python
data["planted_species"] = planted_species
df["planting"] = ...        # df is the accumulator, not the file just read
df["plantation"] = ...
```

Every previously-accumulated row had its `planting`/`plantation` overwritten by the
current file's value. `load_data` is not on the live path (`load_data_parallel` is), so no
published result is affected, but the function was a trap for anyone switching to the
serial loader while debugging.

### 4. Volume columns stayed `object` dtype

```python
summaries.loc[:, col] = pd.to_numeric(summaries[col], errors="coerce")
```

Assigning through `.loc` writes the numbers back into the existing object-dtype column, so
`to_numeric` had no effect on the dtype. Every later multiplication and `groupby` then ran
element-by-element in Python instead of in NumPy. Numerically identical, but it is a large
part of why stage 2 needs 20 GB and hours for Surselva. Now assigned as a whole column;
`test_summary_end_to_end.py` asserts the dtype.

### 5. The BIO branch of `parse_filename` could not handle planting

It split on `_` expecting exactly two parts and stripped a fixed four characters
(`"dead"`). A BIO file with a planting variant would have raised `ValueError`; an alive
file would have lost a character from the stand id. The general pattern covers BIO
correctly, so the special case is gone.

---

## Open — worth knowing, not changed

### 6. `preprocess_data` relies on pandas index alignment for the planting weight

```python
mask = summaries['planting']
summaries.loc[mask*summaries['species_count']>1, 'weight'] /= summaries.loc[mask, 'species_count']-1
```

The left-hand selection is a strict subset of the right-hand one, so pandas' index
alignment happens to give the right answer, and `test_preprocess_weights.py` confirms the
weights sum to 1. But `mask * species_count > 1` is an integer comparison standing in for a
boolean one, and the two `.loc` selections are different sets. It works; it is not obvious.
I left the arithmetic exactly as it is because changing it would change published numbers —
the tests now make any future change verifiable.

### 7. SorSim output encoding is decoded by luck

SorSim writes Windows-1252 (`Längenklasse`, `Stärkenklasse`). The code reads it without an
`encoding=` argument and then renames the mojibake:

```python
summaries.rename({"L�ngenklasse": "Laengenklasse", ...})
```

This depends on the reader producing U+FFFD rather than raising `UnicodeDecodeError`, which
is a function of the pandas version and the locale. A different Euler stack could break it.
The robust fix is `pd.read_csv(..., encoding="cp1252")` plus proper names, but that changes
the column names in every existing summary file, so it is a deliberate migration, not a
drive-by edit.

### 8. An unknown `Staerkenklasse` raises `KeyError` mid-job

`add_sawmill_diameter_info` uses `dict[x]`, not `dict.get(x)` — the code comments even say
so. SorSim writes `Restholz 1`, `Restholz 2`, … in the per-tree block and plain `Restholz`
in the aggregated block, so a change in SorSim's aggregation would abort stage 2 hours in.
Tested as current behaviour (`test_an_unknown_diameter_class_fails_loudly`) — failing loudly
is defensible, but consider a clearer error message.

### 9. Dead code in `map_species`

`"Acer"` is matched twice (→ `Ahorn` first, so the later `Ubrige Laubholzer` branch is
unreachable), and `Pseudotsuga menziesii` → `Ubrige Nadelholzer` while
`species_quality_mapping` in stage 2 maps `Tanne` → `Douglasie`. Neither is a bug today —
`test_every_species_in_the_template_is_mapped` guards the outcome that matters — but the
species mapping is spread over three places (`tools.map_species`, `baumart2code`,
`species_quality_mapping`) and is the first thing to check if a new region has species the
current ones do not.

### 10. Diameter classes are quality proxies, not sawmill limits

The source comment is explicit: sawmills take roughly 15–18 cm up to 50–60 cm, while the
code splits `4`–`8` as "for sawmills". Documented here so the assumption is not mistaken
for a measurement.

### 11. `sim_area (m2)` is hard-coded

`100 patches × 625 m² = 62 500 m²` for every stand, replacing an earlier
`n.patches`-based computation (commit *"updating code based on the fact 100 patches are
simulated for each stand"*). If the new region is simulated with a different patch count,
this constant in `augment_with_stand_data` must change, and there is nothing that would
warn you.

### 12. Stands missing from `stand.details.csv` produce `NaN`, not an error

`augment_with_stand_data` prints `Stand <id> not found in stand data` and continues; those
rows end up with `NaN` area and `NaN` volume, which then vanish from the sums. Check the
join before launching — see [03-runbook.md](03-runbook.md) §1.3.
