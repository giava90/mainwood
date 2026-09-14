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

## Fixed on 2026-09-13

### 9. Stage 1 deadlocked on its own shared failure list — *the Jurapark hang*

A 48-core Jurapark run (60,870 files) produced 7,906 outputs and then stopped
completely: zero files for more than five hours, 0% CPU, while SLURM reported it
`RUNNING`. The job log had the answer:

```
Exception in thread Thread-1 (accepter):
  File ".../multiprocessing/managers.py", line 194, in accepter
    t.start()
RuntimeError: can't start new thread
```

`process_files` kept the failed-file list in a `multiprocessing.Manager` list, and
`run_sorsim` tested `file in failed` for **every** file. Each test is a remote call
into the manager process, and the manager spawns a thread per connection. At 48
workers over 60,870 files it exhausted the thread limit; once the accepter died the
manager stopped accepting, and every worker blocked forever on its next call.

Two things made this hard to read from outside:

- `MaxRSS` was 50 GB of 122 GB requested, so it was never a memory problem — the
  obvious first theory, and wrong.
- `sstat` returned nothing for the live job, and `sacct` showed `TotalCPU
  00:00:00` until the job ended, so "is it using CPU" could not be answered while
  it mattered. The completed record shows `2-15:04:34` — it worked, then flatlined.

Failures are now ordinary return values collected from `starmap`, and phase 1's
failures are filtered out before phase 2 rather than re-queried per file. There is
no shared object left to exhaust. `tests/test_no_shared_state.py` pins it,
including that no `failed` parameter and no `Manager()` returns to either
converter.

This also removed 60,870 remote calls from the run, each transferring a growing
list — a likely part of why 48 cores managed only 20 files/h/core while 16 cores
managed 72.7.

## Checked, not a defect

### Jurapark BAU looked like the WOOD scenario — *checked 2026-09-14, no code issue*

The species composition of Jurapark BAU under RCP 8.5 showed Buche falling 42% to
2% and Foehre rising 8% to 48%, which reads more like the paper's WOOD scenario
than its BAU.

Checked and cleared:

* the stage 1 code path is clean — one scenario variable drives the input path,
  the output path and both subfolders, so a `mgmt_BAU` input cannot produce a
  WOOD-labelled output;
* the preflight log for that run printed the resolved input folder as
  `/cluster/work/climate/amauri/Jurapark/Results/mgmt_BAU/dead.trees/`.

One thing I had wrong, corrected by the person running these: **planting is not
scenario-specific — it happens under BAU too.** What is WOOD-only is
*plantations*, the small planted conifer stands of Douglasie and Weisstanne. So
`planted_species` values other than 999 in a BAU summary are expected and carry no
signal about which folder was read. `diagnose_summary.py` said otherwise and has
been corrected.

Their note on the remaining surprise, left as written:

> I am still surprised by the shift in species composition from Beech to Foehre.
> Beech is an hardwood species. However, this surprise is linked to a lack on
> ecological understanding from my side.

Worth keeping alongside it: Foehre is not a WOOD signature either — WOOD plants
Douglasie and Weisstanne, which appear as `Ubrige Nadelholz` and `Tanne` and were
only 5% and 10% in that figure. And the paper's "Buche und Fichte stay above 50%
under BAU" is stated for **RCP 4.5**, while this figure is RCP 8.5, where the paper
says the decline is stronger without giving a number.

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

### 7. SorSim writes replacement characters into its own output

*Corrected 2026-09-10 after checking the bytes — an earlier version of this file
claimed the files were Windows-1252 and recommended `encoding="cp1252"`. They are not,
and that change would have corrupted the data.*

SorSim writes the file with the UTF-8 replacement character (`EF BF BD`) already baked
in, so `Längenklasse` reaches disk as `L�ngenklasse`. Reading it back:

| read as | column name | species |
|---|---|---|
| default (what the code did) | `L�ngenklasse` | `F�hre` |
| `utf-8` + `encoding_errors="replace"` | `L�ngenklasse` | `F�hre` |
| `cp1252` | `Lï¿½ngenklasse` | `Fï¿½hre` |

`preprocess_data` then maps `�` → `oe`, which is where `Foehre`, `Loerche` and
`Ubrige Laubolz` come from — those spellings are not typos, they are the round trip.

The reader now passes `encoding="utf-8", encoding_errors="replace"` explicitly. That is
what pandas was doing implicitly, so the values are unchanged, but it no longer depends
on the pandas version choosing to replace rather than raise. The umlauts cannot be
recovered — the information is gone before the file is written.

### 8. An unknown `Staerkenklasse` is fatal (by design)

`add_sawmill_diameter_info` used `dict[x]`, not `dict.get(x)`. SorSim writes `Restholz 1`,
`Restholz 2`, … in the per-tree block and plain `Restholz` in the aggregated block, so a
change in SorSim's aggregation would abort stage 2 hours in.

Still deliberately fatal — silently bucketing an unknown class would corrupt the volumes —
but the vectorisation (2026-09-10) now checks the distinct values up front and raises
`KeyError: Unknown Staerkenklasse values: [...]` naming them, instead of failing on
whichever row happened to hit it first.

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
