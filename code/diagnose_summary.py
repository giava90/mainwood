"""Report what a summary actually contains, in units that compare across regions.

Written to answer one question: a Jurapark BAU result looked more like the paper's
WOOD scenario than its BAU. Volume totals cannot settle that -- regions differ in
area and stand count -- so this reports m3 per hectare per year alongside the two
things that distinguish the scenarios in the paper:

* **planting is NOT a discriminator.** Corrected 2026-09-14 by the person who
  runs these: planting happens under BAU too. What is WOOD-only is *plantations*
  -- the small planted conifer stands of Douglasie and Weisstanne, up to 30% of
  forest area in Vaud and 18% in Entlebuch. So a BAU summary carrying
  ``planted_species`` values other than 999 is expected, and says nothing about
  which folder was read. The ``plantation`` flag is the WOOD-only one.
* **species composition.** Under BAU, Buche and Fichte decline but stay
  substantial; under WOOD they fall below roughly 10% of assortments.

Usage:
    python diagnose_summary.py Jurapark BAU
    python diagnose_summary.py Jurapark BAU --cohort alive
    python diagnose_summary.py Jurapark BAU --compare Vaud_BAU --compare Vaud_WOOD
    python diagnose_summary.py --all            # every summary in the folder

Reads whichever of .parquet / .csv exists, Parquet first, and takes --data like
the other scripts (default: MAINWOOD_SUMMARY_DIR).
"""

import argparse
import os
import sys

import pandas as pd

import paths
from summary_io import list_summaries, locate_summary

VOLUME = "Volumen OR [m3]"

#: The species the paper tracks as the BAU/WOOD discriminator.
BAU_INDICATOR_SPECIES = ("Buche", "Fichte")

#: Planted-conifer species in the WOOD scenario. "Ubrige Nadelholz" is mostly
#: Douglasie per the paper.
WOOD_PLANTED_HINTS = ("Tanne", "Ubrige Nadelholz", "Douglasie")


def find_summary(data_dir, case_study, management, cohort="dead"):
    """Locate one summary, searching parquet/ then csv/ then the folder itself."""
    suffix = "" if cohort == "dead" else f"_{cohort}"
    return locate_summary(data_dir, f"{case_study}_{management}{suffix}")


def stand_area(case_study):
    """Total area from stand.details.csv, or None if it cannot be read."""
    path = paths.stand_details_path(case_study)
    if not os.path.isfile(path):
        return None
    stands = pd.read_csv(path)
    if "area_ha" not in stands.columns:
        return None
    return float(stands["area_ha"].sum())


def describe(path, case_study, management):
    """One summary, reported in comparable units.

    Returns:
        dict: The headline numbers, for the comparison table.
    """
    reader = pd.read_parquet if path.endswith(".parquet") else pd.read_csv
    frame = reader(path)

    print(f"\n{'=' * 72}\n{os.path.basename(path)}  ({len(frame):,} rows)\n{'=' * 72}")

    simtypes = sorted(frame["simtype"].astype(str).unique())
    years = frame["year"].astype(int)
    print(f"  simtypes      {simtypes}")
    print(f"  years         {years.min()}-{years.max()}")
    print(f"  stands        {frame['stand'].nunique():,}")

    # --- planting: the clearest BAU/WOOD discriminator ---------------------
    if "planted_species" in frame.columns:
        planted = frame["planted_species"].astype(str)
        counts = planted.value_counts()
        non_default = counts[counts.index != "999"]
        share = frame.loc[planted != "999", VOLUME].sum() / frame[VOLUME].sum() * 100
        print(f"  planted_species values: {list(counts.index[:8])}")
        if len(non_default):
            print(f"  {len(non_default)} planted species present, {share:.1f}% of volume")
            print("      Expected under BAU as well as WOOD -- planting is not")
            print("      scenario-specific. Plantations are; see the flag below.")
        else:
            print("  no planting (all 999)")
    if "plantation" in frame.columns:
        pct = 100 * frame["plantation"].mean()
        print(f"  plantation flag set on {pct:.1f}% of rows"
              + ("   <-- plantations are the WOOD-only feature" if pct else ""))

    # --- volume in comparable units ---------------------------------------
    area = stand_area(case_study)
    span = max(1, years.max() - years.min() + 1)
    total = frame[VOLUME].sum()
    print(f"\n  total volume  {total:,.0f} m3 over {span} years")
    if area:
        print(f"  stand area    {area:,.1f} ha")
        print(f"  ==> {total / area:,.1f} m3/ha over the period")
        print(f"  ==> {total / area / span:,.3f} m3/ha/year   <-- compare this across regions")
    else:
        print("  stand area    unavailable -- cannot normalise")

    # --- species composition, early vs late --------------------------------
    if "Baumart" in frame.columns:
        early = frame[years <= 2050]
        late = frame[years >= 2120]
        print("\n  species share of volume (% ):")
        print(f"    {'species':<20}{'<=2050':>10}{'>=2120':>10}")
        both = (early.groupby("Baumart", observed=True)[VOLUME].sum()
                / max(early[VOLUME].sum(), 1) * 100).to_frame("early").join(
                (late.groupby("Baumart", observed=True)[VOLUME].sum()
                 / max(late[VOLUME].sum(), 1) * 100).to_frame("late"), how="outer").fillna(0)
        for species, row in both.sort_values("early", ascending=False).head(10).iterrows():
            print(f"    {str(species):<20}{row['early']:>9.1f}{row['late']:>10.1f}")

        indicator = both.reindex(list(BAU_INDICATOR_SPECIES)).fillna(0)
        print(f"\n    {' + '.join(BAU_INDICATOR_SPECIES)}: "
              f"{indicator['early'].sum():.1f}% -> {indicator['late'].sum():.1f}%")
        print("      The paper: under BAU these stay substantial; under WOOD they")
        print("      fall below roughly 10% of assortments.")

    return {
        "summary": os.path.basename(path),
        "rows": len(frame),
        "stands": frame["stand"].nunique(),
        "m3_per_ha_per_year": (total / area / span) if area else None,
        "planted_pct_volume": share if "planted_species" in frame.columns else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case_study", nargs="?")
    parser.add_argument("management", nargs="?")
    parser.add_argument("--cohort", default="dead")
    parser.add_argument("--data", default=None)
    parser.add_argument("--compare", action="append", default=[],
                        help="repeatable, as <Region>_<scenario>")
    parser.add_argument("--all", action="store_true", help="every summary in --data")
    args = parser.parse_args(argv)

    data_dir = os.path.abspath(args.data or paths.summary_dir())
    print(f"data {data_dir}")

    targets = []
    if args.all:
        for stem in list_summaries(data_dir):
            if "_" in stem:
                case_study, _, rest = stem.partition("_")
                targets.append((case_study, rest))
    else:
        if not args.case_study or not args.management:
            parser.error("give <case_study> <management>, or --all")
        targets.append((args.case_study, args.management))
        for other in args.compare:
            case_study, _, management = other.partition("_")
            targets.append((case_study, management))

    rows = []
    for case_study, management in targets:
        cohort = "alive" if management.endswith("_alive") else args.cohort
        management = management.replace("_alive", "")
        path = find_summary(data_dir, case_study, management, cohort)
        if path is None:
            print(f"\nno summary for {case_study} / {management} / {cohort}")
            continue
        rows.append(describe(path, case_study, management))

    if len(rows) > 1:
        print(f"\n{'=' * 72}\nComparable across regions\n{'=' * 72}")
        print(f"  {'summary':<32}{'m3/ha/yr':>12}{'planted % vol':>15}")
        for row in rows:
            m3 = f"{row['m3_per_ha_per_year']:.3f}" if row["m3_per_ha_per_year"] else "n/a"
            planted = f"{row['planted_pct_volume']:.1f}" if row["planted_pct_volume"] is not None else "n/a"
            print(f"  {row['summary']:<32}{m3:>12}{planted:>15}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
