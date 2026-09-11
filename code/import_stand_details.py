"""Turn a ForClim `manag4giacomo` delivery into a region's ``stand.details.csv``.

The deliveries are not uniform. Across the five regions the file is named
``manag_areas_all.csv``, ``manag_areas_all7.csv`` or ``manag_areas_all12.csv``;
the region folder is lower-case; the column sets differ entirely; and Jurapark
spells the area column ``Area_ha`` while everyone else writes ``area_ha``. Copying
by hand means noticing all of that every time, and the one thing that must be
right -- the area column -- is the one that silently scales every volume in the
summary if it is wrong or absent.

Usage:
    python import_stand_details.py <Region> <delivery.csv>          # write it
    python import_stand_details.py <Region> <delivery.csv> --check  # compare only

Example:
    python import_stand_details.py Jurapark \
        ../data/manag4giacomo/manag4giacomo/jurapark/manag_areas_all.csv

Writes ``<MAINWOOD_STAND_DETAILS>`` for that region, which by default is the
tracked copy at ``data/<Region>/stand.details.csv``. Commit the result -- see
docs/03-runbook.md section 1.3.
"""

import hashlib
import os
import sys

import pandas as pd

import paths
import regions

#: Columns stage 2 cannot work without.
REQUIRED = ("fsID", "area_ha")

#: Optional, enables the altitude-split figures only.
OPTIONAL = ("Above1000m",)


def md5(path):
    """Checksum of a file, for recording which delivery a copy came from."""
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def normalise(frame):
    """Rename the area column to ``area_ha`` whatever case the delivery used.

    Returns:
        tuple[pandas.DataFrame, list[str]]: The frame and a list of notes.
    """
    notes = []
    if "area_ha" not in frame.columns:
        candidates = [c for c in frame.columns if c.lower() == "area_ha"]
        if candidates:
            frame = frame.rename(columns={candidates[0]: "area_ha"})
            notes.append(f"renamed {candidates[0]!r} -> 'area_ha'")
    return frame, notes


def validate(frame, region):
    """Check the frame can actually drive stage 2.

    Returns:
        list[str]: Blocking problems, empty if the frame is usable.
    """
    problems = []
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        problems.append(
            f"missing required column(s) {missing}; delivery has {sorted(frame.columns)}"
        )
        return problems

    if frame["area_ha"].isna().any():
        problems.append(f"{int(frame['area_ha'].isna().sum())} rows with missing area_ha")
    if frame["fsID"].isna().any():
        problems.append(f"{int(frame['fsID'].isna().sum())} rows with missing fsID")
    if frame["fsID"].duplicated().any():
        problems.append(f"{int(frame['fsID'].duplicated().sum())} duplicate fsID values")
    return problems


def describe(frame, region, source):
    """The provenance worth pasting into a commit message."""
    lines = [
        f"  region      {region}",
        f"  source      {source}",
        f"  md5         {md5(source)}",
        f"  stands      {len(frame)} ({frame['fsID'].nunique()} unique fsID)",
    ]
    if "area_ha" in frame.columns:
        area = frame["area_ha"]
        lines.append(f"  area_ha     {area.sum():,.1f} ha total, min {area.min():.4f}, max {area.max():.2f}")
        zero = int((area <= 0).sum())
        if zero:
            lines.append(f"  note        {zero} stand(s) with area_ha <= 0 contribute nothing")
    for column in OPTIONAL:
        if column not in frame.columns:
            lines.append(f"  note        no {column} column -- the altitude-split figures will not work")
    return "\n".join(lines)


def main(argv):
    region = regions.check_case_study(argv[1], include_all=False)
    source = argv[2]
    check_only = "--check" in argv[3:]

    if not os.path.isfile(source):
        raise SystemExit(f"Delivery not found: {source}")

    frame, notes = normalise(pd.read_csv(source))
    for note in notes:
        print(f"  {note}")

    problems = validate(frame, region)
    print(describe(frame, region, source))
    if problems:
        print("\nBlocking problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    target = paths.stand_details_path(region)

    if os.path.isfile(target):
        existing = pd.read_csv(target)
        same = existing.equals(frame)
        print(f"\n  existing    {target}")
        print(f"  identical   {same}")
        if not same:
            print(f"  existing has {len(existing)} rows, delivery has {len(frame)}")
            if "area_ha" in existing.columns:
                print(f"  existing area_ha total {existing['area_ha'].sum():,.1f} ha")
            else:
                print("  existing has NO area_ha column -- it is an older delivery")
    elif check_only:
        print(f"\n  {target} does not exist yet")

    if check_only:
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    frame.to_csv(target, index=False)
    print(f"\nWrote {target}")
    print("Commit it:  git add " + target + " && git commit")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv))
