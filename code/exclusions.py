"""Which stands cannot be computed, and why.

A stand is only usable if ``stand.details.csv`` can tell us its area: stage 2
rescales patch volumes to the real stand area through ``area_ha``, so a stand the
join cannot find, or one whose area is zero, cannot produce a meaningful volume.

Until now both cases went through the whole pipeline anyway. A missing stand
produced ``NaN`` volumes after one printed warning; a zero-area stand produced a
confident ``0``, which is worse, because it is indistinguishable from a stand that
genuinely harvested nothing and it silently drags down any per-stand mean.

Both are now excluded at **stage 1**, before SorSim runs, so no cluster time is
spent on a stand whose result would be discarded -- and the run leaves behind a
report naming every excluded stand and the reason, which is the list to hand back
to the ForClim side and to re-run once corrected data arrives.

Exclusion is deliberately narrow. It covers only stands that cannot be computed,
never stands whose numbers are merely surprising.
"""

import datetime as dt
import os

#: Reason codes written into the report. Stable strings -- downstream code and
#: the ForClim side both read them.
NOT_IN_STAND_DETAILS = "not in stand.details.csv"
NON_POSITIVE_AREA = "area_ha <= 0"

#: Columns of the report, in order.
REPORT_COLUMNS = (
    "stand",
    "reason",
    "area_ha",
    "n_files",
    "example_file",
    "case_study",
    "scenario",
    "cohort",
    "checked_at",
)


def load_stand_areas(path):
    """Read ``fsID -> area_ha`` from a ``stand.details.csv``.

    Args:
        path (str): Path to the file.

    Returns:
        dict[str, float] | None: Areas keyed by stand id as a string (the form the
        file names parse to), or ``None`` if the file is missing or has no
        ``area_ha`` column -- in which case nothing can be judged and the caller
        must not exclude anything.
    """
    import pandas as pd

    if not os.path.isfile(path):
        return None

    stands = pd.read_csv(path)
    if "fsID" not in stands.columns or "area_ha" not in stands.columns:
        return None

    return {
        str(fsid): area
        for fsid, area in zip(stands["fsID"].astype(str), stands["area_ha"])
    }


def classify(stand, areas):
    """Decide whether one stand can be computed.

    Args:
        stand (str): Stand id as parsed from a ForClim file name.
        areas (dict[str, float]): From :func:`load_stand_areas`.

    Returns:
        tuple[str, float] | None: ``(reason, area)`` if the stand must be
        excluded, or ``None`` if it is usable. ``area`` is ``None`` when the
        stand is absent entirely.
    """
    if stand not in areas:
        return NOT_IN_STAND_DETAILS, None

    area = areas[stand]
    # NaN fails every comparison, so test it explicitly rather than relying on
    # `area <= 0` to catch it.
    if area is None or area != area:
        return NON_POSITIVE_AREA, area
    if area <= 0:
        return NON_POSITIVE_AREA, area
    return None


def partition(files, parse, areas, case_study, scenario, cohort):
    """Split ForClim files into the ones to process and the ones to report.

    Args:
        files (list[str]): File names that already matched the cohort convention.
        parse (callable): ``parse_forclim_filename``-like, returning ``(stand, simtype)``.
        areas (dict[str, float] | None): From :func:`load_stand_areas`. ``None``
            means we cannot judge, so every file is kept.
        case_study (str): Region, recorded in the report.
        scenario (str): Management scenario, recorded in the report.
        cohort (str): ``dead`` or ``alive``, recorded in the report.

    Returns:
        tuple[list[str], list[dict]]: ``(keep, report_rows)``. Report rows are one
        per excluded *stand*, not per file, with the file count alongside.
    """
    if areas is None:
        return list(files), []

    checked_at = dt.datetime.now().isoformat(timespec="seconds")
    keep = []
    excluded = {}

    for name in files:
        parsed = parse(name)
        if parsed is None:
            keep.append(name)
            continue
        stand = parsed[0]

        verdict = classify(stand, areas)
        if verdict is None:
            keep.append(name)
            continue

        reason, area = verdict
        row = excluded.setdefault(stand, {
            "stand": stand,
            "reason": reason,
            "area_ha": "" if area is None else area,
            "n_files": 0,
            "example_file": name,
            "case_study": case_study,
            "scenario": scenario,
            "cohort": cohort,
            "checked_at": checked_at,
        })
        row["n_files"] += 1

    rows = sorted(excluded.values(), key=lambda r: (r["reason"], _sortable(r["stand"])))
    return keep, rows


def _sortable(stand):
    """Sort stand ids numerically where they are numeric, lexically otherwise."""
    return (0, int(stand), "") if stand.isdigit() else (1, 0, stand)


def report_path(output_folder_path, case_study, scenario, cohort):
    """Where the report for one run goes.

    Deliberately the region root and **not** ``outputs/<scenario>/``: stage 2
    reads every file in that folder, so a report dropped there would be handed to
    the summary parser.
    """
    name = f"excluded_stands_{case_study}_{scenario}_{cohort}.csv"
    return os.path.join(output_folder_path, name)


def write_report(path, rows):
    """Write the exclusion report, or remove a stale one when nothing is excluded.

    A leftover report from a previous run would otherwise be read as current, and
    "nothing was excluded this time" is exactly the state worth being able to
    trust.

    Args:
        path (str): From :func:`report_path`.
        rows (list[dict]): From :func:`partition`.

    Returns:
        str | None: The path written, or ``None`` if there was nothing to report.
    """
    import csv

    if not rows:
        if os.path.isfile(path):
            os.remove(path)
        return None

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def summarise(rows, kept, total):
    """A short human-readable account for the job log.

    Returns:
        str: Multi-line text, empty when nothing was excluded.
    """
    if not rows:
        return ""

    by_reason = {}
    for row in rows:
        by_reason.setdefault(row["reason"], []).append(row["stand"])

    lines = [
        f"Excluded {len(rows)} stand(s) covering {total - kept} file(s); "
        f"{kept} of {total} files will be processed."
    ]
    for reason, stands in sorted(by_reason.items()):
        shown = ", ".join(sorted(stands, key=_sortable)[:10])
        more = "" if len(stands) <= 10 else f" (+{len(stands) - 10} more)"
        lines.append(f"  {reason}: {len(stands)} stand(s) -- {shown}{more}")
    return "\n".join(lines)
