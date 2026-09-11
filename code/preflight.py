"""Check a run will work before handing it eight hours of walltime.

Every failure this script reports has actually happened on this pipeline, and
each one used to surface only after the job had been queued and started:

* the input folder is right but empty, or does not exist at all;
* the files are there but none match ``dataSim.<cohort><stand>_<simtype>``, so
  stage 1 skips all of them and writes an empty ``outputs/``;
* ``stand.details.csv`` is missing stands, so stage 2 produces ``NaN`` volumes
  and a plot of nothing (this one only ever printed a warning);
* the output tree does not exist yet.

Usage:
    python preflight.py <case_study> <scenario> [cohort] [summary_format]
    python preflight.py Entlebuch WOOD
    python preflight.py All ALL alive
    python preflight.py Jurapark WOOD dead csv

Exit status is 0 if the run can proceed, 1 if anything would fail. Use it as a
gate in a submit script: ``python preflight.py ... && ./run_conversion.sh ...``
"""

import os
import sys

import paths
import regions
from naming import COHORTS, parse_forclim_filename

OK, WARN, FAIL = "ok  ", "warn", "FAIL"


def check_input_folder(folder, case_study, cohort):
    """Confirm the folder exists and holds files this cohort's parser accepts.

    Returns:
        tuple[str, str]: ``(status, message)``.
    """
    if not os.path.isdir(folder):
        return FAIL, f"input folder does not exist: {folder}"

    names = [e.name for e in os.scandir(folder) if e.is_file()]
    if not names:
        return FAIL, f"input folder is empty: {folder}"

    matching = [n for n in names if parse_forclim_filename(n, case_study, cohort) is not None]
    if not matching:
        return FAIL, (
            f"{len(names)} files in {folder}, but none match the {cohort}-cohort "
            f"naming convention. First few: {', '.join(sorted(names)[:3])}. "
            f"If the cohort token differs, change COHORT_TOKEN in naming.py."
        )
    if len(matching) < len(names):
        return WARN, f"{len(matching)} of {len(names)} files match the {cohort} convention in {folder}"
    return OK, f"{len(matching)} input files in {folder}"


def check_stand_details(case_study, folder, cohort, sample_limit=2000):
    """Confirm ``stand.details.csv`` exists and covers the stands on disk.

    A stand present in the file names but absent from ``stand.details.csv`` is
    the silent ``NaN`` failure: stage 2 warns once and carries on. All findings
    are collected rather than returned at the first one, so a warning about
    zero-area stands never hides a genuine join failure underneath it.

    Returns:
        list[tuple[str, str]]: One ``(status, message)`` per finding.
    """
    import pandas as pd

    path = paths.stand_details_path(case_study)
    if not os.path.isfile(path):
        return [(FAIL, f"missing {path}")]

    stands = pd.read_csv(path)
    missing_cols = {"fsID", "area_ha"} - set(stands.columns)
    if missing_cols:
        return [(FAIL, (
            f"{path} lacks required column(s): {sorted(missing_cols)}. "
            f"It has {sorted(stands.columns)[:6]}... -- this looks like an older "
            f"delivery from the ForClim side; stage 2 cannot rescale patch volumes "
            f"to stand area without area_ha."
        ))]

    findings = []

    # A missing area_ha poisons every sum it touches; a zero one silently drops
    # that stand from the totals. Only the first is worth refusing to run over.
    n_missing = int(stands["area_ha"].isna().sum())
    if n_missing:
        findings.append((FAIL, f"{path}: {n_missing} rows with missing area_ha -- these give NaN volumes"))
    zero_area = stands.loc[stands["area_ha"] <= 0, "fsID"].tolist()
    if zero_area:
        preview = ", ".join(str(s) for s in zero_area[:5])
        findings.append((WARN, (
            f"{len(zero_area)} of {len(stands)} stands have area_ha <= 0 "
            f"({preview}) -- they contribute zero volume and vanish from the totals"
        )))

    if not os.path.isdir(folder):
        findings.append((WARN, f"{len(stands)} stands listed; input folder absent, join not checked"))
        return findings

    known = set(stands["fsID"].astype(str))
    seen, unknown = set(), set()
    for entry in os.scandir(folder):
        if not entry.is_file() or len(seen) >= sample_limit:
            continue
        parsed = parse_forclim_filename(entry.name, case_study, cohort)
        if parsed is None:
            continue
        stand = parsed[0]
        seen.add(stand)
        if stand not in known:
            unknown.add(stand)

    if unknown:
        findings.append((FAIL, (
            f"{len(unknown)} stand(s) in the input file names are absent from "
            f"stand.details.csv (e.g. {', '.join(sorted(unknown)[:5])}) -- these "
            f"would give NaN volumes"
        )))
    else:
        # The area total is the number every volume in the summary is scaled by, so
        # print it: it is the one value worth checking against what the ForClim side
        # said they sent, and a swapped-in file shows up here immediately.
        findings.append((OK, (
            f"{len(stands)} stands listed, {stands['area_ha'].sum():,.1f} ha total, "
            f"all {len(seen)} stands on disk join"
        )))
    return findings


def check_output_tree(output_folder_path, scenario):
    """Create the output tree if needed and report whether it already had files."""
    created = paths.ensure_output_tree(output_folder_path, scenario)
    outputs = created[-1]
    existing = sum(1 for _ in os.scandir(outputs)) if os.path.isdir(outputs) else 0
    if existing:
        return WARN, f"{outputs} already holds {existing} files -- they will be overwritten"
    return OK, f"output tree ready: {outputs}"


def check_summary_dependencies(summary_format="parquet"):
    """Confirm stage 2 can write its output and read the quality table.

    Both of these fail at the *end* of stage 2 -- after every file has been read
    and summarised -- so an absent import costs the whole run. The Euler module
    stack does not necessarily carry either: a checkout there skipped the 18
    Parquet tests because pyarrow was missing, which is the same absence that
    would abort a real run at the final write.

    Args:
        summary_format (str): The format stage 2 will write.

    Returns:
        list[tuple[str, str]]: One ``(status, message)`` per finding.
    """
    import importlib.util

    findings = []

    if summary_format == "parquet":
        if importlib.util.find_spec("pyarrow") is None:
            findings.append((FAIL, (
                "summary format is parquet but pyarrow is not installed -- stage 2 "
                "would fail at the final write, after all the work. Either "
                "`pip install --user pyarrow`, or pass a format: "
                "./run_analysis.sh <region> <scenario> dead False csv"
            )))
        else:
            findings.append((OK, f"pyarrow available (summary format {summary_format})"))
    else:
        findings.append((OK, f"summary format {summary_format} needs no extra package"))

    # stage 2 reads data/fraction_quality.xlsx through pandas -> openpyxl
    if importlib.util.find_spec("openpyxl") is None:
        findings.append((FAIL, (
            "openpyxl is not installed -- stage 2 cannot read fraction_quality.xlsx "
            "(the quality split). `pip install --user openpyxl`"
        )))
    else:
        findings.append((OK, "openpyxl available (fraction_quality.xlsx)"))

    return findings


def check_java():
    """Stage 1 needs a JVM; on Euler that means the openjdk module is loaded."""
    import shutil

    if shutil.which("java") is None:
        return FAIL, "java not on PATH -- module load stack/2024-06 openjdk/21.0.3_9"
    return OK, "java available"


def main(argv):
    case_study_input = argv[1]
    scenario_input = argv[2]
    cohort = argv[3] if len(argv) > 3 else "dead"
    summary_format = argv[4] if len(argv) > 4 else "parquet"

    if cohort not in COHORTS:
        raise SystemExit(f"Invalid cohort {cohort!r}; expected one of {list(COHORTS)}.")

    regions.check_case_study(case_study_input)
    regions.check_scenario(scenario_input)
    case_studies = regions.resolve_case_studies(case_study_input)
    scenarios = regions.resolve_scenarios(scenario_input)

    local_env = paths.load_local_env()
    print("Configuration:")
    print(f"  local.env  {paths.LOCAL_ENV_PATH if os.path.isfile(paths.LOCAL_ENV_PATH) else '(none -- repository defaults)'}")
    print(f"  data root  {paths.data_root(local_env)}")
    print()

    failures = 0
    java_status, java_message = check_java()
    print(f"[{java_status}] {java_message}")
    failures += java_status == FAIL

    for status, message in check_summary_dependencies(summary_format):
        print(f"[{status}] {message}")
        failures += status == FAIL

    for cs in case_studies:
        for ms in scenarios:
            print(f"\n--- {cs} / {ms} / {cohort} ---")
            folder = paths.input_folder(cs, ms, cohort, local_env)
            out = paths.output_folder(cs, ms, cohort, local_env)
            results = [check_input_folder(folder, cs, cohort)]
            results += check_stand_details(cs, folder, cohort)
            results.append(check_output_tree(out, ms))
            for status, message in results:
                print(f"[{status}] {message}")
                failures += status == FAIL

    print()
    if failures:
        print(f"{failures} blocking problem(s) -- do not submit.")
        return 1
    print("All checks passed. Safe to submit.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv))
