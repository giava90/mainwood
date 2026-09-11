"""Create the folder skeleton a run writes into, wherever it is configured to write.

Replaces ``bash_code_to_create_folder_structure_for_data.sh``, which hardcoded the
four scenarios, had to be run from inside the region folder, and knew nothing about
where the run would actually write.

This reads the same templates the pipeline does, so a fresh
``/cluster/scratch/<user>/mainwood/`` is built to match whatever ``code/local.env``
says -- and re-running it after a scratch purge rebuilds exactly what was lost.

Usage:
    python setup_data_tree.py                        # every region, every scenario
    python setup_data_tree.py Jurapark               # one region
    python setup_data_tree.py Jurapark WOOD          # one region, one scenario
    python setup_data_tree.py --dry-run              # show, create nothing
    python setup_data_tree.py --inputs               # also make inputs/ (see below)

``inputs/`` is only needed when ForClim files are copied onto scratch rather than
read from the delivery folder in place. It is not created unless you ask, so an
empty ``inputs/`` never sits there looking like the run has nowhere to read from.

Stage 1 already creates ``intermediate/`` and ``outputs/`` on demand and stage 2
creates the summary folder, so nothing here is strictly required -- it exists so
you can see the layout, and check it is landing on scratch rather than on your
home quota rather than on scratch, before committing hours of walltime to it.
"""

import os
import sys

import paths
import regions


def planned_folders(case_studies, scenarios, want_inputs, local_env):
    """Every folder the configured run would write into.

    Returns:
        list[tuple[str, str]]: ``(purpose, path)`` pairs, in creation order.
    """
    plan = []
    for case_study in case_studies:
        root = paths.output_folder(case_study, scenarios[0], local_env=local_env)
        for scenario in scenarios:
            if want_inputs:
                plan.append(("inputs", os.path.join(root, "inputs", scenario)))
            plan.append(("intermediate", os.path.join(root, "intermediate", scenario)))
            plan.append(("outputs", os.path.join(root, "outputs", scenario)))
    plan.append(("summaries", paths.summary_dir(local_env)))
    return plan


def main(argv):
    flags = {a for a in argv[1:] if a.startswith("--")}
    positional = [a for a in argv[1:] if not a.startswith("--")]

    unknown = flags - {"--dry-run", "--inputs"}
    if unknown:
        raise SystemExit(f"Unknown option(s): {sorted(unknown)}\n\n{__doc__}")

    dry_run = "--dry-run" in flags
    want_inputs = "--inputs" in flags

    case_studies = (
        regions.resolve_case_studies(regions.check_case_study(positional[0]))
        if positional else list(regions.CASE_STUDIES)
    )
    scenarios = (
        regions.resolve_scenarios(regions.check_scenario(positional[1]))
        if len(positional) > 1 else list(regions.SCENARIOS)
    )

    local_env = paths.load_local_env()
    print("Configuration:")
    print(f"  local.env   {paths.LOCAL_ENV_PATH if os.path.isfile(paths.LOCAL_ENV_PATH) else '(none -- repository defaults)'}")
    print(f"  region root {paths.output_folder('<region>', '<scenario>', local_env=local_env)}")
    print(f"  summaries   {paths.summary_dir(local_env)}")
    print(f"  regions     {', '.join(case_studies)}")
    print(f"  scenarios   {', '.join(scenarios)}")
    print()

    plan = planned_folders(case_studies, scenarios, want_inputs, local_env)
    created = existed = 0
    for purpose, folder in plan:
        if os.path.isdir(folder):
            existed += 1
            continue
        if dry_run:
            print(f"  would create  [{purpose}] {folder}")
        else:
            os.makedirs(folder, exist_ok=True)
            print(f"  created       [{purpose}] {folder}")
        created += 1

    print()
    verb = "would create" if dry_run else "created"
    print(f"{verb} {created} folder(s); {existed} already existed.")

    if not want_inputs:
        print("inputs/ not created -- pass --inputs if you copy ForClim files onto scratch.")

    # Only worth flagging when the two disagree: assortments on scratch but the
    # summaries built from them landing somewhere else. Stage 2 hardcoded
    # ../data/summaries_for_plots/ while taking its input root as an argument, so
    # that combination used to be the default on Euler.
    summaries = paths.summary_dir(local_env)
    region_root = paths.output_folder(case_studies[0], scenarios[0], local_env=local_env)
    if "/scratch/" in region_root and "/scratch/" not in summaries:
        print()
        print("WARNING: assortments are on scratch but the summaries are not:")
        print(f"           assortments  {region_root}")
        print(f"           summaries    {summaries}")
        print("         The summaries are the deliverable. Extrapolating from the measured")
        print("         table in docs/09-summary-format.md, a Surselva-sized region runs to")
        print("         about 1.0 GB per scenario as Parquet and 5.5 GB as CSV -- so all")
        print("         four scenarios are roughly 4 GB, or 22 GB if you asked for CSV.")
        print("         Home quotas are far smaller than scratch; check yours with lquota.")
        print("         Set MAINWOOD_SUMMARY_DIR in code/local.env.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
