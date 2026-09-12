"""One measured stage 1 run, for the scaling grid. Submitted by scaling_benchmark.sh.

Each grid point runs in its own output folder. Nine concurrent stage 1 jobs on the
same region and scenario would otherwise write into the same
``outputs/<scenario>/``, overwrite each other's files, and time each other's I/O --
which would measure the collision, not the code.

Each point also writes its own result file rather than appending to a shared one,
so nine jobs finishing at once cannot interleave a line.

Usage:
    python scaling_run.py <scenario> <case_study> <cohort> <cores> <samples> <bench_root>

Writes ``<bench_root>/results/scaling_<cores>c_<samples>s.csv``.
"""

import csv
import datetime as dt
import os
import socket
import subprocess
import sys
import time

import paths
from convert_data import PHASE_MARKER

#: Columns of a result row, in order.
RESULT_COLUMNS = (
    "cores",
    "samples",
    "elapsed_s",
    "convert_s",
    "sorsim_s",
    "files_produced",
    "files_per_s",
    "case_study",
    "scenario",
    "cohort",
    "exit_code",
    "host",
    "job_id",
    "finished_at",
)


def output_root(bench_root, cores, samples):
    """An output tree private to this grid point."""
    return os.path.join(bench_root, f"{cores}c_{samples}s")


def results_dir(bench_root):
    return os.path.join(bench_root, "results")


def result_path(bench_root, cores, samples):
    return os.path.join(results_dir(bench_root), f"scaling_{cores}c_{samples}s.csv")


def parse_phase_timing(stdout):
    """Pull the phase split out of stage 1's machine-readable line.

    The two phases do not scale alike -- phase 1 is pandas, phase 2 spawns a JVM
    per file and dominates at scale -- so a single elapsed figure is what made a
    200-file benchmark under-predict a 60,870-file run by hours.

    Returns:
        dict[str, float]: ``{"convert_s": ..., "sorsim_s": ...}``, empty if the
        line is absent (an older stage 1, or a run that died before printing it).
    """
    for line in reversed(stdout.splitlines()):
        if not line.startswith(PHASE_MARKER):
            continue
        fields = {}
        for token in line.split()[1:]:
            key, _, value = token.partition("=")
            if key in ("convert_s", "sorsim_s"):
                try:
                    fields[key] = float(value)
                except ValueError:
                    pass
        return fields
    return {}


def count_outputs(output_root_path, case_study, scenario):
    """How many SorSim files the run actually produced.

    The real measure of work done: ``samples`` is what was asked for, this is what
    arrived. They differ when the input folder holds fewer files, or when some
    stand was excluded.
    """
    folder = os.path.join(output_root_path, case_study, "outputs", scenario)
    if not os.path.isdir(folder):
        return 0
    return sum(1 for entry in os.scandir(folder) if entry.is_file())


def main(argv):
    scenario, case_study, cohort = argv[1], argv[2], argv[3]
    cores, samples = int(argv[4]), int(argv[5])
    bench_root = argv[6]

    root = output_root(bench_root, cores, samples)
    os.makedirs(root, exist_ok=True)
    os.makedirs(results_dir(bench_root), exist_ok=True)

    env = dict(os.environ)
    env["MAINWOOD_SAMPLE_SIZE"] = str(samples)
    # Private output tree for this point. The input template is inherited, so every
    # point reads the same ForClim files -- the first `samples` of them.
    env["MAINWOOD_OUTPUT_TEMPLATE"] = os.path.join(root, "{case_study}") + os.sep

    command = [
        sys.executable, "convert_data.py",
        scenario, "True", str(cores), "False", case_study, cohort,
    ]
    print(f"[{cores} cores / {samples} samples] {' '.join(command)}", flush=True)

    start = time.perf_counter()
    completed = subprocess.run(
        command, env=env, cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - start

    # Echo the child's output so the job log still reads as it did.
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", flush=True)

    phases = parse_phase_timing(completed.stdout or "")

    produced = count_outputs(root, case_study, scenario)
    row = {
        "cores": cores,
        "samples": samples,
        "elapsed_s": round(elapsed, 3),
        "convert_s": round(phases["convert_s"], 3) if "convert_s" in phases else "",
        "sorsim_s": round(phases["sorsim_s"], 3) if "sorsim_s" in phases else "",
        "files_produced": produced,
        "files_per_s": round(produced / elapsed, 4) if elapsed > 0 else 0,
        "case_study": case_study,
        "scenario": scenario,
        "cohort": cohort,
        "exit_code": completed.returncode,
        "host": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    path = result_path(bench_root, cores, samples)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_COLUMNS))
        writer.writeheader()
        writer.writerow(row)

    print(f"{elapsed:.1f} s, {produced} files, {row['files_per_s']} files/s -> {path}",
          flush=True)
    return completed.returncode


if __name__ == "__main__":
    if len(sys.argv) < 7:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv))
