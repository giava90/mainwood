"""The stage 1 scaling grid (code/scaling_run.py, code/plot_scaling.py).

The grid runs every (cores x samples) point as its own SLURM job. Two properties
make the measurement meaningful rather than noise, and both are pinned here: each
point writes into its own output tree, and each writes its own result file.
Without the first, nine concurrent jobs overwrite each other's SorSim output and
time each other's I/O. Without the second, nine jobs finishing at once interleave
lines in a shared file.
"""

import csv
import os

import pytest

import plot_scaling
import scaling_run


def write_result(root, cores, samples, elapsed, exit_code=0, produced=None):
    """Write one result file the way scaling_run does."""
    produced = samples if produced is None else produced
    os.makedirs(scaling_run.results_dir(root), exist_ok=True)
    row = {
        "cores": cores, "samples": samples, "elapsed_s": elapsed,
        "files_produced": produced,
        "files_per_s": round(produced / elapsed, 4) if elapsed else 0,
        "case_study": "Jurapark", "scenario": "BAU", "cohort": "dead",
        "exit_code": exit_code, "host": "eu-a2p-1", "job_id": "1",
        "finished_at": "2026-09-11T17:00:00",
    }
    path = scaling_run.result_path(root, cores, samples)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scaling_run.RESULT_COLUMNS))
        writer.writeheader()
        writer.writerow(row)
    return path


def test_every_grid_point_gets_its_own_output_tree(tmp_path):
    """Nine jobs sharing outputs/<scenario>/ would measure the collision."""
    roots = {
        scaling_run.output_root(str(tmp_path), cores, samples)
        for cores in (5, 10, 20) for samples in (40, 80, 200)
    }
    assert len(roots) == 9


def test_every_grid_point_gets_its_own_result_file(tmp_path):
    """Nine jobs appending to one file would interleave lines."""
    paths_ = {
        scaling_run.result_path(str(tmp_path), cores, samples)
        for cores in (5, 10, 20) for samples in (40, 80, 200)
    }
    assert len(paths_) == 9


def test_results_are_read_back_with_numbers_not_strings(tmp_path):
    """exit_code stayed a string once, so `!= 0` was true for every row and the
    whole grid reported as failed."""
    write_result(str(tmp_path), 5, 40, 21.2)
    rows = plot_scaling.load_results(str(tmp_path))

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row["cores"], int)
    assert isinstance(row["samples"], int)
    assert isinstance(row["elapsed_s"], float)
    assert isinstance(row["exit_code"], int)
    assert row["exit_code"] == 0


def test_results_are_sorted_by_samples_then_cores(tmp_path):
    for cores in (20, 5, 10):
        for samples in (200, 40):
            write_result(str(tmp_path), cores, samples, 10.0)
    rows = plot_scaling.load_results(str(tmp_path))
    assert [(r["samples"], r["cores"]) for r in rows] == [
        (40, 5), (40, 10), (40, 20), (200, 5), (200, 10), (200, 20)
    ]


def test_a_malformed_row_is_skipped_not_fatal(tmp_path, capsys):
    """One bad file must not lose the other eight points."""
    write_result(str(tmp_path), 5, 40, 21.2)
    bad = scaling_run.result_path(str(tmp_path), 10, 40)
    with open(bad, "w", encoding="utf-8", newline="") as handle:
        handle.write("cores,samples,elapsed_s,files_produced,files_per_s,exit_code\n")
        handle.write("ten,40,x,40,1.0,0\n")

    rows = plot_scaling.load_results(str(tmp_path))
    assert len(rows) == 1
    assert "Skipping malformed row" in capsys.readouterr().out


def test_plot_writes_a_png(tmp_path):
    for cores in (5, 10, 20):
        for samples in (40, 80, 200):
            write_result(str(tmp_path), cores, samples, samples * 1.9 / cores + 6)
    rows = plot_scaling.load_results(str(tmp_path))

    out = str(tmp_path / "scaling.png")
    plot_scaling.plot(rows, out, " — Jurapark / BAU / dead")

    assert os.path.isfile(out)
    assert os.path.getsize(out) > 5000          # a real figure, not an empty canvas


def test_a_single_core_count_still_plots(tmp_path):
    """The ideal-linear reference needs two points; one must not crash."""
    write_result(str(tmp_path), 5, 40, 21.2)
    rows = plot_scaling.load_results(str(tmp_path))
    out = str(tmp_path / "one.png")
    plot_scaling.plot(rows, out)
    assert os.path.isfile(out)


def test_more_sample_sizes_than_hues_is_refused(tmp_path):
    """A 4th series must not become a generated hue."""
    for samples in (40, 80, 200, 400):
        write_result(str(tmp_path), 5, samples, 10.0)
    rows = plot_scaling.load_results(str(tmp_path))

    with pytest.raises(SystemExit, match="categorical slots"):
        plot_scaling.plot(rows, str(tmp_path / "too_many.png"))


def test_missing_results_say_what_to_do(tmp_path):
    assert plot_scaling.load_results(str(tmp_path)) == []


def test_count_outputs_reports_what_actually_arrived(tmp_path):
    """`samples` is what was asked for; this is what the run produced. They differ
    when the input folder is smaller, or a stand was excluded."""
    root = str(tmp_path)
    assert scaling_run.count_outputs(root, "Jurapark", "BAU") == 0

    folder = tmp_path / "Jurapark" / "outputs" / "BAU"
    folder.mkdir(parents=True)
    for n in range(3):
        (folder / f"sorsim_output{n}_1_planted_00.csv").write_text("x", encoding="utf-8")

    assert scaling_run.count_outputs(root, "Jurapark", "BAU") == 3
