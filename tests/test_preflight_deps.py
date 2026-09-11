"""The dependency gate in code/preflight.py.

Both imports checked here fail at the *end* of stage 2, after every file has been
read and summarised, so an absent package costs the whole run rather than failing
fast. An Euler checkout reported 130 passed / 1 skipped where a local run gave
148 -- the 18 Parquet tests module-skipping on a missing pyarrow. That is the
same absence that would abort a real stage 2 at its final write.
"""

import importlib.util
import os

import pytest

import preflight


@pytest.fixture
def without(monkeypatch):
    """Make find_spec report a named package as absent."""
    real = importlib.util.find_spec

    def make(missing):
        def fake(name, *args, **kwargs):
            return None if name == missing else real(name, *args, **kwargs)
        monkeypatch.setattr(importlib.util, "find_spec", fake)

    return make


def statuses(findings):
    return [status for status, _ in findings]


def test_parquet_without_pyarrow_blocks_the_run(without):
    without("pyarrow")
    findings = preflight.check_summary_dependencies("parquet")
    assert preflight.FAIL in statuses(findings)
    message = " ".join(m for _, m in findings)
    assert "pyarrow" in message
    assert "csv" in message          # the message offers the way out


def test_csv_format_does_not_need_pyarrow(without):
    """The documented workaround must actually pass the gate."""
    without("pyarrow")
    findings = preflight.check_summary_dependencies("csv")
    assert preflight.FAIL not in statuses(findings)


def test_missing_openpyxl_blocks_any_format(without):
    without("openpyxl")
    for fmt in ("parquet", "csv"):
        findings = preflight.check_summary_dependencies(fmt)
        assert preflight.FAIL in statuses(findings), fmt
        assert "openpyxl" in " ".join(m for _, m in findings)


def test_a_complete_environment_passes(monkeypatch):
    """Both packages importable -> nothing blocks.

    Simulated rather than observed. An earlier version of this test asserted that
    *this* machine had pyarrow, so it failed on Euler -- the one environment the
    whole check exists to describe. A test of the gate must not depend on which
    side of the gate the machine running it is on.
    """
    real = importlib.util.find_spec

    def everything_present(name, *args, **kwargs):
        if name in ("pyarrow", "openpyxl"):
            return object()
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", everything_present)

    findings = preflight.check_summary_dependencies("parquet")
    assert preflight.FAIL not in statuses(findings)


def test_missing_stands_warn_rather_than_block(tmp_path, monkeypatch):
    """Stage 1 excludes stands the join cannot find, so preflight must not refuse
    a run that would actually succeed. It reported FAIL for Jurapark/BAU, where
    17 of ~7939 stands are absent."""
    import pandas as pd

    import paths

    details = tmp_path / "stand.details.csv"
    pd.DataFrame({"fsID": [1, 2], "area_ha": [1.5, 2.0]}).to_csv(details, index=False)
    monkeypatch.setenv("MAINWOOD_STAND_DETAILS", str(details))

    folder = tmp_path / "in"
    folder.mkdir()
    for stand in ("1", "9999"):
        (folder / f"dataSim.dead{stand}_1_planted_00.csv.gz").write_text("x", encoding="utf-8")

    out = tmp_path / "out"
    out.mkdir()

    findings = preflight.check_stand_details(
        "Jurapark", str(folder), "dead", "BAU", str(out)
    )
    assert preflight.FAIL not in [s for s, _ in findings]
    assert preflight.WARN in [s for s, _ in findings]


def test_the_full_list_of_missing_stands_is_written(tmp_path, monkeypatch):
    """The printed message names five; the file is what goes to the ForClim side."""
    import csv

    import pandas as pd

    details = tmp_path / "stand.details.csv"
    pd.DataFrame({"fsID": [1], "area_ha": [1.5]}).to_csv(details, index=False)
    monkeypatch.setenv("MAINWOOD_STAND_DETAILS", str(details))

    folder = tmp_path / "in"
    folder.mkdir()
    absent = [str(n) for n in range(500, 520)]          # 20 > the 5 shown
    for stand in absent:
        (folder / f"dataSim.dead{stand}_1_planted_00.csv.gz").write_text("x", encoding="utf-8")

    out = tmp_path / "out"
    out.mkdir()
    preflight.check_stand_details("Jurapark", str(folder), "dead", "BAU", str(out))

    path = preflight.preview_report_path(str(out), "Jurapark", "BAU", "dead")
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert sorted(r["stand"] for r in rows) == sorted(absent)
    assert {r["reason"] for r in rows} == {"not in stand.details.csv"}


def test_preview_report_is_named_apart_from_the_run_report(tmp_path):
    """A preview of what would be excluded must not be mistaken for the record of
    a run that actually happened."""
    import exclusions

    preview = preflight.preview_report_path(str(tmp_path), "Jurapark", "BAU", "dead")
    actual = exclusions.report_path(str(tmp_path), "Jurapark", "BAU", "dead")
    assert preview != actual
    assert "preflight" in os.path.basename(preview)
