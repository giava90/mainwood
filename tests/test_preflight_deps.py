"""The dependency gate in code/preflight.py.

Both imports checked here fail at the *end* of stage 2, after every file has been
read and summarised, so an absent package costs the whole run rather than failing
fast. An Euler checkout reported 130 passed / 1 skipped where a local run gave
148 -- the 18 Parquet tests module-skipping on a missing pyarrow. That is the
same absence that would abort a real stage 2 at its final write.
"""

import importlib.util

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
