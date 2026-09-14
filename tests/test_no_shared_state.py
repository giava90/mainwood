"""Stage 1 reports failures without shared state (code/convert_data.py).

A real run died of this. `process_files` kept the failed-file list in a
``multiprocessing.Manager`` list, and `run_sorsim` tested ``file in failed`` for
**every** file -- one remote call each, into a manager that spawns a thread per
connection. At 48 workers over 60,870 files the manager hit the thread limit::

    Exception in thread Thread-1 (accepter):
      File ".../multiprocessing/managers.py", line 194, in accepter
        t.start()
    RuntimeError: can't start new thread

After that it stopped accepting connections, every worker blocked forever on its
next call, and the job sat at 0% CPU for five hours while SLURM reported RUNNING.
MaxRSS was 50 GB of 122 GB, so it was never a memory problem.

Failures now come back as ordinary return values, so there is no shared object to
exhaust. These tests pin the contract that replaced it.
"""

import inspect

import pytest

import convert_data
import convert_data_from_intermediate


class SerialPool:
    """A drop-in for multiprocessing.Pool that runs in this process.

    The orchestration under test is the collecting and filtering of failures, not
    multiprocessing itself -- and a real Pool cannot pickle a test's local
    functions on a spawn platform. Running serially keeps these deterministic.
    """

    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


@pytest.fixture
def serial_pool(monkeypatch):
    monkeypatch.setattr(convert_data, "Pool", SerialPool)
    return SerialPool


def test_no_manager_anywhere_in_stage_one():
    """The import itself is the regression: reintroducing it brings back the hang."""
    for module in (convert_data, convert_data_from_intermediate):
        source = inspect.getsource(module)
        code = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "Manager()" not in code, module.__name__
        assert "manager.list()" not in code, module.__name__


def test_workers_take_no_shared_collection():
    """A `failed` parameter is how the shared list got in. Keep it out of the
    signatures so it cannot be reintroduced by habit."""
    for func in (convert_data.convert_forclim, convert_data.run_sorsim,
                 convert_data_from_intermediate.run_sorsim):
        assert "failed" not in inspect.signature(func).parameters, func.__qualname__


def test_convert_forclim_returns_the_name_it_could_not_parse():
    """Reporting by return value, not by mutating shared state."""
    result = convert_data.convert_forclim(
        "not-a-forclim-name.csv", "in/", "out/", "Misox", "BAU", cohort="alive")
    assert result == "not-a-forclim-name.csv"


def test_run_sorsim_returns_the_name_it_could_not_parse():
    result = convert_data.run_sorsim(
        "not-a-forclim-name.csv", "out/", "Misox", "BAU", cohort="alive")
    assert result == "not-a-forclim-name.csv"


def test_a_usable_name_is_not_reported_as_failed(monkeypatch, tmp_path):
    """convert_forclim returns None when the conversion command succeeds."""
    monkeypatch.setattr(convert_data, "run_command", lambda command: True)
    result = convert_data.convert_forclim(
        "dataSim_1000_scen7.csv", str(tmp_path) + "/", str(tmp_path) + "/",
        "Misox", "BAU", cohort="alive")
    assert result is None


def test_a_failing_command_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(convert_data, "run_command", lambda command: False)
    result = convert_data.convert_forclim(
        "dataSim_1000_scen7.csv", str(tmp_path) + "/", str(tmp_path) + "/",
        "Misox", "BAU", cohort="alive")
    assert result == "dataSim_1000_scen7.csv"


def test_phase_two_skips_what_phase_one_could_not_convert(serial_pool, monkeypatch, tmp_path):
    """The old code asked the shared list about every file. Now phase 1's failures
    are filtered out before phase 2 runs, so SorSim is never handed a tree list
    that was never written."""
    seen = {"convert": [], "sorsim": []}

    def fake_convert(file, *args, **kwargs):
        seen["convert"].append(file)
        return file if file.endswith("_bad.csv") else None

    def fake_sorsim(file, *args, **kwargs):
        seen["sorsim"].append(file)
        return None

    monkeypatch.setattr(convert_data, "convert_forclim", fake_convert)
    monkeypatch.setattr(convert_data, "run_sorsim", fake_sorsim)

    files = ["a_ok.csv", "b_bad.csv", "c_ok.csv"]
    failed = convert_data.process_files(
        files, str(tmp_path) + "/", str(tmp_path) + "/", "Misox", "BAU",
        num_cores=1, sample=False, save_intermediate=False, cohort="alive")

    assert sorted(seen["convert"]) == sorted(files)      # phase 1 sees all three
    assert seen["sorsim"] == ["a_ok.csv", "c_ok.csv"]    # phase 2 skips the failure
    assert failed == ["b_bad.csv"]


def test_failures_from_both_phases_are_returned(serial_pool, monkeypatch, tmp_path):
    monkeypatch.setattr(convert_data, "convert_forclim",
                        lambda file, *a, **k: file if "conv" in file else None)
    monkeypatch.setattr(convert_data, "run_sorsim",
                        lambda file, *a, **k: file if "sor" in file else None)

    files = ["conv_fail.csv", "sor_fail.csv", "fine.csv"]
    failed = convert_data.process_files(
        files, str(tmp_path) + "/", str(tmp_path) + "/", "Misox", "BAU",
        num_cores=1, sample=False, save_intermediate=False, cohort="alive")

    assert sorted(failed) == ["conv_fail.csv", "sor_fail.csv"]


def test_phase_two_reports_the_count_it_actually_processed(serial_pool, monkeypatch, tmp_path, capsys):
    """It printed len(files) while running len(for_sorsim) -- a small lie that
    would have made the phase timings wrong whenever anything failed."""
    monkeypatch.setattr(convert_data, "convert_forclim",
                        lambda file, *a, **k: file if "bad" in file else None)
    monkeypatch.setattr(convert_data, "run_sorsim", lambda file, *a, **k: None)

    convert_data.process_files(
        ["ok1.csv", "bad.csv", "ok2.csv"], str(tmp_path) + "/", str(tmp_path) + "/",
        "Misox", "BAU", num_cores=1, sample=False, save_intermediate=False,
        cohort="alive")

    out = capsys.readouterr().out
    assert "Phase 1 (ForClim -> tree lists)" in out and "for 3 files" in out
    assert "Phase 2 (SorSim)" in out and "for 2 files" in out
    assert "files=2" in out          # the machine-readable line agrees
