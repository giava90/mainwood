"""Stands that cannot be computed (code/exclusions.py).

Two kinds, both of which used to travel through the whole pipeline: a stand
missing from ``stand.details.csv`` produced ``NaN`` volumes after one printed
warning, and a stand whose ``area_ha`` is zero produced a confident ``0`` that
looks exactly like a stand which genuinely harvested nothing. They are now
dropped at stage 1, before SorSim runs, and reported.
"""

import csv
import os

import pytest

import exclusions
from naming import parse_forclim_filename


AREAS = {"100": 1.5, "2501": 0.0, "3471": -2.0, "8000": float("nan")}


def parse(case_study="Entlebuch", cohort="dead"):
    return lambda name: parse_forclim_filename(name, case_study, cohort)


def test_a_usable_stand_is_not_excluded():
    assert exclusions.classify("100", AREAS) is None


@pytest.mark.parametrize("stand", ["2501", "3471", "8000"])
def test_zero_negative_and_nan_areas_are_all_excluded(stand):
    """NaN fails every comparison, so `area <= 0` alone would let it through."""
    verdict = exclusions.classify(stand, AREAS)
    assert verdict is not None
    assert verdict[0] == exclusions.NON_POSITIVE_AREA


def test_a_stand_absent_from_stand_details_is_excluded():
    reason, area = exclusions.classify("999", AREAS)
    assert reason == exclusions.NOT_IN_STAND_DETAILS
    assert area is None


def test_partition_keeps_usable_files_and_groups_the_rest_by_stand():
    files = [
        "dataSim.dead100_1_planted_00.csv",
        "dataSim.dead100_7_planted_02.csv",
        "dataSim.dead2501_1_planted_00.csv",
        "dataSim.dead2501_7_planted_02.csv",
        "dataSim.dead999_1_planted_00.csv",
    ]
    keep, rows = exclusions.partition(files, parse(), AREAS, "Entlebuch", "WOOD", "dead")

    assert sorted(keep) == [
        "dataSim.dead100_1_planted_00.csv",
        "dataSim.dead100_7_planted_02.csv",
    ]
    # one row per stand, not per file
    assert [r["stand"] for r in rows] == ["2501", "999"]
    assert {r["stand"]: r["n_files"] for r in rows} == {"2501": 2, "999": 1}


def test_nothing_is_excluded_when_stand_details_cannot_be_read():
    """Without areas nothing can be judged, so the run must not silently drop
    every stand -- that would look like a successful empty run."""
    files = ["dataSim.dead100_1_planted_00.csv", "dataSim.dead999_1_planted_00.csv"]
    keep, rows = exclusions.partition(files, parse(), None, "Entlebuch", "WOOD", "dead")
    assert keep == files
    assert rows == []


def test_load_stand_areas_returns_none_without_an_area_column(tmp_path):
    """The older Vaud delivery had no area_ha at all."""
    import pandas as pd

    path = tmp_path / "stand.details.csv"
    pd.DataFrame({"fsID": [1, 2], "area": [1.0, 2.0]}).to_csv(path, index=False)
    assert exclusions.load_stand_areas(str(path)) is None


def test_load_stand_areas_keys_are_strings(tmp_path):
    """File names parse to strings; fsID is usually an int in the CSV."""
    import pandas as pd

    path = tmp_path / "stand.details.csv"
    pd.DataFrame({"fsID": [100, 2501], "area_ha": [1.5, 0.0]}).to_csv(path, index=False)
    areas = exclusions.load_stand_areas(str(path))
    assert areas == {"100": 1.5, "2501": 0.0}


def test_report_is_written_with_a_stable_header(tmp_path):
    _, rows = exclusions.partition(
        ["dataSim.dead2501_1_planted_00.csv"], parse(), AREAS, "Entlebuch", "WOOD", "dead"
    )
    path = exclusions.write_report(str(tmp_path / "excluded.csv"), rows)

    with open(path, encoding="utf-8", newline="") as handle:
        read = list(csv.DictReader(handle))

    assert list(read[0]) == list(exclusions.REPORT_COLUMNS)
    assert read[0]["stand"] == "2501"
    assert read[0]["reason"] == exclusions.NON_POSITIVE_AREA
    assert read[0]["case_study"] == "Entlebuch"


def test_a_stale_report_is_removed_when_nothing_is_excluded(tmp_path):
    """"Nothing was excluded this time" is the state that must be trustworthy;
    a leftover file from a previous run would be read as current."""
    path = str(tmp_path / "excluded.csv")
    exclusions.write_report(path, [{"stand": "2501", "reason": "x", "area_ha": 0,
                                    "n_files": 1, "example_file": "f", "case_study": "E",
                                    "scenario": "WOOD", "cohort": "dead", "checked_at": "t"}])
    assert os.path.isfile(path)

    assert exclusions.write_report(path, []) is None
    assert not os.path.isfile(path)


def test_report_does_not_land_where_stage_two_reads(tmp_path):
    """Stage 2 reads every file in outputs/<scenario>/, so the report must not
    be written there or the summary parser is handed it."""
    path = exclusions.report_path(str(tmp_path), "Entlebuch", "WOOD", "dead")
    assert "outputs" not in os.path.relpath(path, str(tmp_path)).split(os.sep)
    assert os.path.basename(path) == "excluded_stands_Entlebuch_WOOD_dead.csv"


def test_cohorts_get_separate_reports(tmp_path):
    dead = exclusions.report_path(str(tmp_path), "Jurapark", "WOOD", "dead")
    alive = exclusions.report_path(str(tmp_path), "Jurapark", "WOOD", "alive")
    assert dead != alive


def test_summary_names_the_stands_and_the_counts():
    _, rows = exclusions.partition(
        ["dataSim.dead2501_1_planted_00.csv", "dataSim.dead999_1_planted_00.csv"],
        parse(), AREAS, "Entlebuch", "WOOD", "dead",
    )
    text = exclusions.summarise(rows, kept=4, total=6)
    assert "2 stand(s)" in text
    assert "2501" in text and "999" in text
    assert exclusions.NON_POSITIVE_AREA in text
    assert exclusions.NOT_IN_STAND_DETAILS in text


def test_summary_is_empty_when_nothing_was_excluded():
    assert exclusions.summarise([], kept=6, total=6) == ""


def test_a_float_fsid_column_still_joins(tmp_path):
    """The September 2026 Jurapark delivery arrived with fsID as float64. Keying
    on str(fsID) made every key "1432.0", so every stand parsed from a file name
    was "not in stand.details.csv" -- stage 1 would have excluded the whole region
    and produced an empty run that looked successful."""
    import pandas as pd

    path = tmp_path / "stand.details.csv"
    pd.DataFrame({"fsID": [1432.0, 7934.0, 720900.0],
                  "area_ha": [1.5, 2.0, 3.0]}).to_csv(path, index=False)

    areas = exclusions.load_stand_areas(str(path))
    assert set(areas) == {"1432", "7934", "720900"}
    for stand in ("1432", "7934", "720900"):
        assert exclusions.classify(stand, areas) is None


def test_a_float_fsid_delivery_excludes_nothing_it_should_not(tmp_path):
    """End to end through partition, which is what stage 1 calls."""
    import pandas as pd

    path = tmp_path / "stand.details.csv"
    pd.DataFrame({"fsID": [100.0, 2501.0], "area_ha": [1.5, 0.0]}).to_csv(path, index=False)
    areas = exclusions.load_stand_areas(str(path))

    files = ["dataSim.dead100_1_planted_00.csv", "dataSim.dead2501_1_planted_00.csv"]
    keep, rows = exclusions.partition(files, parse(), areas, "Jurapark", "BAU", "dead")

    assert keep == ["dataSim.dead100_1_planted_00.csv"]
    assert [r["stand"] for r in rows] == ["2501"]
    assert rows[0]["reason"] == exclusions.NON_POSITIVE_AREA
