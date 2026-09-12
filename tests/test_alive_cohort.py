"""The alive cohort, which arrives from a different pipeline than the dead one.

The alive delivery differs in four ways that each broke something:

  * the file name is ``dataSim_4810_scen7.csv`` -- underscores, no cohort token,
    simtype written as ``scen<n>``. The historical parser returned ``None``, so
    stage 1 matched zero files and produced an empty run;
  * it is a snapshot of the **first** simulated year, 2015. The 2020 floor in
    ``preprocess_data`` would have discarded every row;
  * it carries only simtype 7, so the figures' ``simtype == '1'`` filter left
    nothing to plot;
  * there is no planting and one simulation per stand, so the planting weights do
    not apply -- and the name, having no ``planted`` in it, was being read as the
    single-species plantation form.

Fixtures here mirror the real Misox delivery in data/Misox/inputs/BAU/alive.
"""

import numpy as np
import pandas as pd
import pytest

from naming import parse_forclim_filename, parse_scen_filename, sorsim_output_filename
from summarize_and_create_plots import (
    augment_with_stand_data,
    compact_dtypes,
    parse_filename,
    preprocess_data,
)

import paths


# --------------------------------------------------------------- names ----

@pytest.mark.parametrize(
    "filename, expected",
    [
        ("dataSim_4810_scen7.csv", ("4810", "7")),
        ("dataSim_1000_scen7.csv", ("1000", "7")),
        ("dataSim_1000_scen12.csv", ("1000", "12")),
        ("dataSim_1000_scen7.csv.gz", ("1000", "7")),
    ],
)
def test_the_alive_delivery_name_parses(filename, expected):
    assert parse_scen_filename(filename) == expected
    assert parse_forclim_filename(filename, "Misox", "alive") == expected


@pytest.mark.parametrize("filename", [
    "dataSim.dead207_1_planted_06.csv.gz",      # the historical dead form
    "notadatafile.csv",
    "dataSim_abc_scen7.csv",                    # stand must be digits
    "dataSim_1000_scenX.csv",                   # simtype must be digits
])
def test_names_that_are_not_the_alive_form_are_rejected(filename):
    assert parse_scen_filename(filename) is None


def test_the_historical_form_still_wins():
    """Both parsers are tried; the dot form must not be captured by the new one."""
    assert parse_forclim_filename("dataSim.alive207_1_planted_06.csv.gz", "Misox", "alive") == (
        "207", "1_planted_06"
    )


def test_the_output_name_is_the_alive_one():
    stand, simtype = parse_scen_filename("dataSim_4810_scen7.csv")
    assert sorsim_output_filename(stand, simtype, "alive") == "sorsim_alive_output4810_7.csv"


# --------------------------------------------------------------- paths ----

def test_case_study_lower_placeholder():
    """The delivery nests a lower-case region inside a capitalised one:
    .../raw/Misox/alive.data/misox/"""
    template = "/nfs/.../raw/{case_study}/alive.data/{case_study_lower}/"
    assert paths.expand(template, "Misox", "BAU", "alive") == (
        "/nfs/.../raw/Misox/alive.data/misox/"
    )


# ------------------------------------------------------- stage 2 names ----

def test_an_alive_output_is_not_read_as_a_plantation():
    """Without the alive branch, a name with no 'planted' in it falls through to
    the single-species plantation form: planted_species became the simtype and
    plantation became True for every alive file."""
    stand, simtype, planted, plantation, cohort = parse_filename(
        "sorsim_alive_output1000_7.csv", "BAU"
    )
    assert (stand, simtype, cohort) == ("1000", "7", "alive")
    assert planted == "999"
    assert plantation is False


def test_the_dead_forms_are_unchanged():
    assert parse_filename("sorsim_output240_7_planted_01.csv", "BAU") == (
        "240", "7", "01", False, "dead"
    )
    assert parse_filename("sorsim_output59_PMen_1.csv", "BAU") == (
        "59", "1", "PMen", True, "dead"
    )


# ------------------------------------------------- weights and the year ----

def make_alive_summary(stand="1000", year=2015, volume=10.0, n=2):
    """What process_file hands preprocess_data for the alive cohort."""
    return pd.DataFrame(
        {
            "#Gruppierungsmerkmal": [f"#{year}"] * n,
            "Baumart": ["Fichte"] * n,
            "Volumen OR [m3]": [volume] * n,
            "Volumen IR [m3]": [volume] * n,
            "Wert [CHF]": [100.0] * n,
            "simtype": ["7"] * n,
            "stand": [stand] * n,
            "planted_species": ["999"] * n,
            "planting": [False] * n,
            "plantation": [False] * n,
        }
    )


def test_the_2015_snapshot_survives_preprocessing():
    """The 2020 floor is a dead-cohort rule. Applied to the alive cohort it
    discards every row and writes an empty summary that reports success."""
    out = preprocess_data(make_alive_summary(year=2015), "BAU", cohort="alive")
    assert not out.empty
    assert sorted(out["year"].unique()) == [2015]


def test_the_dead_cohort_still_drops_the_spin_up():
    kept = preprocess_data(make_alive_summary(year=2015), "BAU", cohort="dead")
    assert kept.empty


def test_alive_rows_are_weighted_one():
    """One simulation per stand and no planting, so each row is its stand's only
    observation. The volume must come through unscaled."""
    out = preprocess_data(make_alive_summary(volume=10.0), "BAU", cohort="alive")
    assert out["Volumen OR [m3]"].tolist() == pytest.approx([10.0, 10.0])


def test_alive_weighting_does_not_depend_on_the_planting_arithmetic():
    """A plantation flag set by mistake must not rescale an alive row."""
    frame = make_alive_summary()
    frame["plantation"] = True
    out = preprocess_data(frame, "BAU", cohort="alive")
    assert out["Volumen OR [m3]"].tolist() == pytest.approx([10.0, 10.0])


# -------------------------------------------------- the join regression ----

def test_augment_works_on_a_compacted_frame():
    """compact_dtypes makes `stand` a Categorical. Mapping over a Categorical
    returns a Categorical, which cannot then be multiplied by the area -- so
    replacing astype(str) with a plain .map() broke stage 2 for BOTH cohorts
    with `TypeError: unsupported operand type(s) for *: 'Categorical' and 'int'`.
    No test covered augment on a compacted frame, so nothing caught it.
    """
    summaries = pd.DataFrame(
        {
            "stand": ["1000", "1001"],
            "simtype": ["7", "7"],
            "planted_species": ["999", "999"],
            "cohort": ["alive", "alive"],
            "year": [2015, 2015],
            "Volumen OR [m3]": [10.0, 20.0],
            "Volumen IR [m3]": [11.0, 21.0],
        }
    )
    summaries = compact_dtypes(summaries)
    assert str(summaries["stand"].dtype) == "category"     # the precondition

    stand_data = pd.DataFrame({"fsID": [1000, 1001], "area_ha": [1.0, 2.0]})
    out = augment_with_stand_data(summaries, stand_data)

    assert out["area"].tolist() == pytest.approx([1.0, 2.0])
    assert out["Volumen OR [m3]"].notna().all()
    # 10 m3 over 62500 m2 rescaled to 1 ha
    assert out["Volumen OR [m3]"].iloc[0] == pytest.approx(10.0 / 62500 * 1.0 * 10000)


def test_augment_joins_a_float_fsid_on_a_compacted_frame():
    """Both fixes at once: float fsID from the delivery, Categorical stand from
    compaction."""
    summaries = compact_dtypes(pd.DataFrame({
        "stand": ["1000"], "simtype": ["7"], "planted_species": ["999"],
        "cohort": ["alive"], "year": [2015],
        "Volumen OR [m3]": [10.0], "Volumen IR [m3]": [11.0],
    }))
    stand_data = pd.DataFrame({"fsID": [1000.0], "area_ha": [1.0]})
    out = augment_with_stand_data(summaries, stand_data)
    assert out["area"].tolist() == pytest.approx([1.0])


def test_an_alive_file_with_a_planting_suffix_still_parses_it():
    """The no-planting branch must be guarded: the earlier alive support handles
    sorsim_alive_output240_7_planted_01, and that form keeps its species."""
    assert parse_filename("sorsim_alive_output240_7_planted_01.csv", "WOOD") == (
        "240", "7", "01", False, "alive"
    )


# ------------------------------------------- per-cohort input templates ----

@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(paths.DEFAULTS) + [
        "MAINWOOD_INPUT_TEMPLATE_ALIVE", "MAINWOOD_INPUT_TEMPLATE_DEAD"
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(paths, "LOCAL_ENV_PATH", str(tmp_path / "absent.env"))
    return monkeypatch


DEAD_TEMPLATE = "/cluster/work/climate/amauri/{case_study}/Results/mgmt_{scenario}/{cohort}.trees/"
ALIVE_TEMPLATE = "/nfs/x/raw/{case_study}/alive.data/{case_study_lower}/"


def test_the_alive_override_wins_for_alive_only(clean_env):
    """The cohorts are not in the same tree, so one template with a {cohort}
    placeholder cannot reach both. Without the override, an alive preflight
    resolved to .../mgmt_BAU/alive.trees/ and failed as a missing folder."""
    clean_env.setenv("MAINWOOD_INPUT_TEMPLATE", DEAD_TEMPLATE)
    clean_env.setenv("MAINWOOD_INPUT_TEMPLATE_ALIVE", ALIVE_TEMPLATE)

    assert paths.input_folder("Entlebuch", "BAU", "dead") == (
        "/cluster/work/climate/amauri/Entlebuch/Results/mgmt_BAU/dead.trees/"
    )
    assert paths.input_folder("Entlebuch", "BAU", "alive") == (
        "/nfs/x/raw/Entlebuch/alive.data/entlebuch/"
    )


def test_without_an_override_both_cohorts_share_the_template(clean_env):
    """Unchanged behaviour for a machine that declares only the shared one."""
    clean_env.setenv("MAINWOOD_INPUT_TEMPLATE", DEAD_TEMPLATE)
    assert paths.input_folder("Entlebuch", "BAU", "alive") == (
        "/cluster/work/climate/amauri/Entlebuch/Results/mgmt_BAU/alive.trees/"
    )


def test_an_empty_override_is_ignored(clean_env):
    """An uncommented-but-blank line in local.env must not blank the template."""
    clean_env.setenv("MAINWOOD_INPUT_TEMPLATE", DEAD_TEMPLATE)
    clean_env.setenv("MAINWOOD_INPUT_TEMPLATE_ALIVE", "")
    assert "mgmt_BAU" in paths.input_folder("Entlebuch", "BAU", "alive")
