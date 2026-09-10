"""End-to-end read of a real SorSim output file.

``minimal/testdata/outputs/deadCohorts_sample.csv`` is a genuine SorSim result
(2300 lines, per-tree block followed by the aggregated ``#Gruppierungsmerkmal``
block). Reading it exercises the part of the summariser that no unit test can:
finding the aggregated block, re-heading the frame, and surviving the file's
Windows-1252 umlauts.
"""

import os
import shutil

import pandas as pd
import pytest

from summarize_and_create_plots import (
    add_sawmill_diameter_info,
    augment_with_stand_data,
    load_data_parallel,
    preprocess_data,
    process_file,
)

SAMPLE = os.path.join("minimal", "testdata", "outputs", "deadCohorts_sample.csv")


@pytest.fixture
def outputs_dir(tmp_path, repo_root):
    """A folder holding the sample file under a real pipeline name."""
    source = os.path.join(repo_root, SAMPLE)
    if not os.path.exists(source):
        pytest.skip("SorSim sample output not available")
    folder = tmp_path / "outputs" / "WOOD"
    folder.mkdir(parents=True)
    shutil.copy(source, folder / "sorsim_output42_1_planted_00.csv")
    return str(folder)


def test_process_file_finds_the_aggregated_block(outputs_dir):
    df = process_file((outputs_dir, "sorsim_output42_1_planted_00.csv", "WOOD"))
    assert df is not None and not df.empty
    assert "#Gruppierungsmerkmal" in df.columns
    assert "Baumart" in df.columns
    # metadata taken from the file name
    assert df["stand"].unique().tolist() == ["42"]
    assert df["simtype"].unique().tolist() == ["1"]
    assert df["planted_species"].unique().tolist() == ["00"]
    assert df["cohort"].unique().tolist() == ["dead"]


def test_process_file_returns_none_for_a_file_without_the_block(tmp_path):
    stray = tmp_path / "sorsim_output42_1_planted_00.csv"
    stray.write_text("#ID;Baumart\n1;Fichte\n", encoding="utf-8")
    assert process_file((str(tmp_path), stray.name, "WOOD")) is None


def test_cohort_filter_selects_the_right_files(outputs_dir):
    shutil.copy(
        os.path.join(outputs_dir, "sorsim_output42_1_planted_00.csv"),
        os.path.join(outputs_dir, "sorsim_alive_output42_1_planted_00.csv"),
    )
    dead = load_data_parallel(outputs_dir, "WOOD", cohort="dead")
    alive = load_data_parallel(outputs_dir, "WOOD", cohort="alive")
    assert dead["cohort"].unique().tolist() == ["dead"]
    assert alive["cohort"].unique().tolist() == ["alive"]
    assert len(dead) == len(alive)


def test_the_whole_summary_chain_runs_on_a_real_file(outputs_dir):
    """process_file -> preprocess_data -> augment_with_stand_data -> classes."""
    df = process_file((outputs_dir, "sorsim_output42_1_planted_00.csv", "WOOD"))
    summaries = preprocess_data(df, "WOOD")

    assert not summaries.empty
    assert summaries["year"].min() >= 2020
    for column in ("Volumen OR [m3]", "Volumen IR [m3]", "Wert [CHF]"):
        assert pd.api.types.is_float_dtype(summaries[column])
    # only one variant exists, so nothing is scaled down
    assert summaries["Baumart"].notna().all()

    stand_data = pd.DataFrame({"fsID": [42], "area_ha": [2.5], "Above1000m": [1]})
    summaries = augment_with_stand_data(summaries, stand_data)
    assert summaries["area"].unique().tolist() == [2.5]

    summaries = add_sawmill_diameter_info(summaries)
    assert set(summaries["diameter_class"]) <= {"<20cm", "20-40cm", ">40cm"}


def test_umlaut_columns_are_normalised(outputs_dir):
    """SorSim writes Laengenklasse/Staerkenklasse with Windows-1252 umlauts."""
    df = process_file((outputs_dir, "sorsim_output42_1_planted_00.csv", "WOOD"))
    summaries = preprocess_data(df, "WOOD")
    assert "Laengenklasse" in summaries.columns
    assert "Staerkenklasse" in summaries.columns
