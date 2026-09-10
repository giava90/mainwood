"""Stand rescaling, diameter classes and the sawmill quality split.

These three steps turn per-patch SorSim volumes into the numbers that end up in
``summaries_for_plots``. They are pure lookups and arithmetic, and they are the
places where a new case study region is most likely to break: a missing
``area_ha`` column or an unknown ``Staerkenklasse`` raises a KeyError deep inside
an 8-hour job.
"""

import pandas as pd
import pytest

from summarize_and_create_plots import (
    add_sawmill_diameter_info,
    augment_with_stand_data,
    calculate_biomass_for_sawmills,
    create_quality_mapping,
    map_species_for_quality,
    split_by_soft_hard,
)

SOFT = ["Tanne", "Loerche", "Fichte", "Foehre", "Ubrige Nadelholz"]
HARD = ["Buche", "Eiche", "Esche", "Ahorn", "Ubrige Laubolz"]

#: The simulated area is always 100 patches of 625 m2.
SIM_AREA_M2 = 100 * 625


@pytest.fixture
def summaries():
    return pd.DataFrame(
        {
            "stand": ["1", "2"],
            "Baumart": ["Fichte", "Buche"],
            "Staerkenklasse": ["4", "1a"],
            "Volumen OR [m3]": [SIM_AREA_M2 / 10000.0, SIM_AREA_M2 / 10000.0],  # 1 m3/ha
            "Volumen IR [m3]": [SIM_AREA_M2 / 10000.0, SIM_AREA_M2 / 10000.0],
        }
    )


@pytest.fixture
def stand_data():
    return pd.DataFrame({"fsID": [1, 2], "area_ha": [3.0, 10.0], "Above1000m": [1, 0]})


def test_volume_is_rescaled_from_the_simulated_patches_to_the_real_stand_area(summaries, stand_data):
    """1 m3/ha simulated becomes area_ha m3 for the stand."""
    out = augment_with_stand_data(summaries, stand_data)
    assert out["Volumen OR [m3]"].tolist() == pytest.approx([3.0, 10.0])
    assert out["Volumen IR [m3]"].tolist() == pytest.approx([3.0, 10.0])
    assert out["sim_area (m2)"].unique().tolist() == [SIM_AREA_M2]


def test_altitude_flag_is_carried_over_when_the_region_has_one(summaries, stand_data):
    out = augment_with_stand_data(summaries, stand_data)
    assert out["Above1000m"].tolist() == [1, 0]


def test_regions_without_an_altitude_column_still_work(summaries, stand_data):
    """Surselva and Vaud have no Above1000m column in stand.details.csv."""
    out = augment_with_stand_data(summaries, stand_data.drop(columns=["Above1000m"]))
    assert "Above1000m" not in out.columns
    assert out["area"].tolist() == pytest.approx([3.0, 10.0])


def test_a_stand_missing_from_stand_details_yields_no_area(summaries, stand_data, capsys):
    """The run continues but the volume becomes NaN -- the printed warning is the
    only signal, so it is worth checking a new region's stand list up front."""
    out = augment_with_stand_data(summaries, stand_data[stand_data["fsID"] == 1])
    assert out["area"].isna().tolist() == [False, True]
    assert "Stand 2 not found" in capsys.readouterr().out


@pytest.mark.parametrize(
    "staerkenklasse, diameter_class, for_sawmills",
    [
        ("1a", "<20cm", False),
        ("1b", "<20cm", False),
        ("2a", "20-40cm", False),
        ("3b", "20-40cm", False),
        ("4", ">40cm", True),
        ("8", ">40cm", True),
        ("Restholz", "<20cm", False),
    ],
)
def test_diameter_classes(staerkenklasse, diameter_class, for_sawmills):
    df = pd.DataFrame({"Staerkenklasse": [staerkenklasse]})
    out = add_sawmill_diameter_info(df)
    assert out["diameter_class"].iloc[0] == diameter_class
    assert bool(out["is_for_sawmills_diameter"].iloc[0]) == for_sawmills


def test_an_unknown_diameter_class_fails_loudly():
    """SorSim writes 'Restholz 1' in the per-tree block; only the aggregated
    'Restholz' is expected here. A new value must not be silently mapped."""
    with pytest.raises(KeyError):
        add_sawmill_diameter_info(pd.DataFrame({"Staerkenklasse": ["Restholz 1"]}))


def test_soft_and_hard_wood_split():
    df = pd.DataFrame({"Baumart": ["Fichte", "Buche", "Something else"]})
    out = split_by_soft_hard(df, SOFT, HARD)
    assert out["is_soft"].tolist() == [True, False, False]
    assert out["is_hard"].tolist() == [False, True, False]


def test_quality_mapping_is_a_fraction():
    quality_df = pd.DataFrame({"Baumart": ["Fichte", "Buche"], "For Sawmills": [80.0, 40.0]})
    mapping = create_quality_mapping(quality_df)
    assert mapping == pytest.approx({"Fichte": 0.8, "Buche": 0.4})


def test_species_are_mapped_onto_the_species_used_by_the_quality_table():
    df = pd.DataFrame({"Baumart": ["Foehre", "Esche", "Fichte"]})
    out = map_species_for_quality(df, {"Foehre": "Kiefer", "Esche": "Eiche"})
    assert out["baumart_for_quality"].tolist() == ["Kiefer", "Eiche", "Fichte"]


def test_only_sawmill_diameters_contribute_high_quality_volume():
    mapping = {"Fichte": 0.8}
    big = {"is_for_sawmills_diameter": True, "baumart_for_quality": "Fichte", "Volumen OR [m3]": 10.0}
    small = {"is_for_sawmills_diameter": False, "baumart_for_quality": "Fichte", "Volumen OR [m3]": 10.0}
    assert calculate_biomass_for_sawmills(pd.Series(big), mapping) == pytest.approx(8.0)
    assert calculate_biomass_for_sawmills(pd.Series(small), mapping) == 0


def test_species_without_a_quality_entry_contribute_nothing_to_high_quality():
    row = pd.Series({"is_for_sawmills_diameter": True, "baumart_for_quality": "Unknown", "Volumen OR [m3]": 10.0})
    assert calculate_biomass_for_sawmills(row, {"Fichte": 0.8}) == 0


def test_quality_table_covers_every_species_the_pipeline_maps_to(repo_root):
    """The mapping in process_combination must land on species the Excel knows,
    or that volume quietly drops out of the 'high quality' series."""
    openpyxl = pytest.importorskip("openpyxl")
    from summarize_and_create_plots import load_quality_data

    quality_df = load_quality_data(f"{repo_root}/data/fraction_quality.xlsx")
    if quality_df.empty:
        pytest.skip("fraction_quality.xlsx not available")
    mapping = create_quality_mapping(quality_df)
    targets = {"Douglasie", "Fichte", "Kiefer", "Eiche", "Buche", "Birke"}
    assert targets <= set(mapping)
