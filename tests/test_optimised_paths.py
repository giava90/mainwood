"""The vectorised replacements must agree with the per-row versions they replaced.

`calculate_biomass_for_sawmills` is kept in the module precisely so the fast path
can be checked against it row by row.
"""

import numpy as np
import pandas as pd
import pytest

from summarize_and_create_plots import (
    CATEGORICAL_COLUMNS,
    calculate_biomass_for_sawmills,
    calculate_sawmill_split,
    compact_dtypes,
)

FRACTIONS = {"Douglasie": 0.75, "Fichte": 0.8, "Kiefer": 0.6, "Eiche": 0.4, "Buche": 0.45}


@pytest.fixture
def frame():
    rng = np.random.default_rng(7)
    n = 5_000
    return pd.DataFrame(
        {
            "baumart_for_quality": rng.choice(
                ["Fichte", "Eiche", "Buche", "Kiefer", "Unmapped species"], n
            ),
            "is_for_sawmills_diameter": rng.random(n) > 0.5,
            "Volumen OR [m3]": rng.random(n) * 10,
        }
    )


def test_vectorised_sawmill_split_matches_the_per_row_function(frame):
    """The fast path is only safe if it reproduces the old numbers exactly."""
    expected = frame.apply(lambda x: calculate_biomass_for_sawmills(x, FRACTIONS), axis=1)
    out = calculate_sawmill_split(frame.copy(), FRACTIONS)
    assert np.array_equal(out["Volumen OR [m3]_for_sawmills"].to_numpy(), expected.to_numpy())


def test_the_two_parts_add_back_up_to_the_total(frame):
    out = calculate_sawmill_split(frame.copy(), FRACTIONS)
    total = out["Volumen OR [m3]_for_sawmills"] + out["Volumen OR [m3]_not_for_sawmills"]
    assert np.allclose(total.to_numpy(), frame["Volumen OR [m3]"].to_numpy())


def test_species_missing_from_the_quality_table_contribute_nothing(frame):
    out = calculate_sawmill_split(frame.copy(), FRACTIONS)
    unmapped = out[out["baumart_for_quality"] == "Unmapped species"]
    assert (unmapped["Volumen OR [m3]_for_sawmills"] == 0).all()


def test_small_diameters_contribute_nothing(frame):
    out = calculate_sawmill_split(frame.copy(), FRACTIONS)
    small = out[~out["is_for_sawmills_diameter"]]
    assert (small["Volumen OR [m3]_for_sawmills"] == 0).all()
    assert np.allclose(small["Volumen OR [m3]_not_for_sawmills"], small["Volumen OR [m3]"])


def test_compact_dtypes_shrinks_the_frame_without_changing_values():
    df = pd.DataFrame(
        {
            "Baumart": ["Fichte", "Buche"] * 2_000,
            "stand": ["1", "2"] * 2_000,
            "simtype": ["1"] * 4_000,
            "cohort": ["dead"] * 4_000,
            "year": [2050] * 4_000,
            "Volumen OR [m3]": np.arange(4_000, dtype=float),
        }
    )
    before = df.memory_usage(deep=True).sum()
    out = compact_dtypes(df.copy())
    assert out.memory_usage(deep=True).sum() < before / 2
    assert out["Baumart"].tolist() == df["Baumart"].tolist()
    assert out["Volumen OR [m3]"].tolist() == df["Volumen OR [m3]"].tolist()
    assert out["year"].tolist() == df["year"].tolist()


def test_diameter_class_stays_a_plain_column():
    """The plots group by diameter_class; a categorical key would make pandas
    emit the full product of categories instead of the observed rows."""
    assert "diameter_class" not in CATEGORICAL_COLUMNS


def test_compact_dtypes_survives_missing_columns():
    """Older summaries have no cohort column."""
    df = pd.DataFrame({"Baumart": ["Fichte"], "Volumen OR [m3]": [1.0]})
    out = compact_dtypes(df)
    assert out["Baumart"].tolist() == ["Fichte"]


def test_compact_dtypes_on_an_empty_frame():
    assert compact_dtypes(pd.DataFrame()).empty


def test_year_fits_in_int16():
    """Simulations run to ~2300; int16 tops out at 32767."""
    df = pd.DataFrame({"year": [2020, 2300]})
    assert compact_dtypes(df)["year"].tolist() == [2020, 2300]
