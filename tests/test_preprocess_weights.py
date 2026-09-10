"""The planting weights in ``preprocess_data``.

ForClim simulates each stand once without planting and once per planted species.
The weights collapse those variants back into one expected volume per stand, and
they must sum to 1 over the variants of a given (stand, simtype) -- otherwise
every reported volume is off by a constant factor that no plot would reveal.

The split is 90% no-planting / 10% planting, spread over the planted species.
"""

import numpy as np
import pandas as pd
import pytest

from summarize_and_create_plots import preprocess_data


def make_raw_summary(planted_species, plantation=False, year=2050, volume=10.0, stand="1", simtype="1"):
    """Build the frame ``process_file`` hands to ``preprocess_data``.

    One row per planted-species variant, each carrying the same volume, so the
    weights alone decide the result.
    """
    n = len(planted_species)
    return pd.DataFrame(
        {
            "#Gruppierungsmerkmal": [f"#{year}"] * n,
            "Baumart": ["Fichte"] * n,
            "Volumen OR [m3]": [volume] * n,
            "Volumen IR [m3]": [volume] * n,
            "Wert [CHF]": [100.0] * n,
            "simtype": [simtype] * n,
            "stand": [stand] * n,
            "planted_species": list(planted_species),
            "planting": [s != "999" for s in planted_species],
            "plantation": [plantation] * n,
        }
    )


def total_volume(df):
    return df["Volumen OR [m3]"].sum()


def test_bio_has_no_planting_so_weights_are_one():
    raw = make_raw_summary(["999"])
    out = preprocess_data(raw, "BIO")
    assert total_volume(out) == pytest.approx(10.0)


def test_no_planting_variant_keeps_ninety_percent():
    """Two variants: no planting (0.9) and one planted species (0.1)."""
    raw = make_raw_summary(["999", "00"])
    out = preprocess_data(raw, "WOOD")
    volumes = dict(zip(out["planted_species"], out["Volumen OR [m3]"]))
    assert volumes["999"] == pytest.approx(9.0)
    assert volumes["00"] == pytest.approx(1.0)


def test_the_planting_share_is_split_over_the_planted_species():
    """0.9 + 2 x 0.05 = 1: the weights of one (stand, simtype) sum to one."""
    raw = make_raw_summary(["999", "00", "01"])
    out = preprocess_data(raw, "WOOD")
    volumes = dict(zip(out["planted_species"], out["Volumen OR [m3]"]))
    assert volumes["999"] == pytest.approx(9.0)
    assert volumes["00"] == pytest.approx(0.5)
    assert volumes["01"] == pytest.approx(0.5)
    assert total_volume(out) == pytest.approx(10.0)


@pytest.mark.parametrize("n_planted", [1, 2, 3, 5])
def test_weights_sum_to_one_for_any_number_of_planted_species(n_planted):
    species = ["999"] + [f"{i:02d}" for i in range(n_planted)]
    out = preprocess_data(make_raw_summary(species), "WOOD")
    assert total_volume(out) == pytest.approx(10.0)


def test_a_single_variant_gets_the_full_weight():
    """A stand simulated only once must not be scaled down to 0.9."""
    out = preprocess_data(make_raw_summary(["00"]), "WOOD")
    assert total_volume(out) == pytest.approx(10.0)


def test_plantation_stands_split_the_whole_volume_over_their_species():
    """Plantation stands have no no-planting variant: each species gets 1/n."""
    out = preprocess_data(make_raw_summary(["PMen", "AAlb"], plantation=True), "WOOD")
    assert total_volume(out) == pytest.approx(10.0)
    assert out["Volumen OR [m3]"].tolist() == pytest.approx([5.0, 5.0])


def test_weights_are_computed_per_stand_and_simtype():
    """Stands with different numbers of variants must not contaminate each other."""
    raw = pd.concat(
        [
            make_raw_summary(["999", "00"], stand="1"),
            make_raw_summary(["999", "00", "01"], stand="2"),
        ],
        ignore_index=True,
    )
    out = preprocess_data(raw, "WOOD")
    per_stand = out.groupby("stand")["Volumen OR [m3]"].sum()
    assert per_stand["1"] == pytest.approx(10.0)
    assert per_stand["2"] == pytest.approx(10.0)


def test_value_and_inner_volume_are_weighted_like_the_outer_volume():
    out = preprocess_data(make_raw_summary(["999", "00"]), "WOOD")
    assert out["Volumen IR [m3]"].sum() == pytest.approx(10.0)
    assert out["Wert [CHF]"].sum() == pytest.approx(100.0)


def test_years_before_2020_are_dropped():
    """The simulations start before the study period; only 2020 on is reported."""
    raw = pd.concat(
        [make_raw_summary(["999"], year=2015), make_raw_summary(["999"], year=2020)],
        ignore_index=True,
    )
    out = preprocess_data(raw, "BIO")
    assert out["year"].tolist() == [2020]
    assert out["year"].dtype.kind == "i"


def test_separator_rows_without_a_species_are_dropped():
    """SorSim writes bare '#' rows between species blocks."""
    raw = make_raw_summary(["999", "00"])
    raw.loc[len(raw)] = {**raw.iloc[0].to_dict(), "Baumart": np.nan}
    out = preprocess_data(raw, "WOOD")
    assert len(out) == 2
