"""File-name rules of the pipeline (code/naming.py).

Every stand/simtype pair travels through three file names, and a mismatch
anywhere silently produces empty output folders rather than an error. These
tests pin the rules against the real names found in ``data/*/inputs`` and
``data/*/outputs``.
"""

import pytest

from naming import (
    intermediate_filename,
    parse_forclim_filename,
    parse_intermediate_filename,
    sorsim_output_filename,
    strip_data_suffix,
)


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("dataSim.dead207_1_planted_06.csv.gz", "dataSim.dead207_1_planted_06"),
        ("dataSim.dead207_1_planted_06.csv", "dataSim.dead207_1_planted_06"),
        ("dataSim.dead207_1_planted_06.zip", "dataSim.dead207_1_planted_06"),
        ("dataSim.dead207_1_planted_06", "dataSim.dead207_1_planted_06"),
    ],
)
def test_strip_data_suffix(filename, expected):
    assert strip_data_suffix(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "dataSim.dead207_1_planted_06.csv.gz",
        "dataSim.dead207_1_planted_06.csv",
        "dataSim.dead207_1_planted_06.zip",
    ],
)
def test_compression_does_not_change_the_parsed_simtype(filename):
    """Regression: uncompressed inputs used to yield ``1_planted_06.c``.

    Before 2025-09 the suffix was chopped by character count *after* matching,
    so only ``.csv.gz`` inputs parsed correctly. Plain ``.csv`` files produced a
    corrupted simtype, which is why ``data/Vaud/outputs/WOOD/`` came out empty
    while its input folder held a ``.csv`` file.
    """
    assert parse_forclim_filename(filename, "Vaud") == ("207", "1_planted_06")


def test_region_name_may_appear_in_the_forclim_filename():
    assert parse_forclim_filename("dataSim.deadEntlebuch59_1.csv.gz", "Entlebuch") == ("59", "1")


def test_bio_style_name_without_planting():
    """BIO runs have no planting variants; the general pattern covers them."""
    assert parse_forclim_filename("dataSim.dead100_1.csv.gz", "Entlebuch") == ("100", "1")


def test_alive_and_dead_are_parsed_from_their_own_token():
    assert parse_forclim_filename("dataSim.alive207_1.csv.gz", "Vaud", "alive") == ("207", "1")
    # an alive file must not be picked up by a dead-cohort run, and vice versa
    assert parse_forclim_filename("dataSim.alive207_1.csv.gz", "Vaud", "dead") is None
    assert parse_forclim_filename("dataSim.dead207_1.csv.gz", "Vaud", "alive") is None


def test_unrelated_files_are_rejected_rather_than_crashing():
    assert parse_forclim_filename("README.txt", "Vaud") is None
    assert parse_forclim_filename("assortments_summaries.csv", "Vaud") is None


def test_intermediate_name_is_cohort_specific():
    """Alive and dead runs of one stand must not overwrite each other."""
    assert intermediate_filename("207", "1_planted_06") == "deadCohorts207_1_planted_06.csv"
    assert intermediate_filename("207", "1_planted_06", "alive") == "aliveCohorts207_1_planted_06.csv"


def test_dead_sorsim_output_name_is_unchanged():
    """Earlier runs are addressable by exactly this name; do not change it."""
    assert sorsim_output_filename("207", "1_planted_06") == "sorsim_output207_1_planted_06.csv"


def test_alive_sorsim_output_name_carries_the_flag():
    assert sorsim_output_filename("207", "1_planted_06", "alive") == "sorsim_alive_output207_1_planted_06.csv"


@pytest.mark.parametrize("cohort", ["dead", "alive"])
def test_intermediate_name_round_trips(cohort):
    name = intermediate_filename("1234", "7_planted_999", cohort)
    assert parse_intermediate_filename(name) == (cohort, "1234", "7_planted_999")


def test_parse_intermediate_rejects_other_files():
    assert parse_intermediate_filename("sorsim_output207_1.csv") is None


@pytest.mark.parametrize(
    "value, expected",
    [
        (1432, "1432"),
        (1432.0, "1432"),          # the September 2026 Jurapark delivery, float64
        ("1432", "1432"),
        ("1432.0", "1432"),
        (" 1432 ", "1432"),
        (720900.0, "720900"),      # the large ids that were reported missing
    ],
)
def test_stand_key_canonicalises_both_sides_of_the_join(value, expected):
    """A file name parses to "1432"; fsID may be int or float depending on the
    delivery. str(1432.0) is "1432.0", which matches nothing -- so every stand
    reads as missing and stage 1 excludes the entire region."""
    from naming import stand_key
    assert stand_key(value) == expected


def test_stand_key_leaves_non_numeric_ids_alone():
    """A region using letters in its ids must keep working."""
    from naming import stand_key
    assert stand_key("A12") == "A12"
    assert stand_key("") == ""
