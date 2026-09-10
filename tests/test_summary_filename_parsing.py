"""The summariser must read back exactly what stage 1 wrote.

``code/naming.py`` builds SorSim output names; ``summarize_and_create_plots.
parse_filename`` takes them apart again. The two live in different scripts and
have drifted apart before, so the round trip is pinned here, together with the
awkward real-world name variants found in ``data/*/outputs/``.
"""

import pytest

from naming import sorsim_output_filename
from summarize_and_create_plots import file_cohort, parse_filename


@pytest.mark.parametrize("cohort", ["dead", "alive"])
@pytest.mark.parametrize(
    "stand, simtype",
    [("207", "1_planted_06"), ("1000", "7_planted_999"), ("59", "1")],
)
def test_stage1_names_round_trip_through_the_summariser(stand, simtype, cohort):
    """What convert_data.py writes, summarize_and_create_plots.py must parse."""
    filename = sorsim_output_filename(stand, simtype, cohort)
    parsed_stand, parsed_simtype, _species, _plantation, parsed_cohort = parse_filename(filename, "WOOD")
    assert parsed_stand == stand
    assert parsed_cohort == cohort
    # simtype in the summariser is only the leading RCP token; the planting
    # variant is carried separately in planted_species
    assert simtype.startswith(parsed_simtype)


def test_planted_variant():
    stand, simtype, species, plantation, cohort = parse_filename("sorsim_output240_7_planted_01.csv", "WOOD")
    assert (stand, simtype, species, plantation, cohort) == ("240", "7", "01", False, "dead")


def test_no_planting_variant_is_species_999():
    _stand, _simtype, species, plantation, _cohort = parse_filename("sorsim_output240_7_planted_999.csv", "WOOD")
    assert species == "999"
    assert plantation is False


def test_single_species_plantation_naming():
    """e.g. data/Entlebuch/outputs/WOOD/sorsim_output59_PMen_1.csv"""
    stand, simtype, species, plantation, cohort = parse_filename("sorsim_output59_PMen_1.csv", "WOOD")
    assert (stand, simtype, species, plantation, cohort) == ("59", "1", "PMen", True, "dead")


def test_bio_has_no_planting():
    stand, simtype, species, plantation, cohort = parse_filename("sorsim_output10_1.csv", "BIO")
    assert (stand, simtype, species, plantation, cohort) == ("10", "1", "999", False, "dead")


def test_alive_flag_does_not_disturb_the_planting_suffix():
    stand, simtype, species, plantation, cohort = parse_filename("sorsim_alive_output240_7_planted_01.csv", "WOOD")
    assert (stand, simtype, species, plantation, cohort) == ("240", "7", "01", False, "alive")


def test_file_cohort_filters_a_mixed_folder():
    files = [
        "sorsim_output240_7_planted_01.csv",
        "sorsim_alive_output240_7_planted_01.csv",
    ]
    assert [f for f in files if file_cohort(f, "WOOD") == "dead"] == [files[0]]
    assert [f for f in files if file_cohort(f, "WOOD") == "alive"] == [files[1]]


def test_file_cohort_survives_an_unreadable_name():
    """One stray file must not abort a run over thousands of files."""
    assert file_cohort("notes.csv", "WOOD") is None
