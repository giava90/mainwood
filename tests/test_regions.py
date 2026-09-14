"""The single list of regions and scenarios (code/regions.py).

Onboarding a region used to mean editing five copies of the same list, and they
had already drifted (the now-deleted ``plot_only`` dropped ``HYBRID``, the
others kept it).
These tests pin that there is now one list, that the wildcard tokens keep their
two different spellings, and that the one deliberate divergence stays possible.
"""

import pytest

import regions


def test_jurapark_is_registered():
    """The 2026-09 region. Spelled as it appears in the ForClim file names."""
    assert "Jurapark" in regions.CASE_STUDIES


def test_every_entry_point_sees_the_same_regions():
    """The whole point of the module: one list, no per-script copies."""
    import convert_data
    import convert_data_from_intermediate
    import preflight
    import summarize_and_create_plots

    for module in (convert_data, convert_data_from_intermediate,
                   summarize_and_create_plots, preflight):
        assert module.regions.CASE_STUDIES is regions.CASE_STUDIES


def test_wildcards_keep_their_different_spellings():
    """`All` for regions, `ALL` for scenarios -- what existing jobs already pass."""
    assert regions.ALL_CASE_STUDIES == "All"
    assert regions.ALL_SCENARIOS == "ALL"
    assert "All" in regions.valid_case_studies()
    assert "ALL" in regions.valid_scenarios()


def test_wildcard_expands_to_every_member_without_itself():
    assert regions.resolve_case_studies("All") == list(regions.CASE_STUDIES)
    assert regions.resolve_scenarios("ALL") == list(regions.SCENARIOS)
    assert "All" not in regions.resolve_case_studies("All")
    assert "ALL" not in regions.resolve_scenarios("ALL")


def test_a_single_value_resolves_to_itself():
    assert regions.resolve_case_studies("Jurapark") == ["Jurapark"]
    assert regions.resolve_scenarios("WOOD") == ["WOOD"]


def test_a_script_can_narrow_the_scenario_list():
    """A deliberate per-script narrowing stays possible, and stays visible.

    plot_only used this to drop HYBRID before it was deleted; the capability is
    kept because the next script to need it should declare it here rather than
    keeping its own diverging copy of the list."""
    assert "HYBRID" not in regions.valid_scenarios(exclude=("HYBRID",))
    assert "HYBRID" in regions.valid_scenarios()
    assert regions.resolve_scenarios("ALL", exclude=("HYBRID",)) == ["BAU", "WOOD", "BIO"]


@pytest.mark.parametrize("bad", ["jurapark", "JURAPARK", "Jura Park", "Entlebuch2", ""])
def test_unknown_region_is_rejected_with_the_options_listed(bad):
    with pytest.raises(ValueError, match="Invalid case study"):
        regions.check_case_study(bad)


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="Invalid management scenario"):
        regions.check_scenario("TIMBER")


def test_known_values_pass_validation_unchanged():
    assert regions.check_case_study("Jurapark") == "Jurapark"
    assert regions.check_scenario("BAU") == "BAU"
