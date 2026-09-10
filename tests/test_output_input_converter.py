"""ForClim cohort table -> SorSim tree list (minimal/functions/tools.py).

This is the step that decides *which trees* enter the assortment calculation, so
the dead/alive distinction and the tree-count expansion are worth pinning: an
error here changes every downstream volume without ever raising.
"""

import pandas as pd
import pytest

from functions.tools import map_species, output_input_converter

SORSIM_COLUMNS = ["#ID", "Baumart", "Baumart-Code", "Beschrieb", "Aufnahme-Datum", "BHD", "D7m", "SchaftLaenge_m"]


def test_dead_cohorts_keep_only_harvested_trees(forclim_dead_cohorts, minimal_dir):
    """``type == 2`` is the harvested cohort; trees that died standing are excluded."""
    out = output_input_converter(forclim_dead_cohorts.copy(), dead_cohorts=True, folder_path_sorsim=minimal_dir)
    # 3 + 1 harvested trees, the 5 trees of the type==1 cohort are dropped
    assert len(out) == 4
    assert set(out["Baumart"]) == {"Fichte"}


def test_dead_cohorts_can_keep_every_dead_tree(forclim_dead_cohorts, minimal_dir):
    out = output_input_converter(
        forclim_dead_cohorts.copy(), dead_cohorts=True, only_harvested=False, folder_path_sorsim=minimal_dir
    )
    assert len(out) == 3 + 1 + 5


def test_alive_cohorts_use_the_trees_column_and_no_type_filter(forclim_alive_cohorts, minimal_dir):
    """The alive cohort is the standing stock: every cohort counts."""
    out = output_input_converter(forclim_alive_cohorts.copy(), dead_cohorts=False, folder_path_sorsim=minimal_dir)
    assert len(out) == 4 + 2
    assert set(out["Baumart"]) == {"Fichte", "Tanne"}


def test_output_has_the_column_layout_sorsim_expects(forclim_dead_cohorts, minimal_dir):
    out = output_input_converter(forclim_dead_cohorts.copy(), dead_cohorts=True, folder_path_sorsim=minimal_dir)
    assert list(out.columns) == SORSIM_COLUMNS
    assert list(out["#ID"]) == list(range(len(out)))


def test_height_is_converted_from_centimetres_to_metres(forclim_dead_cohorts, minimal_dir):
    out = output_input_converter(forclim_dead_cohorts.copy(), dead_cohorts=True, folder_path_sorsim=minimal_dir)
    assert sorted(out["SchaftLaenge_m"].unique()) == [9.0, 25.0]


def test_species_codes_follow_the_sorsim_table(forclim_alive_cohorts, minimal_dir):
    out = output_input_converter(forclim_alive_cohorts.copy(), dead_cohorts=False, folder_path_sorsim=minimal_dir)
    codes = dict(zip(out["Baumart"], out["Baumart-Code"]))
    assert codes == {"Fichte": 100, "Tanne": 101}
    assert out["Baumart-Code"].dtype.kind == "i"


def test_species_outside_the_template_are_dropped(forclim_alive_cohorts, minimal_dir):
    """An unknown speciesid must not reach SorSim as a NaN code."""
    df = forclim_alive_cohorts.copy()
    df.loc[len(df)] = {"year": 2020, "run": 3, "speciesid": 99, "type": 1, "trees": 7, "diameter": 20.0, "height": 1500.0}
    out = output_input_converter(df, dead_cohorts=False, folder_path_sorsim=minimal_dir)
    assert len(out) == 4 + 2
    assert out["Baumart-Code"].notna().all()


@pytest.mark.parametrize(
    "latin, german",
    [
        ("Picea abies", "Fichte"),
        ("Abies alba", "Tanne"),
        ("Pinus sylvestris", "Foehre"),
        ("Larix decidua", "Laerche"),
        ("Fagus sylvatica", "Buche"),
        ("Quercus robur", "Eiche"),
        ("Fraxinus excelsior", "Esche"),
        ("Acer pseudoplatanus", "Ahorn"),
        ("Pseudotsuga menziesii", "Ubrige Nadelholzer"),
        ("Tilia cordata", "Ubrige Laubholzer"),
    ],
)
def test_species_mapping(latin, german):
    assert map_species([latin])[latin] == german


def test_every_species_in_the_template_is_mapped(minimal_dir):
    """A new species in templateSpec_v2.txt silently becomes 'Ubrige' (code -1)
    and is then dropped, so this guards against losing trees unnoticed."""
    template = pd.read_csv(minimal_dir + "templateSpec_v2.txt", sep="\t", index_col=0)
    mapping = map_species(template.index.tolist())
    unmapped = [k for k, v in mapping.items() if v == "Ubrige"]
    assert unmapped == []
