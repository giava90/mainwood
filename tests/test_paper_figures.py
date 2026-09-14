"""The paper figures (code/make_paper_figures.py, code/plotting_tools_for_paper.py).

These are not the figures stage 2 draws. The paper's versions take fixed axes, a
year window and per-panel legends, and they are kept in their own module whose
function bodies are copied verbatim from the code that produced the published
figures. The point of the separation is that the difference between the two is
explicit rather than accidental.
"""

import os

import pandas as pd
import pytest

import make_paper_figures as mpf
import plotting_tools_for_paper as ptp


def test_the_module_carries_only_what_the_figures_need():
    """Carrying the EWP helpers too would mean maintaining a second copy of code
    nothing in this repository runs."""
    functions = {n for n in dir(ptp) if callable(getattr(ptp, n)) and not n.startswith("_")}
    assert {"plot_biomass_by_diameter_class", "plot_percentages_of_wood",
            "plot_change_in_species_comp", "prepare_data_for_sank_plot"} <= functions
    assert not any("ewp" in n.lower() for n in functions)


def make_summary(simtype="1", years=(2050, 2120), n_species=2):
    """A summary shaped like the deliverable, minimal but plottable."""
    rows = []
    for year in years:
        for species, soft in (("Fichte", True), ("Buche", False))[:n_species]:
            for diameter in ("<20cm", "20-40cm", ">40cm"):
                rows.append({
                    "year": year, "simtype": simtype, "stand": "1",
                    "Baumart": species, "diameter_class": diameter,
                    "is_soft": soft, "is_hard": not soft,
                    "Volumen OR [m3]": 100.0,
                    "Volumen OR [m3]_for_sawmills": 60.0,
                    "Volumen OR [m3]_not_for_sawmills": 40.0,
                })
    return pd.DataFrame(rows)


def test_simtype_is_matched_as_text(tmp_path):
    """The notebook compared it to an integer. The pipeline writes the column as a
    string, so an integer comparison matches nothing and every figure comes out
    empty -- silently."""
    make_summary(simtype="1").to_parquet(tmp_path / "Vaud_BAU.parquet")

    frame = mpf.load_summary(str(tmp_path), "Vaud", "BAU", "1")
    assert frame is not None and not frame.empty

    # and an integer argument must behave the same way, not fall through to empty
    frame = mpf.load_summary(str(tmp_path), "Vaud", "BAU", 1)
    assert frame is not None and not frame.empty


def test_a_simtype_that_is_absent_returns_nothing(tmp_path, capsys):
    make_summary(simtype="1").to_parquet(tmp_path / "Vaud_BAU.parquet")
    assert mpf.load_summary(str(tmp_path), "Vaud", "BAU", "7") is None
    assert "nothing left" in capsys.readouterr().out


def test_parquet_is_preferred_but_csv_still_works(tmp_path):
    make_summary().to_csv(tmp_path / "Vaud_BAU.csv", index=False)
    assert mpf.load_summary(str(tmp_path), "Vaud", "BAU", "1") is not None


def test_a_missing_summary_is_skipped_not_fatal(tmp_path, capsys):
    assert mpf.load_summary(str(tmp_path), "Nowhere", "BAU", "1") is None
    assert "no summary" in capsys.readouterr().out


def test_the_year_window_is_applied(tmp_path):
    """The published figures cover 2020-2160; a 2015 row must not sneak in."""
    make_summary(years=(2015, 2050, 2200)).to_parquet(tmp_path / "Vaud_BAU.parquet")
    frame = mpf.load_summary(str(tmp_path), "Vaud", "BAU", "1")
    assert sorted(frame["year"].unique()) == [2050]


def test_savefig_is_redirected_to_one_folder(tmp_path):
    """The plotting functions write to inconsistent relative paths; everything has
    to land in --outdir regardless of the working directory."""
    import matplotlib.pyplot as plt

    original = plt.savefig
    try:
        written = mpf.redirect_savefig(str(tmp_path / "out"))
        fig = plt.figure()
        plt.savefig("../figures/deep/name.png")
        plt.close(fig)

        assert len(written) == 1
        assert os.path.dirname(written[0]) == str(tmp_path / "out")
        assert os.path.basename(written[0]) == "name.png"
        assert os.path.isfile(written[0])
    finally:
        plt.savefig = original


def test_no_figures_is_reported_as_a_failure(tmp_path, capsys):
    """An empty run used to exit 0 having drawn nothing, which reads as success."""
    code = mpf.main(["--data", str(tmp_path), "--outdir", str(tmp_path / "out"),
                     "--case-study", "Nowhere"])
    assert code == 1
    assert "No figures written" in capsys.readouterr().out


# ---------------------------------------- discovery and the y-axis limit ----

def write(tmp_path, name):
    make_summary().to_parquet(tmp_path / name) if name.endswith(".parquet") \
        else make_summary().to_csv(tmp_path / name, index=False)


def test_regions_are_discovered_from_the_summaries(tmp_path):
    """A new region should need no code change."""
    for name in ("Vaud_BAU.parquet", "Vaud_WOOD.parquet", "Surselva_BIO.csv",
                 "Jurapark_BAU.parquet"):
        write(tmp_path, name)

    found = mpf.discover(str(tmp_path))
    assert found == {"Jurapark": ["BAU"], "Surselva": ["BIO"], "Vaud": ["BAU", "WOOD"]}


def test_alive_summaries_are_not_discovered(tmp_path):
    """The alive cohort is a single 2015 snapshot; these figures bin by decade,
    so there is nothing for them to draw."""
    write(tmp_path, "Misox_BAU.parquet")
    write(tmp_path, "Misox_BAU_alive.parquet")

    assert mpf.discover(str(tmp_path)) == {"Misox": ["BAU"]}


def test_unrelated_files_are_ignored(tmp_path):
    write(tmp_path, "Vaud_BAU.parquet")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "nounderscore.parquet").write_bytes(b"x")

    assert mpf.discover(str(tmp_path)) == {"Vaud": ["BAU"]}


def test_the_published_regions_keep_their_published_axis():
    """170,000 is an empirical constant for the two regions in the paper. Losing
    it would silently change figures that are already published."""
    for region in ("Vaud", "Entlebuch"):
        y_max, why = mpf.resolve_y_max(region, None)
        assert y_max == 170_000
        assert "published" in why


def test_any_other_region_is_auto_scaled():
    """Applying Vaud's limit to a region with a different harvest scale gives
    bars squashed into the floor or clipped off the top."""
    y_max, why = mpf.resolve_y_max("Surselva", None)
    assert y_max is None
    assert "no published value" in why


def test_the_axis_can_be_forced_either_way():
    assert mpf.resolve_y_max("Vaud", "auto") == (None, "requested with --y-max auto")
    y_max, why = mpf.resolve_y_max("Surselva", "250000")
    assert y_max == 250_000 and "--y-max" in why


def test_an_empty_data_folder_says_so(tmp_path):
    with pytest.raises(SystemExit, match="No summaries found"):
        mpf.main(["--data", str(tmp_path), "--outdir", str(tmp_path / "out")])
