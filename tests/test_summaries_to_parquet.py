"""CSV to Parquet conversion, and the one read order (code/summaries_to_parquet.py).

The summaries exist as tens of GB of CSV and reading them dominates every figure
run. Converting them is only safe if two things hold: the files are far larger
than RAM so nothing may load one whole, and `planted_species` carries both `999`
and codes like `00`, so a per-chunk dtype guess would disagree across a chunk
boundary and fail partway through a multi-GB write.
"""

import os

import pandas as pd
import pytest

import summaries_to_parquet as s2p
from summary_io import list_summaries, locate_summary, summary_search_path


def make_csv(path, rows=50, planted=("999", "00", "PMen")):
    """A summary CSV shaped like the real ones, index column included."""
    frame = pd.DataFrame({
        "Baumart": ["Fichte", "Buche"] * (rows // 2),
        "simtype": ["1"] * rows,
        "stand": [str(i) for i in range(rows)],
        "planted_species": [planted[i % len(planted)] for i in range(rows)],
        "plantation": [False] * rows,
        "year": [2050] * rows,
        "Volumen OR [m3]": [1.5] * rows,
    })
    frame.to_csv(path)          # with the unnamed index column
    return frame


def test_conversion_round_trips_rows_and_volume(tmp_path):
    csv = tmp_path / "Region_BAU.csv"
    original = make_csv(csv)
    parquet = tmp_path / "Region_BAU.parquet"

    result = s2p.convert(str(csv), str(parquet), verbose=False)
    assert result["rows"] == len(original)

    back = pd.read_parquet(parquet)
    assert len(back) == len(original)
    assert back["Volumen OR [m3]"].sum() == pytest.approx(original["Volumen OR [m3]"].sum())


def test_the_unnamed_index_column_is_dropped(tmp_path):
    """The old CSVs carry a meaningless RangeIndex as the first column; Parquet
    written by summary_io drops it, so the converter must too."""
    csv = tmp_path / "Region_BAU.csv"
    make_csv(csv)
    parquet = tmp_path / "Region_BAU.parquet"
    s2p.convert(str(csv), str(parquet), verbose=False)

    back = pd.read_parquet(parquet)
    assert not any(c == "" or c.startswith("Unnamed") for c in back.columns)


def test_planted_species_survives_a_chunk_boundary(tmp_path):
    """The real failure mode: one chunk all '999' (integer-looking), the next
    carrying 'PMen'. Inferring per chunk gives int then string and the write dies
    partway through a multi-GB file."""
    csv = tmp_path / "Region_WOOD.csv"
    rows = 40
    pd.DataFrame({
        "Baumart": ["Fichte"] * rows,
        "planted_species": ["999"] * (rows // 2) + ["PMen"] * (rows // 2),
        "year": [2050] * rows,
        "Volumen OR [m3]": [1.0] * rows,
    }).to_csv(csv)

    parquet = tmp_path / "Region_WOOD.parquet"
    s2p.convert(str(csv), str(parquet), chunk_rows=10, verbose=False)

    back = pd.read_parquet(parquet)
    assert len(back) == rows
    assert set(back["planted_species"].astype(str)) == {"999", "PMen"}


def test_conversion_streams_rather_than_loading_everything(tmp_path, monkeypatch):
    """Nothing may hold more than one chunk: the files are larger than RAM."""
    csv = tmp_path / "Region_BAU.csv"
    make_csv(csv, rows=100)

    seen = {}
    original = pd.read_csv

    def spy(*args, **kwargs):
        seen["chunksize"] = kwargs.get("chunksize")
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    s2p.convert(str(csv), str(tmp_path / "out.parquet"), chunk_rows=10, verbose=False)
    assert seen["chunksize"] == 10


def test_verify_catches_a_mismatch(tmp_path):
    csv = tmp_path / "Region_BAU.csv"
    make_csv(csv, rows=20)
    parquet = tmp_path / "Region_BAU.parquet"
    s2p.convert(str(csv), str(parquet), verbose=False)

    ok, *_ = s2p.verify(str(csv), str(parquet))
    assert ok

    # truncate the parquet to half the rows and the check must fail
    pd.read_parquet(parquet).head(5).to_parquet(parquet)
    ok, *_ = s2p.verify(str(csv), str(parquet))
    assert not ok


# ------------------------------------------------------- the read order ----

def test_parquet_is_preferred_over_csv_in_a_split_folder(tmp_path):
    (tmp_path / "parquet").mkdir()
    (tmp_path / "csv").mkdir()
    make_csv(tmp_path / "csv" / "Region_BAU.csv")
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "parquet" / "Region_BAU.parquet")

    assert locate_summary(str(tmp_path), "Region_BAU").endswith(".parquet")


def test_csv_is_the_fallback_per_file(tmp_path):
    """A half-converted folder must work: Parquet where it exists, CSV elsewhere."""
    (tmp_path / "parquet").mkdir()
    (tmp_path / "csv").mkdir()
    make_csv(tmp_path / "csv" / "Region_BAU.csv")
    make_csv(tmp_path / "csv" / "Region_WOOD.csv")
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "parquet" / "Region_BAU.parquet")

    assert locate_summary(str(tmp_path), "Region_BAU").endswith(".parquet")
    assert locate_summary(str(tmp_path), "Region_WOOD").endswith(".csv")


def test_an_unsplit_folder_still_works(tmp_path):
    """Folders that were never split must keep working unchanged."""
    make_csv(tmp_path / "Region_BAU.csv")
    assert locate_summary(str(tmp_path), "Region_BAU").endswith(".csv")
    assert summary_search_path(str(tmp_path)) == [str(tmp_path)]


def test_a_stem_present_in_both_is_listed_once(tmp_path):
    (tmp_path / "parquet").mkdir()
    (tmp_path / "csv").mkdir()
    make_csv(tmp_path / "csv" / "Region_BAU.csv")
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "parquet" / "Region_BAU.parquet")

    found = list_summaries(str(tmp_path))
    assert list(found) == ["Region_BAU"]
    assert found["Region_BAU"].endswith(".parquet")


def test_a_missing_stem_is_none(tmp_path):
    assert locate_summary(str(tmp_path), "Nowhere_BAU") is None
