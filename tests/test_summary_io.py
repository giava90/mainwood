"""Reading, writing and converting the summaries deliverable.

The summaries go to collaborators whose pipelines read CSV, so the contract that
matters is: whatever format we store, ``summary_to_csv.py`` must hand them back
exactly the CSV the pipeline produced before the format switch.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from summary_io import (
    SUMMARY_FORMATS,
    find_summary,
    read_summary,
    strip_summary_extension,
    summary_path,
    write_summary,
)

pyarrow = pytest.importorskip("pyarrow", reason="Parquet support needs pyarrow")

CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")


@pytest.fixture
def summaries():
    """A frame shaped like a real summary, with the awkward bits included."""
    rng = np.random.default_rng(3)
    n = 2_000
    return pd.DataFrame(
        {
            "Baumart": pd.Categorical(rng.choice(["Fichte", "Buche", "Foehre"], n)),
            "Staerkenklasse": pd.Categorical(rng.choice(["1a", "4", "Restholz"], n)),
            "Volumen OR [m3]": rng.random(n) * 1e-3,
            "Wert [CHF]": rng.random(n) * 1e5,
            "Anzahl": rng.integers(1, 50, n),
            "simtype": pd.Categorical(rng.choice(["1", "7"], n)),
            "stand": pd.Categorical(rng.integers(1, 500, n).astype(str)),
            "cohort": pd.Categorical(["dead"] * n),
            "year": rng.integers(2020, 2300, n).astype("int16"),
            "diameter_class": rng.choice(["<20cm", "20-40cm", ">40cm"], n),
            "is_soft": rng.random(n) > 0.5,
            "plantation": rng.random(n) > 0.9,
        }
    )


def test_summary_path_extensions():
    assert summary_path("/x/Vaud_BAU") == "/x/Vaud_BAU.parquet"
    assert summary_path("/x/Vaud_BAU", "csv") == "/x/Vaud_BAU.csv"
    assert summary_path("/x/Vaud_BAU", "csv.gz") == "/x/Vaud_BAU.csv.gz"


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError):
        summary_path("/x/Vaud_BAU", "feather")


@pytest.mark.parametrize("fmt", SUMMARY_FORMATS)
def test_every_format_round_trips_the_values(summaries, tmp_path, fmt):
    base = str(tmp_path / "Region_BAU")
    written = write_summary(summaries, base, fmt=fmt)
    assert os.path.exists(written)
    back = read_summary(written)
    assert len(back) == len(summaries)
    assert np.allclose(back["Volumen OR [m3]"], summaries["Volumen OR [m3]"])
    assert back["Baumart"].astype(str).tolist() == summaries["Baumart"].astype(str).tolist()


def test_parquet_is_bit_exact_for_floats(summaries, tmp_path):
    """CSV rounds to ~16 significant digits; Parquet stores the bits."""
    parquet = write_summary(summaries, str(tmp_path / "p"), fmt="parquet")
    csv = write_summary(summaries, str(tmp_path / "c"), fmt="csv")
    from_parquet = read_summary(parquet)["Volumen OR [m3]"].to_numpy()
    from_csv = read_summary(csv)["Volumen OR [m3]"].to_numpy()
    original = summaries["Volumen OR [m3]"].to_numpy()
    assert np.array_equal(from_parquet, original)
    assert np.allclose(from_csv, original)  # close, but not bit-identical


def test_parquet_keeps_dtypes_that_csv_loses(summaries, tmp_path):
    back = read_summary(write_summary(summaries, str(tmp_path / "p"), fmt="parquet"))
    assert back["year"].dtype == "int16"
    assert str(back["Baumart"].dtype) == "category"
    assert back["is_soft"].dtype == bool


def test_csv_keeps_the_historical_layout(summaries, tmp_path):
    """The unnamed leading index column the old files had."""
    csv = write_summary(summaries, str(tmp_path / "c"), fmt="csv")
    raw = pd.read_csv(csv)
    assert raw.columns[0] == "Unnamed: 0"
    assert raw["Unnamed: 0"].tolist() == list(range(len(summaries)))
    assert set(raw["is_soft"].unique()) <= {True, False}


def test_read_summary_finds_the_file_without_an_extension(summaries, tmp_path):
    write_summary(summaries, str(tmp_path / "Region_BAU"), fmt="parquet")
    assert len(read_summary(str(tmp_path / "Region_BAU"))) == len(summaries)


def test_read_summary_prefers_parquet_when_both_exist(summaries, tmp_path):
    base = str(tmp_path / "Region_BAU")
    write_summary(summaries, base, fmt="csv")
    write_summary(summaries, base, fmt="parquet")
    assert find_summary(base).endswith(".parquet")


def test_read_summary_can_select_columns(summaries, tmp_path):
    parquet = write_summary(summaries, str(tmp_path / "p"), fmt="parquet")
    back = read_summary(parquet, columns=["year", "Volumen OR [m3]"])
    assert list(back.columns) == ["year", "Volumen OR [m3]"]


def test_missing_summary_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No summary found"):
        read_summary(str(tmp_path / "Nowhere_BAU"))


def test_strip_summary_extension():
    assert strip_summary_extension("/x/a.csv.gz") == "/x/a"
    assert strip_summary_extension("/x/a.parquet") == "/x/a"
    assert strip_summary_extension("/x/a") == "/x/a"


# --------------------------------------------------------------------------
# summary_to_csv.py: what the collaborators actually run
# --------------------------------------------------------------------------

def run_converter(*args):
    result = subprocess.run(
        [sys.executable, os.path.join(CODE_DIR, "summary_to_csv.py"), *args],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_converted_csv_is_byte_identical_to_the_old_output(summaries, tmp_path):
    """The whole point: an existing R pipeline needs no changes."""
    reference = tmp_path / "reference.csv"
    summaries.to_csv(reference)

    parquet = write_summary(summaries, str(tmp_path / "Region_BAU"), fmt="parquet")
    out_dir = tmp_path / "out"
    run_converter(parquet, "-o", str(out_dir))

    produced = out_dir / "Region_BAU.csv"
    assert produced.read_bytes() == reference.read_bytes()


def test_converter_index_is_continuous_across_chunks(summaries, tmp_path):
    """A chunked write must not restart the row numbering at each chunk."""
    parquet = write_summary(summaries, str(tmp_path / "Region_BAU"), fmt="parquet")
    out_dir = tmp_path / "out"
    run_converter(parquet, "-o", str(out_dir), "--chunk-rows", "100")
    raw = pd.read_csv(out_dir / "Region_BAU.csv")
    assert raw["Unnamed: 0"].tolist() == list(range(len(summaries)))


def test_converter_gzip_output(summaries, tmp_path):
    parquet = write_summary(summaries, str(tmp_path / "Region_BAU"), fmt="parquet")
    out_dir = tmp_path / "out"
    run_converter(parquet, "-o", str(out_dir), "--gzip")
    produced = out_dir / "Region_BAU.csv.gz"
    assert produced.exists()
    assert len(pd.read_csv(produced)) == len(summaries)


def test_converter_can_drop_the_index_column(summaries, tmp_path):
    parquet = write_summary(summaries, str(tmp_path / "Region_BAU"), fmt="parquet")
    out_dir = tmp_path / "out"
    run_converter(parquet, "-o", str(out_dir), "--no-index")
    raw = pd.read_csv(out_dir / "Region_BAU.csv")
    assert raw.columns[0] == "Baumart"


def test_converter_handles_a_whole_folder(summaries, tmp_path):
    for name in ("Vaud_BAU", "Vaud_WOOD"):
        write_summary(summaries, str(tmp_path / name), fmt="parquet")
    out_dir = tmp_path / "out"
    run_converter(str(tmp_path), "-o", str(out_dir))
    assert sorted(p.name for p in out_dir.glob("*.csv")) == ["Vaud_BAU.csv", "Vaud_WOOD.csv"]


def test_parquet_falls_back_to_csv_when_pyarrow_is_missing(tmp_path, monkeypatch, capsys):
    """Stage 2 imports pyarrow only at the final write, and it is absent from the
    Euler module stack -- so this is the difference between keeping hours of
    finished work and losing all of it. The fallback is loud, and find_summary
    picks the file up either way."""
    import pandas as pd

    import summary_io

    frame = pd.DataFrame({"stand": ["42"], "volume": [1.5]})
    base = str(tmp_path / "Jurapark_WOOD")

    def no_pyarrow(*args, **kwargs):
        raise ImportError("No module named 'pyarrow'")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_pyarrow)

    written = summary_io.write_summary(frame, base, fmt="parquet")

    assert written.endswith(".csv")
    assert os.path.isfile(written)
    assert summary_io.find_summary(base) == written

    out = capsys.readouterr().out
    assert "pyarrow is not installed" in out
    assert "has been written as CSV instead" in out

    back = pd.read_csv(written, index_col=0)
    assert back["stand"].astype(str).tolist() == ["42"]
    assert back["volume"].tolist() == [1.5]
