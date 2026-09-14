"""Convert a folder of summary CSVs to Parquet, without loading them into memory.

The summaries exist as tens of GB of CSV -- 9.7 GB for a single Entlebuch_WOOD --
and reading them is what makes a figure run take minutes instead of seconds. This
is the counterpart to ``summary_to_csv.py``, which goes the other way for
collaborators whose pipeline reads CSV.

Usage:
    python summaries_to_parquet.py <folder>
    python summaries_to_parquet.py <folder> --out <folder>
    python summaries_to_parquet.py <folder> --organise      # also move the CSVs
    python summaries_to_parquet.py <folder> --only Misox_BIO

Layout produced::

    <out>/parquet/<Region>_<scenario>.parquet
    <out>/csv/<Region>_<scenario>.csv          (only with --organise)

``--organise`` moves the originals rather than copying them, so it is a rename
within the same filesystem and costs nothing. It is off by default: the readers
search ``parquet/``, then ``csv/``, then the folder itself, so conversion alone is
enough to make everything prefer Parquet.

Two things this has to get right:

* **Memory.** The files are far larger than RAM, so they are streamed in chunks
  and appended to the Parquet file row group by row group. Nothing holds more than
  one chunk.
* **Schema.** ``planted_species`` carries both ``999`` and codes like ``00`` and
  ``PMen``, which is the "Columns (10) have mixed types" warning pandas emits. Left
  to infer per chunk, one chunk would be integer and the next string, and the write
  would fail partway through a multi-GB file. Every text-like column is read as a
  string explicitly.
"""

import argparse
import os
import shutil
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

#: Columns that must be read as text. ``planted_species`` is the one that actually
#: breaks; the rest are declared so a chunk boundary can never reinterpret them.
TEXT_COLUMNS = (
    "Baumart", "Laengenklasse", "Staerkenklasse", "simtype", "stand",
    "planted_species", "diameter_class", "cohort",
)

#: Rows per chunk. About 300 MB of frame for these summaries.
CHUNK_ROWS = 2_000_000

#: snappy over zstd for the same reason summary_io uses it: these files are handed
#: to other people, and every Parquet reader supports snappy.
COMPRESSION = "snappy"


def dtypes_for(path):
    """Read the header and pin the text columns that are present."""
    header = pd.read_csv(path, nrows=0)
    return {c: "string" for c in TEXT_COLUMNS if c in header.columns}, list(header.columns)


def convert(csv_path, parquet_path, chunk_rows=CHUNK_ROWS, verbose=True):
    """Stream one CSV into one Parquet file.

    Returns:
        dict: Row count, byte sizes and elapsed seconds.
    """
    dtypes, columns = dtypes_for(csv_path)
    # The first column is the unnamed RangeIndex the old CSVs carried; dropping it
    # is what summary_io already does when it writes Parquet with index=False.
    index_col = 0 if columns and columns[0] in ("", "Unnamed: 0") else None

    start = time.perf_counter()
    writer = None
    rows = 0
    try:
        reader = pd.read_csv(csv_path, dtype=dtypes, index_col=index_col,
                             chunksize=chunk_rows)
        for number, chunk in enumerate(reader, start=1):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema,
                                          compression=COMPRESSION)
            else:
                # Later chunks must match the schema the writer opened with.
                table = table.cast(writer.schema)
            writer.write_table(table)
            rows += len(chunk)
            if verbose:
                print(f"    chunk {number}: {rows:,} rows", end="\r", flush=True)
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - start
    csv_bytes = os.path.getsize(csv_path)
    parquet_bytes = os.path.getsize(parquet_path)
    if verbose:
        print(f"    {rows:,} rows | {csv_bytes / 1e9:.2f} GB -> "
              f"{parquet_bytes / 1e9:.2f} GB "
              f"({csv_bytes / max(parquet_bytes, 1):.1f}x smaller) | {elapsed:,.0f} s")
    return {"rows": rows, "csv_bytes": csv_bytes, "parquet_bytes": parquet_bytes,
            "seconds": elapsed}


def verify(csv_path, parquet_path, column="Volumen OR [m3]"):
    """Check the Parquet matches the CSV on row count and a column total.

    Reads only the one column back, so this stays cheap on a multi-GB file.
    """
    table = pq.read_table(parquet_path, columns=[column])
    parquet_rows = table.num_rows
    parquet_sum = pa.compute.sum(table[column]).as_py()

    csv_rows, csv_sum = 0, 0.0
    for chunk in pd.read_csv(csv_path, usecols=[column], chunksize=CHUNK_ROWS):
        csv_rows += len(chunk)
        csv_sum += float(chunk[column].sum())

    rows_ok = csv_rows == parquet_rows
    # float accumulation order differs, so compare relatively
    sum_ok = abs(csv_sum - parquet_sum) <= max(abs(csv_sum), 1.0) * 1e-9
    return rows_ok and sum_ok, csv_rows, parquet_rows, csv_sum, parquet_sum


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", help="folder holding the summary CSVs")
    parser.add_argument("--out", default=None,
                        help="where parquet/ and csv/ go (default: alongside)")
    parser.add_argument("--organise", action="store_true",
                        help="move the CSVs into <out>/csv/ after converting")
    parser.add_argument("--only", action="append", default=[],
                        help="repeatable stem, e.g. Misox_BIO")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the row-count and column-total check")
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    args = parser.parse_args(argv)

    folder = os.path.abspath(args.folder)
    out = os.path.abspath(args.out or folder)
    parquet_dir = os.path.join(out, "parquet")
    csv_dir = os.path.join(out, "csv")
    os.makedirs(parquet_dir, exist_ok=True)

    names = sorted(
        entry for entry in os.listdir(folder)
        if entry.endswith(".csv")
        and (not args.only or os.path.splitext(entry)[0] in args.only)
    )
    if not names:
        raise SystemExit(f"No CSV summaries in {folder}"
                         + (f" matching {args.only}" if args.only else ""))

    print(f"from {folder}")
    print(f"to   {parquet_dir}")
    print(f"{len(names)} file(s)\n")

    totals = {"csv": 0, "parquet": 0, "rows": 0, "seconds": 0.0}
    failures = []
    for name in names:
        stem = os.path.splitext(name)[0]
        csv_path = os.path.join(folder, name)
        parquet_path = os.path.join(parquet_dir, stem + ".parquet")

        if os.path.isfile(parquet_path):
            print(f"  {stem}: already converted -- skipped")
            continue

        print(f"  {stem}:")
        try:
            result = convert(csv_path, parquet_path, args.chunk_rows)
        except Exception as exc:                      # noqa: BLE001 - report and continue
            print(f"    FAILED: {exc}")
            if os.path.isfile(parquet_path):
                os.remove(parquet_path)               # no half-written file left behind
            failures.append(stem)
            continue

        if not args.no_verify:
            ok, csv_rows, pq_rows, csv_sum, pq_sum = verify(csv_path, parquet_path)
            if ok:
                print(f"    verified: {pq_rows:,} rows, volume total matches")
            else:
                print(f"    VERIFY FAILED: rows {csv_rows:,} vs {pq_rows:,}, "
                      f"sum {csv_sum:,.3f} vs {pq_sum:,.3f}")
                failures.append(stem)
                continue

        for key, value in (("csv", result["csv_bytes"]), ("parquet", result["parquet_bytes"]),
                           ("rows", result["rows"]), ("seconds", result["seconds"])):
            totals[key] += value

        if args.organise:
            os.makedirs(csv_dir, exist_ok=True)
            shutil.move(csv_path, os.path.join(csv_dir, name))
            print(f"    moved the CSV to {csv_dir}")

    print(f"\n{totals['rows']:,} rows | {totals['csv'] / 1e9:.1f} GB -> "
          f"{totals['parquet'] / 1e9:.1f} GB "
          f"({totals['csv'] / max(totals['parquet'], 1):.1f}x smaller) | "
          f"{totals['seconds'] / 60:,.1f} min")
    if failures:
        print(f"\n{len(failures)} file(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
