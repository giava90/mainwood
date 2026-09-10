"""Convert Parquet summaries back to CSV.

For collaborators whose pipeline reads CSV, and for anyone who wants the old
files back. The output is byte-compatible with the CSVs this pipeline wrote
before 2026-09: same columns, same order, same unnamed leading index column,
same ``True``/``False`` spelling.

The conversion streams row group by row group, so converting a 6 GB summary does
not need 6 GB of memory.

Usage
-----
    # one file
    python summary_to_csv.py ../data/summaries_for_plots/Surselva_WOOD.parquet

    # a whole folder, into the same folder
    python summary_to_csv.py ../data/summaries_for_plots/

    # somewhere else, gzip-compressed (pandas and R read .csv.gz directly)
    python summary_to_csv.py ../data/summaries_for_plots/ -o /tmp/csv --gzip

Requires pandas and pyarrow. If you only have R, see ``code/read_summaries.R``.
"""

import argparse
import glob
import os
import sys
import time


def convert_file(source, destination, chunk_rows=500_000, gzip_output=False, with_index=True):
    """Converts one Parquet summary to CSV, streaming through row groups.

    Args:
        source (str): Path to the .parquet file.
        destination (str): Path to the .csv (or .csv.gz) to write.
        chunk_rows (int): Rows held in memory at a time.
        gzip_output (bool): Write gzip-compressed CSV.
        with_index (bool): Write the leading unnamed index column, as the
            historical CSVs had. Use False for a cleaner file.

    Returns:
        tuple[int, float]: Rows written and seconds taken.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    started = time.perf_counter()
    parquet_file = pq.ParquetFile(source)
    total_rows = parquet_file.metadata.num_rows
    written = 0

    compression = "gzip" if gzip_output else None
    open_mode = "w"
    for batch in parquet_file.iter_batches(batch_size=chunk_rows):
        frame = batch.to_pandas()
        # continue the row numbering across chunks so the index column matches a
        # single-shot to_csv of the whole table
        frame.index = range(written, written + len(frame))
        frame.to_csv(
            destination,
            mode=open_mode,
            header=(open_mode == "w"),
            index=with_index,
            compression=compression,
        )
        written += len(frame)
        open_mode = "a"
        print(f"\r  {os.path.basename(source)}: {written:,}/{total_rows:,} rows", end="", flush=True)

    elapsed = time.perf_counter() - started
    size_mb = os.path.getsize(destination) / 1e6
    print(f"\r  {os.path.basename(source)}: {written:,} rows -> "
          f"{os.path.basename(destination)} ({size_mb:,.0f} MB) in {elapsed:,.1f} s")
    return written, elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert Parquet summaries to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1] if "Usage" in __doc__ else None,
    )
    parser.add_argument("source", help="A .parquet file, or a folder containing them")
    parser.add_argument("-o", "--out-dir", default=None,
                        help="Where to write the CSVs (default: next to the source)")
    parser.add_argument("--gzip", action="store_true",
                        help="Write .csv.gz instead of .csv (about 5x smaller; R reads it directly)")
    parser.add_argument("--no-index", action="store_true",
                        help="Omit the leading unnamed index column the old CSVs had")
    parser.add_argument("--chunk-rows", type=int, default=500_000,
                        help="Rows per chunk (default 500000)")
    args = parser.parse_args(argv)

    if os.path.isdir(args.source):
        sources = sorted(glob.glob(os.path.join(args.source, "*.parquet")))
        if not sources:
            print(f"No .parquet files in {args.source}")
            return 1
    else:
        sources = [args.source]

    out_dir = args.out_dir or (args.source if os.path.isdir(args.source)
                               else os.path.dirname(os.path.abspath(args.source)))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Converting {len(sources)} file(s) -> {out_dir}")
    total_rows, total_time = 0, 0.0
    for source in sources:
        stem = os.path.splitext(os.path.basename(source))[0]
        destination = os.path.join(out_dir, stem + (".csv.gz" if args.gzip else ".csv"))
        rows, elapsed = convert_file(source, destination,
                                     chunk_rows=args.chunk_rows,
                                     gzip_output=args.gzip,
                                     with_index=not args.no_index)
        total_rows += rows
        total_time += elapsed
    print(f"Done: {total_rows:,} rows in {total_time:,.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
