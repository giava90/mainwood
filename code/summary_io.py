"""Reading and writing the ``summaries_for_plots`` deliverable.

The summaries are the product of the pipeline and are shared with collaborators
whose own pipelines are written in R. They were plain CSV until 2026-09; Parquet
is now the default because it is ~5x smaller, ~7x faster to write, and bit-exact
for floats (CSV loses the last ~1e-16 through decimal rounding).

Nothing is locked in:

* ``--format csv`` writes exactly what was written before;
* ``code/summary_to_csv.py`` converts any Parquet summary back to that same CSV;
* ``code/read_summaries.R`` reads either format straight into R.

:func:`read_summary` accepts a path with or without an extension and picks
whichever file is actually there, so downstream code does not care which format
a given region was written in.
"""

import os

import pandas as pd

#: Formats ``write_summary`` accepts.
SUMMARY_FORMATS = ("parquet", "csv", "csv.gz")

#: Extensions :func:`read_summary` probes, best first.
SUMMARY_EXTENSIONS = (".parquet", ".csv", ".csv.gz")

#: Default Parquet compression. snappy is understood by every Parquet reader,
#: including older R arrow builds; zstd is ~20% smaller but less universally
#: supported, and these files are handed to other people.
DEFAULT_COMPRESSION = "snappy"


def summary_path(base_path, fmt="parquet"):
    """Full path for a summary written in ``fmt``.

    Args:
        base_path (str): Path without extension, e.g. ``../data/summaries_for_plots/Vaud_BAU``.
        fmt (str): One of :data:`SUMMARY_FORMATS`.

    Returns:
        str: ``base_path`` with the matching extension.
    """
    if fmt not in SUMMARY_FORMATS:
        raise ValueError(f"Unknown summary format {fmt!r}. Use one of {list(SUMMARY_FORMATS)}.")
    return f"{base_path}.{fmt}"


def write_summary(summaries, base_path, fmt="parquet", compression=DEFAULT_COMPRESSION):
    """Writes the summary table in the requested format.

    The CSV branch reproduces the historical layout exactly, including the
    unnamed leading index column, so a file written today is interchangeable with
    one written before the format switch.

    Args:
        summaries (pandas.DataFrame): The finished summary table.
        base_path (str): Path without extension.
        fmt (str): "parquet" (default), "csv" or "csv.gz".
        compression (str): Parquet compression codec.

    Returns:
        str: The path written.
    """
    path = summary_path(base_path, fmt)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fmt == "parquet":
        # index=False: the old CSVs carried a meaningless RangeIndex as an unnamed
        # first column. summary_to_csv.py writes it back when converting, so the
        # CSV a collaborator receives still looks the way it always did.
        summaries.to_parquet(path, index=False, compression=compression)
    else:
        summaries.to_csv(path)
    return path


def find_summary(base_path):
    """Finds an existing summary file for ``base_path``, whatever its format.

    Args:
        base_path (str): Path with or without an extension.

    Returns:
        str or None: The path that exists, preferring Parquet.
    """
    for extension in SUMMARY_EXTENSIONS:
        if base_path.endswith(extension) and os.path.exists(base_path):
            return base_path
    stem = strip_summary_extension(base_path)
    for extension in SUMMARY_EXTENSIONS:
        candidate = stem + extension
        if os.path.exists(candidate):
            return candidate
    return None


def strip_summary_extension(path):
    """Removes a summary extension from ``path`` if it has one."""
    for extension in SUMMARY_EXTENSIONS:
        if path.endswith(extension):
            return path[: -len(extension)]
    return path


def read_summary(path, columns=None):
    """Reads a summary written in either format.

    Args:
        path (str): Path with or without an extension.
        columns (list, optional): Only these columns. With Parquet this actually
            skips the rest on disk, which is the cheap way to plot one variable
            out of a very large summary.

    Returns:
        pandas.DataFrame: The summary table.

    Raises:
        FileNotFoundError: If no summary exists for that path.
    """
    found = find_summary(path)
    if found is None:
        raise FileNotFoundError(
            f"No summary found for {path!r} (tried {', '.join(SUMMARY_EXTENSIONS)})"
        )
    if found.endswith(".parquet"):
        return pd.read_parquet(found, columns=columns)
    # index_col=0 drops the unnamed RangeIndex column the old CSVs carry
    df = pd.read_csv(found, index_col=0, low_memory=False)
    return df[columns] if columns else df
