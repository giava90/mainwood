"""Single source of truth for the file names used along the ForClim -> SorSim pipeline.

Every stage of the pipeline agrees on three names, all derived from the same
``(stand, simtype)`` pair that is parsed out of the ForClim file name:

    ForClim output      dataSim.<cohort><stand>_<simtype>[.csv|.csv.gz]
    SorSim input        <cohort>Cohorts<stand>_<simtype>.csv        (intermediate/)
    SorSim output       sorsim_output<stand>_<simtype>.csv          (outputs/, dead)
                        sorsim_alive_output<stand>_<simtype>.csv    (outputs/, alive)

Keeping the three builders here means ``convert_forclim`` (which writes the
intermediate file) and ``run_sorsim`` (which reads it back) can never disagree,
and adding the alive cohort did not require touching either of them.

The dead-cohort names are exactly the ones used by every run up to 2025, so the
existing ``data/<region>/outputs/<scenario>/`` archives stay readable.
"""

import re

#: Cohorts we can process. ``dead`` = harvested/dying trees (``dtrees`` column in
#: the ForClim output), ``alive`` = standing stock (``trees`` column).
COHORTS = ("dead", "alive")

#: The token ForClim puts in the file name for each cohort: ``dataSim.dead207_1``.
COHORT_TOKEN = {"dead": "dead", "alive": "alive"}

#: Suffixes a ForClim output can carry. Longest first, so ``.csv.gz`` wins over ``.gz``.
DATA_SUFFIXES = (".csv.gz", ".csv", ".gz", ".zip")


def strip_data_suffix(filename):
    """Remove a trailing data suffix (``.csv``, ``.csv.gz``, ``.gz``, ``.zip``).

    Args:
        filename (str): A file name, with or without a suffix.

    Returns:
        str: The file name without its data suffix.
    """
    for suffix in DATA_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def forclim_pattern(case_study, cohort="dead"):
    """Build the regex that pulls ``stand`` and ``simtype`` out of a ForClim name.

    ForClim writes either ``dataSim.dead207_1_planted_06.csv.gz`` or, for some
    regions, the region name in between: ``dataSim.deadEntlebuch59_1.csv.gz``.
    Both are accepted, as is a leading underscore before the stand id.

    Args:
        case_study (str): Region name, optionally present in the file name.
        cohort (str): One of :data:`COHORTS`.

    Returns:
        re.Pattern: Compiled pattern with groups ``(stand, _simtype)``.
    """
    token = COHORT_TOKEN[cohort]
    return re.compile(
        r"dataSim\." + token + r"(?:" + re.escape(case_study) + r")?(\d+|_?\d+)(_.*)"
    )


#: The alive cohort is delivered from a different pipeline with a different name:
#: ``dataSim_4810_scen7.csv`` -- underscores rather than a dot, no cohort token,
#: and the simtype written as ``scen<n>``. It carries no cohort marker at all, so
#: the caller's ``cohort`` argument is what decides; the two folders are separate
#: (``alive.data/``) and stage 1 is invoked per cohort, so nothing is ambiguous.
SCEN_PATTERN = re.compile(r"^dataSim_(\d+)_scen(\d+)$")


def parse_scen_filename(filename):
    """Extract ``(stand, simtype)`` from the ``dataSim_<stand>_scen<n>`` form.

    Args:
        filename (str): e.g. ``dataSim_4810_scen7.csv``.

    Returns:
        tuple[str, str] | None: ``("4810", "7")``, or ``None`` if it is not this form.
    """
    match = SCEN_PATTERN.match(strip_data_suffix(filename))
    if match is None:
        return None
    return match.group(1), match.group(2)


def parse_forclim_filename(filename, case_study, cohort="dead"):
    """Extract ``(stand, simtype)`` from a ForClim output file name.

    The suffix is stripped *before* matching, so ``.csv`` and ``.csv.gz`` inputs
    give the same answer. (Until 2025-09 the suffix was chopped off by character
    count after matching, which silently corrupted ``simtype`` for uncompressed
    inputs -- see docs/06-known-issues.md.)

    Args:
        filename (str): e.g. ``dataSim.dead207_1_planted_06.csv.gz``.
        case_study (str): Region name, e.g. ``Vaud``.
        cohort (str): One of :data:`COHORTS`.

    Returns:
        tuple[str, str] | None: ``(stand, simtype)``, or ``None`` if the name does
        not belong to this cohort/region (caller should skip the file).
    """
    match = forclim_pattern(case_study, cohort).search(strip_data_suffix(filename))
    if match is not None:
        return match.group(1).lstrip("_"), match.group(2)[1:]
    # The alive delivery uses dataSim_<stand>_scen<n> instead. Tried second so the
    # historical form always wins where both could match.
    return parse_scen_filename(filename)


def intermediate_filename(stand, simtype, cohort="dead"):
    """Name of the SorSim tree list written into ``intermediate/<scenario>/``."""
    return f"{cohort}Cohorts{stand}_{simtype}.csv"


def sorsim_output_filename(stand, simtype, cohort="dead"):
    """Name of the SorSim assortment file written into ``outputs/<scenario>/``.

    Dead-cohort names are unchanged from earlier runs; alive-cohort files carry
    an ``alive`` flag right after the ``sorsim`` prefix, where the summariser can
    strip it without disturbing the ``_planted_<species>`` suffix.
    """
    if cohort == "alive":
        return f"sorsim_alive_output{stand}_{simtype}.csv"
    return f"sorsim_output{stand}_{simtype}.csv"


#: Matches an intermediate SorSim tree list, e.g. ``deadCohorts207_1_planted_06.csv``.
INTERMEDIATE_PATTERN = re.compile(r"^(dead|alive)Cohorts(\d+)_(.+)$")


def parse_intermediate_filename(filename):
    """Extract ``(cohort, stand, simtype)`` from an intermediate tree list name.

    This is the inverse of :func:`intermediate_filename`, used when re-running
    SorSim from tree lists that were kept with ``save_intermediate=True``.

    Args:
        filename (str): e.g. ``deadCohorts207_1_planted_06.csv``.

    Returns:
        tuple[str, str, str] | None: ``(cohort, stand, simtype)`` or ``None``.
    """
    match = INTERMEDIATE_PATTERN.match(strip_data_suffix(filename))
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def stand_key(value):
    """Canonical string form of a stand id, whichever side it came from.

    The stand parsed out of a ForClim file name is a string of digits (``"1432"``).
    The ``fsID`` column of ``stand.details.csv`` is whatever pandas inferred, and
    that is not stable across deliveries: the September 2026 Jurapark file reads
    as ``float64``, so ``str(fsID)`` gives ``"1432.0"`` and every join against the
    file names fails. Silently -- the stands are all "missing", so stage 1 would
    have excluded every one of them and produced an empty run.

    Both sides go through here instead.

    Args:
        value: A stand id as ``int``, ``float``, ``str`` or numpy scalar.

    Returns:
        str: ``"1432"`` for ``1432``, ``1432.0``, ``"1432"`` and ``"1432.0"``.
        Non-numeric ids are returned stripped but otherwise untouched, so a
        region that uses letters in its ids still works.
    """
    text = str(value).strip()
    if not text:
        return text
    try:
        number = float(text)
    except ValueError:
        return text
    if number != number:            # NaN
        return text
    if number.is_integer():
        return str(int(number))
    return text
