"""The regions and management scenarios the pipeline knows about.

Until 2026-09 this list was copy-pasted into five entry points
(``convert_data``, ``convert_data_from_intermediate``, ``summarize_and_create_plots``,
``plot_only``, ``preflight``), and they had already drifted apart: ``plot_only``
silently dropped ``HYBRID`` while the others kept it. Onboarding a region meant
finding all five and getting all five right; miss one and the run aborts on an
argument check hours after you thought you were done.

Adding a region is now one line in :data:`CASE_STUDIES`.

Both axes have a wildcard token that means "every one of them": ``All`` for
regions, ``ALL`` for scenarios. The two spellings are not a typo -- they are what
every existing script and SLURM submission already passes, so they are kept.
"""

#: Every case study region, in onboarding order. Spelled exactly as in the
#: ForClim file names -- ``naming.forclim_pattern`` matches against this.
CASE_STUDIES = ("Entlebuch", "Vaud", "Surselva", "Misox", "Jurapark")

#: Every management scenario.
SCENARIOS = ("BAU", "WOOD", "BIO", "HYBRID")

#: The wildcard tokens. Different case, by existing convention.
ALL_CASE_STUDIES = "All"
ALL_SCENARIOS = "ALL"


def valid_case_studies(include_all=True):
    """Accepted values for the ``case_study`` argument."""
    return list(CASE_STUDIES) + ([ALL_CASE_STUDIES] if include_all else [])


def valid_scenarios(include_all=True, exclude=()):
    """Accepted values for the ``management_scenario`` argument.

    Args:
        include_all (bool): Include the ``ALL`` wildcard.
        exclude (tuple): Scenarios this particular entry point does not support.
            ``plot_only`` uses it to keep its long-standing exclusion of
            ``HYBRID`` visible rather than hidden in a diverged copy of the list.
    """
    return [s for s in SCENARIOS if s not in exclude] + ([ALL_SCENARIOS] if include_all else [])


def check_case_study(value, include_all=True):
    """Validate a region name, raising the message the entry points used to raise."""
    allowed = valid_case_studies(include_all)
    if value not in allowed:
        raise ValueError(f"Invalid case study. Please provide a valid case study {allowed}.")
    return value


def check_scenario(value, include_all=True, exclude=()):
    """Validate a scenario name, raising the message the entry points used to raise."""
    allowed = valid_scenarios(include_all, exclude)
    if value not in allowed:
        raise ValueError(f"Invalid management scenario. Please provide a valid management scenario {allowed}.")
    return value


def resolve_case_studies(value):
    """Expand ``All`` into every region; otherwise the single region asked for."""
    return list(CASE_STUDIES) if value == ALL_CASE_STUDIES else [value]


def resolve_scenarios(value, exclude=()):
    """Expand ``ALL`` into every scenario; otherwise the single scenario asked for."""
    if value == ALL_SCENARIOS:
        return [s for s in SCENARIOS if s not in exclude]
    return [value]
