"""Where the pipeline reads and writes, resolved per machine instead of per edit.

Until 2026-09 the Euler paths lived as commented-out lines at the bottom of
``convert_data.py`` and as literals inside ``run_*.sh``. Switching between the
laptop and the cluster therefore meant editing tracked files, which turned every
``git pull`` on Euler into a conflict (and produced the stale ``euler`` branch).

Now the paths come from four environment variables, each a template expanded
with ``{case_study}``, ``{scenario}`` and ``{cohort}``:

====================================  =========================================
``MAINWOOD_INPUT_TEMPLATE``           ForClim output read by stage 1
``MAINWOOD_INTERMEDIATE_TEMPLATE``    SorSim tree lists read by the re-run path
``MAINWOOD_OUTPUT_TEMPLATE``          region root that holds ``outputs/``
``MAINWOOD_DATA_ROOT``                stage 2's ``folder_data`` default
====================================  =========================================

The defaults are exactly the repository-relative paths used before, so a machine
that sets nothing behaves as it always did. Values are read from the environment
first and from an optional, git-ignored ``code/local.env`` second, so both
``./run_conversion.sh`` and a bare ``python convert_data.py ...`` see the same
configuration. See docs/03-runbook.md section 0.
"""

import os
import re

#: Absolute path to ``code/``, so the dotenv lookup does not depend on the cwd.
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Per-machine overrides, git-ignored. ``code/local.env.example`` is the template.
LOCAL_ENV_PATH = os.path.join(CODE_DIR, "local.env")

#: Fallbacks: the paths the pipeline used before the templates existed.
DEFAULTS = {
    "MAINWOOD_INPUT_TEMPLATE": "../data/{case_study}/inputs/{scenario}/",
    "MAINWOOD_INTERMEDIATE_TEMPLATE": "../data/{case_study}/intermediate/{scenario}/",
    "MAINWOOD_OUTPUT_TEMPLATE": "../data/{case_study}/",
    "MAINWOOD_DATA_ROOT": "../data",
    "MAINWOOD_STAND_DETAILS": "../data/{case_study}/stand.details.csv",
    "MAINWOOD_SUMMARY_DIR": "../data/summaries_for_plots",
    "MAINWOOD_SAMPLE_SIZE": "50",
}

#: Placeholders a template may contain. ``case_study_lower`` exists because the
#: alive delivery nests a lower-case region inside a capitalised one:
#: ``.../raw/Misox/alive.data/misox/``.
PLACEHOLDERS = ("case_study", "case_study_lower", "scenario", "cohort")

#: A settings key the shell can also assign. Kept in step with code/load_env.sh.
SETTING_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def load_local_env(path=None):
    """Read ``code/local.env`` into a dict of ``KEY: value``.

    A deliberately small ``.env`` reader: ``KEY=value`` per line, ``#`` comments
    and blank lines skipped, optional surrounding quotes stripped, and an
    optional leading ``export``. No interpolation -- a path is a path.

    Args:
        path (str | None): File to read. Defaults to :data:`LOCAL_ENV_PATH`.

    Returns:
        dict[str, str]: Parsed settings, empty if the file does not exist.
    """
    path = LOCAL_ENV_PATH if path is None else path
    if not os.path.isfile(path):
        return {}

    settings = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, _, value = line.partition("=")
            key = key.strip()
            # local.env is also sourced by code/load_env.sh, which skips anything
            # that is not a shell-assignable name. Skip the same lines here, or the
            # two readers disagree about the same file.
            if not SETTING_NAME.match(key):
                continue
            value = value.split(" #", 1)[0].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            settings[key] = value
    return settings


def setting(name, local_env=None):
    """Resolve one setting: real environment, then ``local.env``, then default.

    The environment wins so that a one-off run can override the file without
    editing it (``MAINWOOD_INPUT_TEMPLATE=... python convert_data.py ...``).

    Args:
        name (str): One of the keys in :data:`DEFAULTS`.
        local_env (dict | None): Pre-parsed ``local.env``, to avoid re-reading it.

    Returns:
        str: The resolved value.
    """
    if name in os.environ and os.environ[name] != "":
        return os.environ[name]
    if local_env is None:
        local_env = load_local_env()
    if local_env.get(name):
        return local_env[name]
    return DEFAULTS[name]


def optional_setting(name, local_env=None):
    """Resolve a setting that has no default, returning ``None`` when unset.

    :func:`setting` is deliberately strict about unknown names; this is for the
    per-cohort overrides, which exist only when a machine declares them.
    """
    if os.environ.get(name):
        return os.environ[name]
    if local_env is None:
        local_env = load_local_env()
    return local_env.get(name) or None


def input_template(cohort="dead", local_env=None):
    """The input template for one cohort.

    ``MAINWOOD_INPUT_TEMPLATE_ALIVE`` (or ``_DEAD``) wins over the shared
    ``MAINWOOD_INPUT_TEMPLATE`` when it is set. The two cohorts are not always in
    the same tree: the dead cohort is under the ForClim results folder while the
    alive delivery lives on a different filesystem entirely, so a single template
    with a ``{cohort}`` placeholder cannot reach both.

    Args:
        cohort (str): ``dead`` or ``alive``.
        local_env (dict | None): Pre-parsed ``local.env``.

    Returns:
        str: The template to expand.
    """
    specific = optional_setting(f"MAINWOOD_INPUT_TEMPLATE_{str(cohort).upper()}", local_env)
    if specific:
        return specific
    return setting("MAINWOOD_INPUT_TEMPLATE", local_env)


def expand(template, case_study, scenario, cohort="dead", trailing_slash=True):
    """Fill a path template, failing loudly on an unknown placeholder.

    A typo such as ``{region}`` would otherwise reach ``os.scandir`` as a
    literal and surface eight hours later as an empty output folder.

    Args:
        template (str): Path template, e.g. ``/cluster/work/{case_study}/``.
        case_study (str): Region name.
        scenario (str): Management scenario.
        cohort (str): ``dead`` or ``alive``.

    Args:
        trailing_slash (bool): Append a separator. True for folders, which callers
            join to by concatenation; False for a file path.

    Returns:
        str: The expanded path.

    Raises:
        ValueError: If the template contains a placeholder we do not define.
    """
    values = {
        "case_study": case_study,
        "case_study_lower": str(case_study).lower(),
        "scenario": scenario,
        "cohort": cohort,
    }
    try:
        expanded = template.format(**values)
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder {exc} in path template {template!r}. "
            f"Available placeholders: {', '.join('{' + p + '}' for p in PLACEHOLDERS)}."
        ) from exc
    if not trailing_slash:
        return expanded
    return expanded if expanded.endswith(("/", os.sep)) else expanded + "/"


def input_folder(case_study, scenario, cohort="dead", local_env=None):
    """Folder holding the ForClim output for one (region, scenario, cohort)."""
    return expand(input_template(cohort, local_env), case_study, scenario, cohort)


def intermediate_folder(case_study, scenario, cohort="dead", local_env=None):
    """Folder holding the saved SorSim tree lists, read when re-running SorSim."""
    return expand(setting("MAINWOOD_INTERMEDIATE_TEMPLATE", local_env), case_study, scenario, cohort)


def output_folder(case_study, scenario, cohort="dead", local_env=None):
    """Region root that stage 1 writes ``intermediate/`` and ``outputs/`` under."""
    return expand(setting("MAINWOOD_OUTPUT_TEMPLATE", local_env), case_study, scenario, cohort)


def data_root(local_env=None):
    """Stage 2's ``folder_data``: the root holding ``<region>/outputs/<scenario>/``."""
    return setting("MAINWOOD_DATA_ROOT", local_env)


def stand_details_path(case_study, local_env=None):
    """The ``stand.details.csv`` for one region -- a file path, not a folder.

    This is the join key and the area column behind every volume in the summary,
    so stage 2 and preflight have to agree on which file they mean. They did not:
    ``summarize_and_create_plots`` hardcoded ``../data/<region>/stand.details.csv``
    while ``preflight`` looked under ``MAINWOOD_DATA_ROOT``, which on Euler is
    ``/cluster/scratch/...`` -- a different file, or none. Both now come here.

    The file arrives from the ForClim side as a one-off delivery (``manag4giacomo``)
    which they do not host, so the tracked copy under ``data/<region>/`` is the
    canonical one -- small, not regenerable, and versioned because it scales every
    number in the output. That is the default and normally stays.

    Override it only to point at a delivery that has not been committed yet::

        MAINWOOD_STAND_DETAILS='/cluster/scratch/giacomov/incoming/{case_study}/stand.details.csv'
    """
    template = setting("MAINWOOD_STAND_DETAILS", local_env)
    return expand(template, case_study, scenario="", cohort="", trailing_slash=False)


def summary_dir(local_env=None):
    """Folder holding the ``<region>_<scenario>.parquet`` deliverable.

    Stage 2 hardcoded ``../data/summaries_for_plots/`` while taking its *input*
    root as an argument, so on Euler it read the assortments from
    ``/cluster/scratch/...`` and then wrote the summaries back inside the
    repository -- onto the home quota, which is far smaller than scratch. A
    Surselva-sized region is about 1.0 GB per scenario as Parquet and 5.5 GB as
    CSV, extrapolated from the measured table in docs/09-summary-format.md.

    Returns:
        str: Folder path, without a trailing separator.
    """
    value = setting("MAINWOOD_SUMMARY_DIR", local_env)
    return value.rstrip("/").rstrip(os.sep)


def sample_size(local_env=None):
    """How many files ``use_sample=True`` processes.

    Was a literal 50 inside ``process_files``; the Euler checkout had it edited to
    100, which is the kind of change that then blocks a ``git pull``.

    Returns:
        int: A positive file count.

    Raises:
        ValueError: If the configured value is not a positive integer.
    """
    raw = setting("MAINWOOD_SAMPLE_SIZE", local_env)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"MAINWOOD_SAMPLE_SIZE must be a positive integer, got {raw!r}.")
    if value <= 0:
        raise ValueError(f"MAINWOOD_SAMPLE_SIZE must be a positive integer, got {value}.")
    return value


def ensure_output_tree(output_folder_path, scenario):
    """Create the ``intermediate/`` and ``outputs/`` folders stage 1 writes into.

    This replaces the manual ``mkdir`` skeleton step
    (``bash_code_to_create_folder_structure_for_data.sh``): a missing folder used
    to abort the run after SorSim had already been handed the first file.

    Args:
        output_folder_path (str): Region root, as returned by :func:`output_folder`.
        scenario (str): Management scenario, the sub-folder under each.

    Returns:
        list[str]: The folders that now exist.
    """
    created = []
    for kind in ("intermediate", "outputs"):
        folder = os.path.join(output_folder_path, kind, scenario)
        os.makedirs(folder, exist_ok=True)
        created.append(folder)
    return created


def describe(case_study, scenario, cohort="dead"):
    """One-line summary of the resolved paths, printed at the top of a run.

    Eight-hour jobs are worth three lines of log: reading the wrong folder is
    the failure mode this whole module exists to prevent.
    """
    local_env = load_local_env()
    return (
        f"  config     {LOCAL_ENV_PATH if os.path.isfile(LOCAL_ENV_PATH) else '(no local.env -- using defaults)'}\n"
        f"  input      {input_folder(case_study, scenario, cohort, local_env)}\n"
        f"  output     {output_folder(case_study, scenario, cohort, local_env)}"
    )
