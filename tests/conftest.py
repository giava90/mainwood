"""Shared fixtures and import plumbing for the test suite.

The pipeline scripts are plain scripts, not an installed package, so the tests
put ``code/`` and ``minimal/`` on ``sys.path`` the same way the scripts see each
other when they are launched from their own folder.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, "code")
MINIMAL_DIR = os.path.join(REPO_ROOT, "minimal")

for _path in (CODE_DIR, MINIMAL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# matplotlib must not try to open a window on Euler or in CI
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session")
def repo_root():
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def minimal_dir():
    """Absolute path to ``minimal/``, where the SorSim templates live."""
    return MINIMAL_DIR + os.sep


@pytest.fixture
def forclim_dead_cohorts():
    """A tiny ForClim dead-cohort table, shaped like ``dataSim.dead*.csv``.

    Three cohorts: two harvested (``type == 2``) and one that died standing
    (``type == 1``), which the dead-cohort conversion must drop.
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "year": [2020, 2020, 2020],
            "run": [1, 1, 2],
            "speciesid": [2, 2, 0],  # Picea abies, Picea abies, Abies alba
            "type": [2, 2, 1],
            "dtrees": [3, 1, 5],
            "diameter": [30.0, 12.0, 44.0],
            "height": [2500.0, 900.0, 3100.0],  # centimetres, as ForClim writes them
        }
    )


@pytest.fixture
def forclim_alive_cohorts():
    """A tiny ForClim alive-cohort table, shaped like the standing-stock output."""
    import pandas as pd

    return pd.DataFrame(
        {
            "year": [2020, 2020],
            "run": [1, 2],
            "speciesid": [2, 0],
            "type": [1, 1],
            "trees": [4, 2],
            "diameter": [30.0, 44.0],
            "height": [2500.0, 3100.0],
        }
    )
