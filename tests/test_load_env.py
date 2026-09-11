"""The shell side of the per-machine config (code/load_env.sh).

``code/paths.py`` has always given the real environment precedence over
``local.env``. The first version of the SLURM wrappers sourced the file with
``set -a``, which assigns unconditionally -- so

    MAINWOOD_INPUT_TEMPLATE=/other/path/ ./run_conversion.sh WOOD Surselva

was silently ignored, and the shell and Python disagreed about the same setting.
These tests pin the two to the same precedence, and to the same parsing rules.
"""

import os
import shutil
import subprocess

import pytest

import paths

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")

CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
LOADER = os.path.join(CODE_DIR, "load_env.sh")

SAMPLE = """# a comment
export MAINWOOD_INPUT_TEMPLATE='/from/file/{case_study}/'
MAINWOOD_DATA_ROOT="/scratch/root/"   # trailing comment
CONVERT_CORES=4
BAD-KEY=ignored

"""


def run_loader(tmp_path, env_file_text, names, extra_env=None):
    """Source load_env.sh over a local.env and report the resulting values."""
    env_file = tmp_path / "local.env"
    env_file.write_text(env_file_text, encoding="utf-8")

    script = tmp_path / "probe.sh"
    echoes = "\n".join('echo "%s=${%s:-<unset>}"' % (n, n) for n in names)
    script.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        '. "%s"\n' % LOADER.replace("\\", "/")
        + 'load_local_env "%s"\n' % str(env_file).replace("\\", "/")
        + echoes + "\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    # keep the parent's own MAINWOOD_* out of the probe
    for key in list(env):
        if key.startswith("MAINWOOD_") or key.startswith("CONVERT_"):
            del env[key]
    env.update(extra_env or {})

    out = subprocess.run(
        [BASH, str(script)], capture_output=True, text=True, env=env, check=True
    ).stdout
    return dict(line.split("=", 1) for line in out.strip().splitlines())


def test_file_supplies_values_when_environment_is_empty(tmp_path):
    got = run_loader(tmp_path, SAMPLE, ["MAINWOOD_INPUT_TEMPLATE", "MAINWOOD_DATA_ROOT", "CONVERT_CORES"])
    assert got["MAINWOOD_INPUT_TEMPLATE"] == "/from/file/{case_study}/"
    assert got["MAINWOOD_DATA_ROOT"] == "/scratch/root/"
    assert got["CONVERT_CORES"] == "4"


def test_environment_wins_over_the_file(tmp_path):
    """The regression: a one-off override must survive the file being loaded."""
    got = run_loader(
        tmp_path, SAMPLE,
        ["MAINWOOD_INPUT_TEMPLATE", "CONVERT_CORES", "MAINWOOD_DATA_ROOT"],
        extra_env={"MAINWOOD_INPUT_TEMPLATE": "/from/env/{case_study}/", "CONVERT_CORES": "42"},
    )
    assert got["MAINWOOD_INPUT_TEMPLATE"] == "/from/env/{case_study}/"
    assert got["CONVERT_CORES"] == "42"
    assert got["MAINWOOD_DATA_ROOT"] == "/scratch/root/"   # untouched key still loads


def test_missing_file_is_not_an_error(tmp_path):
    script = tmp_path / "probe.sh"
    script.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        '. "%s"\n' % LOADER.replace("\\", "/")
        + 'load_local_env "%s"\necho "ok"\n' % str(tmp_path / "absent.env").replace("\\", "/"),
        encoding="utf-8",
    )
    out = subprocess.run([BASH, str(script)], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


def test_shell_and_python_parse_the_file_identically(tmp_path):
    """One config file, two readers -- they must not drift apart again."""
    env_file = tmp_path / "local.env"
    env_file.write_text(SAMPLE, encoding="utf-8")

    from_python = paths.load_local_env(str(env_file))
    keys = ["MAINWOOD_INPUT_TEMPLATE", "MAINWOOD_DATA_ROOT", "CONVERT_CORES"]
    from_shell = run_loader(tmp_path, SAMPLE, keys)

    for key in keys:
        assert from_python[key] == from_shell[key], key
    assert "BAD-KEY" not in from_python
