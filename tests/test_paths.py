"""Per-machine path configuration (code/paths.py).

These pin the two properties that matter: a machine that configures nothing
behaves exactly as the pipeline did before the templates existed, and a
configured machine never needs an edit to a tracked file.
"""

import os

import pytest

import paths


@pytest.fixture
def no_env(monkeypatch, tmp_path):
    """A clean slate: no MAINWOOD_* in the environment, no local.env on disk."""
    for key in paths.DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(paths, "LOCAL_ENV_PATH", str(tmp_path / "absent.env"))
    return tmp_path


def test_defaults_match_the_pre_template_paths(no_env):
    """The repository-relative defaults are what convert_data.py used to hardcode."""
    assert paths.input_folder("Entlebuch", "WOOD") == "../data/Entlebuch/inputs/WOOD/"
    assert paths.intermediate_folder("Entlebuch", "WOOD") == "../data/Entlebuch/intermediate/WOOD/"
    assert paths.output_folder("Entlebuch", "WOOD") == "../data/Entlebuch/"
    assert paths.data_root() == "../data"


def test_euler_template_expands_with_cohort(no_env, monkeypatch):
    """The Euler layout, including the per-cohort ForClim sub-folder."""
    monkeypatch.setenv(
        "MAINWOOD_INPUT_TEMPLATE",
        "/cluster/work/climate/amauri/{case_study}/Results/mgmt_{scenario}/{cohort}.trees/",
    )
    assert paths.input_folder("Vaud", "BAU", "dead") == (
        "/cluster/work/climate/amauri/Vaud/Results/mgmt_BAU/dead.trees/"
    )
    assert paths.input_folder("Vaud", "BAU", "alive") == (
        "/cluster/work/climate/amauri/Vaud/Results/mgmt_BAU/alive.trees/"
    )


def test_trailing_separator_is_added(no_env, monkeypatch):
    """Callers join with plain concatenation, so the trailing slash must be there."""
    monkeypatch.setenv("MAINWOOD_OUTPUT_TEMPLATE", "/scratch/{case_study}")
    assert paths.output_folder("Misox", "BIO") == "/scratch/Misox/"


def test_unknown_placeholder_is_rejected(no_env, monkeypatch):
    """A typo must fail now, not eight hours in with an empty output folder."""
    monkeypatch.setenv("MAINWOOD_INPUT_TEMPLATE", "/data/{region}/")
    with pytest.raises(ValueError, match="Unknown placeholder"):
        paths.input_folder("Vaud", "BAU")


def test_local_env_is_read_when_environment_is_unset(no_env, monkeypatch):
    env_file = no_env / "local.env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "export MAINWOOD_DATA_ROOT='/cluster/scratch/giacomov/mainwood/'\n"
        "MAINWOOD_OUTPUT_TEMPLATE=/scratch/{case_study}/   # trailing comment\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "LOCAL_ENV_PATH", str(env_file))

    assert paths.data_root() == "/cluster/scratch/giacomov/mainwood/"
    assert paths.output_folder("Vaud", "BAU") == "/scratch/Vaud/"


def test_environment_wins_over_local_env(no_env, monkeypatch):
    """A one-off override must not require editing local.env."""
    env_file = no_env / "local.env"
    env_file.write_text("MAINWOOD_DATA_ROOT=/from/file\n", encoding="utf-8")
    monkeypatch.setattr(paths, "LOCAL_ENV_PATH", str(env_file))
    monkeypatch.setenv("MAINWOOD_DATA_ROOT", "/from/environment")

    assert paths.data_root() == "/from/environment"


def test_missing_local_env_is_not_an_error(no_env):
    assert paths.load_local_env(str(no_env / "nope.env")) == {}


def test_ensure_output_tree_creates_both_folders(tmp_path):
    """Replaces the manual mkdir skeleton, and is safe to call on a live tree."""
    root = str(tmp_path) + os.sep
    created = paths.ensure_output_tree(root, "WOOD")

    assert [os.path.basename(os.path.dirname(c)) for c in created] == ["intermediate", "outputs"]
    for folder in created:
        assert os.path.isdir(folder)

    # idempotent: a re-run over an existing tree must not raise
    paths.ensure_output_tree(root, "WOOD")


def test_sample_size_defaults_to_fifty(no_env):
    """The literal that used to sit inside process_files."""
    assert paths.sample_size() == 50


def test_sample_size_is_configurable(no_env, monkeypatch):
    """The Euler checkout had this edited to 100 in the source, blocking pulls."""
    monkeypatch.setenv("MAINWOOD_SAMPLE_SIZE", "100")
    assert paths.sample_size() == 100


@pytest.mark.parametrize("bad", ["nope", "0", "-5", "", "12.5"])
def test_sample_size_rejects_non_positive_integers(no_env, monkeypatch, bad):
    monkeypatch.setenv("MAINWOOD_SAMPLE_SIZE", bad)
    if bad == "":
        assert paths.sample_size() == 50   # empty means "unset", fall through to default
    else:
        with pytest.raises(ValueError, match="positive integer"):
            paths.sample_size()


def test_local_env_skips_keys_the_shell_cannot_assign(no_env):
    """load_env.sh skips these, so the Python reader must too."""
    env_file = no_env / "local.env"
    env_file.write_text("BAD-KEY=x\nGOOD_KEY=y\n2ND=z\n", encoding="utf-8")
    parsed = paths.load_local_env(str(env_file))
    assert "BAD-KEY" not in parsed
    assert parsed["GOOD_KEY"] == "y"
