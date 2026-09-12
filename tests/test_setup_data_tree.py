"""The folder skeleton (code/setup_data_tree.py).

It builds what a run writes into, from the same templates the pipeline reads, so a
freshly purged scratch is rebuilt to match whatever local.env says.

Two shapes matter and are easy to get wrong: ``inputs/<scenario>/`` is per
scenario, while ``alive.trees/`` sits at the region root with no scenario at all --
the alive delivery is one 2015 snapshot per region, taken before management
diverges, so a per-scenario copy would be four copies of one thing.
"""

import os

import pytest

import paths
import setup_data_tree


@pytest.fixture
def configured(monkeypatch, tmp_path):
    """A region root and summary folder under tmp_path, nothing external."""
    root = tmp_path / "mainwood"
    for key in list(paths.DEFAULTS) + [
        "MAINWOOD_INPUT_TEMPLATE_ALIVE", "MAINWOOD_INPUT_TEMPLATE_DEAD"
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(paths, "LOCAL_ENV_PATH", str(tmp_path / "absent.env"))

    as_posix = str(root).replace("\\", "/")
    monkeypatch.setenv("MAINWOOD_OUTPUT_TEMPLATE", as_posix + "/{case_study}/")
    monkeypatch.setenv("MAINWOOD_SUMMARY_DIR", as_posix + "/summaries")
    monkeypatch.setenv("MAINWOOD_INPUT_TEMPLATE", as_posix + "/{case_study}/inputs/{scenario}/")
    monkeypatch.setenv("MAINWOOD_INPUT_TEMPLATE_ALIVE", as_posix + "/{case_study}/alive.trees/")
    return root


def purposes(plan):
    return [purpose for purpose, _ in plan]


def test_inputs_are_created_by_default(configured):
    """ForClim data is staged onto scratch rather than read from NFS in place, so
    the folder it is copied into should exist without asking."""
    plan = setup_data_tree.planned_folders(["Misox"], ["BAU"], want_inputs=True, local_env={})
    assert "inputs" in purposes(plan)


def test_no_inputs_skips_them(configured):
    plan = setup_data_tree.planned_folders(["Misox"], ["BAU"], want_inputs=False, local_env={})
    assert "inputs" not in purposes(plan)
    assert "alive.trees" not in purposes(plan)
    assert "outputs" in purposes(plan)


def test_alive_trees_is_one_folder_per_region_not_per_scenario(configured):
    """Four scenarios must not produce four alive folders."""
    plan = setup_data_tree.planned_folders(
        ["Misox"], ["BAU", "WOOD", "BIO", "HYBRID"], want_inputs=True, local_env={})

    alive = [path for purpose, path in plan if purpose == "alive.trees"]
    assert len(alive) == 1
    assert os.path.basename(alive[0]) == "alive.trees"
    # and it is at the region root, not under inputs/
    assert "inputs" not in alive[0].replace("\\", "/").split("/")

    per_scenario = [path for purpose, path in plan if purpose == "inputs"]
    assert len(per_scenario) == 4


def test_the_tree_is_actually_created_and_is_idempotent(configured):
    assert setup_data_tree.main(["setup_data_tree.py", "Misox", "BAU"]) == 0

    root = configured / "Misox"
    for expected in ("inputs/BAU", "intermediate/BAU", "outputs/BAU", "alive.trees"):
        assert (root / expected).is_dir(), expected

    # re-running must not raise
    assert setup_data_tree.main(["setup_data_tree.py", "Misox", "BAU"]) == 0


def test_dry_run_creates_nothing(configured):
    assert setup_data_tree.main(["setup_data_tree.py", "Misox", "BAU", "--dry-run"]) == 0
    assert not (configured / "Misox").exists()


def test_an_external_input_template_is_reported_not_created(configured, monkeypatch, tmp_path):
    """The ForClim results tree and the NFS export belong to someone else. Say so
    rather than creating directories there."""
    monkeypatch.setenv(
        "MAINWOOD_INPUT_TEMPLATE",
        "/cluster/work/climate/amauri/{case_study}/Results/mgmt_{scenario}/{cohort}.trees/")

    external = setup_data_tree.external_inputs(["Misox"], ["BAU"], {})
    folders = [folder for _, folder in external]
    assert any("amauri" in f for f in folders)
    assert not any("alive.trees" in f for f in folders)   # that one is in-tree


def test_nothing_is_reported_when_both_templates_are_in_tree(configured):
    assert setup_data_tree.external_inputs(["Misox"], ["BAU"], {}) == []


def test_an_unknown_flag_is_refused(configured):
    with pytest.raises(SystemExit, match="Unknown option"):
        setup_data_tree.main(["setup_data_tree.py", "--inpts"])
