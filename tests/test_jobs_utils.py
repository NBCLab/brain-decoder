import argparse
import os.path as op
import sys

ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jobs import utils


def test_str_to_bool_accepts_common_true_false_values():
    assert utils.str_to_bool("true") is True
    assert utils.str_to_bool("YES") is True
    assert utils.str_to_bool("0") is False
    assert utils.str_to_bool("No") is False


def test_str_to_bool_rejects_unknown_value():
    try:
        utils.str_to_bool("maybe")
    except argparse.ArgumentTypeError:
        return

    raise AssertionError("Expected argparse.ArgumentTypeError for unknown boolean input.")


def test_get_default_project_dir_points_to_repo_root():
    project_dir = utils.get_default_project_dir()
    assert op.basename(project_dir) == "brain-decoder"
    assert op.isdir(op.join(project_dir, "jobs"))
    assert op.isfile(op.join(project_dir, "pyproject.toml"))


def test_resolve_project_paths_defaults_to_repo_layout():
    project_dir, data_dir, results_dir = utils.resolve_project_paths()
    assert data_dir == op.join(project_dir, "data")
    assert results_dir == op.join(project_dir, "results")


def test_add_common_job_args_uses_expected_defaults():
    parser = utils.add_common_job_args(argparse.ArgumentParser())
    args = parser.parse_args([])

    assert args.project_dir == utils.get_default_project_dir()
    assert args.sections == utils.DEFAULT_SECTIONS
    assert args.model_ids == utils.DEFAULT_MODEL_IDS
    assert args.reduced is True
    assert args.topk == 20
    assert args.standardize is False
