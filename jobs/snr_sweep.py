"""Run or document the HCP subject-level group-size sensitivity analysis."""

import argparse
import os
import os.path as op

import pandas as pd

from jobs.utils import resolve_project_paths


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Group-size sensitivity analysis for subject-level HCP maps."
    )
    parser.add_argument("--project_dir", dest="project_dir", default=None)
    parser.add_argument("--data_dir", dest="data_dir", default=None)
    parser.add_argument("--results_dir", dest="results_dir", default=None)
    parser.add_argument(
        "--subject_dir",
        dest="subject_dir",
        default=None,
        help="Directory containing subject-level HCP maps.",
    )
    parser.add_argument(
        "--output_fn",
        dest="output_fn",
        required=True,
        help="CSV report path.",
    )
    return parser


def main(project_dir=None, data_dir=None, results_dir=None, subject_dir=None, output_fn=None):
    _, data_dir, _ = resolve_project_paths(project_dir, data_dir, results_dir)
    subject_dir = op.join(data_dir, "hcp_subject") if subject_dir is None else op.abspath(subject_dir)

    if not op.isdir(subject_dir):
        report_df = pd.DataFrame(
            [
                {
                    "status": "blocked",
                    "reason": "subject_level_hcp_maps_missing",
                    "expected_path": subject_dir,
                }
            ]
        )
    else:
        report_df = pd.DataFrame(
            [
                {
                    "status": "ready",
                    "reason": "subject_level_hcp_maps_present",
                    "expected_path": subject_dir,
                }
            ]
        )

    os.makedirs(op.dirname(op.abspath(output_fn)), exist_ok=True)
    report_df.to_csv(output_fn, index=False)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
