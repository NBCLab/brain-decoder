"""Aggregate ROI decoding outputs or document missing ROI assets."""

import argparse
import json
import os
import os.path as op
from glob import glob

import pandas as pd

from jobs.utils import resolve_project_paths


def _get_parser():
    parser = argparse.ArgumentParser(description="ROI follow-up summary for striatum and related seeds.")
    parser.add_argument("--project_dir", dest="project_dir", default=None)
    parser.add_argument("--data_dir", dest="data_dir", default=None)
    parser.add_argument("--results_dir", dest="results_dir", default=None)
    parser.add_argument("--prediction_dir", dest="prediction_dir", default=None)
    parser.add_argument("--output_fn", dest="output_fn", required=True)
    return parser


def _top_prediction(csv_path):
    df = pd.read_csv(csv_path)
    first = df.iloc[0]
    value_col = "prob" if "prob" in df.columns else "weight" if "weight" in df.columns else "corr"
    return {"pred": first["pred"], "score": float(first[value_col])}


def main(project_dir=None, data_dir=None, results_dir=None, prediction_dir=None, output_fn=None):
    _, data_dir, results_dir = resolve_project_paths(project_dir, data_dir, results_dir)
    prediction_dir = (
        op.join(results_dir, "predictions_rois") if prediction_dir is None else op.abspath(prediction_dir)
    )
    seed_dir = op.join(data_dir, "seed-regions")

    if not op.isdir(seed_dir) and not op.isdir(prediction_dir):
        output_df = pd.DataFrame(
            [
                {
                    "status": "blocked",
                    "reason": "roi_seed_maps_missing",
                    "expected_seed_dir": seed_dir,
                    "expected_prediction_dir": prediction_dir,
                }
            ]
        )
    else:
        rows = []
        for task_csv in sorted(glob(op.join(prediction_dir, "*_pred-task_brainclip.csv"))):
            stem = op.basename(task_csv).replace("_pred-task_brainclip.csv", "")
            concept_csv = task_csv.replace("_pred-task_brainclip.csv", "_pred-concept_brainclip.csv")
            process_csv = task_csv.replace("_pred-task_brainclip.csv", "_pred-process_brainclip.csv")
            row = {"roi_label": stem, "status": "ready"}
            row["top_task_json"] = json.dumps(_top_prediction(task_csv))
            if op.exists(concept_csv):
                row["top_concept_json"] = json.dumps(_top_prediction(concept_csv))
            if op.exists(process_csv):
                row["top_domain_json"] = json.dumps(_top_prediction(process_csv))
            rows.append(row)
        output_df = pd.DataFrame(rows)

    os.makedirs(op.dirname(op.abspath(output_fn)), exist_ok=True)
    output_df.to_csv(output_fn, index=False)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
