"""Analyze which task-level features are associated with per-term decoding performance."""

import argparse
import os
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from jobs.utils import build_cognitiveatlas, resolve_project_paths


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Correlate per-term decoding performance with simple task-level features."
    )
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        default=None,
        help="Path to the repository root.",
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        default=None,
        help="Optional explicit data directory.",
    )
    parser.add_argument(
        "--results_dir",
        dest="results_dir",
        default=None,
        help="Optional explicit results directory.",
    )
    parser.add_argument(
        "--per_term_fn",
        dest="per_term_fn",
        default=None,
        help="Per-term CSV from jobs/per_term_eval.py.",
    )
    parser.add_argument(
        "--ibc_mapping_fn",
        dest="ibc_mapping_fn",
        default=None,
        help="IBC reduced mapping CSV.",
    )
    parser.add_argument(
        "--cnp_mapping_fn",
        dest="cnp_mapping_fn",
        default=None,
        help="CNP reduced mapping CSV.",
    )
    parser.add_argument(
        "--output_table_fn",
        dest="output_table_fn",
        required=True,
        help="Path to the per-term feature table CSV.",
    )
    parser.add_argument(
        "--output_summary_fn",
        dest="output_summary_fn",
        required=True,
        help="Path to the correlation/regression summary CSV.",
    )
    return parser


def _safe_zscore(values):
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / std


def _term_specificity(image_paths):
    specificities = []
    for image_path in image_paths:
        data = np.asanyarray(nib.load(image_path).dataobj, dtype=np.float32)
        data = np.abs(data[np.isfinite(data)])
        if data.size == 0:
            continue
        l1 = float(data.sum())
        l2_sq = float((data**2).sum())
        if l1 == 0 or l2_sq == 0:
            continue
        effective_voxels = (l1**2) / l2_sq
        specificity = 1.0 - (effective_voxels / data.size)
        specificities.append(specificity)
    if not specificities:
        return np.nan
    return float(np.mean(specificities))


def _load_task_feature_frame(data_dir, ibc_mapping_fn, cnp_mapping_fn):
    cognitiveatlas = build_cognitiveatlas(data_dir, reduced=True)
    task_df = pd.DataFrame(
        {
            "term": cognitiveatlas.task_names,
            "definition_length_chars": [len(text) for text in cognitiveatlas.task_definitions],
            "definition_length_words": [len(text.split()) for text in cognitiveatlas.task_definitions],
            "n_linked_concepts": [len(idxs) for idxs in cognitiveatlas.task_to_concept_idxs],
            "n_linked_domains": [len(idxs) for idxs in cognitiveatlas.task_to_process_idxs],
        }
    )

    counts = np.load(op.join(data_dir, "vocabulary", "vocabulary-cogatlasred_task-names_section-body_counts.npy"))
    if counts.ndim == 2:
        counts = counts.sum(axis=1)
    emb = np.load(
        op.join(
            data_dir,
            "vocabulary",
            "vocabulary-cogatlasred_task-combined_embedding-BrainGPT-7B-v0.2.npy",
        )
    )
    task_df["training_article_count"] = counts
    task_df["embedding_norm"] = np.linalg.norm(emb, axis=1)

    mapping_frames = []
    for dataset, mapping_fn in [
        ("ibc", ibc_mapping_fn),
        ("cnp", cnp_mapping_fn),
    ]:
        if mapping_fn is None:
            continue
        mapping_df = pd.read_csv(mapping_fn)
        mapping_df = mapping_df.loc[:, ["task_name", "local_path"]].copy()
        mapping_df["dataset"] = dataset
        mapping_frames.append(mapping_df.rename(columns={"task_name": "term"}))
    mappings_df = pd.concat(mapping_frames, ignore_index=True)

    specificity_rows = []
    grouped = mappings_df.groupby("term", dropna=False, sort=True)
    for term, group_df in tqdm(grouped, total=grouped.ngroups, desc="term specificity"):
        specificity_rows.append(
            {
                "term": term,
                "mean_map_specificity": _term_specificity(group_df["local_path"].tolist()),
                "n_eval_maps": int(len(group_df)),
                "n_eval_datasets": int(group_df["dataset"].nunique()),
            }
        )

    specificity_df = pd.DataFrame(specificity_rows)
    return task_df.merge(specificity_df, on="term", how="left")


def _fit_standardized_regression(df, predictors, target):
    work_df = df.loc[:, predictors + [target]].dropna().copy()
    if len(work_df) < 3:
        return []
    x = np.column_stack([_safe_zscore(work_df[col]) for col in predictors] + [np.ones(len(work_df))])
    y = _safe_zscore(work_df[target].to_numpy())
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ coef
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    rows = []
    for predictor, beta in zip(predictors, coef[:-1]):
        rows.append(
            {
                "analysis": "standardized_regression",
                "target": target,
                "feature": predictor,
                "value": float(beta),
                "n_terms": int(len(work_df)),
                "r2": r2,
            }
        )
    return rows


def main(
    project_dir=None,
    data_dir=None,
    results_dir=None,
    per_term_fn=None,
    ibc_mapping_fn=None,
    cnp_mapping_fn=None,
    output_table_fn=None,
    output_summary_fn=None,
):
    _, data_dir, results_dir = resolve_project_paths(project_dir, data_dir, results_dir)
    per_term_fn = (
        op.join(results_dir, "per_term_cross_dataset_reduced_full.csv")
        if per_term_fn is None
        else op.abspath(per_term_fn)
    )
    ibc_mapping_fn = (
        op.join(data_dir, "ibc", "mapping_reduced.csv")
        if ibc_mapping_fn is None
        else op.abspath(ibc_mapping_fn)
    )
    cnp_mapping_fn = (
        op.join(data_dir, "cnp", "mapping_reduced.csv")
        if cnp_mapping_fn is None
        else op.abspath(cnp_mapping_fn)
    )

    per_term_df = pd.read_csv(per_term_fn)
    per_term_df = per_term_df.loc[
        (per_term_df["backend"] == "brainclip")
        & (per_term_df["level"] == "task")
        & (per_term_df["sub_category"] == "combined")
        & (per_term_df["section"] == "body")
    ].copy()

    feature_df = _load_task_feature_frame(data_dir, ibc_mapping_fn, cnp_mapping_fn)
    result_df = per_term_df.merge(feature_df, on="term", how="left")
    os.makedirs(op.dirname(op.abspath(output_table_fn)), exist_ok=True)
    result_df.to_csv(output_table_fn, index=False)

    feature_columns = [
        "training_article_count",
        "definition_length_words",
        "embedding_norm",
        "n_linked_concepts",
        "n_linked_domains",
        "mean_map_specificity",
        "n_eval_maps",
    ]
    summary_rows = []
    for target in ["mean_hit_at_k", "mean_reciprocal_rank"]:
        for feature in feature_columns:
            work_df = result_df.loc[:, [feature, target]].dropna()
            if len(work_df) < 3:
                continue
            rho, p_value = spearmanr(work_df[feature], work_df[target])
            summary_rows.append(
                {
                    "analysis": "spearman",
                    "target": target,
                    "feature": feature,
                    "value": float(rho),
                    "p_value": float(p_value),
                    "n_terms": int(len(work_df)),
                    "r2": np.nan,
                }
            )
        summary_rows.extend(_fit_standardized_regression(result_df, feature_columns, target))

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(op.dirname(op.abspath(output_summary_fn)), exist_ok=True)
    summary_df.to_csv(output_summary_fn, index=False)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
