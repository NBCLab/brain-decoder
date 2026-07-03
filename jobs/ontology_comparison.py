"""Compare reduced versus full Cognitive Atlas decoding performance."""

import argparse
import json
import os
import os.path as op

import numpy as np
import pandas as pd

from jobs.utils import build_cognitiveatlas, resolve_project_paths


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Summarize reduced versus full ontology structure and decoding performance."
    )
    parser.add_argument("--project_dir", dest="project_dir", default=None)
    parser.add_argument("--data_dir", dest="data_dir", default=None)
    parser.add_argument("--results_dir", dest="results_dir", default=None)
    parser.add_argument("--reduced_eval_fn", dest="reduced_eval_fn", required=True)
    parser.add_argument("--reduced_details_fn", dest="reduced_details_fn", required=True)
    parser.add_argument("--full_eval_fn", dest="full_eval_fn", required=True)
    parser.add_argument("--full_details_fn", dest="full_details_fn", required=True)
    parser.add_argument("--full_mapping_fn", dest="full_mapping_fn", required=True)
    parser.add_argument("--dataset_name", dest="dataset_name", required=True)
    parser.add_argument("--ontology_stats_fn", dest="ontology_stats_fn", required=True)
    parser.add_argument("--comparison_fn", dest="comparison_fn", required=True)
    parser.add_argument(
        "--term_table_fn",
        dest="term_table_fn",
        default=None,
        help="Optional path for a task-level retained/removed/enriched ontology table.",
    )
    return parser


def _ontology_stats(data_dir):
    rows = []
    for label, reduced in [("reduced", True), ("full", False)]:
        atlas = build_cognitiveatlas(data_dir, reduced=reduced)
        rows.append(
            {
                "ontology": label,
                "n_tasks": int(len(atlas.task_names)),
                "n_concepts": int(len(atlas.concept_names)),
                "n_domains": int(len(atlas.process_names)),
                "n_task_concept_edges": int(sum(len(idxs) for idxs in atlas.task_to_concept_idxs)),
                "n_concept_domain_edges": int(sum(len(idxs) for idxs in atlas.process_to_concept_idxs)),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_from_details(details_df):
    grouped = details_df.groupby(
        [
            "dataset",
            "source",
            "category",
            "sub_category",
            "section",
            "model_id",
            "model_name",
            "vocabulary_label",
            "backend",
            "level",
            "k",
        ],
        dropna=False,
        sort=False,
    )
    return (
        grouped.agg(
            n_images=("prediction_label", "size"),
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_hit_at_k=("hit_at_k", "mean"),
            median_best_rank=("best_rank", lambda x: np.nanmedian(np.where(np.isfinite(x), x, np.nan))),
        )
        .reset_index()
    )


def _task_term_table(data_dir):
    full_atlas = build_cognitiveatlas(data_dir, reduced=False)
    reduced_atlas = build_cognitiveatlas(data_dir, reduced=True)

    full_task_to_concepts = {
        task_name: {full_atlas.concept_names[idx] for idx in concept_idxs}
        for task_name, concept_idxs in zip(full_atlas.task_names, full_atlas.task_to_concept_idxs)
    }
    reduced_task_to_concepts = {
        task_name: {reduced_atlas.concept_names[idx] for idx in concept_idxs}
        for task_name, concept_idxs in zip(reduced_atlas.task_names, reduced_atlas.task_to_concept_idxs)
    }

    rows = []
    all_tasks = sorted(set(full_task_to_concepts) | set(reduced_task_to_concepts))
    for task_name in all_tasks:
        full_concepts = full_task_to_concepts.get(task_name, set())
        reduced_concepts = reduced_task_to_concepts.get(task_name, set())
        shared_concepts = full_concepts & reduced_concepts

        if task_name not in reduced_task_to_concepts:
            status = "removed_in_reduced"
        elif full_concepts == reduced_concepts:
            status = "retained_same_concepts"
        elif reduced_concepts > full_concepts:
            status = "retained_enriched_concepts"
        elif reduced_concepts < full_concepts:
            status = "retained_pruned_concepts"
        else:
            status = "retained_rewired_concepts"

        rows.append(
            {
                "task_name": task_name,
                "in_full_ontology": task_name in full_task_to_concepts,
                "in_reduced_ontology": task_name in reduced_task_to_concepts,
                "status": status,
                "n_full_concepts": len(full_concepts),
                "n_reduced_concepts": len(reduced_concepts),
                "n_shared_concepts": len(shared_concepts),
                "full_concepts_json": json.dumps(sorted(full_concepts)),
                "reduced_concepts_json": json.dumps(sorted(reduced_concepts)),
            }
        )

    return pd.DataFrame(rows)


def main(
    project_dir=None,
    data_dir=None,
    results_dir=None,
    reduced_eval_fn=None,
    reduced_details_fn=None,
    full_eval_fn=None,
    full_details_fn=None,
    full_mapping_fn=None,
    dataset_name=None,
    ontology_stats_fn=None,
    comparison_fn=None,
    term_table_fn=None,
):
    _, data_dir, _ = resolve_project_paths(project_dir, data_dir, results_dir)
    reduced_eval_df = pd.read_csv(op.abspath(reduced_eval_fn))
    reduced_details_df = pd.read_csv(op.abspath(reduced_details_fn))
    full_eval_df = pd.read_csv(op.abspath(full_eval_fn))
    full_details_df = pd.read_csv(op.abspath(full_details_fn))
    full_mapping_df = pd.read_csv(op.abspath(full_mapping_fn))

    stats_df = _ontology_stats(data_dir)
    os.makedirs(op.dirname(op.abspath(ontology_stats_fn)), exist_ok=True)
    stats_df.to_csv(ontology_stats_fn, index=False)

    if term_table_fn is not None:
        term_table_df = _task_term_table(data_dir)
        os.makedirs(op.dirname(op.abspath(term_table_fn)), exist_ok=True)
        term_table_df.to_csv(term_table_fn, index=False)

    reduced_eval_df = reduced_eval_df.assign(ontology="reduced", evaluation_scope="all_images")
    full_eval_df = full_eval_df.assign(ontology="full", evaluation_scope="all_images")

    overlap_labels = set(
        full_mapping_df.loc[full_mapping_df["in_reduced_ontology"].fillna(False), "prediction_label"].astype(str)
    )
    full_overlap_df = full_details_df.loc[full_details_df["prediction_label"].astype(str).isin(overlap_labels)].copy()
    full_overlap_summary_df = _aggregate_from_details(full_overlap_df).assign(
        ontology="full",
        evaluation_scope="overlap_with_reduced",
    )
    reduced_overlap_df = reduced_details_df.loc[
        reduced_details_df["prediction_label"].astype(str).isin(overlap_labels)
    ].copy()
    reduced_overlap_summary_df = _aggregate_from_details(reduced_overlap_df).assign(
        ontology="reduced",
        evaluation_scope="overlap_with_reduced",
    )

    comparison_df = pd.concat(
        [reduced_eval_df, full_eval_df, reduced_overlap_summary_df, full_overlap_summary_df],
        ignore_index=True,
        sort=False,
    )
    comparison_df.insert(0, "dataset_name", dataset_name)

    os.makedirs(op.dirname(op.abspath(comparison_fn)), exist_ok=True)
    comparison_df.to_csv(comparison_fn, index=False)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
