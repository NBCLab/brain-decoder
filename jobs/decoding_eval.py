"""Evaluate decoding predictions against explicit dataset mappings."""

import argparse
import itertools
import json
import os
import os.path as op
from glob import glob

import numpy as np
import pandas as pd

from jobs.utils import (
    DEFAULT_MODEL_IDS,
    DEFAULT_SECTIONS,
    build_cognitiveatlas,
    infer_prediction_label,
    parse_name_list,
    resolve_project_paths,
)

LEGACY_HCP_DOMAIN_TO_LABEL = {
    "EMOTION": "emotion",
    "GAMBLING": "gambling",
    "LANGUAGE": "language",
    "MOTOR": "motor",
    "RELATIONAL": "relational",
    "SOCIAL": "social",
    "WM": "working memory",
}


def _recall_at_n(true_labels, pred_labels, n):
    if not true_labels:
        return np.nan
    return len(set(true_labels) & set(pred_labels[:n])) / len(true_labels)


def _best_rank(true_labels, pred_labels):
    true_labels = set(true_labels)
    for rank, pred in enumerate(pred_labels, start=1):
        if pred in true_labels:
            return rank
    return np.inf


def _resolve_column(df, requested_name, candidates, required=False):
    names = [requested_name] if requested_name is not None else []
    names.extend(candidates)
    for name in names:
        if name and name in df.columns:
            return name
    if required:
        raise KeyError(f"Could not find any of the required columns: {names}.")
    return None


def _build_record_from_task(prediction_label, task_names, cognitiveatlas, dataset_name, row_id):
    task_names = parse_name_list(task_names)
    if not task_names:
        raise ValueError(f"Missing task labels for record {row_id}.")

    task_idx = cognitiveatlas.get_task_idx_from_names(task_names)
    task_idxs = [task_idx] if isinstance(task_idx, (int, np.integer)) else list(task_idx)
    concept_idx = np.unique(
        np.concatenate([cognitiveatlas.task_to_concept_idxs[idx] for idx in task_idxs])
        if task_idxs
        else np.array([], dtype=int)
    )
    concept_idx = concept_idx.astype(int, copy=False)
    domain_idx = np.unique(
        np.concatenate([cognitiveatlas.task_to_process_idxs[idx] for idx in task_idxs])
        if task_idxs
        else np.array([], dtype=int)
    )
    domain_idx = domain_idx.astype(int, copy=False)
    concept_names = cognitiveatlas.get_concept_names_from_idx(concept_idx).tolist()
    domain_names = cognitiveatlas.get_process_names_from_idx(domain_idx).tolist()
    return {
        "dataset": dataset_name,
        "prediction_label": prediction_label,
        "task": task_names,
        "concept": concept_names,
        "domain": domain_names,
    }


def _load_mapping_records(
    dataset_name,
    mapping_fn,
    mapping_label_column,
    mapping_task_column,
    mapping_concepts_column,
    mapping_domains_column,
    mapping_filename_column,
    mapping_label_delimiter,
    mapping_label_token_index,
    cognitiveatlas,
):
    mapping_df = pd.read_csv(mapping_fn)
    label_column = _resolve_column(
        mapping_df,
        mapping_label_column,
        ["prediction_label", "task_code", "image_label", "label"],
        required=False,
    )
    filename_column = _resolve_column(
        mapping_df,
        mapping_filename_column,
        ["filename", "image_path", "image", "path"],
        required=label_column is None,
    )
    task_column = _resolve_column(
        mapping_df,
        mapping_task_column,
        ["task_name", "task"],
        required=True,
    )
    concepts_column = _resolve_column(
        mapping_df,
        mapping_concepts_column,
        ["concepts", "concept", "true_concepts"],
        required=False,
    )
    domains_column = _resolve_column(
        mapping_df,
        mapping_domains_column,
        ["domains", "domain", "true_domains"],
        required=False,
    )

    records = []
    for row_idx, row in mapping_df.iterrows():
        if label_column is not None:
            prediction_label = str(row[label_column]).strip()
        else:
            prediction_label = infer_prediction_label(
                row[filename_column],
                delimiter=mapping_label_delimiter,
                token_index=mapping_label_token_index,
            )

        task_names = parse_name_list(row[task_column])
        if not task_names:
            raise ValueError(f"Row {row_idx} in {mapping_fn} does not define any task labels.")

        if concepts_column is None or domains_column is None:
            if cognitiveatlas is None:
                raise ValueError(
                    "A CognitiveAtlas object is required when the mapping file does not define "
                    "explicit concepts/domains columns."
                )
            record = _build_record_from_task(
                prediction_label,
                task_names,
                cognitiveatlas,
                dataset_name=dataset_name,
                row_id=f"{mapping_fn}:{row_idx}",
            )
        else:
            record = {
                "dataset": dataset_name,
                "prediction_label": prediction_label,
                "task": task_names,
                "concept": parse_name_list(row[concepts_column]),
                "domain": parse_name_list(row[domains_column]),
            }

        records.append(record)

    return records


def _load_legacy_hcp_records(image_dir, ground_truth_fn):
    with open(ground_truth_fn, "r") as file_obj:
        ground_truth = json.load(file_obj)

    images = sorted(glob(op.join(image_dir, "*.nii.gz")))
    records = []
    for img_fn in images:
        image_name = op.basename(img_fn).split(".")[0]
        task_code = image_name.split("_")[1]
        domain_label = LEGACY_HCP_DOMAIN_TO_LABEL[task_code]
        gt = ground_truth.get(domain_label) or ground_truth.get(domain_label.replace(" ", "_"))
        if gt is None:
            raise KeyError(f"Could not find ground truth for HCP domain {domain_label!r}.")
        records.append(
            {
                "dataset": "hcp",
                "prediction_label": task_code,
                "task": parse_name_list(gt["task"]),
                "concept": parse_name_list(gt["concept"]),
                "domain": parse_name_list(gt["domain"]),
            }
        )
    return records


def _get_parser():
    parser = argparse.ArgumentParser(description="Evaluate decoding predictions against ground truth.")
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
        "--dataset_name",
        dest="dataset_name",
        default="hcp",
        help="Dataset label to attach to evaluation rows.",
    )
    parser.add_argument(
        "--sections",
        dest="sections",
        nargs="+",
        default=list(DEFAULT_SECTIONS),
        help="Text sections to evaluate.",
    )
    parser.add_argument(
        "--model_ids",
        dest="model_ids",
        nargs="+",
        default=list(DEFAULT_MODEL_IDS),
        help="One or more embedding model identifiers to evaluate.",
    )
    parser.add_argument(
        "--sources",
        dest="sources",
        nargs="+",
        default=["cogatlasred", "cogatlas"],
        help="Vocabulary sources to evaluate.",
    )
    parser.add_argument(
        "--categories",
        dest="categories",
        nargs="+",
        default=["task"],
        help="Vocabulary categories to evaluate.",
    )
    parser.add_argument(
        "--sub_categories",
        dest="sub_categories",
        nargs="+",
        default=["combined", "names"],
        help="Vocabulary embedding variants to evaluate.",
    )
    parser.add_argument(
        "--models",
        dest="models",
        nargs="+",
        default=["neurosynth", "gclda", "brainclip"],
        help="Prediction backends expected in the output directory.",
    )
    parser.add_argument(
        "--prediction_dir",
        dest="prediction_dir",
        default=None,
        help="Optional explicit prediction directory.",
    )
    parser.add_argument(
        "--mapping_fn",
        dest="mapping_fn",
        default=None,
        help="CSV describing evaluation items. Preferred for IBC/CNP and explicit HCP mappings.",
    )
    parser.add_argument(
        "--mapping_label_column",
        dest="mapping_label_column",
        default=None,
        help="Optional mapping CSV column containing the prediction label prefix.",
    )
    parser.add_argument(
        "--mapping_task_column",
        dest="mapping_task_column",
        default=None,
        help="Optional mapping CSV column containing task labels.",
    )
    parser.add_argument(
        "--mapping_concepts_column",
        dest="mapping_concepts_column",
        default=None,
        help="Optional mapping CSV column containing concept labels.",
    )
    parser.add_argument(
        "--mapping_domains_column",
        dest="mapping_domains_column",
        default=None,
        help="Optional mapping CSV column containing domain labels.",
    )
    parser.add_argument(
        "--mapping_filename_column",
        dest="mapping_filename_column",
        default=None,
        help="Optional mapping CSV column used to derive the prediction label when no label column exists.",
    )
    parser.add_argument(
        "--mapping_label_delimiter",
        dest="mapping_label_delimiter",
        default="_",
        help="Delimiter used when deriving prediction labels from filenames.",
    )
    parser.add_argument(
        "--mapping_label_token_index",
        dest="mapping_label_token_index",
        type=int,
        default=None,
        help="Optional token index used when deriving prediction labels from filenames.",
    )
    parser.add_argument(
        "--image_dir",
        dest="image_dir",
        default=None,
        help="Legacy HCP fallback: path to evaluation images used to infer labels from filenames.",
    )
    parser.add_argument(
        "--ground_truth_fn",
        dest="ground_truth_fn",
        default=None,
        help="Legacy HCP fallback: path to the HCP ground-truth JSON mapping file.",
    )
    parser.add_argument(
        "--task_k",
        dest="task_k",
        type=int,
        default=4,
        help="Recall@K cutoff for task predictions.",
    )
    parser.add_argument(
        "--concept_k",
        dest="concept_k",
        type=int,
        default=4,
        help="Recall@K cutoff for concept predictions.",
    )
    parser.add_argument(
        "--domain_k",
        dest="domain_k",
        type=int,
        default=2,
        help="Recall@K cutoff for domain predictions.",
    )
    parser.add_argument(
        "--output_fn",
        dest="output_fn",
        default=None,
        help="Optional explicit aggregate output CSV path.",
    )
    parser.add_argument(
        "--details_output_fn",
        dest="details_output_fn",
        default=None,
        help="Optional explicit detailed output CSV path.",
    )
    return parser


def main(
    project_dir=None,
    data_dir=None,
    results_dir=None,
    dataset_name="hcp",
    sections=None,
    model_ids=None,
    sources=None,
    categories=None,
    sub_categories=None,
    models=None,
    prediction_dir=None,
    mapping_fn=None,
    mapping_label_column=None,
    mapping_task_column=None,
    mapping_concepts_column=None,
    mapping_domains_column=None,
    mapping_filename_column=None,
    mapping_label_delimiter="_",
    mapping_label_token_index=None,
    image_dir=None,
    ground_truth_fn=None,
    task_k=4,
    concept_k=4,
    domain_k=2,
    output_fn=None,
    details_output_fn=None,
):
    _, data_dir, results_dir = resolve_project_paths(project_dir, data_dir, results_dir)
    sections = list(DEFAULT_SECTIONS) if sections is None else sections
    model_ids = list(DEFAULT_MODEL_IDS) if model_ids is None else model_ids
    sources = ["cogatlasred", "cogatlas"] if sources is None else sources
    categories = ["task"] if categories is None else categories
    sub_categories = ["combined", "names"] if sub_categories is None else sub_categories
    models = ["neurosynth", "gclda", "brainclip"] if models is None else models

    prediction_dir = (
        op.join(results_dir, f"predictions_{dataset_name}") if prediction_dir is None else op.abspath(prediction_dir)
    )
    image_dir = op.join(data_dir, "hcp", "neurovault") if image_dir is None else op.abspath(image_dir)
    ground_truth_fn = (
        op.join(data_dir, "hcp", "ground_truth.json")
        if ground_truth_fn is None
        else op.abspath(ground_truth_fn)
    )
    output_fn = (
        op.join(results_dir, f"eval-{dataset_name}_results.csv") if output_fn is None else op.abspath(output_fn)
    )
    details_output_fn = (
        op.join(results_dir, f"eval-{dataset_name}_details.csv")
        if details_output_fn is None
        else op.abspath(details_output_fn)
    )
    os.makedirs(op.dirname(output_fn), exist_ok=True)
    os.makedirs(op.dirname(details_output_fn), exist_ok=True)

    detailed_rows = []
    aggregate_rows = []

    for section, model_id, source, category, sub_category in itertools.product(
        sections,
        model_ids,
        sources,
        categories,
        sub_categories,
    ):
        reduced = source == "cogatlasred"
        cognitiveatlas = None
        if mapping_fn is not None:
            mapping_preview_df = pd.read_csv(op.abspath(mapping_fn), nrows=1)
            preview_concepts_column = _resolve_column(
                mapping_preview_df,
                mapping_concepts_column,
                ["concepts", "concept", "true_concepts"],
                required=False,
            )
            preview_domains_column = _resolve_column(
                mapping_preview_df,
                mapping_domains_column,
                ["domains", "domain", "true_domains"],
                required=False,
            )
            needs_cognitiveatlas = preview_concepts_column is None or preview_domains_column is None
        else:
            needs_cognitiveatlas = True

        if needs_cognitiveatlas:
            concept_to_process_fn = op.join(data_dir, "cognitive_atlas", "concept_to_process.json")
            cognitiveatlas = build_cognitiveatlas(data_dir, reduced, concept_to_process_fn)
        model_name = model_id.split("/")[-1]
        vocabulary_label = (
            f"vocabulary-{source}_{category}-{sub_category}_embedding-{model_name}_section-{section}"
        )

        if mapping_fn is not None:
            records = _load_mapping_records(
                dataset_name=dataset_name,
                mapping_fn=op.abspath(mapping_fn),
                mapping_label_column=mapping_label_column,
                mapping_task_column=mapping_task_column,
                mapping_concepts_column=mapping_concepts_column,
                mapping_domains_column=mapping_domains_column,
                mapping_filename_column=mapping_filename_column,
                mapping_label_delimiter=mapping_label_delimiter,
                mapping_label_token_index=mapping_label_token_index,
                cognitiveatlas=cognitiveatlas,
            )
        else:
            if not op.exists(ground_truth_fn):
                raise FileNotFoundError(
                    f"Ground-truth file not found at {ground_truth_fn}. Pass --mapping_fn or --ground_truth_fn."
                )
            records = _load_legacy_hcp_records(image_dir=image_dir, ground_truth_fn=ground_truth_fn)

        for backend in models:
            if backend != "brainclip" and sub_category != "names":
                continue

            backend_level_rows = []
            for record in records:
                file_base = f"{record['prediction_label']}_{vocabulary_label}"

                task_path = op.join(prediction_dir, f"{file_base}_pred-task_{backend}.csv")
                if not op.exists(task_path):
                    raise FileNotFoundError(f"Prediction file not found: {task_path}")
                task_pred_df = pd.read_csv(task_path)
                task_preds = task_pred_df["pred"].tolist()
                task_rank = _best_rank(record["task"], task_preds)
                task_recall = _recall_at_n(record["task"], task_preds, task_k)
                task_row = {
                    "dataset": record["dataset"],
                    "prediction_label": record["prediction_label"],
                    "source": source,
                    "category": category,
                    "sub_category": sub_category,
                    "section": section,
                    "model_id": model_id,
                    "model_name": model_name,
                    "vocabulary_label": vocabulary_label,
                    "backend": backend,
                    "level": "task",
                    "k": task_k,
                    "num_true_labels": len(record["task"]),
                    "recall_at_k": task_recall,
                    "hit_at_k": float(task_rank <= task_k),
                    "best_rank": task_rank,
                    "true_labels_json": json.dumps(record["task"]),
                    "top_predictions_json": json.dumps(task_preds),
                }
                detailed_rows.append(task_row)
                backend_level_rows.append(task_row)

                if backend != "brainclip":
                    continue

                level_specs = [
                    ("concept", concept_k, record["concept"], op.join(prediction_dir, f"{file_base}_pred-concept_{backend}.csv")),
                    ("domain", domain_k, record["domain"], op.join(prediction_dir, f"{file_base}_pred-process_{backend}.csv")),
                ]
                for level_name, k_value, true_labels, pred_path in level_specs:
                    if not op.exists(pred_path):
                        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
                    pred_df = pd.read_csv(pred_path)
                    pred_labels = pred_df["pred"].tolist()
                    best_rank = _best_rank(true_labels, pred_labels)
                    recall = _recall_at_n(true_labels, pred_labels, k_value)
                    level_row = {
                        "dataset": record["dataset"],
                        "prediction_label": record["prediction_label"],
                        "source": source,
                        "category": category,
                        "sub_category": sub_category,
                        "section": section,
                        "model_id": model_id,
                        "model_name": model_name,
                        "vocabulary_label": vocabulary_label,
                        "backend": backend,
                        "level": level_name,
                        "k": k_value,
                        "num_true_labels": len(true_labels),
                        "recall_at_k": recall,
                        "hit_at_k": float(best_rank <= k_value),
                        "best_rank": best_rank,
                        "true_labels_json": json.dumps(true_labels),
                        "top_predictions_json": json.dumps(pred_labels),
                    }
                    detailed_rows.append(level_row)
                    backend_level_rows.append(level_row)

            backend_df = pd.DataFrame(backend_level_rows)
            for level_name, level_df in backend_df.groupby("level", sort=False):
                finite_ranks = level_df["best_rank"].replace(np.inf, np.nan)
                aggregate_rows.append(
                    {
                        "dataset": dataset_name,
                        "source": source,
                        "category": category,
                        "sub_category": sub_category,
                        "section": section,
                        "model_id": model_id,
                        "model_name": model_name,
                        "vocabulary_label": vocabulary_label,
                        "backend": backend,
                        "level": level_name,
                        "k": int(level_df["k"].iloc[0]),
                        "n_images": int(len(level_df)),
                        "mean_recall_at_k": float(level_df["recall_at_k"].mean()),
                        "mean_hit_at_k": float(level_df["hit_at_k"].mean()),
                        "median_best_rank": float(np.nanmedian(finite_ranks)),
                    }
                )

    pd.DataFrame(aggregate_rows).to_csv(output_fn, index=False)
    pd.DataFrame(detailed_rows).to_csv(details_output_fn, index=False)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
