"""Audit emotion-task coverage and summarize emotion-map decoding performance."""

import argparse
import json
import os
import os.path as op

import pandas as pd


EMOTION_KEYWORDS = [
    "emotion",
    "emotional",
    "fear",
    "anger",
    "disgust",
    "happiness",
    "happy",
    "sad",
    "face",
]


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Summarize fine-grained emotion coverage and decoding on available benchmark maps."
    )
    parser.add_argument("--ibc_mapping_fn", dest="ibc_mapping_fn", required=True)
    parser.add_argument("--details_fn", dest="details_fn", required=True)
    parser.add_argument("--full_vocab_fn", dest="full_vocab_fn", required=True)
    parser.add_argument("--reduced_vocab_fn", dest="reduced_vocab_fn", required=True)
    parser.add_argument("--concept_snapshot_fn", dest="concept_snapshot_fn", required=True)
    parser.add_argument("--coverage_fn", dest="coverage_fn", required=True)
    parser.add_argument("--summary_fn", dest="summary_fn", required=True)
    return parser


def _contains_emotion(text):
    text = str(text).lower()
    return any(keyword in text for keyword in EMOTION_KEYWORDS)


def _parse_json_list(value):
    if pd.isna(value):
        return []
    return list(json.loads(value))


def _topk_emotion_hits(top_predictions_json):
    predictions = _parse_json_list(top_predictions_json)
    return [pred for pred in predictions if _contains_emotion(pred)]


def main(
    ibc_mapping_fn,
    details_fn,
    full_vocab_fn,
    reduced_vocab_fn,
    concept_snapshot_fn,
    coverage_fn,
    summary_fn,
):
    ibc_mapping_df = pd.read_csv(op.abspath(ibc_mapping_fn))
    details_df = pd.read_csv(op.abspath(details_fn))

    emotion_maps_df = ibc_mapping_df.loc[
        ibc_mapping_df["task_family"].astype(str).str.contains("emotion|emotional", case=False, na=False)
        | ibc_mapping_df["task_name"].astype(str).map(_contains_emotion)
        | ibc_mapping_df["contrast_definition"].astype(str).map(_contains_emotion)
    ].copy()
    emotion_labels = set(emotion_maps_df["prediction_label"].astype(str))

    with open(op.abspath(full_vocab_fn), "r") as file_obj:
        full_vocab = [line.strip() for line in file_obj]
    with open(op.abspath(reduced_vocab_fn), "r") as file_obj:
        reduced_vocab = [line.strip() for line in file_obj]
    with open(op.abspath(concept_snapshot_fn), "r") as file_obj:
        concepts = json.load(file_obj)

    concept_names = [
        concept.get("name", "")
        for concept in concepts
        if concept.get("name") and concept.get("definition_text")
    ]
    coverage_rows = []
    for source_name, terms in [
        ("full_task_vocabulary", full_vocab),
        ("reduced_task_vocabulary", reduced_vocab),
        ("concept_snapshot", concept_names),
    ]:
        hits = [term for term in terms if _contains_emotion(term)]
        for term in hits:
            coverage_rows.append({"source": source_name, "term": term})

    coverage_df = pd.DataFrame(coverage_rows)
    os.makedirs(op.dirname(op.abspath(coverage_fn)), exist_ok=True)
    coverage_df.to_csv(coverage_fn, index=False)

    emotion_details_df = details_df.loc[
        (details_df["dataset"].astype(str).isin(["ibc", "ibc_full"]))
        & (details_df["backend"] == "brainclip")
        & (details_df["prediction_label"].astype(str).isin(emotion_labels))
    ].copy()
    emotion_details_df["n_emotion_predictions_top20"] = emotion_details_df["top_predictions_json"].map(
        lambda value: len(_topk_emotion_hits(value))
    )
    emotion_details_df["top_emotion_predictions_json"] = emotion_details_df["top_predictions_json"].map(
        lambda value: json.dumps(_topk_emotion_hits(value))
    )
    for column in ["recall_at_k", "hit_at_k", "best_rank", "n_emotion_predictions_top20"]:
        emotion_details_df[column] = pd.to_numeric(emotion_details_df[column], errors="coerce")

    grouped = emotion_details_df.groupby(
        ["sub_category", "level"],
        dropna=False,
        sort=False,
    )
    summary_df = (
        grouped.agg(
            n_images=("prediction_label", "size"),
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_hit_at_k=("hit_at_k", "mean"),
            median_best_rank=("best_rank", "median"),
            mean_n_emotion_predictions_top20=("n_emotion_predictions_top20", "mean"),
        )
        .reset_index()
    )
    summary_df.insert(0, "n_emotion_maps", len(emotion_labels))
    summary_df.insert(
        1,
        "ground_truth_granularity_note",
        "Available IBC emotion maps are coarse face/shape or emotional localizer contrasts rather than specific emotion labels.",
    )
    os.makedirs(op.dirname(op.abspath(summary_fn)), exist_ok=True)
    summary_df.to_csv(summary_fn, index=False)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
