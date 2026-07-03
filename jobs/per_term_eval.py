"""Compute per-term decoding performance with a permutation null baseline."""

import argparse
import json
import os
import os.path as op

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


GROUP_COLUMNS = [
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
]


def _parse_json_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return list(json.loads(value))


def _term_rank(term, predictions):
    for rank, pred in enumerate(predictions, start=1):
        if pred == term:
            return rank
    return np.inf


def _make_term_observations(details_df):
    rows = []
    for row in details_df.itertuples(index=False):
        true_labels = _parse_json_list(row.true_labels_json)
        predictions = _parse_json_list(row.top_predictions_json)
        for term in true_labels:
            rank = _term_rank(term, predictions)
            rows.append(
                {
                    "dataset": row.dataset,
                    "source": row.source,
                    "category": row.category,
                    "sub_category": row.sub_category,
                    "section": row.section,
                    "model_id": row.model_id,
                    "model_name": row.model_name,
                    "vocabulary_label": row.vocabulary_label,
                    "backend": row.backend,
                    "level": row.level,
                    "k": row.k,
                    "term": term,
                    "rank": rank,
                    "hit_at_k": float(rank <= row.k),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_term_rows(term_df):
    grouped = term_df.groupby(GROUP_COLUMNS + ["term"], dropna=False, sort=False)
    return (
        grouped.agg(
            n_images=("term", "size"),
            mean_hit_at_k=("hit_at_k", "mean"),
            mean_rank=("rank", lambda x: np.nanmean(np.where(np.isfinite(x), x, np.nan))),
            median_rank=("rank", lambda x: np.nanmedian(np.where(np.isfinite(x), x, np.nan))),
            mean_reciprocal_rank=("rank", lambda x: np.mean(np.where(np.isfinite(x), 1.0 / x, 0.0))),
        )
        .reset_index()
    )


def _permutation_null(details_df, actual_df, n_permutations, random_seed):
    rng = np.random.default_rng(random_seed)
    actual_keyed = actual_df.set_index(GROUP_COLUMNS + ["term"])
    accum = {}

    grouped = details_df.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    for group_key, group_df in tqdm(grouped, total=grouped.ngroups, desc="permutation groups"):
        group_rows = group_df.copy()
        prediction_lists = [
            _parse_json_list(value) for value in group_rows["top_predictions_json"].tolist()
        ]
        true_lists = [_parse_json_list(value) for value in group_rows["true_labels_json"].tolist()]
        k_value = int(group_rows["k"].iloc[0])
        term_counts = {}
        for true_terms in true_lists:
            for term in true_terms:
                term_counts[term] = term_counts.get(term, 0) + 1

        observed_hits = {
            term: actual_keyed.loc[group_key + (term,), "mean_hit_at_k"] for term in term_counts
        }

        sum_means = {term: 0.0 for term in term_counts}
        sum_sq_means = {term: 0.0 for term in term_counts}
        ge_counts = {term: 0 for term in term_counts}

        for _ in range(n_permutations):
            permuted_indices = rng.permutation(len(prediction_lists))
            hit_sums = {term: 0.0 for term in term_counts}

            for row_idx, true_terms in enumerate(true_lists):
                permuted_predictions = prediction_lists[permuted_indices[row_idx]]
                topk_predictions = set(permuted_predictions[:k_value])
                for term in true_terms:
                    hit_sums[term] += float(term in topk_predictions)

            for term, count in term_counts.items():
                perm_mean = hit_sums[term] / count
                sum_means[term] += perm_mean
                sum_sq_means[term] += perm_mean**2
                ge_counts[term] += int(perm_mean >= observed_hits[term])

        for term, count in term_counts.items():
            key = group_key + (term,)
            null_mean = sum_means[term] / n_permutations
            variance = max(sum_sq_means[term] / n_permutations - null_mean**2, 0.0)
            accum[key] = {
                "null_mean_hit_at_k": null_mean,
                "null_std_hit_at_k": float(np.sqrt(variance)),
                "empirical_p_value": (ge_counts[term] + 1) / (n_permutations + 1),
                "n_null_permutations": n_permutations,
            }

    return pd.DataFrame(
        [
            dict(zip(GROUP_COLUMNS + ["term"], key), **value)
            for key, value in accum.items()
        ]
    )


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate detailed decoding evaluations into per-term scores with a null baseline."
    )
    parser.add_argument(
        "--details_fns",
        dest="details_fns",
        nargs="+",
        required=True,
        help="One or more detailed evaluation CSVs emitted by jobs/decoding_eval.py.",
    )
    parser.add_argument(
        "--levels",
        dest="levels",
        nargs="+",
        default=["task", "concept", "domain"],
        help="Prediction levels to analyze.",
    )
    parser.add_argument(
        "--backends",
        dest="backends",
        nargs="+",
        default=None,
        help="Optional backend filter, e.g. brainclip or gclda.",
    )
    parser.add_argument(
        "--n_permutations",
        dest="n_permutations",
        type=int,
        default=1000,
        help="Number of within-group permutations for the null baseline.",
    )
    parser.add_argument(
        "--random_seed",
        dest="random_seed",
        type=int,
        default=0,
        help="Random seed for permutation generation.",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        type=float,
        default=0.05,
        help="Significance threshold used for the above-chance summary flag.",
    )
    parser.add_argument(
        "--output_fn",
        dest="output_fn",
        required=True,
        help="Path to the per-term output CSV.",
    )
    parser.add_argument(
        "--summary_output_fn",
        dest="summary_output_fn",
        default=None,
        help="Optional path to a group-level summary CSV.",
    )
    return parser


def main(
    details_fns,
    levels=None,
    backends=None,
    n_permutations=1000,
    random_seed=0,
    alpha=0.05,
    output_fn=None,
    summary_output_fn=None,
):
    levels = ["task", "concept", "domain"] if levels is None else levels
    details_df = pd.concat([pd.read_csv(path) for path in details_fns], ignore_index=True)
    details_df = details_df.loc[details_df["level"].isin(levels)].reset_index(drop=True)
    if backends is not None:
        details_df = details_df.loc[details_df["backend"].isin(backends)].reset_index(drop=True)

    term_df = _make_term_observations(details_df)
    actual_df = _aggregate_term_rows(term_df)
    null_df = _permutation_null(details_df, actual_df, n_permutations, random_seed)
    result_df = actual_df.merge(null_df, on=GROUP_COLUMNS + ["term"], how="left")
    result_df["normalized_hit_at_k"] = result_df["mean_hit_at_k"] / result_df["null_mean_hit_at_k"].replace(
        0,
        np.nan,
    )
    result_df["above_chance"] = (
        (result_df["mean_hit_at_k"] > result_df["null_mean_hit_at_k"])
        & (result_df["empirical_p_value"] <= alpha)
    )

    os.makedirs(op.dirname(op.abspath(output_fn)), exist_ok=True)
    result_df.to_csv(output_fn, index=False)

    if summary_output_fn is not None:
        summary_df = (
            result_df.groupby(GROUP_COLUMNS, dropna=False, sort=False)
            .agg(
                n_terms=("term", "size"),
                n_terms_above_chance=("above_chance", "sum"),
                mean_term_hit_at_k=("mean_hit_at_k", "mean"),
                mean_term_null_hit_at_k=("null_mean_hit_at_k", "mean"),
            )
            .reset_index()
        )
        os.makedirs(op.dirname(op.abspath(summary_output_fn)), exist_ok=True)
        summary_df.to_csv(summary_output_fn, index=False)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
