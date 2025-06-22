import argparse
import itertools
import os
import os.path as op

import numpy as np
from nimare import decode
from nimare.annotate.gclda import GCLDAModel
from nimare.dataset import Dataset
from nimare.meta.cbma.mkda import MKDAChi2


def _get_parser():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--category",
        dest="category",
        default="task",
        help="Category of annotations to use for training",
    )
    parser.add_argument(
        "--section",
        dest="section",
        default="abstract",
        help="Section of annotations to use for training",
    )
    parser.add_argument(
        "--baseline",
        dest="baseline",
        default="gclda",
        help="Baseline model to train (neurosynth, gclda)",
    )
    parser.add_argument(
        "--model_id",
        dest="model_id",
        default="BrainGPT/BrainGPT-7B-v0.2",
        help="Model to selecte indices of the training set",
    )
    parser.add_argument(
        "--reduced",
        dest="reduced",
        default=True,
        help="Use reduced vocabulary",
    )
    return parser


def _get_counts(dset, feature_group):
    annotations_df = dset.annotations

    if not feature_group.endswith("__"):
        feature_group += "__"
    feature_names = annotations_df.columns.values
    feature_names = [f for f in feature_names if f.startswith(feature_group)]

    counts_df = annotations_df.copy()
    counts_df = counts_df.set_index("id")
    counts_df = counts_df[feature_names]

    # Remove feature group prefix
    vocabulary_names = [f.replace(feature_group, "") for f in feature_names]
    counts_df.columns = vocabulary_names

    return counts_df


def main(
    project_dir,
    category="task",
    section="body",
    baseline="gclda",
    model_id="BrainGPT/BrainGPT-7B-v0.2",
    reduced=False,
):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results", "baseline")
    braindec_dir = op.join(project_dir, "results", "pubmed")
    source = "cogatlasred" if reduced else "cogatlas"

    model_name = model_id.split("/")[-1]
    os.makedirs(results_dir, exist_ok=True)
    n_cores = -1

    dset = Dataset.load(op.join(data_dir, f"dset-pubmed_{source}-annotated_nimare.pkl"))
    indices_fn = op.join(
        braindec_dir,
        f"model-clip_section-{section}_embedding-{model_name}_best-indices.npz",
    )
    indices_dict = np.load(indices_fn)
    train_indices = indices_dict["train"]
    sel_ids = dset.ids[train_indices]
    dset = dset.slice(sel_ids)

    print(f"Training {baseline} model for {category} in {section}")

    model_fn = op.join(
        results_dir,
        f"model-{baseline}_{source}-{category}_embedding-{model_name}_section-{section}.pkl",
    )

    if baseline == "neurosynth":
        feature_group = f"{source}-{category}_section-{section}_annot-tfidf"
        frequency_threshold = 0.001
        decoder = decode.CorrelationDecoder(
            frequency_threshold=frequency_threshold,
            meta_estimator=MKDAChi2,
            feature_group=feature_group,
            target_image="z_desc-association",
            n_cores=n_cores,
        )
        decoder.fit(dset)
        decoder.save(model_fn)

    elif baseline == "gclda":
        feature_group = f"{source}-{category}_section-{section}_annot-counts"

        counts_df = _get_counts(dset, feature_group)
        model = GCLDAModel(
            counts_df,
            dset.coordinates,
            mask=dset.masker.mask_img,
            n_topics=25,
            n_regions=4,
            symmetric=True,
        )
        model.fit(n_iters=1000, loglikely_freq=10)
        model.save(model_fn)

    else:
        raise ValueError(f"Invalid model name: {baseline}")


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    # main(**kwargs)

    reduced = False
    categories = ["task"]  # "concept"
    model_ids = [
        "BrainGPT/BrainGPT-7B-v0.2",
        "mistralai/Mistral-7B-v0.1",
        "BrainGPT/BrainGPT-7B-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
    ]
    sections = ["body", "abstract"]
    baselines = ["neurosynth", "gclda"]
    for category, section, baseline, model_id in itertools.product(
        categories, sections, baselines, model_ids
    ):
        print(f"Training {baseline} model for {category} in {section} with {model_id}")
        main(
            kwargs["project_dir"],
            category=category,
            section=section,
            baseline=baseline,
            model_id=model_id,
            reduced=reduced,
        )


if __name__ == "__main__":
    _main()
