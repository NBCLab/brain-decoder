import argparse
import itertools
import os.path as op

import numpy as np
import pandas as pd
from scipy.special import softmax
from utils import _read_vocabulary


def _get_parser():
    parser = argparse.ArgumentParser(description="Run gradient-decoding workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    return parser


def _get_prior_prob_old(doc_emb, emb, temperature=10, n_top_docs=10):
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    doc_emb = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8)

    names_doc_similarity = emb @ doc_emb.T

    return np.mean(softmax(names_doc_similarity * temperature, axis=0), axis=1)


def main(project_dir):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    reduced = False
    voc_dir = op.join(data_dir, "vocabulary")
    source = "cogatlasred" if reduced else "cogatlas"
    text_dir = op.join(data_dir, "text")
    braindec_dir = op.join(project_dir, "results", "pubmed")
    categories = ["task"]  # "concept"
    sub_categories = ["names", "definitions", "combined"]
    n_top_docs = 10

    model_ids = [
        "BrainGPT/BrainGPT-7B-v0.2",
        "mistralai/Mistral-7B-v0.1",
        "BrainGPT/BrainGPT-7B-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
    ]
    sections = ["body", "abstract"]
    for section, model_id in itertools.product(sections, model_ids):
        model_name = model_id.split("/")[-1]

        doc_emb_fn = op.join(text_dir, f"text-raw_section-{section}_embedding-{model_name}.npy")
        doc_emb = np.load(doc_emb_fn)

        # Only calculate priors on the training set
        indices_fn = op.join(
            braindec_dir,
            f"model-clip_section-{section}_embedding-{model_name}_best-indices.npz",
        )
        indices_dict = np.load(indices_fn)
        train_indices = indices_dict["train"]
        doc_emb = doc_emb[train_indices]

        for category in categories:
            names_fn = op.join(voc_dir, f"vocabulary-{source}_{category}.txt")

            for sub_category in sub_categories:
                lb = f"vocabulary-{source}_{category}-{sub_category}_embedding-{model_name}"
                emb_fn = op.join(voc_dir, f"{lb}.npy")
                prior_fn = op.join(voc_dir, f"{lb}_section-{section}_prior.npy")

                names, emb = _read_vocabulary(names_fn, emb_fn)

                prior_prob = _get_prior_prob_old(doc_emb, emb, n_top_docs=n_top_docs)
                np.save(prior_fn, prior_prob)

                df = pd.DataFrame({"name": names, "prior": prior_prob})
                df = df.sort_values("prior", ascending=False)
                df.to_csv(prior_fn.replace(".npy", ".csv"), index=False)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
