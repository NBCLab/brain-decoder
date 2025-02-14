import argparse
import json
import os.path as op

import nimare
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
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


def count_top_n_appearances(similarity_matrix, n_top_docs=10):
    """
    Count how many times each task appears in the top N most similar documents.

    Parameters:
    similarity_matrix (numpy.ndarray): Matrix of shape (n_tasks, n_docs) containing similarity
    n (int): Number of top documents to consider (default: 10)

    Returns:
    numpy.ndarray: Array containing the count of appearances in top N for each task
    """
    # Get the indices of top N similar documents for each task
    top_n_indices = np.argsort(-similarity_matrix, axis=0)[:n_top_docs, :]

    # Initialize array to store counts
    task_counts = np.zeros(similarity_matrix.shape[0], dtype=int)

    # Count appearances in top N for each task
    for i in range(similarity_matrix.shape[0]):
        # Get documents where this task appears in their top N
        task_counts[i] = (top_n_indices == i).sum()

    return task_counts, top_n_indices


def _get_prior_prob_old(doc_emb, emb, temperature=10, n_top_docs=10):
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    doc_emb = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8)

    names_doc_similarity = emb @ doc_emb.T
    # mean_similarity = np.mean(names_doc_similarity, axis=1)
    prior = np.mean(softmax(names_doc_similarity * temperature, axis=0), axis=1)

    # prior = softmax(mean_similarity)
    names_counts, top_n_indices = count_top_n_appearances(
        names_doc_similarity,
        n_top_docs=n_top_docs,
    )

    # return names_counts, top_n_indices, names_counts / doc_emb.shape[0]
    return prior, top_n_indices


def main(project_dir):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    voc_dir = op.join(data_dir, "vocabulary")
    text_dir = op.join(data_dir, "text")
    source = "cogatlas"  # cogatlas, neurosynth
    section = "body"  # abstract, body
    n_top_docs = 10
    model_id = "BrainGPT/BrainGPT-7B-v0.2"
    model_name = model_id.split("/")[-1]

    dset = nimare.dataset.Dataset.load(op.join(data_dir, "dset-pubmed_nimare.pkl"))
    pmids = dset.texts["study_id"].values
    doc_emb_fn = op.join(text_dir, f"text-raw_section-{section}_embedding-{model_name}.npy")
    doc_emb = np.load(doc_emb_fn)
    # TODO: Use the training indices to slice the docuemnt and calculate the prior
    #       only on the training set. Add section to the file name

    for category in ["task", "concept"]:
        for sub_category in ["names", "definitions", "combined"]:
            lb = f"vocabulary-{source}_{category}-{sub_category}_embedding-{model_name}"
            emb_fn = op.join(voc_dir, f"{lb}.npy")
            names_fn = op.join(voc_dir, f"{lb}.txt")
            prior_fn = op.join(voc_dir, f"{lb}_section-{section}_prior.npy")
            top_fn = op.join(voc_dir, f"{lb}_section-{section}_top.npy")
            pmid_to_task_fn = op.join(voc_dir, f"{lb}_section-{section}_top.json")

            names, emb = _read_vocabulary(names_fn, emb_fn)

            prior_prob, topindx = _get_prior_prob_old(doc_emb, emb, n_top_docs=n_top_docs)
            np.save(prior_fn, prior_prob)
            np.save(top_fn, topindx)

            names = np.array(names)
            df = pd.DataFrame({"name": names, "prior": prior_prob})
            df = df.sort_values("prior", ascending=False)
            df.to_csv(prior_fn.replace(".npy", ".csv"), index=False)

            pmid_to_task = {}
            for study_i in range(pmids.shape[0]):
                pmid = pmids[study_i]
                pmid_to_task[pmid] = list(names[topindx[:, study_i]])

            with open(pmid_to_task_fn, "w") as f:
                json.dump(pmid_to_task, f, indent=4)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
