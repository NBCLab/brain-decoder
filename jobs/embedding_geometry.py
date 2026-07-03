"""Project task and HCP image embeddings into 2D and summarize latent-space structure."""

import argparse
import os
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from braindec.embedding import ImageEmbedding
from braindec.model import build_model
from jobs.utils import build_cognitiveatlas, get_model_name, resolve_project_paths


DEFAULT_MODEL_IDS = [
    "BrainGPT/BrainGPT-7B-v0.2",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b-chat-hf",
]


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Inspect shared text-image embedding geometry on HCP representative maps."
    )
    parser.add_argument("--project_dir", dest="project_dir", default=None)
    parser.add_argument("--data_dir", dest="data_dir", default=None)
    parser.add_argument("--results_dir", dest="results_dir", default=None)
    parser.add_argument(
        "--hcp_mapping_fn",
        dest="hcp_mapping_fn",
        default=None,
        help="Benchmark mapping CSV with representative HCP maps.",
    )
    parser.add_argument(
        "--model_ids",
        dest="model_ids",
        nargs="+",
        default=list(DEFAULT_MODEL_IDS),
        help="Models to compare.",
    )
    parser.add_argument(
        "--section",
        dest="section",
        default="body",
        help="Model section to use.",
    )
    parser.add_argument(
        "--source",
        dest="source",
        default="cogatlasred",
        help="Vocabulary source to use.",
    )
    parser.add_argument("--coords_fn", dest="coords_fn", required=True)
    parser.add_argument("--distance_fn", dest="distance_fn", required=True)
    parser.add_argument("--plot_fn", dest="plot_fn", required=True)
    parser.add_argument("--device", dest="device", default=None)
    return parser


def _normalize_rows(array):
    norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-8
    return array / norms


def _first_domain(atlas, task_name):
    task_idx = atlas.get_task_idx_from_names(task_name)
    process_idxs = atlas.task_to_process_idxs[task_idx]
    if len(process_idxs) == 0:
        return None
    return atlas.get_process_names_from_idx(process_idxs)[0]


def main(
    project_dir=None,
    data_dir=None,
    results_dir=None,
    hcp_mapping_fn=None,
    model_ids=None,
    section="body",
    source="cogatlasred",
    coords_fn=None,
    distance_fn=None,
    plot_fn=None,
    device=None,
):
    _, data_dir, results_dir = resolve_project_paths(project_dir, data_dir, results_dir)
    hcp_mapping_fn = (
        op.join(".runs", "hcp_benchmark_best", "data", "hcp", "benchmark_mapping.csv")
        if hcp_mapping_fn is None
        else op.abspath(hcp_mapping_fn)
    )
    hcp_mapping_fn = op.abspath(hcp_mapping_fn)
    atlas = build_cognitiveatlas(data_dir, reduced=(source == "cogatlasred"))
    hcp_df = pd.read_csv(hcp_mapping_fn)

    image_emb_gene = ImageEmbedding(standardize=False, nilearn_dir=op.join(data_dir, "nilearn"))
    hcp_images = [nib.load(path) for path in hcp_df["image_path"].tolist()]
    image_inputs = image_emb_gene(hcp_images)
    image_inputs = _normalize_rows(image_inputs)

    coords_rows = []
    distance_rows = []
    for model_id in model_ids:
        model_name = get_model_name(model_id)
        model_path = op.join(
            results_dir,
            "pubmed",
            f"model-clip_section-{section}_embedding-{model_name}_best.pth",
        )
        vocab_emb = np.load(
            op.join(
                data_dir,
                "vocabulary",
                f"vocabulary-{source}_task-combined_embedding-{model_name}.npy",
            )
        )
        with open(op.join(data_dir, "vocabulary", f"vocabulary-{source}_task.txt"), "r") as file_obj:
            task_vocab = [line.strip() for line in file_obj]

        model = build_model(model_path, device=device or "cpu")
        with torch.no_grad():
            text_features = (
                model.encode_text(torch.from_numpy(vocab_emb).float().to(model.device))
                .cpu()
                .numpy()
            )
            image_features = (
                model.encode_image(torch.from_numpy(image_inputs).float().to(model.device))
                .cpu()
                .numpy()
            )
        text_features = _normalize_rows(text_features)
        image_features = _normalize_rows(image_features)

        domain_labels = [_first_domain(atlas, task_name) for task_name in task_vocab]
        combined = np.vstack([text_features, image_features])
        pca = PCA(n_components=min(20, combined.shape[1], combined.shape[0] - 1), random_state=0)
        reduced = pca.fit_transform(combined)
        perplexity = max(2, min(15, combined.shape[0] - 1))
        coords = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=0,
        ).fit_transform(reduced)

        text_coords = coords[: len(task_vocab)]
        image_coords = coords[len(task_vocab) :]
        for idx, task_name in enumerate(task_vocab):
            coords_rows.append(
                {
                    "model_name": model_name,
                    "point_type": "task_text",
                    "label": task_name,
                    "group": domain_labels[idx],
                    "x": float(text_coords[idx, 0]),
                    "y": float(text_coords[idx, 1]),
                }
            )
        for idx, row in hcp_df.iterrows():
            coords_rows.append(
                {
                    "model_name": model_name,
                    "point_type": "hcp_image",
                    "label": row["task_name"],
                    "group": row["domain_key"],
                    "x": float(image_coords[idx, 0]),
                    "y": float(image_coords[idx, 1]),
                }
            )

        task_to_feature = {task_name: text_features[idx] for idx, task_name in enumerate(task_vocab)}
        cosine = text_features @ text_features.T
        same_domain = []
        different_domain = []
        for i in range(len(task_vocab)):
            for j in range(i + 1, len(task_vocab)):
                if domain_labels[i] is None or domain_labels[j] is None:
                    continue
                if domain_labels[i] == domain_labels[j]:
                    same_domain.append(cosine[i, j])
                else:
                    different_domain.append(cosine[i, j])
        matched_cosines = []
        for idx, row in hcp_df.iterrows():
            if row["task_name"] not in task_to_feature:
                continue
            matched_cosines.append(float(np.dot(image_features[idx], task_to_feature[row["task_name"]])))
        distance_rows.extend(
            [
                {
                    "model_name": model_name,
                    "metric": "mean_within_domain_task_cosine",
                    "value": float(np.mean(same_domain)),
                },
                {
                    "model_name": model_name,
                    "metric": "mean_between_domain_task_cosine",
                    "value": float(np.mean(different_domain)),
                },
                {
                    "model_name": model_name,
                    "metric": "mean_hcp_image_to_matched_task_cosine",
                    "value": float(np.mean(matched_cosines)),
                },
            ]
        )

    coords_df = pd.DataFrame(coords_rows)
    distance_df = pd.DataFrame(distance_rows)
    os.makedirs(op.dirname(op.abspath(coords_fn)), exist_ok=True)
    coords_df.to_csv(coords_fn, index=False)
    distance_df.to_csv(distance_fn, index=False)

    model_names = [get_model_name(model_id) for model_id in model_ids]
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5), squeeze=False)
    for ax, model_name in zip(axes[0], model_names):
        sub = coords_df.loc[coords_df["model_name"] == model_name]
        text_sub = sub.loc[sub["point_type"] == "task_text"]
        image_sub = sub.loc[sub["point_type"] == "hcp_image"]
        ax.scatter(text_sub["x"], text_sub["y"], s=12, alpha=0.35, label="Task text")
        ax.scatter(image_sub["x"], image_sub["y"], s=60, marker="x", label="HCP image")
        for row in image_sub.itertuples(index=False):
            ax.text(row.x, row.y, row.label, fontsize=7)
        ax.set_title(model_name)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(op.dirname(op.abspath(plot_fn)), exist_ok=True)
    fig.savefig(plot_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
