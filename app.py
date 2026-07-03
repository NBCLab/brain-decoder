"""Gradio Space: upload a NIfTI brain activation map, get NiCLIP task predictions."""

import logging
import os.path as op
import time

import numpy as np
from nilearn.image import new_img_like

import gradio as gr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("niclip-space")

MODEL_NAME = "BrainGPT-7B-v0.2"
SECTION = "body"
SOURCE = "cogatlasred"

_RESOURCES = {}


def load_resources():
    """Download assets and build the model/embedder once, warming the DiFuMo atlas cache."""
    t0 = time.perf_counter()
    from braindec.fetcher import download_bundle, get_data_dir

    work_dir = get_data_dir()
    download_bundle("example_prediction", destination_root=work_dir)
    logger.info("bundle ready in %.1fs", time.perf_counter() - t0)

    data_dir = op.join(work_dir, "data")
    results_dir = op.join(work_dir, "results")
    voc_dir = op.join(data_dir, "vocabulary")
    voc_label = f"vocabulary-{SOURCE}_task-combined_embedding-{MODEL_NAME}"

    model_fn = op.join(
        results_dir, "pubmed", f"model-clip_section-{SECTION}_embedding-{MODEL_NAME}_best.pth"
    )
    vocabulary_fn = op.join(voc_dir, f"vocabulary-{SOURCE}_task.txt")
    vocabulary_emb_fn = op.join(voc_dir, f"{voc_label}.npy")
    vocabulary_prior_fn = op.join(voc_dir, f"{voc_label}_section-{SECTION}_prior.npy")

    from braindec.model import build_model
    from braindec.utils import _get_device

    device = _get_device()
    t1 = time.perf_counter()
    try:
        model = build_model(model_fn, device=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load CLIP model from {model_fn}") from exc
    logger.info("model built in %.1fs (device=%s)", time.perf_counter() - t1, device)

    from braindec.embedding import ImageEmbedding

    image_embedder = ImageEmbedding(
        standardize=False, nilearn_dir=op.join(data_dir, "nilearn"), space="MNI152"
    )

    with open(vocabulary_fn) as fh:
        vocabulary = [line.strip() for line in fh]
    vocabulary_emb = np.load(vocabulary_emb_fn)
    vocabulary_prior = np.load(vocabulary_prior_fn)

    t2 = time.perf_counter()
    dummy = new_img_like(
        image_embedder.atlas_maps,
        np.zeros(image_embedder.atlas_maps.shape[:3], dtype=np.float32),
    )
    image_embedder(dummy)
    logger.info("image_embedder warmup in %.1fs", time.perf_counter() - t2)

    _RESOURCES.update(
        model=model,
        device=device,
        image_embedder=image_embedder,
        vocabulary=vocabulary,
        vocabulary_emb=vocabulary_emb,
        vocabulary_prior=vocabulary_prior,
        data_dir=data_dir,
    )
    logger.info("TOTAL startup: %.1fs", time.perf_counter() - t0)


def predict(nifti_path, topk):
    if nifti_path is None:
        raise gr.Error("Please upload a .nii or .nii.gz file.")
    lower = nifti_path.lower()
    if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
        raise gr.Error("File must be a NIfTI image (.nii or .nii.gz).")

    from braindec.predict import image_to_labels

    r = _RESOURCES
    try:
        task_df = image_to_labels(
            nifti_path,
            model_path=None,
            vocabulary=r["vocabulary"],
            vocabulary_emb=r["vocabulary_emb"],
            prior_probability=r["vocabulary_prior"],
            topk=int(topk),
            logit_scale=20.0,
            model=r["model"],
            image_emb_gene=r["image_embedder"],
            data_dir=r["data_dir"],
            device=r["device"],
        )
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("prediction failed")
        raise gr.Error(f"Decoding failed: {exc}") from exc

    return task_df[["pred", "prob", "bayes_factor"]].round(4)


def build_ui():
    with gr.Blocks(title="NiCLIP: Brain Activation Decoder") as demo:
        gr.Markdown(
            "# NiCLIP — Functional Brain Decoding\n"
            "Upload a NIfTI brain activation map (group- or subject-level "
            "z-stat/t-stat) to predict the cognitive tasks most associated with it. "
            "[Paper](https://doi.org/10.1101/2025.06.14.659706)"
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label="Activation map (.nii / .nii.gz)", type="filepath")
                topk = gr.Slider(3, 20, value=10, step=1, label="Top-k tasks")
                run_btn = gr.Button("Decode", variant="primary")
            with gr.Column(scale=2):
                task_out = gr.Dataframe(label="Predicted tasks  P(T|A)")

        run_btn.click(
            fn=predict,
            inputs=[file_in, topk],
            outputs=[task_out],
            concurrency_limit=1,
        )
    return demo


def main():
    load_resources()
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch()


if __name__ == "__main__":
    main()
