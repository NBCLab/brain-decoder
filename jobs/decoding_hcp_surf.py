import os
import os.path as op
from glob import glob

import numpy as np
from neuromaps.datasets import fetch_atlas
from nibabel.gifti import GiftiDataArray
from nilearn import datasets
from nilearn.surface import PolyMesh, SurfaceImage, load_surf_mesh
from nimare.decode.continuous import CorrelationDecoder
from nimare.extract import download_nidm_pain
from nimare.utils import get_resource_path
from utils import _read_vocabulary

from braindec.plot import plot_surf
from braindec.predict import image_to_labels
from braindec.utils import _zero_medial_wall

# TASKS = ["WM", "GAMBLING", "MOTOR", "LANGUAGE", "SOCIAL", "RELATIONAL", "EMOTION"]
TASKS = ["WM"]
CONTRASTS = {
    "WM": ["2BK-0BK"],
    # "GAMBLING": ["REWARD-PUNISH"],
    # "MOTOR": ["AVG"],
    # "LANGUAGE": ["STORY-MATH"],
    # "SOCIAL": ["TOM-RANDOM"],
    # "RELATIONAL": ["REL"],
    # "EMOTION": ["FACES-SHAPES"],
}


def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    voc_dir = op.join(data_dir, "vocabulary")
    results_dir = op.join(project_dir, "results")
    tasks_dir = op.join(data_dir, "hcp")
    section = "body"  # abstract, body
    voc_source = "cogatlas"  # cogatlas, neurosynth
    sub_category = "combined"  # "names", "combined"
    topk = 20  # top k predictions
    standardize = False
    logit_scale = 20  # None
    device = "cpu"
    space = "fsLR"
    density = "32k"
    model_id = "BrainGPT/BrainGPT-7B-v0.2"  # BrainGPT/BrainGPT-7B-v0.2, mistralai/Mistral-7B-v0.1
    model_name = model_id.split("/")[-1]

    image_dir = op.join(tasks_dir, "tfMRI")

    atlas = fetch_atlas(space, density)
    lh_surfaces, rh_surfaces = atlas["midthickness"]
    lh_mesh = load_surf_mesh(lh_surfaces)
    rh_mesh = load_surf_mesh(rh_surfaces)
    mesh = PolyMesh(
        left=lh_mesh,
        right=rh_mesh,
    )

    # --------------------------------------------------------------------
    # Set path to AI Decoder model and vocabulary
    model_path = op.join(
        results_dir,
        "pubmed",
        f"model-clip_section-{section}_embedding-{model_name}_best.pth",
    )

    for task in TASKS:
        output_dir = op.join(results_dir, "predictions_hcp", task)
        os.makedirs(output_dir, exist_ok=True)
        for contrast in CONTRASTS[task]:
            image_lh_fns = glob(op.join(image_dir, f"*_tfMRI_{task}_{contrast}_lh.gii"))
            image_rh_fns = glob(op.join(image_dir, f"*_tfMRI_{task}_{contrast}_rh.gii"))
            assert len(image_lh_fns) == len(image_rh_fns)
            assert len(image_lh_fns) == 1
            image_lh_fn = image_lh_fns[0]
            image_rh_fn = image_rh_fns[0]

            data = {"left": image_lh_fn, "right": image_rh_fn}
            _zero_medial_wall
            image = SurfaceImage(
                mesh=mesh,
                data=data,
            )
            image.data.parts["left"] = image.data.parts["left"].reshape(-1, 1)
            image.data.parts["right"] = image.data.parts["right"].reshape(-1, 1)

            plot_surf(
                (image_lh_fn, image_rh_fn),
                op.join(output_dir, f"task-{task}_contrast-{contrast}_map.png"),
                vmax=1,
            )

            for category in ["task", "concept"]:
                vocabulary_lb = (
                    f"vocabulary-{voc_source}_{category}-{sub_category}_embedding-{model_name}"
                )
                vocabulary_fn = op.join(voc_dir, f"{vocabulary_lb}.txt")
                vocabulary_emb_fn = op.join(voc_dir, f"{vocabulary_lb}.npy")
                vocabulary_prior_fn = op.join(
                    voc_dir, f"{vocabulary_lb}_section-{section}_prior.npy"
                )
                vocabulary, vocabulary_emb, vocabulary_prior = _read_vocabulary(
                    vocabulary_fn,
                    vocabulary_emb_fn,
                    vocabulary_prior_fn,
                )

                predictions_df = image_to_labels(
                    image,
                    model_path,
                    vocabulary,
                    vocabulary_emb,
                    vocabulary_prior,
                    topk=topk,
                    logit_scale=logit_scale,
                    data_dir=data_dir,
                    device=device,
                    standardize=standardize,
                    space=space,
                    density=density,
                )

                out_fn = (
                    f"task-{task}_contrast-{contrast}_section-{section}_"
                    f"embedding-{model_name}_cogatlas-{category}_brainclip.csv"
                )
                predictions_df.to_csv(op.join(output_dir, out_fn), index=False)


if __name__ == "__main__":
    main()
