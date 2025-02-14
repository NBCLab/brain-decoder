import os
import os.path as op
from glob import glob

import numpy as np
from nilearn import datasets
from nimare.decode.continuous import CorrelationDecoder
from nimare.extract import download_nidm_pain
from nimare.utils import get_resource_path

from braindec.plot import plot_surf
from braindec.predict import image_to_labels

# TASKS = ["bart", "pamret", "scap", "stopsignal", "taskswitch"]
TASKS = ["taskswitch"]
# TASKS = ["bart", "pamret", "scap", "stopsignal"]
CONTRASTS = {
    # "bart": ["Accept", "AcceptParametric", "AcceptRT", "Explode", "Reject"],
    # "bart": ["Explode"],
    # "pamret": ["All", "All_RT", "TruePos-FalsePos", "TruePos-FalsePos", "TruePos-TrueNeg"],
    # "pamret": ["All"],
    # "scap": ["All", "All_rt", "LinearUp_delay", "LinearUp_load"],
    # "scap": ["All"],
    # "stopsignal": ["Go", "GoRT", "Go-StopSuccess", "StopSuccess-StopUnsuccess"],
    # "stopsignal": ["Go"],
    # "taskswitch": ["ALL", "ALL_rt", "CONGRUENT-INCONGRUENT", "SWITCH-NOSWITCH"],
    "taskswitch": ["SWITCH-NOSWITCH"],
}


def _read_vocabulary(vocabulary_fn, vocabulary_emb_fn, vocabulary_prior_fn):
    vocabulary_emb = np.load(vocabulary_emb_fn)
    vocabulary_prior = np.load(vocabulary_prior_fn)
    with open(vocabulary_fn, "r") as f:
        vocabulary = [line.strip() for line in f]
    return vocabulary, vocabulary_emb, vocabulary_prior


def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    voc_dir = op.join(data_dir, "vocabulary")
    results_dir = op.join(project_dir, "results")
    tasks_dir = op.join(data_dir, "tasks")
    level = "group"
    section = "body"  # abstract, body
    voc_source = "cogatlas"  # cogatlas, neurosynth
    sub_category = "combined"  # "names", "combined"
    topk = 20  # top k predictions
    standardize = False
    logit_scale = 20  # None
    device = "cpu"
    model_id = "BrainGPT/BrainGPT-7B-v0.2"  # BrainGPT/BrainGPT-7B-v0.2, mistralai/Mistral-7B-v0.1
    model_name = model_id.split("/")[-1]

    image_dir = op.join(tasks_dir, f"{level}-level")

    # --------------------------------------------------------------------
    # Load baseline model
    ns_model_fn = op.join(data_dir, "neurosynth", "neurosynth-abstract.pkl.gz")
    if not op.isfile(ns_model_fn):
        metamaps_dir = op.join(data_dir, "neurosynth", "metamaps")
        mask_img = op.join(get_resource_path(), "templates", "MNI152_2x2x2_brainmask.nii.gz")

        decoder = CorrelationDecoder()
        decoder.load_imgs(metamaps_dir, mask=mask_img)
        decoder.save(ns_model_fn)
    else:
        decoder = CorrelationDecoder.load(ns_model_fn)

    # --------------------------------------------------------------------
    # Set path to AI Decoder model and vocabulary
    model_path = op.join(
        results_dir,
        "pubmed",
        f"model-clip_section-{section}_embedding-{model_name}_best.pth",
    )

    for task in TASKS:
        output_dir = op.join(results_dir, f"predictions_cnp_{level}-level", task)
        os.makedirs(output_dir, exist_ok=True)
        for contrast in CONTRASTS[task]:
            images = [
                op.join(image_dir, f"sub-group_task-{task}_contrast-{contrast}_tstat.nii.gz")
            ]

            # --------------------------------------------------------------------
            # Decode images
            for img_i, img in enumerate(images):
                # Plot map for debugging
                plot_surf(
                    img,
                    op.join(output_dir, f"{img_i:02d}_task-{task}_contrast-{contrast}_map.png"),
                    vmax=8,
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
                        img,
                        model_path,
                        vocabulary,
                        vocabulary_emb,
                        vocabulary_prior,
                        topk=topk,
                        standardize=standardize,
                        logit_scale=logit_scale,
                        data_dir=data_dir,
                        device=device,
                    )

                    out_fn = (
                        f"{img_i:02d}_task-{task}_contrast-{contrast}_"
                        f"{vocabulary_lb}_section-{section}_brainclip.csv"
                    )
                    predictions_df.to_csv(op.join(output_dir, out_fn), index=False)

                ns_predictions_df = decoder.transform(img)
                ns_predictions_df = ns_predictions_df.sort_values(by="r", ascending=False).head(
                    topk
                )
                out_baseline_fn = (
                    f"{img_i:02d}_task-{task}_contrast-{contrast}_section-{section}_neurosynth.csv"
                )
                ns_predictions_df.to_csv(op.join(output_dir, out_baseline_fn), index=True)


if __name__ == "__main__":
    main()
