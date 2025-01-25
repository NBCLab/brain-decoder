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


def _read_vocabulary(vocabulary_fn, vocabulary_emb_fn):
    vocabulary_emb = np.load(vocabulary_emb_fn)
    with open(vocabulary_fn, "r") as f:
        vocabulary = [line.strip() for line in f]
    return vocabulary, vocabulary_emb


def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    nilearn_dir = op.join(data_dir, "nilearn")
    nimare_dir = op.join(data_dir, "nimare")
    results_dir = op.join(project_dir, "results")
    images_source = "pain"  # mixed-gambles, localizer, pain
    content = "abstract"
    source = "cogatlas"  # cogatlas, neurosynth
    topk = 10  # top k predictions

    output_dir = op.join(results_dir, f"predictions_{source}_standardized", images_source)
    os.makedirs(output_dir, exist_ok=True)

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
    model_id = "BrainGPT/BrainGPT-7B-v0.2"  # BrainGPT/BrainGPT-7B-v0.2, mistralai/Mistral-7B-v0.1
    model_name = model_id.split("/")[-1]
    model_path = op.join(
        results_dir, "neurostore", f"best_clip-model_{content}_standardized_{model_name}.pth"
    )

    # --------------------------------------------------------------------
    # Load images to decode
    if images_source == "mixed-gambles":
        data = datasets.fetch_mixed_gambles(n_subjects=16, data_dir=nilearn_dir)
        images = data.zmaps
    elif images_source == "localizer":
        data = datasets.fetch_localizer_contrasts(["left vs right button press"], n_subjects=10)
        images = data.cmaps
    elif images_source == "pain":
        pain_data_dir = download_nidm_pain(data_dir=nimare_dir)
        images = sorted(glob(op.join(pain_data_dir, "*", "TStatistic.nii.gz")))
    else:
        raise ValueError(f"Images source {images_source} not supported.")

    # --------------------------------------------------------------------
    # Decode images
    for img_i, img in enumerate(images):
        # Plot map for debugging
        plot_surf(img, op.join(output_dir, f"{img_i:02d}_map.png"), vmax=8)

        for category in ["task", "concept"]:
            vocabulary_fn = op.join(data_dir, f"vocabulary-{source}-{category}.txt")
            vocabulary_emb_fn = op.join(
                data_dir, f"vocabulary-{source}-{category}_embedding-{model_name}.npy"
            )
            vocabulary, vocabulary_emb = _read_vocabulary(vocabulary_fn, vocabulary_emb_fn)

            predictions_df = image_to_labels(
                img,
                model_path,
                vocabulary,
                vocabulary_emb,
                topk=topk,
                data_dir=data_dir,
            )

            predictions_df.to_csv(
                op.join(
                    output_dir, f"{img_i:02d}_predictions_{content}_{category}_{model_name}.csv"
                ),
                index=False,
            )

        ns_predictions_df = decoder.transform(img)
        ns_predictions_df = ns_predictions_df.sort_values(by="r", ascending=False).head(topk)
        ns_predictions_df.to_csv(
            op.join(output_dir, f"{img_i:02d}_predictions_{content}_neurosynth.csv"),
            index=True,
        )


if __name__ == "__main__":
    main()
