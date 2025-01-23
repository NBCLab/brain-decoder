import os
import os.path as op

import numpy as np
from nilearn.datasets import fetch_mixed_gambles

from braindec.predict import image_to_labels


def main(project_dir):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    nilearn_dir = op.join(data_dir, "nilearn")
    results_dir = op.join(project_dir, "results")
    images_source = "mixed-gambles"
    output_dir = op.join(results_dir, "predictions", images_source)
    content = "abstract"
    source = "cogatlas"

    os.makedirs(output_dir, exist_ok=True)

    model_id = "mistralai/Mistral-7B-v0.1"
    model_name = model_id.split("/")[-1]

    if images_source == "mixed-gambles":
        data = fetch_mixed_gambles(n_subjects=16, data_dir=nilearn_dir)
        images = data.zmaps
    else:
        raise ValueError(f"Images source {images_source} not supported.")

    model_path = op.join(results_dir, "neurostore", f"best_clip-model_{content}_{model_name}.pth")
    vocabulary_emb = np.load(op.join(data_dir, f"vocabulary-{source}_embedding_{model_name}.npy"))

    with open(op.join(data_dir, f"vocabulary-{source}_{model_name}.txt"), "r") as f:
        vocabulary = [line.strip() for line in f]

    for img_i, img in enumerate(images):
        predictions_df = image_to_labels(
            img,
            model_path,
            vocabulary,
            vocabulary_emb,
            data_dir=data_dir,
        )

        predictions_df.to_csv(
            op.join(output_dir, f"{img_i:02d}_predictions_{content}_{model_name}.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
