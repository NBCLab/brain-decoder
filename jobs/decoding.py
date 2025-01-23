import os.path as op

import numpy as np

from braindec.dataset import _get_vocabulary
from braindec.predict import image_to_labels


def main(project_dir):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    output_dir = op.join(results_dir, "predictions")
    metamaps_dir = "/Users/julioaperaza/Documents/GitHub/large-scale-ibma/data/neurosynth/metamaps"
    content = "abstract"
    source = "cogatlas"
    term = "working_memory"
    model_id = "mistralai/Mistral-7B-v0.1"
    model_name = model_id.split("/")[-1]

    metamap = op.join(metamaps_dir, f"{term}.nii.gz")

    model_path = op.join(results_dir, "neurostore", f"best_clip-model_{content}_{model_name}.pth")
    vocabulary = _get_vocabulary(source=source, data_dir=data_dir)
    vocabulary_emb = np.load(op.join(data_dir, f"vocabulary_embedding_{model_name}.npy"))

    predictions_df = image_to_labels(
        metamap,
        model_path,
        vocabulary,
        vocabulary_emb,
        data_dir=data_dir,
    )

    predictions_df.to_csv(
        op.join(output_dir, f"{term}_predictions_{content}_{model_name}.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
