import itertools
import os
import os.path as op
from glob import glob

import numpy as np
import pandas as pd

from braindec.cogatlas import CognitiveAtlas


def _recall_at_n(true_lb, pred_lb, n):
    if isinstance(true_lb, int):
        true_lb = [true_lb]

    return len(np.intersect1d(true_lb, pred_lb[:n])) / len(true_lb)


def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    sections = ["body"]
    # sections = ["abstract", "body"]
    voc_source = "cogatlas"
    sub_categories = ["combined"]
    # sub_categories = ["names", "definitions", "combined"]
    categories = ["task"]  # ["task", "concept"]
    model_ids = [
        "BrainGPT/BrainGPT-7B-v0.2",
        # "mistralai/Mistral-7B-v0.1",
        # "BrainGPT/BrainGPT-7B-v0.1",
        # "meta-llama/Llama-2-7b-chat-hf",
    ]

    output_dir = op.join(results_dir, "predictions_ibc")
    os.makedirs(output_dir, exist_ok=True)

    cognitiveatlas = CognitiveAtlas(
        data_dir=data_dir,
        task_snapshot=op.join(data_dir, "cognitive_atlas", "task_snapshot-02-19-25.json"),
        concept_snapshot=op.join(data_dir, "cognitive_atlas", "concept_snapshot-02-19-25.json"),
    )

    image_dir = op.join(data_dir, "ibc")
    images = sorted(glob(op.join(image_dir, "*.nii.gz")))
    metadata = pd.read_csv(op.join(data_dir, "ibc", "metadata.csv"))

    eval_results = []
    for section, category, sub_category, model_id in itertools.product(
        sections, categories, sub_categories, model_ids
    ):
        model_name = model_id.split("/")[-1]

        vocabulary_lb = f"vocabulary-{voc_source}_{category}-{sub_category}_embedding-{model_name}_section-{section}"
        task_recalls, concept_recalls, process_recalls = [], [], []
        for _, img_fn in enumerate(images):
            image_name = op.basename(img_fn).split(".")[0]
            file_lb = f"{image_name}_{vocabulary_lb}"

            task_true = metadata.loc[metadata["file"] == image_name, "task"].values[0]
            task_true_idx = cognitiveatlas.task_names.index(task_true)
            concept_true_idx = cognitiveatlas.task_to_concept_idxs[task_true_idx]
            process_true_idx = cognitiveatlas.task_to_process_idxs[task_true_idx]

            task_out_fn = f"{file_lb}_pred-task_brainclip.csv"
            concept_out_fn = f"{file_lb}_pred-concept_brainclip.csv"
            process_out_fn = f"{file_lb}_pred-process_brainclip.csv"

            task_prob_df = pd.read_csv(op.join(output_dir, task_out_fn))
            concept_prob_df = pd.read_csv(op.join(output_dir, concept_out_fn))
            process_prob_df = pd.read_csv(op.join(output_dir, process_out_fn))

            task_pred = task_prob_df["pred"].values
            task_pred_idx = cognitiveatlas.get_task_idx_from_names(task_pred)
            concept_pred = concept_prob_df["pred"].values
            concept_pred_idx = cognitiveatlas.get_concept_idx_from_names(concept_pred)
            process_pred = process_prob_df["pred"].values
            process_pred_idx = cognitiveatlas.get_process_idx_from_names(process_pred)

            task_recalls.append(_recall_at_n(task_true_idx, task_pred_idx, 10))
            concept_recalls.append(_recall_at_n(concept_true_idx, concept_pred_idx, 5))
            process_recalls.append(_recall_at_n(process_true_idx, process_pred_idx, 2))

        mean_task_recall = np.mean(task_recalls)
        mean_concept_recall = np.mean(concept_recalls)
        mean_process_recall = np.mean(process_recalls)

        std_task_recall = np.std(task_recalls)
        std_concept_recall = np.std(concept_recalls)
        std_process_recall = np.std(process_recalls)

        results_dict = {
            "model": vocabulary_lb,
            "task_mean": mean_task_recall,
            "task_std": std_task_recall,
            "concept_mean": mean_concept_recall,
            "concept_std": std_concept_recall,
            "process_mean": mean_process_recall,
            "process_std": std_process_recall,
        }
        eval_results.append(results_dict)

    # Export results to csv
    results_df = pd.DataFrame(eval_results)
    results_df.to_csv(op.join(results_dir, "eval-ibc_results.csv"), index=False)


if __name__ == "__main__":
    main()
