import itertools
import json
import os
import os.path as op
from glob import glob

import numpy as np
import pandas as pd

from braindec.cogatlas import CognitiveAtlas


def _recall_at_n(true_lb, pred_lb, n):
    if isinstance(true_lb, int):
        true_lb = [true_lb]

    # Check if empty
    # if true_lb:
    #     print("Empty true labels")
    #     return 0

    return len(np.intersect1d(true_lb, pred_lb[:n])) / len(true_lb)


def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    reduced = True
    voc_fn = "vocabulary_reduced" if reduced else "vocabulary"
    voc_dir = op.join(data_dir, voc_fn)
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

    domains = [
        "emotion",
        "gambling",
        "language",
        "motor",
        "relational",
        "social",
        "working memory",
    ]
    subdomains = ["task", "concept", "domain"]

    output_dir = op.join(results_dir, "predictions_ibc")
    os.makedirs(output_dir, exist_ok=True)

    concept_to_task_fn = op.join(data_dir, "cognitive_atlas", "concept_to_task.json")
    with open(concept_to_task_fn, "r") as file:
        concept_to_task = json.load(file)

    concept_to_process_fn = op.join(data_dir, "cognitive_atlas", "concept_to_process.json")
    with open(concept_to_process_fn, "r") as file:
        concept_to_process = json.load(file)

    reduced_tasks_fn = op.join(data_dir, "cognitive_atlas", "reduced_tasks.csv")
    reduced_tasks_df = pd.read_csv(reduced_tasks_fn) if reduced else None

    cognitiveatlas = CognitiveAtlas(
        data_dir=data_dir,
        task_snapshot=op.join(data_dir, "cognitive_atlas", "task_snapshot-02-19-25.json"),
        concept_snapshot=op.join(
            data_dir, "cognitive_atlas", "concept_extended_snapshot-02-19-25.json"
        ),
        reduced_tasks=reduced_tasks_df,
        # concept_to_task=concept_to_task,
        concept_to_process=concept_to_process,
    )

    image_dir = op.join(data_dir, "ibc")
    images = sorted(glob(op.join(image_dir, "*.nii.gz")))
    metadata = pd.read_csv(op.join(data_dir, "ibc", "metadata.csv"))

    ground_truth_fn = op.join(data_dir, "ibc", "ground_truth.json")
    with open(ground_truth_fn, "r") as file:
        ground_truth = json.load(file)

    results_dict = {
        "domain": [],
        "model": [],
        "task_mean": [],
        "task_std": [],
        "concept_mean": [],
        "concept_std": [],
        "process_mean": [],
        "process_std": [],
    }
    for section, category, sub_category, model_id in itertools.product(
        sections, categories, sub_categories, model_ids
    ):
        model_name = model_id.split("/")[-1]

        vocabulary_lb = f"vocabulary-{voc_source}_{category}-{sub_category}_embedding-{model_name}_section-{section}"

        temp_results = {dom: {subdom: [] for subdom in subdomains} for dom in domains}
        for _, img_fn in enumerate(images):
            image_name = op.basename(img_fn).split(".")[0]
            file_lb = f"{image_name}_{vocabulary_lb}"
            print(f"Processing {image_name}")

            domain = metadata.loc[metadata["file"] == image_name, "domain"].values[0]
            task_true = ground_truth[domain]["task"]
            concept_true = ground_truth[domain]["concept"]
            process_true = ground_truth[domain]["domain"]

            print(f"\tTrue task: {task_true}")
            task_true_idx = cognitiveatlas.get_task_idx_from_names(task_true)
            concept_true_idx = cognitiveatlas.get_concept_idx_from_names(concept_true)
            process_true_idx = cognitiveatlas.get_process_idx_from_names(process_true)

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

            task_recall = _recall_at_n(task_true_idx, task_pred_idx, 10)
            concept_recall = _recall_at_n(concept_true_idx, concept_pred_idx, 10)
            process_recalls = _recall_at_n(process_true_idx, process_pred_idx, 3)

            temp_results[domain]["task"].append(task_recall)
            temp_results[domain]["concept"].append(concept_recall)
            temp_results[domain]["domain"].append(process_recalls)

        mean_task_recalls = [np.mean(temp_results[dom]["task"]) for dom in domains]
        mean_concept_recalls = [np.mean(temp_results[dom]["concept"]) for dom in domains]
        mean_process_recalls = [np.mean(temp_results[dom]["domain"]) for dom in domains]
        std_task_recalls = [np.std(temp_results[dom]["task"]) for dom in domains]
        std_concept_recalls = [np.std(temp_results[dom]["concept"]) for dom in domains]
        std_process_recalls = [np.std(temp_results[dom]["domain"]) for dom in domains]

        n_domains = len(domains)
        results_dict["domain"].extend(domains)
        results_dict["model"].extend([vocabulary_lb] * n_domains)
        results_dict["task_mean"].extend(mean_task_recalls)
        results_dict["task_std"].extend(std_task_recalls)
        results_dict["concept_mean"].extend(mean_concept_recalls)
        results_dict["concept_std"].extend(std_concept_recalls)
        results_dict["process_mean"].extend(mean_process_recalls)
        results_dict["process_std"].extend(std_process_recalls)

    # Export results to csv
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(op.join(results_dir, "eval-ibc_results.csv"), index=False)


if __name__ == "__main__":
    main()
