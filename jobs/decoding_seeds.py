import itertools
import os
import os.path as op
from glob import glob

import nibabel as nib
import pandas as pd
from nilearn._utils.niimg_conversions import check_same_fov
from nilearn.image import resample_to_img
from nimare.annotate.gclda import GCLDAModel
from nimare.dataset import Dataset
from nimare.decode.continuous import CorrelationDecoder, gclda_decode_map
from utils import _read_vocabulary

from braindec.cogatlas import CognitiveAtlas
from braindec.plot import plot_vol_roi
from braindec.predict import image_to_labels_hierarchical


def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    reduced = True
    voc_dir = op.join(data_dir, "vocabulary")
    source = "cogatlasred" if reduced else "cogatlas"
    results_dir = op.join(project_dir, "results")
    sections = ["abstract", "body"]
    sub_categories = ["combined"]
    categories = ["task"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    model_ids = [
        "BrainGPT/BrainGPT-7B-v0.2",
        "mistralai/Mistral-7B-v0.1",
        "BrainGPT/BrainGPT-7B-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
    ]
    topk = 20  # top k predictions
    standardize = False
    logit_scale = 20  # None
    device = "mps"

    output_dir = op.join(results_dir, "predictions_rois")
    os.makedirs(output_dir, exist_ok=True)

    dset = Dataset.load(op.join(data_dir, f"dset-pubmed_{source}-annotated_nimare.pkl"))

    reduced_tasks_fn = op.join(data_dir, "cognitive_atlas", "reduced_tasks.csv")
    reduced_tasks_df = pd.read_csv(reduced_tasks_fn) if reduced else None

    cognitiveatlas = CognitiveAtlas(
        data_dir=data_dir,
        task_snapshot=op.join(data_dir, "cognitive_atlas", "task_snapshot-02-19-25.json"),
        concept_snapshot=op.join(
            data_dir, "cognitive_atlas", "concept_extended_snapshot-02-19-25.json"
        ),
        reduced_tasks=reduced_tasks_df,
    )

    image_dir = op.join(data_dir, "seed-regions")
    images = sorted(glob(op.join(image_dir, "*.nii.gz")))

    for img_i, img_fn in enumerate(images):
        image_name = op.basename(img_fn).split(".")[0]
        plot_vol_roi(
            img_fn,
            op.join(output_dir, f"{image_name}_map.png"),
            color=colors[img_i],
        )

        img = nib.load(img_fn)
        if not check_same_fov(img, reference_masker=dset.masker.mask_img):
            img = resample_to_img(img, dset.masker.mask_img)

        for section, category, sub_category, model_id in itertools.product(
            sections, categories, sub_categories, model_ids
        ):
            model_name = model_id.split("/")[-1]
            model_path = op.join(
                results_dir,
                "pubmed",
                f"model-clip_section-{section}_embedding-{model_name}_best.pth",
            )
            vocabulary_lb = f"vocabulary-{source}_{category}-{sub_category}_embedding-{model_name}"
            vocabulary_fn = op.join(voc_dir, f"vocabulary-{source}_{category}.txt")
            vocabulary_emb_fn = op.join(voc_dir, f"{vocabulary_lb}.npy")
            vocabulary_prior_fn = op.join(voc_dir, f"{vocabulary_lb}_section-{section}_prior.npy")
            vocabulary, vocabulary_emb, vocabulary_prior = _read_vocabulary(
                vocabulary_fn,
                vocabulary_emb_fn,
                vocabulary_prior_fn,
            )

            file_base_name = f"{image_name}_{vocabulary_lb}_section-{section}"
            task_out_fn = f"{file_base_name}_pred-task_brainclip.csv"
            concept_out_fn = f"{file_base_name}_pred-concept_brainclip.csv"
            process_out_fn = f"{file_base_name}_pred-process_brainclip.csv"

            task_prob_df, concept_prob_df, process_prob_df = image_to_labels_hierarchical(
                img,
                model_path,
                vocabulary,
                vocabulary_emb,
                vocabulary_prior,
                cognitiveatlas.concept_to_task_idxs,
                cognitiveatlas.process_to_concept_idxs,
                cognitiveatlas.concept_names,
                cognitiveatlas.process_names,
                topk=topk,
                standardize=standardize,
                logit_scale=logit_scale,
                data_dir=data_dir,
                device=device,
            )
            task_prob_df.to_csv(op.join(output_dir, task_out_fn), index=False)
            concept_prob_df.to_csv(op.join(output_dir, concept_out_fn), index=False)
            process_prob_df.to_csv(op.join(output_dir, process_out_fn), index=False)

            if sub_category == "names":
                baseline_label = f"{source}-{category}_embedding-{model_name}_section-{section}"
                # Baseline
                # --------------------------------------------------------------------
                ns_out_fn = f"{file_base_name}_pred-task_neurosynth.csv"
                gclda_out_fn = f"{file_base_name}_pred-task_gclda.csv"

                # Load baseline model
                ns_model_fn = op.join(
                    results_dir,
                    "baseline",
                    f"model-neurosynth_{baseline_label}.pkl",
                )
                gclda_model_fn = op.join(
                    results_dir,
                    "baseline",
                    f"model-gclda_{baseline_label}.pkl",
                )

                gclda_model = GCLDAModel.load(gclda_model_fn)
                gclda_predictions_df, _ = gclda_decode_map(gclda_model, img)
                gclda_predictions_df = gclda_predictions_df.sort_values(
                    by="Weight", ascending=False
                ).head(topk)
                gclda_predictions_df = gclda_predictions_df.reset_index()
                gclda_predictions_df.columns = ["pred", "weight"]
                gclda_predictions_df.to_csv(op.join(output_dir, gclda_out_fn), index=False)

                ns_decoder = CorrelationDecoder.load(ns_model_fn)
                ns_predictions_df = ns_decoder.transform(img)
                feature_group = f"{source}-{category}_section-{section}_annot-tfidf__"
                feature_names = ns_predictions_df.index.values
                vocabulary_names = [f.replace(feature_group, "") for f in feature_names]
                ns_predictions_df.index = vocabulary_names

                ns_predictions_df = ns_predictions_df.sort_values(by="r", ascending=False).head(
                    topk
                )
                ns_predictions_df = ns_predictions_df.reset_index()
                ns_predictions_df.columns = ["pred", "corr"]
                ns_predictions_df.to_csv(op.join(output_dir, ns_out_fn), index=False)


if __name__ == "__main__":
    main()
