import os
import os.path as op
from glob import glob

import nibabel as nib
from nilearn._utils.niimg_conversions import check_same_fov
from nilearn.image import resample_to_img
from nimare.annotate.gclda import GCLDAModel
from nimare.dataset import Dataset
from nimare.decode.continuous import CorrelationDecoder, gclda_decode_map
from utils import _read_vocabulary

from braindec.cogatlas import CognitiveAtlas
from braindec.plot import plot_surf
from braindec.predict import image_to_labels


def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    voc_dir = op.join(data_dir, "vocabulary")
    results_dir = op.join(project_dir, "results")
    section = "body"  # abstract, body
    voc_source = "cogatlas"  # cogatlas, neurosynth
    sub_category = "combined"  # "names", "combined"
    categories = ["task"]  # ["task", "concept"]
    topk = 20  # top k predictions
    standardize = False
    logit_scale = 20  # None
    device = "mps"
    model_id = "BrainGPT/BrainGPT-7B-v0.2"  # BrainGPT/BrainGPT-7B-v0.2, mistralai/Mistral-7B-v0.1
    model_name = model_id.split("/")[-1]

    dset = Dataset.load(op.join(data_dir, "dset-pubmed_annotated_nimare.pkl"))

    cognitiveatlas = CognitiveAtlas(
        data_dir=data_dir,
        task_snapshot=op.join(data_dir, "cognitive_atlas", "task_snapshot-02-19-25.json"),
        concept_snapshot=op.join(data_dir, "cognitive_atlas", "concept_snapshot-02-19-25.json"),
    )

    # --------------------------------------------------------------------
    # Set path to AI Decoder model and vocabulary
    model_path = op.join(
        results_dir,
        "pubmed",
        f"model-clip_section-{section}_embedding-{model_name}_best.pth",
    )

    output_dir = op.join(results_dir, "predictions_hcp_nv")
    os.makedirs(output_dir, exist_ok=True)

    image_dir = op.join(data_dir, "hcp", "neurovault")
    images = sorted(glob(op.join(image_dir, "*.nii.gz")))
    # images = images[:1]

    for img_i, img_fn in enumerate(images):
        image_name = op.basename(img_fn).split(".")[0]
        # Plot map for debugging
        plot_surf(
            img_fn,
            op.join(output_dir, f"{img_i:02d}_{image_name}_map.png"),
            vmax=30,
        )

        img = nib.load(img_fn)
        if not check_same_fov(img, reference_masker=dset.masker.mask_img):
            img = resample_to_img(img, dset.masker.mask_img)

        for category in categories:
            vocabulary_lb = (
                f"vocabulary-{voc_source}_{category}-{sub_category}_embedding-{model_name}"
            )
            vocabulary_fn = op.join(voc_dir, f"{vocabulary_lb}.txt")
            vocabulary_emb_fn = op.join(voc_dir, f"{vocabulary_lb}.npy")
            vocabulary_prior_fn = op.join(voc_dir, f"{vocabulary_lb}_section-{section}_prior.npy")
            vocabulary, vocabulary_emb, vocabulary_prior = _read_vocabulary(
                vocabulary_fn,
                vocabulary_emb_fn,
                vocabulary_prior_fn,
            )

            task_out_fn = f"{img_i:02d}_{vocabulary_lb}_section-{section}_pred-task_brainclip.csv"
            concept_out_fn = (
                f"{img_i:02d}_{vocabulary_lb}_section-{section}_pred-concept_brainclip.csv"
            )
            process_out_fn = (
                f"{img_i:02d}_{vocabulary_lb}_section-{section}_pred-process_brainclip.csv"
            )
            ns_out_fn = f"{img_i:02d}_vocabulary-{voc_source}_{category}-names_embedding-{model_name}_section-{section}_neurosynth.csv"
            gclda_out_fn = f"{img_i:02d}_vocabulary-{voc_source}_{category}-names_embedding-{model_name}_section-{section}_gclda.csv"

            task_prob_df, concept_prob_df, process_prob_df = image_to_labels(
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

            # Baseline
            # --------------------------------------------------------------------
            # Load baseline model
            ns_model_fn = op.join(
                results_dir,
                "baseline",
                f"model-neurosynth_{voc_source}-{category}_embedding-{model_name}_section-{section}.pkl",
            )
            gclda_model_fn = op.join(
                results_dir,
                "baseline",
                f"model-gclda_{voc_source}-{category}_embedding-{model_name}_section-{section}.pkl",
            )

            """
            gclda_model = GCLDAModel.load(gclda_model_fn)
            gclda_predictions_df, _ = gclda_decode_map(gclda_model, img)
            gclda_predictions_df = gclda_predictions_df.sort_values(
                by="Weight", ascending=False
            ).head(topk)
            gclda_predictions_df.to_csv(op.join(output_dir, gclda_out_fn), index=True)
            """

            ns_decoder = CorrelationDecoder.load(ns_model_fn)
            ns_predictions_df = ns_decoder.transform(img)
            feature_group = f"{voc_source}-{category}_section-{section}_annot-tfidf__"
            feature_names = ns_predictions_df.index.values
            vocabulary_names = [f.replace(feature_group, "") for f in feature_names]
            ns_predictions_df.index = vocabulary_names

            ns_predictions_df = ns_predictions_df.sort_values(by="r", ascending=False).head(topk)
            ns_predictions_df.to_csv(op.join(output_dir, ns_out_fn), index=True)


if __name__ == "__main__":
    main()
