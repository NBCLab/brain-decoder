import os
import os.path as op

from nimare.annotate.gclda import GCLDAModel
from nimare.decode.continuous import CorrelationDecoder, gclda_decode_map
from nimare.utils import get_resource_path
from utils import _read_vocabulary

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
    categories = ["task"]  # ["task", "concept"]
    sub_category = "combined"  # "names", "combined"
    topk = 20  # top k predictions
    standardize = False
    logit_scale = 10  # 20, None
    device = "mps"
    model_id = "BrainGPT/BrainGPT-7B-v0.2"  # BrainGPT/BrainGPT-7B-v0.2, mistralai/Mistral-7B-v0.1
    model_name = model_id.split("/")[-1]

    # --------------------------------------------------------------------
    # Set path to AI Decoder model and vocabulary
    model_path = op.join(
        results_dir,
        "pubmed",
        f"model-clip_section-{section}_embedding-{model_name}_best.pth",
    )

    meta_dir = "/Users/julioaperaza/Documents/GitHub/large-scale-ibma/data/neurosynth/metamaps"
    metamaps = [
        "working_memory",
        "pain",
        "motor",
        # "dmn",
    ]  # "pain" "working_memory", "pain", "motor", "dmn"
    images = [op.join(meta_dir, f"{metamap}.nii.gz") for metamap in metamaps]

    for img_i, img in enumerate(images):
        output_dir = op.join(results_dir, "predictions_meta-analysis", metamaps[img_i])
        os.makedirs(output_dir, exist_ok=True)

        # Plot map for debugging
        plot_surf(
            img,
            op.join(output_dir, "map.png"),
            vmax=8,
        )

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

            brainclip_out_fn = f"{vocabulary_lb}_section-{section}_brainclip.csv"
            ns_out_fn = f"vocabulary-{voc_source}_{category}-names_embedding-{model_name}_section-{section}_neurosynth.csv"
            gclda_out_fn = f"vocabulary-{voc_source}_{category}-names_embedding-{model_name}_section-{section}_gclda.csv"

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
            predictions_df.to_csv(op.join(output_dir, brainclip_out_fn), index=False)

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

            gclda_model = GCLDAModel.load(gclda_model_fn)
            gclda_predictions_df, _ = gclda_decode_map(gclda_model, img)
            gclda_predictions_df = gclda_predictions_df.sort_values(
                by="Weight", ascending=False
            ).head(topk)
            gclda_predictions_df.to_csv(op.join(output_dir, gclda_out_fn), index=True)

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
