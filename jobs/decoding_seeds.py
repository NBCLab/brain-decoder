import argparse
import itertools
import os
import os.path as op
from glob import glob

from jobs.utils import (
    DEFAULT_MODEL_IDS,
    DEFAULT_SECTIONS,
    add_common_job_args,
    build_cognitiveatlas,
    get_source,
    load_decoding_resources,
    resolve_project_paths,
)


def _get_parser():
    parser = add_common_job_args(argparse.ArgumentParser(description="Decode ROI seed maps"))
    parser.add_argument(
        "--image_dir",
        dest="image_dir",
        default=None,
        help="Optional explicit path to the ROI seed image directory.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default=None,
        help="Optional explicit path to the output directory.",
    )
    parser.add_argument(
        "--categories",
        dest="categories",
        nargs="+",
        default=["task"],
        help="Vocabulary categories to evaluate.",
    )
    parser.add_argument(
        "--sub_categories",
        dest="sub_categories",
        nargs="+",
        default=["combined"],
        help="Vocabulary embedding variants to evaluate.",
    )
    return parser


def main(
    project_dir=None,
    data_dir=None,
    results_dir=None,
    image_dir=None,
    output_dir=None,
    sections=None,
    categories=None,
    sub_categories=None,
    model_ids=None,
    topk=20,
    standardize=False,
    logit_scale=20.0,
    device=None,
    reduced=True,
):
    import nibabel as nib
    from nilearn.image import resample_to_img
    from nimare.annotate.gclda import GCLDAModel
    from nimare.dataset import Dataset
    from nimare.decode.continuous import CorrelationDecoder, gclda_decode_map

    from braindec.plot import plot_vol_roi
    from braindec.predict import image_to_labels_hierarchical
    from braindec.utils import _get_device, images_have_same_fov

    _, data_dir, results_dir = resolve_project_paths(project_dir, data_dir, results_dir)
    source = get_source(reduced)
    voc_dir = op.join(data_dir, "vocabulary")
    image_dir = op.join(data_dir, "seed-regions") if image_dir is None else op.abspath(image_dir)
    output_dir = (
        op.join(results_dir, "predictions_rois") if output_dir is None else op.abspath(output_dir)
    )
    os.makedirs(output_dir, exist_ok=True)

    sections = list(DEFAULT_SECTIONS) if sections is None else sections
    categories = ["task"] if categories is None else categories
    sub_categories = ["combined"] if sub_categories is None else sub_categories
    model_ids = list(DEFAULT_MODEL_IDS) if model_ids is None else model_ids
    device = _get_device() if device is None else device

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]

    dset = Dataset.load(op.join(data_dir, f"dset-pubmed_{source}-annotated_nimare.pkl"))
    cognitiveatlas = build_cognitiveatlas(data_dir, reduced)
    images = sorted(glob(op.join(image_dir, "*.nii.gz")))

    for img_i, img_fn in enumerate(images):
        image_name = op.basename(img_fn).split(".")[0]
        plot_vol_roi(
            img_fn,
            op.join(output_dir, f"{image_name}_map.png"),
            color=colors[img_i % len(colors)],
        )

        img = nib.load(img_fn)
        if not images_have_same_fov(img, dset.masker.mask_img):
            img = resample_to_img(img, dset.masker.mask_img)

        for section, category, sub_category, model_id in itertools.product(
            sections,
            categories,
            sub_categories,
            model_ids,
        ):
            resources = load_decoding_resources(
                results_dir,
                voc_dir,
                source,
                category,
                sub_category,
                model_id,
                section,
            )

            file_base_name = f"{image_name}_{resources['vocabulary_label']}_section-{section}"
            task_out_fn = f"{file_base_name}_pred-task_brainclip.csv"
            concept_out_fn = f"{file_base_name}_pred-concept_brainclip.csv"
            process_out_fn = f"{file_base_name}_pred-process_brainclip.csv"

            task_prob_df, concept_prob_df, process_prob_df = image_to_labels_hierarchical(
                img,
                resources["model_path"],
                resources["vocabulary"],
                resources["vocabulary_emb"],
                resources["vocabulary_prior"],
                cognitiveatlas,
                topk=topk,
                standardize=standardize,
                logit_scale=logit_scale,
                data_dir=data_dir,
                device=device,
            )
            task_prob_df.to_csv(op.join(output_dir, task_out_fn), index=False)
            concept_prob_df.to_csv(op.join(output_dir, concept_out_fn), index=False)
            process_prob_df.to_csv(op.join(output_dir, process_out_fn), index=False)

            if sub_category != "names":
                continue

            baseline_label = f"{source}-{category}_embedding-{resources['model_name']}_section-{section}"
            ns_out_fn = f"{file_base_name}_pred-task_neurosynth.csv"
            gclda_out_fn = f"{file_base_name}_pred-task_gclda.csv"

            ns_model_fn = op.join(results_dir, "baseline", f"model-neurosynth_{baseline_label}.pkl")
            gclda_model_fn = op.join(results_dir, "baseline", f"model-gclda_{baseline_label}.pkl")

            gclda_model = GCLDAModel.load(gclda_model_fn)
            gclda_predictions_df, _ = gclda_decode_map(gclda_model, img)
            gclda_predictions_df = gclda_predictions_df.sort_values(by="Weight", ascending=False).head(
                topk
            )
            gclda_predictions_df = gclda_predictions_df.reset_index()
            gclda_predictions_df.columns = ["pred", "weight"]
            gclda_predictions_df.to_csv(op.join(output_dir, gclda_out_fn), index=False)

            ns_decoder = CorrelationDecoder.load(ns_model_fn)
            ns_predictions_df = ns_decoder.transform(img)
            feature_group = f"{source}-{category}_section-{section}_annot-tfidf__"
            feature_names = ns_predictions_df.index.values
            vocabulary_names = [f.replace(feature_group, "") for f in feature_names]
            ns_predictions_df.index = vocabulary_names
            ns_predictions_df = ns_predictions_df.sort_values(by="r", ascending=False).head(topk)
            ns_predictions_df = ns_predictions_df.reset_index()
            ns_predictions_df.columns = ["pred", "corr"]
            ns_predictions_df.to_csv(op.join(output_dir, ns_out_fn), index=False)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
