"""Decode CNP maps into task, concept, and domain predictions.

Expected inputs:
- NIfTI maps in `data/cnp` by default.
- Prediction labels are derived from filenames and written into filenames like
  `<prediction_label>_<vocabulary_label>_section-<section>_pred-task_brainclip.csv`.

Use a companion mapping CSV with `prediction_label` plus task metadata when running
`jobs/decoding_eval.py` on the resulting outputs.
"""

import argparse
import itertools
import os
import os.path as op
from glob import glob

import pandas as pd
from tqdm.auto import tqdm

from jobs.utils import (
    DEFAULT_MODEL_IDS,
    DEFAULT_SECTIONS,
    add_common_job_args,
    build_cognitiveatlas,
    get_source,
    infer_prediction_label,
    load_decoding_resources,
    resolve_project_paths,
)


def _get_parser():
    parser = add_common_job_args(argparse.ArgumentParser(description="Decode CNP maps"))
    parser.add_argument(
        "--image_dir",
        dest="image_dir",
        default=None,
        help="Optional explicit path to the CNP image directory.",
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
    parser.add_argument(
        "--label_delimiter",
        dest="label_delimiter",
        default="_",
        help="Delimiter used to split image stems when deriving prediction labels.",
    )
    parser.add_argument(
        "--label_token_index",
        dest="label_token_index",
        type=int,
        default=0,
        help="Token index used to derive the prediction label from the image stem.",
    )
    parser.add_argument(
        "--mapping_fn",
        dest="mapping_fn",
        default=None,
        help="Optional mapping CSV with a local_path column used to select images.",
    )
    parser.add_argument(
        "--num_shards",
        dest="num_shards",
        type=int,
        default=1,
        help="Split the selected image list into this many shards.",
    )
    parser.add_argument(
        "--shard_index",
        dest="shard_index",
        type=int,
        default=0,
        help="0-based shard index to decode from the sharded image list.",
    )
    parser.add_argument(
        "--skip_existing",
        dest="skip_existing",
        action="store_true",
        help="Skip image/config outputs when all expected CSVs already exist.",
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
    label_delimiter="_",
    label_token_index=0,
    mapping_fn=None,
    num_shards=1,
    shard_index=0,
    skip_existing=False,
):
    import nibabel as nib
    from nilearn.image import resample_to_img
    from nimare.annotate.gclda import GCLDAModel
    from nimare.decode.continuous import CorrelationDecoder, gclda_decode_map

    from braindec.embedding import ImageEmbedding
    from braindec.model import build_model
    from braindec.predict import image_to_labels_hierarchical
    from braindec.utils import _get_device, images_have_same_fov

    _, data_dir, results_dir = resolve_project_paths(project_dir, data_dir, results_dir)
    source = get_source(reduced)
    voc_dir = op.join(data_dir, "vocabulary")
    image_dir = op.join(data_dir, "cnp") if image_dir is None else op.abspath(image_dir)
    output_dir = (
        op.join(results_dir, "predictions_cnp") if output_dir is None else op.abspath(output_dir)
    )
    os.makedirs(output_dir, exist_ok=True)

    sections = list(DEFAULT_SECTIONS) if sections is None else sections
    categories = ["task"] if categories is None else categories
    sub_categories = ["combined"] if sub_categories is None else sub_categories
    model_ids = list(DEFAULT_MODEL_IDS) if model_ids is None else model_ids
    device = _get_device() if device is None else device
    mask_img = nib.load(op.join(data_dir, "MNI152_2x2x2_brainmask.nii.gz"))

    def _resample_for_reference(image, reference_img):
        if reference_img is None or images_have_same_fov(image, reference_img):
            return image
        return resample_to_img(image, reference_img)

    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards.")

    cognitiveatlas = build_cognitiveatlas(data_dir, reduced)
    if mapping_fn is not None:
        mapping_df = pd.read_csv(op.abspath(mapping_fn))
        if "local_path" not in mapping_df.columns:
            raise KeyError(f"Mapping file {mapping_fn} must include a local_path column.")
        images = sorted(mapping_df["local_path"].dropna().astype(str).tolist())
    else:
        images = sorted(glob(op.join(image_dir, "*.nii.gz")))
    images = [img_fn for idx, img_fn in enumerate(images) if idx % num_shards == shard_index]

    image_records = []
    for img_fn in tqdm(images, desc="load cnp images", unit="img"):
        prediction_label = infer_prediction_label(
            img_fn,
            delimiter=label_delimiter,
            token_index=label_token_index,
        )
        img = nib.load(img_fn)
        if not images_have_same_fov(img, mask_img):
            img = resample_to_img(img, mask_img)
        image_records.append((prediction_label, img))

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
        model = build_model(resources["model_path"], device=device)
        image_emb_gene = ImageEmbedding(
            standardize=standardize,
            nilearn_dir=op.join(data_dir, "nilearn"),
            space="MNI152",
        )

        gclda_model = None
        ns_decoder = None
        ns_mask = None
        if sub_category == "names":
            baseline_label = f"{source}-{category}_embedding-{resources['model_name']}_section-{section}"
            ns_model_fn = op.join(results_dir, "baseline", f"model-neurosynth_{baseline_label}.pkl")
            gclda_model_fn = op.join(results_dir, "baseline", f"model-gclda_{baseline_label}.pkl")

            gclda_model = GCLDAModel.load(gclda_model_fn)
            ns_decoder = CorrelationDecoder.load(ns_model_fn)
            ns_masker = None
            if hasattr(ns_decoder, "results_"):
                ns_masker = getattr(ns_decoder.results_, "masker", None)
            if ns_masker is None and hasattr(ns_decoder, "masker"):
                ns_masker = ns_decoder.masker
            if ns_masker is not None and not hasattr(ns_masker, "clean_args_"):
                clean_kwargs = getattr(ns_masker, "clean_kwargs", None)
                ns_masker.clean_args_ = {} if clean_kwargs is None else clean_kwargs
            if ns_masker is not None:
                ns_mask = getattr(ns_masker, "mask_img", None)
                if ns_mask is None:
                    ns_mask = getattr(ns_masker, "mask_img_", None)

        desc = f"decode cnp {section}/{category}/{sub_category}/{resources['model_name']}"
        for prediction_label, img in tqdm(image_records, desc=desc, unit="img"):
            file_base_name = f"{prediction_label}_{resources['vocabulary_label']}_section-{section}"
            task_out_fn = f"{file_base_name}_pred-task_brainclip.csv"
            concept_out_fn = f"{file_base_name}_pred-concept_brainclip.csv"
            process_out_fn = f"{file_base_name}_pred-process_brainclip.csv"
            expected_outputs = [
                op.join(output_dir, task_out_fn),
                op.join(output_dir, concept_out_fn),
                op.join(output_dir, process_out_fn),
            ]
            if sub_category == "names":
                expected_outputs.extend(
                    [
                        op.join(output_dir, f"{file_base_name}_pred-task_neurosynth.csv"),
                        op.join(output_dir, f"{file_base_name}_pred-task_gclda.csv"),
                    ]
                )
            if skip_existing and all(op.exists(path) for path in expected_outputs):
                continue

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
                model=model,
                image_emb_gene=image_emb_gene,
            )
            task_prob_df.to_csv(op.join(output_dir, task_out_fn), index=False)
            concept_prob_df.to_csv(op.join(output_dir, concept_out_fn), index=False)
            process_prob_df.to_csv(op.join(output_dir, process_out_fn), index=False)

            if sub_category != "names":
                continue

            ns_out_fn = f"{file_base_name}_pred-task_neurosynth.csv"
            gclda_out_fn = f"{file_base_name}_pred-task_gclda.csv"

            gclda_img = _resample_for_reference(img, getattr(gclda_model, "mask", None))
            gclda_predictions_df, _ = gclda_decode_map(gclda_model, gclda_img)
            gclda_predictions_df = gclda_predictions_df.sort_values(by="Weight", ascending=False).head(
                topk
            )
            gclda_predictions_df = gclda_predictions_df.reset_index()
            gclda_predictions_df.columns = ["pred", "weight"]
            gclda_predictions_df.to_csv(op.join(output_dir, gclda_out_fn), index=False)

            ns_img = _resample_for_reference(img, ns_mask)
            ns_predictions_df = ns_decoder.transform(ns_img)
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
