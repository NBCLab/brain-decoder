import argparse
import multiprocessing as mp
import itertools
import json
import os
import os.path as op
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm
from jobs.utils import (
    DEFAULT_MODEL_IDS,
    DEFAULT_SECTIONS,
    build_cognitiveatlas,
    load_decoding_resources,
)

HCP_COLLECTION_ID = 457

HCP_REPRESENTATIVE_MAPS = [
    {
        "domain_key": "emotion",
        "task_code": "EMOTION",
        "image_id": 3128,
        "contrast_label": "Faces vs Shapes",
        "filename": "tfMRI_EMOTION_FACES-SHAPES_zstat1.nii.gz",
        "task_name": "emotion processing fMRI task paradigm",
    },
    {
        "domain_key": "gambling",
        "task_code": "GAMBLING",
        "image_id": 3137,
        "contrast_label": "Reward",
        "filename": "tfMRI_GAMBLING_REWARD_zstat1.nii.gz",
        "task_name": "gambling fMRI task paradigm",
    },
    {
        "domain_key": "language",
        "task_code": "LANGUAGE",
        "image_id": 3142,
        "contrast_label": "Story vs Math",
        "filename": "tfMRI_LANGUAGE_STORY-MATH_zstat1.nii.gz",
        "task_name": "language processing fMRI task paradigm",
    },
    {
        "domain_key": "motor",
        "task_code": "MOTOR",
        "image_id": 3152,
        "contrast_label": "Average",
        "filename": "tfMRI_MOTOR_AVG_zstat1.nii.gz",
        "task_name": "motor fMRI task paradigm",
    },
    {
        "domain_key": "relational",
        "task_code": "RELATIONAL",
        "image_id": 8820,
        "contrast_label": "Relational vs Match",
        "filename": "tfMRI_RELATIONAL_REL-MATCH.nii_tstat1.nii.gz",
        "task_name": "relational processing fMRI task paradigm",
    },
    {
        "domain_key": "social",
        "task_code": "SOCIAL",
        "image_id": 3180,
        "contrast_label": "TOM vs Random",
        "filename": "tfMRI_SOCIAL_TOM-RANDOM_zstat1.nii.gz",
        "task_name": "social cognition (theory of mind) fMRI task paradigm",
    },
    {
        "domain_key": "working_memory",
        "task_code": "WM",
        "image_id": 3190,
        "contrast_label": "2-Back vs 0-Back",
        "filename": "tfMRI_WM_2BK-0BK_zstat1.nii.gz",
        "task_name": "working memory fMRI task paradigm",
    },
]


def _download_file(url, destination, timeout=120):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
    return destination


def _get_parser():
    parser = argparse.ArgumentParser(description="Reproduce the HCP benchmark with published NiCLIP assets")
    parser.add_argument(
        "--workdir",
        dest="workdir",
        default="/tmp/brain-decoder-hcp",
        help="Root directory for downloaded assets and benchmark outputs.",
    )
    parser.add_argument(
        "--sections",
        dest="sections",
        nargs="+",
        default=list(DEFAULT_SECTIONS),
        help="Text sections to evaluate.",
    )
    parser.add_argument(
        "--model_ids",
        dest="model_ids",
        nargs="+",
        default=list(DEFAULT_MODEL_IDS),
        help="Embedding model identifiers to evaluate.",
    )
    parser.add_argument(
        "--sources",
        dest="sources",
        nargs="+",
        default=["cogatlasred", "cogatlas"],
        help="Vocabulary sources to evaluate.",
    )
    parser.add_argument(
        "--sub_categories",
        dest="sub_categories",
        nargs="+",
        default=["combined", "names"],
        help="Vocabulary embedding variants to evaluate.",
    )
    parser.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=20,
        help="Number of predictions to keep per image.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help="Optional device override.",
    )
    parser.add_argument(
        "--devices",
        dest="devices",
        nargs="+",
        default=None,
        help="Optional list of devices to use in parallel (e.g., cuda:0 cuda:1 cuda:2 cuda:3).",
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        default=None,
        help="Number of parallel config workers. Defaults to the number of devices in --devices.",
    )
    parser.add_argument(
        "--max_images",
        dest="max_images",
        type=int,
        default=None,
        help="Optional cap on the number of benchmark maps. Useful for runtime estimation.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing prediction CSVs instead of skipping them.",
    )
    parser.add_argument(
        "--skip_eval",
        dest="skip_eval",
        action="store_true",
        help="Skip aggregate evaluation. Useful when running a subset of images for timing.",
    )
    parser.add_argument(
        "--fetch_timeout",
        dest="fetch_timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds for OSF and NeuroVault downloads.",
    )
    return parser


def _prepare_assets(workdir, fetch_timeout=300):
    from braindec import fetcher

    fetcher.download_asset("brain_mask_mni152_2mm", destination_root=workdir, timeout=fetch_timeout)
    fetcher.download_asset("cognitive_atlas", destination_root=workdir, timeout=fetch_timeout)
    fetcher.download_osf_folder("data/vocabulary", destination_root=workdir, timeout=fetch_timeout)


def _prepare_minimal_assets(workdir, sections, model_ids, sources, sub_categories, fetch_timeout=300):
    from braindec import fetcher
    from jobs.utils import get_model_name

    fetcher.download_asset("brain_mask_mni152_2mm", destination_root=workdir, timeout=fetch_timeout)
    fetcher.download_asset("cognitive_atlas", destination_root=workdir, timeout=fetch_timeout)

    required_pubmed = set()
    required_baseline = set()
    required_vocabulary = set()

    for section in sections:
        for model_id in model_ids:
            model_name = get_model_name(model_id)
            required_pubmed.add(
                f"model-clip_section-{section}_embedding-{model_name}_best.pth"
            )

    for source, sub_category, section, model_id in itertools.product(
        sources,
        sub_categories,
        sections,
        model_ids,
    ):
        model_name = get_model_name(model_id)
        vocabulary_label = f"vocabulary-{source}_task-{sub_category}_embedding-{model_name}"
        required_vocabulary.add(f"vocabulary-{source}_task.txt")
        required_vocabulary.add(f"{vocabulary_label}.npy")
        required_vocabulary.add(f"{vocabulary_label}_section-{section}_prior.npy")

    if "names" in sub_categories:
        for source, section, model_id in itertools.product(sources, sections, model_ids):
            model_name = get_model_name(model_id)
            baseline_label = f"{source}-task_embedding-{model_name}_section-{section}"
            required_baseline.add(f"model-gclda_{baseline_label}.pkl")
            required_baseline.add(f"model-neurosynth_{baseline_label}.pkl")

    def _download_from_listing(folder_path, required_names, allow_csv_prior_fallback=False):
        listing = fetcher.list_remote_assets(remote_path=folder_path, timeout=fetch_timeout)
        items_by_name = {item["attributes"]["name"]: item for item in listing}

        for name in sorted(required_names):
            fallback_name = None
            if name.endswith("_prior.npy") and allow_csv_prior_fallback and name not in items_by_name:
                fallback_name = name[:-4] + ".csv"

            selected_name = name if name in items_by_name else fallback_name
            if selected_name is None:
                raise FileNotFoundError(f"{folder_path}/{name} was not found in published OSF assets.")

            print(f"Downloading {folder_path}/{selected_name}", flush=True)
            fetcher.download_osf_file(
                items_by_name[selected_name]["id"],
                destination_root=workdir,
                timeout=fetch_timeout,
            )

    _download_from_listing("results/pubmed", required_pubmed)
    _download_from_listing("results/baseline", required_baseline)
    _download_from_listing("data/vocabulary", required_vocabulary, allow_csv_prior_fallback=True)


def _prepare_hcp_inputs(workdir, fetch_timeout=300):
    image_dir = Path(workdir) / "data" / "hcp" / "neurovault"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in HCP_REPRESENTATIVE_MAPS:
        url = f"https://neurovault.org/media/images/{HCP_COLLECTION_ID}/{item['filename']}"
        local_path = _download_file(url, image_dir / item["filename"], timeout=fetch_timeout)
        rows.append({**item, "image_path": str(local_path)})
    return pd.DataFrame(rows)


def _prepare_nilearn_assets(workdir):
    from nilearn import datasets

    difumo_kwargs = {
        "dimension": 512,
        "resolution_mm": 2,
        "data_dir": op.join(workdir, "data", "nilearn"),
    }
    try:
        datasets.fetch_atlas_difumo(legacy_format=False, **difumo_kwargs)
    except TypeError:
        datasets.fetch_atlas_difumo(**difumo_kwargs)


def _build_ground_truth(mapping_df, data_dir):
    concept_to_process_fn = op.join(data_dir, "cognitive_atlas", "concept_to_process.json")
    cognitiveatlas = build_cognitiveatlas(data_dir, reduced=True, concept_to_process_fn=concept_to_process_fn)

    records = {}
    doc_rows = []
    for row in mapping_df.to_dict(orient="records"):
        task_idx = cognitiveatlas.get_task_idx_from_names(row["task_name"])
        concept_names = cognitiveatlas.get_concept_names_from_idx(cognitiveatlas.task_to_concept_idxs[task_idx])
        process_names = cognitiveatlas.get_process_names_from_idx(cognitiveatlas.task_to_process_idxs[task_idx])
        records[row["domain_key"]] = {
            "task": [row["task_name"]],
            "concept": concept_names.tolist(),
            "domain": process_names.tolist(),
        }
        doc_rows.append(
            {
                **row,
                "concepts": "; ".join(concept_names.tolist()),
                "domains": "; ".join(process_names.tolist()),
            }
        )

    return records, pd.DataFrame(doc_rows)


def _iter_prediction_jobs(sources, sections, model_ids, sub_categories):
    for source, section, model_id, sub_category in itertools.product(
        sources,
        sections,
        model_ids,
        sub_categories,
    ):
        yield {
            "source": source,
            "section": section,
            "model_id": model_id,
            "sub_category": sub_category,
        }


def _resolve_devices(device=None, devices=None):
    if devices:
        return devices

    if device is not None:
        return [device]

    import torch

    if torch.cuda.is_available():
        return [f"cuda:{device_idx}" for device_idx in range(torch.cuda.device_count())]

    return ["cpu"]


def _run_prediction_job(job, mapping_records, workdir, topk, device, overwrite=False, show_progress=False):
    import nibabel as nib
    from nilearn.image import resample_to_img
    from nimare.annotate.gclda import GCLDAModel
    from nimare.decode.continuous import CorrelationDecoder, gclda_decode_map

    from braindec.embedding import ImageEmbedding
    from braindec.model import build_model
    from braindec.predict import image_to_labels_hierarchical
    from braindec.utils import _get_device, images_have_same_fov

    data_dir = op.join(workdir, "data")
    results_dir = op.join(workdir, "results")
    mask_img = nib.load(op.join(data_dir, "MNI152_2x2x2_brainmask.nii.gz"))
    device = _get_device() if device is None else device

    output_dir = op.join(results_dir, "predictions_hcp_nv")
    os.makedirs(output_dir, exist_ok=True)

    def _resample_for_reference(image, reference_img):
        if reference_img is None or images_have_same_fov(image, reference_img):
            return image
        return resample_to_img(image, reference_img)

    source = job["source"]
    section = job["section"]
    model_id = job["model_id"]
    sub_category = job["sub_category"]
    reduced = source == "cogatlasred"
    concept_to_process_fn = op.join(data_dir, "cognitive_atlas", "concept_to_process.json")
    cognitiveatlas = build_cognitiveatlas(data_dir, reduced=reduced, concept_to_process_fn=concept_to_process_fn)
    resources = load_decoding_resources(
        op.join(workdir, "results"),
        op.join(data_dir, "vocabulary"),
        source,
        "task",
        sub_category,
        model_id,
        section,
    )
    model = build_model(resources["model_path"], device=device)
    image_emb_gene = ImageEmbedding(
        standardize=False,
        nilearn_dir=op.join(data_dir, "nilearn"),
        space="MNI152",
    )
    gclda_model = None
    ns_decoder = None
    ns_masker = None

    if sub_category == "names":
        baseline_label = f"{source}-task_embedding-{resources['model_name']}_section-{section}"
        gclda_model = GCLDAModel.load(op.join(results_dir, "baseline", f"model-gclda_{baseline_label}.pkl"))
        ns_decoder = CorrelationDecoder.load(
            op.join(results_dir, "baseline", f"model-neurosynth_{baseline_label}.pkl")
        )
        if hasattr(ns_decoder, "results_"):
            ns_masker = getattr(ns_decoder.results_, "masker", None)
        if ns_masker is None and hasattr(ns_decoder, "masker"):
            ns_masker = ns_decoder.masker
        if ns_masker is not None and not hasattr(ns_masker, "clean_args_"):
            clean_kwargs = getattr(ns_masker, "clean_kwargs", None)
            ns_masker.clean_args_ = {} if clean_kwargs is None else clean_kwargs

    iterator = mapping_records
    if show_progress:
        iterator = tqdm(
            mapping_records,
            total=len(mapping_records),
            desc=(
                f"images {source}/{section}/{resources['model_name']}/{sub_category}"
            ),
            leave=False,
        )

    started_at = time.perf_counter()
    brainclip_outputs = 0
    gclda_outputs = 0
    ns_outputs = 0

    for row in iterator:
        img = nib.load(row["image_path"])
        if not images_have_same_fov(img, mask_img):
            img = resample_to_img(img, mask_img)

        file_base_name = f"{row['task_code']}_{resources['vocabulary_label']}_section-{section}"
        task_out_fn = op.join(output_dir, f"{file_base_name}_pred-task_brainclip.csv")
        concept_out_fn = op.join(output_dir, f"{file_base_name}_pred-concept_brainclip.csv")
        process_out_fn = op.join(output_dir, f"{file_base_name}_pred-process_brainclip.csv")

        if overwrite or not op.exists(task_out_fn):
            task_prob_df, concept_prob_df, process_prob_df = image_to_labels_hierarchical(
                img,
                resources["model_path"],
                resources["vocabulary"],
                resources["vocabulary_emb"],
                resources["vocabulary_prior"],
                cognitiveatlas,
                topk=topk,
                logit_scale=20.0,
                model=model,
                image_emb_gene=image_emb_gene,
                data_dir=data_dir,
                device=device,
            )
            task_prob_df.to_csv(task_out_fn, index=False)
            concept_prob_df.to_csv(concept_out_fn, index=False)
            process_prob_df.to_csv(process_out_fn, index=False)
            brainclip_outputs += 3

        if sub_category != "names":
            continue

        ns_out_fn = op.join(output_dir, f"{file_base_name}_pred-task_neurosynth.csv")
        gclda_out_fn = op.join(output_dir, f"{file_base_name}_pred-task_gclda.csv")

        if overwrite or not op.exists(gclda_out_fn):
            gclda_img = _resample_for_reference(img, getattr(gclda_model, "mask", None))
            gclda_predictions_df, _ = gclda_decode_map(gclda_model, gclda_img)
            gclda_predictions_df = gclda_predictions_df.sort_values(by="Weight", ascending=False).head(topk)
            gclda_predictions_df = gclda_predictions_df.reset_index()
            gclda_predictions_df.columns = ["pred", "weight"]
            gclda_predictions_df.to_csv(gclda_out_fn, index=False)
            gclda_outputs += 1

        if overwrite or not op.exists(ns_out_fn):
            ns_mask = None
            if ns_masker is not None:
                ns_mask = getattr(ns_masker, "mask_img", None)
                if ns_mask is None:
                    ns_mask = getattr(ns_masker, "mask_img_", None)
            ns_img = _resample_for_reference(img, ns_mask)
            ns_predictions_df = ns_decoder.transform(ns_img)
            feature_group = f"{source}-task_section-{section}_annot-tfidf__"
            vocabulary_names = [feature.replace(feature_group, "") for feature in ns_predictions_df.index.values]
            ns_predictions_df.index = vocabulary_names
            ns_predictions_df = ns_predictions_df.sort_values(by="r", ascending=False).head(topk)
            ns_predictions_df = ns_predictions_df.reset_index()
            ns_predictions_df.columns = ["pred", "corr"]
            ns_predictions_df.to_csv(ns_out_fn, index=False)
            ns_outputs += 1

    elapsed_seconds = time.perf_counter() - started_at
    return {
        **job,
        "device": str(device),
        "num_images": len(mapping_records),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_image": elapsed_seconds / max(len(mapping_records), 1),
        "brainclip_outputs_written": brainclip_outputs,
        "gclda_outputs_written": gclda_outputs,
        "neurosynth_outputs_written": ns_outputs,
    }


def _run_predictions(
    mapping_df,
    workdir,
    sections,
    model_ids,
    sources,
    sub_categories,
    topk,
    device,
    devices,
    num_workers,
    overwrite,
):
    jobs = list(_iter_prediction_jobs(sources, sections, model_ids, sub_categories))
    mapping_records = mapping_df.to_dict(orient="records")
    resolved_devices = _resolve_devices(device=device, devices=devices)
    num_workers = len(resolved_devices) if num_workers is None else num_workers
    num_workers = max(1, min(num_workers, len(jobs)))

    summaries = []
    started_at = time.perf_counter()

    if num_workers == 1:
        for job in tqdm(jobs, desc="configs", unit="config"):
            summaries.append(
                _run_prediction_job(
                    job,
                    mapping_records,
                    workdir,
                    topk=topk,
                    device=resolved_devices[0],
                    overwrite=overwrite,
                    show_progress=True,
                )
            )
    else:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
            future_to_job = {}
            for job_idx, job in enumerate(jobs):
                assigned_device = resolved_devices[job_idx % len(resolved_devices)]
                future = executor.submit(
                    _run_prediction_job,
                    job,
                    mapping_records,
                    workdir,
                    topk,
                    assigned_device,
                    overwrite,
                    False,
                )
                future_to_job[future] = job

            for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="configs", unit="config"):
                summaries.append(future.result())

    elapsed_seconds = time.perf_counter() - started_at
    summary_df = pd.DataFrame(summaries).sort_values(
        by=["source", "section", "model_id", "sub_category"]
    )
    summary_df["total_wall_seconds"] = elapsed_seconds
    runtime_summary_fn = op.join(workdir, "results", "hcp_prediction_runtime.csv")
    summary_df.to_csv(runtime_summary_fn, index=False)

    return op.join(workdir, "results", "predictions_hcp_nv"), runtime_summary_fn


def main(
    workdir,
    sections,
    model_ids,
    sources,
    sub_categories,
    topk=20,
    device=None,
    devices=None,
    num_workers=None,
    max_images=None,
    overwrite=False,
    skip_eval=False,
    fetch_timeout=300,
):
    workdir = op.abspath(workdir)
    _prepare_minimal_assets(
        workdir,
        sections,
        model_ids,
        sources,
        sub_categories,
        fetch_timeout=fetch_timeout,
    )
    mapping_df = _prepare_hcp_inputs(workdir, fetch_timeout=fetch_timeout)
    _prepare_nilearn_assets(workdir)
    if max_images is not None:
        mapping_df = mapping_df.head(max_images).copy()

    ground_truth, documentation_df = _build_ground_truth(mapping_df, op.join(workdir, "data"))
    ground_truth_path = op.join(workdir, "data", "hcp", "ground_truth.json")
    os.makedirs(op.dirname(ground_truth_path), exist_ok=True)
    with open(ground_truth_path, "w") as file_obj:
        json.dump(ground_truth, file_obj, indent=2)

    documentation_df.to_csv(op.join(workdir, "data", "hcp", "benchmark_mapping.csv"), index=False)

    prediction_dir, runtime_summary_fn = _run_predictions(
        mapping_df,
        workdir,
        sections=sections,
        model_ids=model_ids,
        sources=sources,
        sub_categories=sub_categories,
        topk=topk,
        device=device,
        devices=devices,
        num_workers=num_workers,
        overwrite=overwrite,
    )

    if skip_eval:
        print(f"Skipped evaluation. Runtime summary written to {runtime_summary_fn}", flush=True)
        return

    from jobs.decoding_eval import main as eval_main

    eval_main(
        data_dir=op.join(workdir, "data"),
        results_dir=op.join(workdir, "results"),
        sections=sections,
        model_ids=model_ids,
        sources=sources,
        categories=["task"],
        sub_categories=sub_categories,
        models=["neurosynth", "gclda", "brainclip"],
        prediction_dir=prediction_dir,
        image_dir=op.join(workdir, "data", "hcp", "neurovault"),
        ground_truth_fn=ground_truth_path,
        output_fn=op.join(workdir, "results", "eval-hcp-group_results.csv"),
    )


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
