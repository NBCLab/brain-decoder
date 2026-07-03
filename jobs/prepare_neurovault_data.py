"""Download and normalize NeuroVault datasets used for cross-dataset decoding."""

import argparse
import json
import os
import os.path as op
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

IBC_COLLECTIONS = [2138, 6618]
CNP_COLLECTIONS = [2606]
DEFAULT_TIMEOUT = 120


def _slugify(value):
    value = re.sub(r"[^0-9A-Za-z]+", "-", str(value).strip())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "image"


def _request_json(url, timeout=DEFAULT_TIMEOUT):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _iter_collection_images(collection_id, timeout=DEFAULT_TIMEOUT):
    url = f"https://neurovault.org/api/collections/{collection_id}/images/"
    while url:
        payload = _request_json(url, timeout=timeout)
        for row in payload["results"]:
            yield row
        url = payload.get("next")


def _fetch_collection_metadata(collection_id, timeout=DEFAULT_TIMEOUT):
    return _request_json(f"https://neurovault.org/api/collections/{collection_id}/", timeout=timeout)


def _infer_dataset_task_family(dataset_name, row):
    if dataset_name == "cnp":
        name = (row.get("name") or "").strip()
        return name.split()[0].upper() if name else "UNKNOWN"
    if dataset_name == "ibc":
        task = row.get("task") or ""
        return task.strip() or "UNKNOWN"
    return "UNKNOWN"


def _normalize_image_record(dataset_name, collection_id, row):
    image_id = row["id"]
    source_name = row.get("name") or f"image-{image_id}"
    prediction_label = f"nv{image_id}-{_slugify(source_name)}"
    return {
        "dataset": dataset_name,
        "collection_id": collection_id,
        "image_id": image_id,
        "prediction_label": prediction_label,
        "source_name": source_name,
        "file_url": row.get("file"),
        "map_type": row.get("map_type"),
        "modality": row.get("modality"),
        "task_family": _infer_dataset_task_family(dataset_name, row),
        "task_name": row.get("cognitive_paradigm_cogatlas"),
        "task_cogatlas_id": row.get("cognitive_paradigm_cogatlas_id"),
        "task_code": row.get("task"),
        "contrast_definition": row.get("contrast_definition"),
    }


def _download_file(url, destination, timeout=DEFAULT_TIMEOUT, overwrite=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    tmp_destination = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp_destination.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
    tmp_destination.replace(destination)
    return destination


def _download_images(records, output_dir, num_workers=8, timeout=DEFAULT_TIMEOUT, overwrite=False):
    output_dir = Path(output_dir)
    outputs = []
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        future_to_record = {}
        for record in records:
            if not record.get("file_url"):
                continue
            destination = output_dir / f"{record['prediction_label']}.nii.gz"
            future = executor.submit(
                _download_file,
                record["file_url"],
                destination,
                timeout,
                overwrite,
            )
            future_to_record[future] = (record, destination)

        for future in tqdm(as_completed(future_to_record), total=len(future_to_record), desc="download images"):
            record, destination = future_to_record[future]
            future.result()
            outputs.append(
                {
                    **record,
                    "local_path": str(destination),
                    "filename": destination.name,
                }
            )

    outputs_df = pd.DataFrame(outputs).sort_values(by=["collection_id", "image_id"]).reset_index(drop=True)
    return outputs_df


def _write_collection_metadata(dataset_name, collection_id, collection_meta, records, destination_root):
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    with (destination_root / f"collection-{collection_id}.json").open("w") as file_obj:
        json.dump(collection_meta, file_obj, indent=2)
    pd.DataFrame(records).to_csv(destination_root / f"collection-{collection_id}_manifest.csv", index=False)


def _get_parser():
    parser = argparse.ArgumentParser(description="Download and normalize NeuroVault collections.")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        default=op.abspath(op.join(op.dirname(__file__), "..")),
        help="Path to the repository root.",
    )
    parser.add_argument(
        "--datasets",
        dest="datasets",
        nargs="+",
        default=["cnp", "ibc"],
        choices=["cnp", "ibc"],
        help="Datasets to prepare.",
    )
    parser.add_argument(
        "--ibc_collections",
        dest="ibc_collections",
        nargs="+",
        type=int,
        default=[2138, 6618],
        help="IBC NeuroVault collection IDs to index.",
    )
    parser.add_argument(
        "--cnp_collections",
        dest="cnp_collections",
        nargs="+",
        type=int,
        default=[2606],
        help="CNP NeuroVault collection IDs to index.",
    )
    parser.add_argument(
        "--download_ibc_images",
        dest="download_ibc_images",
        action="store_true",
        help="Download IBC NIfTI images in addition to manifests.",
    )
    parser.add_argument(
        "--download_cnp_images",
        dest="download_cnp_images",
        action="store_true",
        help="Download CNP NIfTI images in addition to manifests.",
    )
    parser.add_argument(
        "--ibc_download_collections",
        dest="ibc_download_collections",
        nargs="+",
        type=int,
        default=[2138],
        help="IBC collection IDs whose images should be downloaded when --download_ibc_images is set.",
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        default=8,
        help="Concurrent download worker count.",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing downloaded files.",
    )
    return parser


def main(
    project_dir,
    datasets,
    ibc_collections,
    cnp_collections,
    download_ibc_images=False,
    download_cnp_images=False,
    ibc_download_collections=None,
    num_workers=8,
    timeout=DEFAULT_TIMEOUT,
    overwrite=False,
):
    project_dir = op.abspath(project_dir)
    ibc_download_collections = [2138] if ibc_download_collections is None else ibc_download_collections

    for dataset_name in datasets:
        collection_ids = ibc_collections if dataset_name == "ibc" else cnp_collections
        metadata_dir = Path(project_dir) / "data" / dataset_name / "metadata"
        image_dir = Path(project_dir) / "data" / dataset_name
        metadata_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        all_records = []
        for collection_id in tqdm(collection_ids, desc=f"{dataset_name} collections"):
            collection_meta = _fetch_collection_metadata(collection_id, timeout=timeout)
            records = [
                _normalize_image_record(dataset_name, collection_id, row)
                for row in _iter_collection_images(collection_id, timeout=timeout)
            ]
            _write_collection_metadata(dataset_name, collection_id, collection_meta, records, metadata_dir)
            all_records.extend(records)

        manifest_df = pd.DataFrame(all_records).sort_values(by=["collection_id", "image_id"]).reset_index(drop=True)
        manifest_fn = image_dir / "neurovault_manifest.csv"
        manifest_df.to_csv(manifest_fn, index=False)

        should_download = (dataset_name == "ibc" and download_ibc_images) or (
            dataset_name == "cnp" and download_cnp_images
        )
        if not should_download:
            continue

        if dataset_name == "ibc":
            download_records = [
                record for record in all_records if record["collection_id"] in set(ibc_download_collections)
            ]
        else:
            download_records = list(all_records)

        downloaded_df = _download_images(
            download_records,
            output_dir=image_dir,
            num_workers=num_workers,
            timeout=timeout,
            overwrite=overwrite,
        )
        downloaded_df.to_csv(image_dir / "mapping.csv", index=False)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
