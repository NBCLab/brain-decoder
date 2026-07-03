"""Download published braindec assets from OSF."""

import argparse
import os
import os.path as op
import time
from pathlib import Path

import pandas as pd
import requests

OSF_API_BASE = "https://api.osf.io/v2"
OSF_URL = "https://osf.io/{}/download"
DEFAULT_OSF_NODE = "dsj56"
DEFAULT_PROVIDER = "osfstorage"
CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 2

# Legacy term/classification files used by the old vocabulary fetcher.
OSF_DICT = {
    "source-neuroquery_desc-gclda_features.csv": "trcxs",
    "source-neuroquery_desc-gclda_classification.csv": "93dvg",
    "source-neuroquery_desc-lda_features.csv": "u68w7",
    "source-neuroquery_desc-lda_classification.csv": "mtwvc",
    "source-neuroquery_desc-term_features.csv": "xtjna",
    "source-neuroquery_desc-term_classification.csv": "ypqzx",
    "source-neurosynth_desc-gclda_features.csv": "jcrkd",
    "source-neurosynth_desc-gclda_classification.csv": "p7nd9",
    "source-neurosynth_desc-lda_features.csv": "ve3nj",
    "source-neurosynth_desc-lda_classification.csv": "9mrxb",
    "source-neurosynth_desc-term_features.csv": "hyjrk",
    "source-neurosynth_desc-term_classification.csv": "sd4wy",
}

# Public assets documented in the README.
OSF_ASSETS = {
    "text_embeddings_braingpt_v02_body": {
        "type": "file",
        "file_id": "v748f",
        "description": "Body text embeddings used in the paper.",
    },
    "image_embeddings_difumo512_mkda": {
        "type": "file",
        "file_id": "nu2s7",
        "description": "Normalized MKDA/DiFuMo image embeddings used in the paper.",
    },
    "example_model_braingpt_v02_body": {
        "type": "file",
        "file_id": "u3cxh",
        "description": "Example pretrained CLIP model.",
    },
    "example_vocabulary_cogatlasred_task": {
        "type": "file",
        "file_id": "8m2fz",
        "description": "Reduced Cognitive Atlas task vocabulary.",
    },
    "example_vocabulary_embeddings_cogatlasred_task": {
        "type": "file",
        "file_id": "nza7b",
        "description": "Example vocabulary embeddings for reduced CogAt tasks.",
    },
    "example_vocabulary_prior_cogatlasred_task": {
        "type": "file",
        "file_id": "v82za",
        "description": "Example vocabulary prior for reduced CogAt tasks.",
    },
    "brain_mask_mni152_2mm": {
        "type": "file",
        "file_id": "jzvry",
        "description": "Brain mask used in prediction examples.",
    },
    "cognitive_atlas": {
        "type": "folder",
        "remote_path": "data/cognitive_atlas",
        "description": "Cognitive Atlas snapshots and reduced task mapping.",
    },
    "results_pubmed": {
        "type": "folder",
        "remote_path": "results/pubmed",
        "description": "Published pretrained CLIP outputs from the paper.",
    },
    "results_baseline": {
        "type": "folder",
        "remote_path": "results/baseline",
        "description": "Published baseline decoder models from the paper.",
    },
}

OSF_BUNDLES = {
    "example_prediction": [
        "example_model_braingpt_v02_body",
        "example_vocabulary_cogatlasred_task",
        "example_vocabulary_embeddings_cogatlasred_task",
        "example_vocabulary_prior_cogatlasred_task",
        "brain_mask_mni152_2mm",
        "cognitive_atlas",
    ],
    "training_embeddings": [
        "text_embeddings_braingpt_v02_body",
        "image_embeddings_difumo512_mkda",
    ],
    "paper_results": [
        "results_pubmed",
        "results_baseline",
    ],
    "all_readme_assets": [
        "text_embeddings_braingpt_v02_body",
        "image_embeddings_difumo512_mkda",
        "example_model_braingpt_v02_body",
        "example_vocabulary_cogatlasred_task",
        "example_vocabulary_embeddings_cogatlasred_task",
        "example_vocabulary_prior_cogatlasred_task",
        "brain_mask_mni152_2mm",
        "cognitive_atlas",
        "results_pubmed",
        "results_baseline",
    ],
}


def get_data_dir(data_dir=None):
    """Return the default braindec data directory without importing heavy modules."""
    if data_dir is None:
        data_dir = os.environ.get("BRAINDEC_DATA", os.path.join("~", "braindec-data"))
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _should_retry_request(error):
    response = getattr(error, "response", None)
    if response is None:
        return True
    if response.status_code == 403 and "osf" in response.url:
        return True
    return response.status_code >= 500


def _request_json(url, params=None, timeout=60, max_retries=DEFAULT_MAX_RETRIES):
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            if attempt == max_retries - 1 or not _should_retry_request(error):
                raise
            time.sleep(DEFAULT_RETRY_BACKOFF_SECONDS**attempt)

    raise last_error


def _download_to_file(
    url,
    destination,
    overwrite=False,
    timeout=60,
    chunk_size=CHUNK_SIZE,
    max_retries=DEFAULT_MAX_RETRIES,
):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    tmp_destination = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with tmp_destination.open("wb") as file_obj:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file_obj.write(chunk)
            break
        except requests.RequestException as error:
            last_error = error
            if tmp_destination.exists():
                tmp_destination.unlink()
            if attempt == max_retries - 1 or not _should_retry_request(error):
                raise
            time.sleep(DEFAULT_RETRY_BACKOFF_SECONDS**attempt)
    else:
        raise last_error

    tmp_destination.replace(destination)
    return destination


def _normalize_remote_path(remote_path):
    remote_path = remote_path.strip("/")
    if not remote_path:
        return "/"
    return f"/{remote_path}/"


def _materialized_path_to_local_path(materialized_path, destination_root):
    relative_path = materialized_path.lstrip("/")
    if relative_path.endswith("/"):
        relative_path = relative_path[:-1]
    return Path(destination_root) / relative_path


def _get_osf_url(filename):
    osf_id = OSF_DICT[filename]
    return OSF_URL.format(osf_id)


def _get_osf_file_metadata(file_id, timeout=60):
    return _request_json(f"{OSF_API_BASE}/files/{file_id}/", timeout=timeout)["data"]


def _iter_children(node_id=DEFAULT_OSF_NODE, folder_id=None, provider=DEFAULT_PROVIDER, timeout=60):
    if folder_id is None:
        url = f"{OSF_API_BASE}/nodes/{node_id}/files/{provider}/"
    else:
        url = f"{OSF_API_BASE}/nodes/{node_id}/files/{provider}/{folder_id}/"

    while url:
        payload = _request_json(url, timeout=timeout)
        for item in payload["data"]:
            yield item
        url = payload["links"].get("next")


def _get_folder_item(node_id, remote_path, provider=DEFAULT_PROVIDER, timeout=60):
    normalized_path = _normalize_remote_path(remote_path)
    if normalized_path == "/":
        return None

    folder_id = None
    parts = [part for part in normalized_path.strip("/").split("/") if part]
    materialized_path = "/"
    for part in parts:
        children = list(_iter_children(node_id=node_id, folder_id=folder_id, provider=provider, timeout=timeout))
        match = None
        for child in children:
            attrs = child["attributes"]
            if attrs["kind"] == "folder" and attrs["name"] == part:
                match = child
                break
        if match is None:
            raise FileNotFoundError(f"Remote OSF folder {remote_path!r} was not found in node {node_id}.")
        folder_id = match["id"]
        materialized_path = match["attributes"]["materialized_path"]

    return {
        "id": folder_id,
        "materialized_path": materialized_path,
    }


def _get_remote_item(node_id, remote_path, provider=DEFAULT_PROVIDER, timeout=60):
    normalized_path = remote_path.strip("/")
    if not normalized_path:
        return None

    folder_id = None
    current_item = None
    parts = [part for part in normalized_path.split("/") if part]
    for idx, part in enumerate(parts):
        children = list(_iter_children(node_id=node_id, folder_id=folder_id, provider=provider, timeout=timeout))
        current_item = None
        for child in children:
            if child["attributes"]["name"] == part:
                current_item = child
                break

        if current_item is None:
            raise FileNotFoundError(f"Remote OSF path {remote_path!r} was not found in node {node_id}.")

        is_last = idx == len(parts) - 1
        kind = current_item["attributes"]["kind"]
        if not is_last:
            if kind != "folder":
                raise FileNotFoundError(
                    f"Remote OSF path {remote_path!r} traversed through non-folder component {part!r}."
                )
            folder_id = current_item["id"]

    return current_item


def list_remote_assets(node_id=DEFAULT_OSF_NODE, remote_path="/", provider=DEFAULT_PROVIDER, timeout=60):
    """List files and folders under an OSF path."""
    folder = _get_folder_item(node_id, remote_path, provider=provider, timeout=timeout)
    folder_id = None if folder is None else folder["id"]
    return list(_iter_children(node_id=node_id, folder_id=folder_id, provider=provider, timeout=timeout))


def download_osf_file(
    file_id,
    destination_root=".",
    overwrite=False,
    use_materialized_path=True,
    destination=None,
    timeout=60,
):
    """Download a single OSF file by id."""
    file_data = _get_osf_file_metadata(file_id, timeout=timeout)
    attrs = file_data["attributes"]
    download_url = file_data["links"]["download"]

    if destination is None:
        if use_materialized_path:
            destination = _materialized_path_to_local_path(attrs["materialized_path"], destination_root)
        else:
            destination = Path(destination_root) / attrs["name"]

    destination = Path(destination)
    return _download_to_file(download_url, destination, overwrite=overwrite, timeout=timeout)


def download_osf_folder(
    remote_path,
    destination_root=".",
    node_id=DEFAULT_OSF_NODE,
    provider=DEFAULT_PROVIDER,
    overwrite=False,
    timeout=60,
):
    """Download all files under a folder path from the published OSF project."""
    folder = _get_folder_item(node_id, remote_path, provider=provider, timeout=timeout)
    downloaded = []
    queue = [folder["id"]]

    while queue:
        folder_id = queue.pop(0)
        for item in _iter_children(node_id=node_id, folder_id=folder_id, provider=provider, timeout=timeout):
            attrs = item["attributes"]
            if attrs["kind"] == "folder":
                queue.append(item["id"])
                continue

            destination = _materialized_path_to_local_path(attrs["materialized_path"], destination_root)
            downloaded.append(
                _download_to_file(
                    item["links"]["download"],
                    destination,
                    overwrite=overwrite,
                    timeout=timeout,
                )
            )

    return downloaded


def download_osf_path(
    remote_path,
    destination_root=".",
    node_id=DEFAULT_OSF_NODE,
    provider=DEFAULT_PROVIDER,
    overwrite=False,
    timeout=60,
):
    """Download a published OSF file or folder by its remote path."""
    item = _get_remote_item(node_id=node_id, remote_path=remote_path, provider=provider, timeout=timeout)
    if item["attributes"]["kind"] == "folder":
        return download_osf_folder(
            remote_path,
            destination_root=destination_root,
            node_id=node_id,
            provider=provider,
            overwrite=overwrite,
            timeout=timeout,
        )

    destination = _materialized_path_to_local_path(item["attributes"]["materialized_path"], destination_root)
    return [
        _download_to_file(
            item["links"]["download"],
            destination,
            overwrite=overwrite,
            timeout=timeout,
        )
    ]


def get_available_assets():
    """Return the names of downloadable assets and bundles."""
    return {
        "assets": sorted(OSF_ASSETS),
        "bundles": sorted(OSF_BUNDLES),
    }


def download_asset(name, destination_root=".", overwrite=False, node_id=DEFAULT_OSF_NODE, timeout=60):
    """Download a named asset from the built-in manifest."""
    if name not in OSF_ASSETS:
        raise KeyError(f"Unknown asset {name!r}. Available assets: {sorted(OSF_ASSETS)}")

    asset = OSF_ASSETS[name]
    if asset["type"] == "file":
        return [download_osf_file(asset["file_id"], destination_root=destination_root, overwrite=overwrite, timeout=timeout)]
    if asset["type"] == "folder":
        return download_osf_folder(
            asset["remote_path"],
            destination_root=destination_root,
            node_id=node_id,
            overwrite=overwrite,
            timeout=timeout,
        )

    raise ValueError(f"Unsupported asset type {asset['type']!r}.")


def download_bundle(name, destination_root=".", overwrite=False, node_id=DEFAULT_OSF_NODE, timeout=60):
    """Download a predefined bundle of assets."""
    if name not in OSF_BUNDLES:
        raise KeyError(f"Unknown bundle {name!r}. Available bundles: {sorted(OSF_BUNDLES)}")

    downloaded = []
    for asset_name in OSF_BUNDLES[name]:
        downloaded.extend(
            download_asset(
                asset_name,
                destination_root=destination_root,
                overwrite=overwrite,
                node_id=node_id,
                timeout=timeout,
            )
        )
    return downloaded


def _fetch_vocabulary(
    source="neurosynth",
    subsample=None,
    data_dir=None,
    overwrite=False,
    verbose=1,
):
    """Fetch legacy term features/classifications from OSF and return the vocabulary."""
    subsample = ["Functional"] if subsample is None else subsample
    data_dir = get_data_dir(data_dir)
    vocabulary_dir = get_data_dir(os.path.join(data_dir, "vocabulary"))

    filename = f"source-{source}_desc-term_features.csv"
    features_fn = _download_to_file(
        _get_osf_url(filename),
        Path(vocabulary_dir) / filename,
        overwrite=overwrite,
    )
    del verbose  # preserved for backward compatibility

    df = pd.read_csv(features_fn)

    filename_classification = f"source-{source}_desc-term_classification.csv"
    classification_fn = _download_to_file(
        _get_osf_url(filename_classification),
        Path(vocabulary_dir) / filename_classification,
        overwrite=overwrite,
    )

    classification_df = pd.read_csv(classification_fn, index_col="Classification")
    classification = classification_df.index.tolist()
    keep = [index for index, class_name in enumerate(classification) if class_name in subsample]
    return df.values[keep].flatten().tolist()


def _get_cogatlas_data(url):
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        tasks = response.json()
    except requests.RequestException as error:
        print(f"Error retrieving tasks: {error}")
        return None

    output = {}
    for task in tasks:
        if ("name" in task) and task["name"] and ("definition_text" in task):
            output[task["name"]] = task["definition_text"]
        else:
            print(f"Task {task} does not have a name or definition_text")

    return output


def get_cogatlas_tasks():
    """Fetch task definitions from the Cognitive Atlas API."""
    return _get_cogatlas_data("https://www.cognitiveatlas.org/api/v-alpha/task")


def get_cogatlas_concepts():
    """Fetch concept definitions from the Cognitive Atlas API."""
    return _get_cogatlas_data("https://www.cognitiveatlas.org/api/v-alpha/concept")


def _get_parser():
    parser = argparse.ArgumentParser(description="Download published braindec assets from OSF")
    parser.add_argument(
        "--destination_root",
        dest="destination_root",
        default=".",
        help="Root directory under which OSF materialized paths will be recreated.",
    )
    parser.add_argument(
        "--asset",
        dest="assets",
        nargs="+",
        default=None,
        help="One or more named assets to download.",
    )
    parser.add_argument(
        "--bundle",
        dest="bundles",
        nargs="+",
        default=None,
        help="One or more predefined bundles to download.",
    )
    parser.add_argument(
        "--folder",
        dest="folders",
        nargs="+",
        default=None,
        help="One or more raw OSF folder paths to download, for example data/cognitive_atlas.",
    )
    parser.add_argument(
        "--list",
        dest="list_only",
        action="store_true",
        help="Print available built-in assets and bundles.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing local files.",
    )
    return parser


def _main(argv=None):
    options = _get_parser().parse_args(argv)

    if options.list_only:
        available = get_available_assets()
        print("Assets:")
        for asset in available["assets"]:
            print(f"  - {asset}")
        print("Bundles:")
        for bundle in available["bundles"]:
            print(f"  - {bundle}")
        return

    downloaded = []
    if options.assets:
        for asset in options.assets:
            downloaded.extend(
                download_asset(
                    asset,
                    destination_root=options.destination_root,
                    overwrite=options.overwrite,
                )
            )

    if options.bundles:
        for bundle in options.bundles:
            downloaded.extend(
                download_bundle(
                    bundle,
                    destination_root=options.destination_root,
                    overwrite=options.overwrite,
                )
            )

    if options.folders:
        for folder in options.folders:
            downloaded.extend(
                download_osf_folder(
                    folder,
                    destination_root=options.destination_root,
                    overwrite=options.overwrite,
                )
            )

    if not (options.assets or options.bundles or options.folders or options.list_only):
        raise SystemExit("Select at least one of --asset, --bundle, --folder, or --list.")

    for path in downloaded:
        print(path)


if __name__ == "__main__":
    _main()
