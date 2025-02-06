"""Fetch data."""

import hashlib
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import requests
from nilearn.datasets._utils import fetch_single_file

from braindec.utils import get_data_dir

OSF_URL = "https://osf.io/{}/download"
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


def _get_osf_url(filename):
    osf_id = OSF_DICT[filename]
    return OSF_URL.format(osf_id)


def _mk_tmpdir(data_dir, file, url):
    """Make a temporary directory for fetching."""
    files_pickle = pickle.dumps([(file, url)])
    files_md5 = hashlib.md5(files_pickle).hexdigest()
    temp_dir = os.path.join(data_dir, files_md5)

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    return temp_dir


def _my_fetch_file(data_dir, filename, url, overwrite=False, resume=True, verbose=1):
    """Fetch a file from OSF."""
    path_name = os.path.join(data_dir, filename)
    if not os.path.exists(path_name) or overwrite:
        # Fetch file
        tmpdir = _mk_tmpdir(data_dir, filename, url)
        temp_fn = fetch_single_file(url, tmpdir, resume=resume, verbose=verbose)

        # Move and delete tmpdir
        shutil.move(temp_fn, path_name)
        shutil.rmtree(tmpdir)

    return path_name


def _fetch_vocabulary(
    source="neurosynth",
    subsample=["Functional"],
    data_dir=None,
    overwrite=False,
    resume=True,
    verbose=1,
):
    """Fetch features from OSF.

    Parameters
    ----------
    source : :obj:`str`
        Name of dataset.
    model_nm : :obj:`str`
        Name of model.
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory
    resume : :obj:`bool`, optional
        Whether to resume download of a partly-downloaded file.
        Default=True.
    verbose : :obj:`int`, optional
        Verbosity level (0 means no message).
        Default=1.

    Returns
    -------
    :class:`list` of str
        List of feature names.
    """
    data_dir = get_data_dir(data_dir)
    vocabulary_dir = get_data_dir(os.path.join(data_dir, "vocabulary"))

    filename = f"source-{source}_desc-term_features.csv"
    url = _get_osf_url(filename)

    features_fn = _my_fetch_file(
        vocabulary_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    df = pd.read_csv(features_fn)

    filename_classification = f"source-{source}_desc-term_classification.csv"
    url_classification = _get_osf_url(filename_classification)

    classification_fn = _my_fetch_file(
        vocabulary_dir,
        filename_classification,
        url_classification,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    classification_df = pd.read_csv(classification_fn, index_col="Classification")
    classification = classification_df.index.tolist()

    keep = np.array([c_i for c_i, class_ in enumerate(classification) if class_ in subsample])
    return df.values[keep].flatten().tolist()


def _get_cogatlas_data(url):
    try:
        # Send a GET request to the API
        response = requests.get(url)

        # Raise an exception for bad responses
        response.raise_for_status()

        # Parse the JSON response into a Python dictionary
        tasks = response.json()

    except requests.RequestException as e:
        print(f"Error retrieving tasks: {e}")
        return None

    output = {}
    for task in tasks:
        if ("name" in task) and (task["name"] != "") and ("definition_text" in task):
            output[task["name"]] = task["definition_text"]
        # elif "name" in task and "definition_text" not in task:
        #    output[task["name"]] = ""
        else:
            print(f"Task {task} does not have a name or definition_text")

    return output


def get_cogatlas_tasks():
    # API endpoint for tasks
    return _get_cogatlas_data("https://www.cognitiveatlas.org/api/v-alpha/task")


def get_cogatlas_concepts():
    # API endpoint for concepts
    return _get_cogatlas_data("https://www.cognitiveatlas.org/api/v-alpha/concept")
