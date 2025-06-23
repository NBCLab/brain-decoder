"""Miscellaneous functions used for analyses."""

import os
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from neuromaps import transforms
from neuromaps.datasets import fetch_atlas
from nibabel.gifti import GiftiDataArray
from nilearn.surface import PolyMesh, SurfaceImage, load_surf_mesh
from nimare.utils import get_resource_path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Number of vertices in total without the medial wall
N_VERTICES = {
    "fsLR": {
        "32k": 59412,
        "164k": 298261,
    },
    "fsaverage": {
        "3k": 4661,
        "10k": 18715,
        "41k": 74947,
        "164k": 299881,
    },
    "civet": {
        "41k": 76910,
    },
}

# Number of vertices per hemisphere including the medial wall
N_VERTICES_PH = {
    "fsLR": {
        "32k": 32492,
        "164k": 163842,
    },
    "fsaverage": {
        "3k": 2562,
        "10k": 10242,
        "41k": 40962,
        "164k": 163842,
    },
    "civet": {
        "41k": 40962,
    },
}


def get_data_dir(data_dir=None):
    """Get path to gradec data directory.

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'GRADEC_DATA'; if that is not set, will
        use `~/gradec-data` instead. Default: None

    Returns
    -------
    data_dir : str
        Path to use as data directory

    Notes
    -----
    Taken from Neuromaps.
    https://github.com/netneurolab/neuromaps/blob/abf5a5c3d3d011d644b56ea5c6a3953cedd80b37/
    neuromaps/datasets/utils.py#LL91C1-L115C20
    """
    if data_dir is None:
        data_dir = os.environ.get("BRAINDEC_DATA", os.path.join("~", "braindec-data"))
    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir


def _read_vocabulary(vocabulary_fn, vocabulary_emb_fn, vocabulary_prior_fn=None):
    with open(vocabulary_fn, "r") as f:
        vocabulary = [line.strip() for line in f]

    if vocabulary_prior_fn is not None:
        return vocabulary, np.load(vocabulary_emb_fn), np.load(vocabulary_prior_fn)
    else:
        return vocabulary, np.load(vocabulary_emb_fn)


def _get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")  # Use MPS for mac
    elif torch.cuda.is_available():
        return torch.device("cuda")  # Use CUDA for Nvidia GPUs
    else:
        return torch.device("cpu")  # Default to CPU


def _zero_medial_wall(
    data_lh,
    data_rh,
    space="fsLR",
    density="32k",
    return_arrays=False,
    neuromaps_dir=None,
):
    """Remove medial wall from data in fsLR space."""
    atlas = fetch_atlas(space, density, data_dir=neuromaps_dir, verbose=0)

    medial_lh, medial_rh = atlas["medial"]
    medial_arr_lh = nib.load(medial_lh).agg_data()
    medial_arr_rh = nib.load(medial_rh).agg_data()

    if isinstance(data_lh, str):
        data_lh = nib.load(data_lh)
        data_rh = nib.load(data_rh)

    data_arr_lh = data_lh.agg_data()
    data_arr_rh = data_rh.agg_data()
    if isinstance(data_arr_lh, tuple):
        for vol_i in range(len(data_arr_lh)):
            data_arr_lh[vol_i][np.where(medial_arr_lh == 0)] = 0
            data_arr_rh[vol_i][np.where(medial_arr_rh == 0)] = 0
        data_arr_lh = np.array(data_arr_lh).T
        data_arr_rh = np.array(data_arr_rh).T
    else:
        data_arr_lh[np.where(medial_arr_lh == 0)] = 0
        data_arr_rh[np.where(medial_arr_rh == 0)] = 0

    if return_arrays:
        return data_arr_lh, data_arr_rh, atlas

    data_lh.remove_gifti_data_array(0)
    data_rh.remove_gifti_data_array(0)
    data_lh.add_gifti_data_array(GiftiDataArray(data_arr_lh))
    data_rh.add_gifti_data_array(GiftiDataArray(data_arr_rh))

    return data_lh, data_rh, atlas


def _rm_medial_wall(
    data_lh,
    data_rh,
    space="fsLR",
    density="32k",
    join=True,
    neuromaps_dir=None,
):
    """Remove medial wall from data in fsLR space.

    Data in 32k fs_LR space (e.g., Human Connectome Project data) often in
    GIFTI format include the medial wall in their data arrays, which results
    in a total of 64984 vertices across hemispheres. This function removes
    the medial wall vertices to produce a data array with the full 59412 vertices,
    which is used to perform functional decoding.

    This function was adapted from :func:`surfplot.utils.add_fslr_medial_wall`.

    Parameters
    ----------
    data : numpy.ndarray
        Surface vertices. Must have exactly 32492 vertices per hemisphere.
    join : bool
        Return left and right hemipsheres in the same arrays. Default: True.

    Returns
    -------
    numpy.ndarray
        Vertices with medial wall excluded (59412 vertices total).

    ValueError
    ------
    `data` has the incorrect number of vertices (59412 or 64984 only
        accepted)
    """
    assert data_lh.shape[0] == N_VERTICES_PH[space][density]
    assert data_rh.shape[0] == N_VERTICES_PH[space][density]

    atlas = fetch_atlas(space, density, data_dir=neuromaps_dir, verbose=0)

    medial_lh, medial_rh = atlas["medial"]
    wall_lh = nib.load(medial_lh).agg_data()
    wall_rh = nib.load(medial_rh).agg_data()

    data_lh = data_lh[np.where(wall_lh != 0)]
    data_rh = data_rh[np.where(wall_rh != 0)]

    if not join:
        return data_lh, data_rh

    data = np.hstack((data_lh, data_rh))
    assert data.shape[0] == N_VERTICES[space][density]
    return data


def _vol_to_surf(
    metamap,
    space="fsLR",
    density="32k",
    return_hemis=False,
    return_arrays=False,
    return_atlas=False,
    neuromaps_dir=None,
):
    """Transform 4D metamaps from volume to surface space."""
    if space == "fsLR":
        metamap_lh, metamap_rh = transforms.mni152_to_fslr(metamap, fslr_density=density)
    elif space == "fsaverage":
        metamap_lh, metamap_rh = transforms.mni152_to_fsaverage(metamap, fsavg_density=density)
    elif space == "civet":
        metamap_lh, metamap_rh = transforms.mni152_to_civet(metamap, civet_density=density)

    metamap_lh, metamap_rh, atlas = _zero_medial_wall(
        metamap_lh,
        metamap_rh,
        space=space,
        density=density,
        return_arrays=return_arrays,
        neuromaps_dir=neuromaps_dir,
    )
    if return_hemis and not return_arrays:
        return metamap_lh, metamap_rh
    if return_hemis and return_arrays:
        return metamap_lh, metamap_rh, atlas

    metamap_arr_lh = metamap_lh.agg_data()
    metamap_arr_rh = metamap_rh.agg_data()

    metamap_surf = _rm_medial_wall(
        metamap_arr_lh,
        metamap_arr_rh,
        space=space,
        density=density,
        neuromaps_dir=neuromaps_dir,
    )

    return metamap_surf


def _vol_surfimg(
    vol,
    space="fsLR",
    density="32k",
    neuromaps_dir=None,
):
    lh_data, rh_data, atlas = _vol_to_surf(
        vol,
        space=space,
        density=density,
        neuromaps_dir=neuromaps_dir,
        return_hemis=True,
        return_arrays=True,
        return_atlas=True,
    )

    lh_surfaces, rh_surfaces = atlas["midthickness"]
    lh_mesh = load_surf_mesh(lh_surfaces)
    rh_mesh = load_surf_mesh(rh_surfaces)

    mesh = PolyMesh(
        left=lh_mesh,
        right=rh_mesh,
    )

    data = {
        "left": lh_data,
        "right": rh_data,
    }

    return SurfaceImage(mesh=mesh, data=data)


def _generate_counts(
    text_df,
    vocabulary=None,
    stop_words=None,
    text_column="abstract",
    min_df=0.01,
    max_df=0.99,
):
    """Generate tf-idf/counts weights for unigrams/bigrams derived from textual data.

    Parameters
    ----------
    text_df : (D x 2) :obj:`pandas.DataFrame`
        A DataFrame with two columns ('id' and 'text'). D = document.

    Returns
    -------
    weights_df : (D x T) :obj:`pandas.DataFrame`
        A DataFrame where the index is 'id' and the columns are the
        unigrams/bigrams derived from the data. D = document. T = term.
    """
    if text_column not in text_df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    # Remove rows with empty text cells
    orig_ids = text_df["id"].tolist()
    text_df = text_df.fillna("")
    keep_ids = text_df.loc[text_df[text_column] != "", "id"]
    text_df = text_df.loc[text_df["id"].isin(keep_ids)]

    if len(keep_ids) != len(orig_ids):
        print(f"\t\tRetaining {len(keep_ids)}/{len(orig_ids)} studies", flush=True)

    ids = text_df["id"].tolist()
    text = text_df[text_column].tolist()

    if stop_words is None:
        stoplist = op.join(get_resource_path(), "neurosynth_stoplist.txt")
        with open(stoplist, "r") as fo:
            stop_words = fo.read().splitlines()

    vectorizer_tfidf = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        vocabulary=vocabulary,
        stop_words=stop_words,
    )

    vectorizer_counts = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        vocabulary=vocabulary,
        stop_words=stop_words,
    )

    weights_df_list = []
    for vectorizer in [vectorizer_counts, vectorizer_tfidf]:
        weights = vectorizer.fit_transform(text).toarray()

        names = vectorizer.get_feature_names_out()
        names = [str(name) for name in names]
        weights_df = pd.DataFrame(weights, columns=names, index=ids)
        weights_df.index.name = "id"
        weights_df_list.append(weights_df)

    return weights_df_list
