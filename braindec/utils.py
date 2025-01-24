"""Miscellaneous functions used for analyses."""

import os

import nibabel as nib
import numpy as np
import torch
from neuromaps import transforms
from neuromaps.datasets import fetch_atlas
from nibabel.gifti import GiftiDataArray

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


def _get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")  # Use MPS for mac
    elif torch.cuda.is_available():
        return torch.device("cuda")  # Use CUDA for Nvidia GPUs
    else:
        return torch.device("cpu")  # Default to CPU


def _zero_medial_wall(data_lh, data_rh, space="fsLR", density="32k", neuromaps_dir=None):
    """Remove medial wall from data in fsLR space."""
    atlas = fetch_atlas(space, density, data_dir=neuromaps_dir, verbose=0)

    medial_lh, medial_rh = atlas["medial"]
    medial_arr_lh = nib.load(medial_lh).agg_data()
    medial_arr_rh = nib.load(medial_rh).agg_data()

    data_arr_lh = data_lh.agg_data()
    data_arr_rh = data_rh.agg_data()
    data_arr_lh[np.where(medial_arr_lh == 0)] = 0
    data_arr_rh[np.where(medial_arr_rh == 0)] = 0

    data_lh.remove_gifti_data_array(0)
    data_rh.remove_gifti_data_array(0)
    data_lh.add_gifti_data_array(GiftiDataArray(data_arr_lh))
    data_rh.add_gifti_data_array(GiftiDataArray(data_arr_rh))

    return data_lh, data_rh


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


def _vol_to_surf(metamap, space="fsLR", density="32k", return_hemis=False, neuromaps_dir=None):
    """Transform 4D metamaps from volume to surface space."""
    if space == "fsLR":
        metamap_lh, metamap_rh = transforms.mni152_to_fslr(metamap, fslr_density=density)
    elif space == "fsaverage":
        metamap_lh, metamap_rh = transforms.mni152_to_fsaverage(metamap, fsavg_density=density)
    elif space == "civet":
        metamap_lh, metamap_rh = transforms.mni152_to_civet(metamap, civet_density=density)

    metamap_lh, metamap_rh = _zero_medial_wall(
        metamap_lh,
        metamap_rh,
        space=space,
        density=density,
        neuromaps_dir=neuromaps_dir,
    )
    if return_hemis:
        return metamap_lh, metamap_rh

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
