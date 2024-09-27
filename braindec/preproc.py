import os.path as op

import numpy as np
import torch
from nimare.extract import fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.kernel import MKDAKernel
from torch.utils.data import Dataset

from braindec.utils import get_data_dir


def _get_dataset(dset_nm, data_dir):
    data_dir = get_data_dir(op.join(data_dir, "data", dset_nm))

    if dset_nm == "neurosynth":
        files = fetch_neurosynth(
            data_dir=data_dir,
            version="7",
            overwrite=False,
            source="abstract",
            vocab="terms",
        )
    elif dset_nm == "neuroquery":
        files = fetch_neuroquery(
            data_dir=data_dir,
            version="1",
            overwrite=False,
            source="combined",
            vocab="neuroquery6308",
            type="tfidf",
        )

    dataset_db = files[0]

    dset = convert_neurosynth_to_dataset(
        coordinates_file=dataset_db["coordinates"],
        metadata_file=dataset_db["metadata"],
        annotations_files=dataset_db["features"],
    )
    dset.update_path(data_dir)

    return dset


class MRIDataset(Dataset):
    def __init__(self, project_dir):
        dset = _get_dataset("neurosynth", project_dir)
        kernel = MKDAKernel()
        image_output = kernel.transform(dset, return_type="image")

        self.num_samples = len(image_output)
        self.image_shape = image_output[0].get_fdata().shape

        # Get the image data
        data = [img.get_fdata() for img in image_output]
        self.data = np.array(data).astype(np.float32)

        # Get the image features
        tfidf_columns = dset.annotations.filter(like="terms_abstract_tfidf__")

        # Get the most important term for each image
        terms = tfidf_columns.idxmax(axis=1).to_list()
        terms = [term.replace("terms_abstract_tfidf__", "") for term in terms]
        self.labels = np.array(terms)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.labels[idx], torch.from_numpy(self.data[idx])
