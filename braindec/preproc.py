import os.path as op
from collections import Counter

import numpy as np
import pandas as pd
import torch
from nimare.extract import fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.kernel import MKDAKernel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from braindec.utils import get_data_dir


def _get_dataset(dset_nm, data_dir):
    data_dir = get_data_dir(op.join(data_dir))

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


def create_cuboid_mask(brain_mask):
    # Step 1: Find the bounding box of the brain mask
    nonzero = np.nonzero(brain_mask)
    min_coords = np.min(nonzero, axis=1)
    max_coords = np.max(nonzero, axis=1)

    # Step 2: Calculate dimensions and pad to nearest power of 2 for each axis
    dims = max_coords - min_coords + 1
    target_dims = 2 ** np.ceil(np.log2(dims)).astype(int)

    # Step 3: Create the final cuboid mask
    cuboid_mask = np.zeros(target_dims, dtype=bool)

    # Calculate padding for each dimension
    pad_width = [(0, target_dim - dim) for target_dim, dim in zip(target_dims, dims)]

    # Extract and pad the relevant part of the brain mask
    extracted_brain = brain_mask[
        min_coords[0] : max_coords[0] + 1,
        min_coords[1] : max_coords[1] + 1,
        min_coords[2] : max_coords[2] + 1,
    ]
    padded_brain = np.pad(extracted_brain, pad_width, mode="constant")

    cuboid_mask[:] = padded_brain

    return cuboid_mask


def trim_image(image, mask):
    return image[mask]


class MRIDataset(Dataset):
    def __init__(self, project_dir, dataset="neurosynth"):
        self.project_dir = project_dir
        self.dataset = dataset

        dset = _get_dataset(dataset, project_dir)
        kernel = MKDAKernel()
        image_output = kernel.transform(dset, return_type="image")

        self.num_samples = len(image_output)
        self.image_shape = image_output[0].get_fdata().shape

        # cuboid_mask = create_cuboid_mask(dset.masker.mask_img_.get_fdata())
        # data = [trim_image(img.get_fdata(), cuboid_mask) for img in image_output]

        # Get the image data
        data = [img.get_fdata() for img in image_output]
        data = np.array(data).astype(np.float32)
        self.data = data[:, np.newaxis, :, :, :]

        # Get the image features
        filtered_df = self._filter_terms(dset.annotations)

        # Get the most important term for each image
        terms = filtered_df.idxmax(axis=1).to_list()
        terms = [term.replace("terms_abstract_tfidf__", "") for term in terms]
        self.labels = np.array(terms)

        self.num_classes = len(np.unique(self.labels))

        # Encode labels to integers
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.labels[idx],
            torch.tensor(self.encoded_labels[idx]),
            torch.from_numpy(self.data[idx]),
        )

    def _filter_terms(self, data_df):
        # Get manual annotation of functional terms
        classification_df = pd.read_csv(
            op.join(
                self.project_dir,
                self.dataset,
                "source-neurosynth_desc-term3228_classification.csv",
            )
        )
        functional_mask = (classification_df["Classification"] == "Functional").values

        # Filter the dataframe to keep only the functional terms
        functional_df = data_df.filter(like="terms_abstract_tfidf__").loc[:, functional_mask]

        # Get the term with the highest value for each row
        terms = functional_df.idxmax(axis=1).to_list()
        term_counts = Counter(terms)

        # Filter terms with count greater than 30
        filtered_terms = [term for term, count in term_counts.items() if count > 30]
        filtered_columns = [col for col in functional_df.columns if col in filtered_terms]
        return functional_df[filtered_columns]
