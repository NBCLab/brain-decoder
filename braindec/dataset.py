import json
import os.path as op
from collections import Counter
from glob import glob

import nimare
import numpy as np
import pandas as pd
import torch
from nimare.extract import fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.kernel import MKDAKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from braindec.utils import get_data_dir


def create_balanced_loaders(dataset, batch_size, train_size=0.7, val_size=0.15):
    # Get the targets from the dataset
    targets = dataset.encoded_labels

    # First split: train+val and test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=1 - train_size - val_size,
        stratify=targets,
        random_state=42,
    )

    # Second split: train and val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (train_size + val_size),
        stratify=[targets[i] for i in train_val_idx],
        random_state=42,
    )

    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def create_random_loaders(dataset, batch_size, train_size=0.7, val_size=0.15):
    # Get the targets from the dataset
    targets = dataset.labels

    # First split: train+val and test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=1 - train_size - val_size,
        random_state=42,
    )

    # Second split: train and val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (train_size + val_size),
        random_state=42,
    )

    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def _neurostore_to_nimare(data_dir):
    """
    Convert the NeuroStore dataset to a NiMARE dataset.
    """
    sorted_dirs = sorted(glob(op.join(data_dir, "*")))

    dataset_dict = {}
    for dset_dir in sorted_dirs:
        proc_dir = op.join(dset_dir, "processed")
        if not op.exists(proc_dir):
            continue

        print(f"Processing {dset_dir}")

        extract_dirs = sorted(glob(op.join(proc_dir, "*")))

        if len(extract_dirs) == 0:
            continue

        if len(extract_dirs) > 1:
            print(f"\tMultiple directories found in {proc_dir}")
            extracts = [op.basename(ext) for ext in extract_dirs]
            # Prioritize pubget if available
            if "pubget" in extracts:
                sel_dirs = op.join(proc_dir, "pubget")
            else:
                print(f"\tPubget not found in {extracts}")
                continue
        else:
            print(f"\tOnly one directory found in {proc_dir}")
            sel_dirs = extract_dirs[0]

        coord_fn = op.join(sel_dirs, "coordinates.csv")
        meta_fn = op.join(sel_dirs, "metadata.json")
        text_fn = op.join(sel_dirs, "text.txt")

        if not op.exists(coord_fn) or not op.exists(text_fn):
            print(f"\t\tCoordinates or text file not found: {coord_fn}, {text_fn}")
            continue

        assert op.exists(meta_fn), f"Metadata file not found: {meta_fn}"

        with open(meta_fn, "r") as file:
            metadata = json.load(file)

        with open(text_fn, "r") as file:
            body = file.read()

        try:
            coord_df = pd.read_csv(coord_fn)
        except pd.errors.EmptyDataError:
            print(f"\t\tEmpty coordinates file: {coord_fn}")
            continue

        print(f"\t\t{coord_df.shape[0]} coordinates found")
        if coord_df.empty or body == "":
            continue

        id_ = op.basename(dset_dir)

        if id_ not in dataset_dict:
            dataset_dict[id_] = {}

        if "contrasts" not in dataset_dict[id_]:
            dataset_dict[id_]["contrasts"] = {}

        # For now, lets have only one contrast per study
        contrast_name = "1"

        dataset_dict[id_]["contrasts"][contrast_name] = {
            "coords": {
                "space": "MNI",
                "x": coord_df["x"].values,
                "y": coord_df["y"].values,
                "z": coord_df["z"].values,
            },
            "text": {
                "title": metadata["title"],
                "keywords": metadata["keywords"],
                "abstract": metadata["abstract"],
                "body": body,
            },
        }

    return nimare.dataset.Dataset(dataset_dict)


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
        kernel = MKDAKernel()  # Replace by peark2image
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
        self.soft_labels = normalize(filtered_df.to_numpy(), norm="l1", axis=1)

        # Encode labels to integers
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.labels[idx],
            torch.tensor(self.soft_labels[idx]),
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
