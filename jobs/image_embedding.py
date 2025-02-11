import argparse
import os.path as op

import nimare
import numpy as np
from sklearn.preprocessing import StandardScaler

from braindec.embedding import ImageEmbedding, _coordinates_to_image


def _get_parser():
    parser = argparse.ArgumentParser(description="Calculate image embeddings from coordinates")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--standardize",
        dest="standardize",
        default=False,
        type=bool,
        help="Whether to standardize the image embeddings (default: True)",
    )
    return parser


def main(project_dir, standardize=False):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    nilearn_data = op.join(data_dir, "nilearn")

    dset = nimare.dataset.Dataset.load(op.join(data_dir, "dset-pubmed_nimare.pkl"))
    images = _coordinates_to_image(dset)

    generator = ImageEmbedding(standardize=standardize, data_dir=nilearn_data)
    image_emb = generator(images)

    # Standardize image embeddings
    scaler = StandardScaler()
    image_emb_std = scaler.fit_transform(image_emb)

    # Normalize image embeddings
    image_emb_norm = image_emb / (np.linalg.norm(image_emb, axis=-1) + 1e-8)

    prefix = "coord-MKDA_embedding-DiFuMo.npy"
    np.save(op.join(data_dir, f"image-raw_{prefix}"), image_emb)
    np.save(op.join(data_dir, f"image-standardized_{prefix}"), image_emb_std)
    np.save(op.join(data_dir, f"image-normalized_{prefix}"), image_emb_norm)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
