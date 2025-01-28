import argparse
import os.path as op

import nimare
import numpy as np

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
        default=True,
        type=bool,
        help="Whether to standardize the image embeddings (default: True)",
    )
    return parser


def main(project_dir, standardize=True):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    nilearn_data = op.join(data_dir, "nilearn")

    dset = nimare.dataset.Dataset.load(op.join(data_dir, "dset-pubmed_nimare.pkl"))
    images = _coordinates_to_image(dset)

    generator = ImageEmbedding(standardize=standardize, data_dir=nilearn_data)
    image_embedding_arr = generator(images)

    np.save(op.join(data_dir, "image_coord-MKDA_embedding-DiFuMo.npy"), image_embedding_arr)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
