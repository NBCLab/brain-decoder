import argparse
import os.path as op

import nimare
import numpy as np

from braindec.embedding import ImageEmbedding


def _get_parser():
    parser = argparse.ArgumentParser(description="Run gradient-decoding workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    return parser


def main(project_dir):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    nilearn_data = op.join(data_dir, "nilearn")
    content = "abstract"

    dset = nimare.dataset.Dataset.load(op.join(data_dir, f"neurostore-{content}_dset.pkl"))

    image_emb_gene = ImageEmbedding(data_dir=nilearn_data)
    image_embedding_arr = image_emb_gene(dset)

    print(image_embedding_arr.shape)
    np.save(op.join(data_dir, f"image_embedding_{content}.npy"), image_embedding_arr)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
