import argparse
import os.path as op

import numpy as np
import nimare
from braindec.embedding import ImageEmbedding


def _get_parser():
    parser = argparse.ArgumentParser(description="Run gradient-decoding workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--n_cores",
        dest="n_cores",
        default=4,
        required=False,
        help="CPUs",
    )
    return parser




def main(project_dir, n_cores):
    n_cores = int(n_cores)
    project_dir = op.abspath(project_dir)
    nilearn_data = op.join(project_dir, "data", "nilearn_data")
    dset = nimare.dataset.Dataset.load(op.join(project_dir, "data", "neurostore_nimare.pkl"))

    image_emb_gene = ImageEmbedding(data_dir=nilearn_data, n_jobs=n_cores)
    image_embedding_arr = image_emb_gene(dset)
    
    print(image_embedding_arr.shape)
    np.save(op.join(project_dir, "data", "image_embedding.npy", image_embedding_arr))


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()