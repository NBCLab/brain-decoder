import argparse
import os.path as op

import nimare
import numpy as np

from braindec.embedding import TextEmbedding


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
    content = "abstract"

    dset = nimare.dataset.Dataset.load(op.join(data_dir, f"neurostore-{content}_dset.pkl"))

    model_id = "mistralai/Mistral-7B-v0.1"  # BrainGPT/BrainGPT-7B-v0.2, mistralai/Mistral-7B-v0.1
    model_name = model_id.split("/")[-1]
    generator = TextEmbedding(model_name=model_id)
    text_embedding_arr = generator(dset.texts[content].to_list())  # body

    np.save(op.join(data_dir, f"text_embedding_{content}_{model_name}.npy"), text_embedding_arr)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
