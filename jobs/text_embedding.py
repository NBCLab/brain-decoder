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
    parser.add_argument(
        "--section",
        dest="section",
        default="abstract",
        help="Section to extract text from (default: abstract). Possible values: abstract, body.",
    )
    parser.add_argument(
        "--model_id",
        dest="model_id",
        default="mistralai/Mistral-7B-v0.1",
        help="Model ID for text embedding (default: mistralai/Mistral-7B-v0.1). Possible values: "
        "mistralai/Mistral-7B-v0.1, meta-llama/Llama-2-7b-chat-hf, BrainGPT/BrainGPT-7B-v0.1, "
        "BrainGPT/BrainGPT-7B-v0.2.",
    )
    return parser


def main(project_dir, section="abstract", model_id="mistralai/Mistral-7B-v0.1"):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    model_name = model_id.split("/")[-1]

    dset = nimare.dataset.Dataset.load(op.join(data_dir, "dset-pubmed_nimare.pkl"))

    generator = TextEmbedding(model_name=model_id)
    text_embedding_arr = generator(dset.texts[section].to_list())  # body

    np.save(
        op.join(data_dir, f"text_section-{section}_embedding-{model_name}.npy"),
        text_embedding_arr,
    )


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
