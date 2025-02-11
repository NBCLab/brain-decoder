import argparse
import os.path as op

import nimare
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help="Device to use for computation (default: None). Possible values: cpu, mps, cuda.",
    )
    return parser


def main(project_dir, section="abstract", model_id="mistralai/Mistral-7B-v0.1", device=None):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")

    dset = nimare.dataset.Dataset.load(op.join(data_dir, "dset-pubmed_nimare.pkl"))

    models = [
        "BrainGPT/BrainGPT-7B-v0.2",
        "BrainGPT/BrainGPT-7B-v0.1",
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
    ]
    for use_model in models:
        model_name = use_model.split("/")[-1]

        print(f"Extracting text embedding for {use_model}")
        generator = TextEmbedding(model_name=use_model, device=device)
        text_emb = generator(dset.texts[section].to_list())  # body

        # Standardize text embeddings
        scaler = StandardScaler()
        text_emb_std = scaler.fit_transform(text_emb)

        # Normalize text embeddings
        text_emb_norm = text_emb / (np.linalg.norm(text_emb, axis=-1, keepdims=True) + 1e-8)

        prefix = f"section-{section}_embedding-{model_name}.npy"
        np.save(op.join(data_dir, f"text-raw_{prefix}"), text_emb)
        np.save(op.join(data_dir, f"text-standardized_{prefix}"), text_emb_std)
        np.save(op.join(data_dir, f"text-normalized_{prefix}"), text_emb_norm)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
