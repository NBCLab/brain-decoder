import argparse
import os.path as op

import numpy as np

from braindec.dataset import _get_vocabulary
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


def _write_vocabulary(vocabulary, vocabulary_fn, vocabulary_emb, vocabulary_emb_fn):
    np.save(vocabulary_emb_fn, vocabulary_emb)

    with open(vocabulary_fn, "w") as f:
        for item in vocabulary:
            f.write("%s\n" % item)


def _preprocess_text(text):
    # Remove special characters from the text; e.g., "fMRI", "localizer", "task", "paradigm"
    text = [
        t.replace("fMRI", "").replace("localizer", "").replace("task", "").replace("paradigm", "")
        for t in text
    ]
    return text


def main(project_dir):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    voc_dir = op.join(data_dir, "vocabulary")
    source = "cogatlas"  # cogatlas, neurosynth
    alpha = 0.5

    model_id = "BrainGPT/BrainGPT-7B-v0.2"  # BrainGPT/BrainGPT-7B-v0.2
    # model_id = "meta-llama/Llama-2-7b-chat-hf"  # mistralai/Mistral-7B-v0.1
    model_name = model_id.split("/")[-1]
    generator = TextEmbedding(model_name=model_id)  # , device="cpu"

    if source == "cogatlas":
        tasks_dict, concepts_dict = _get_vocabulary(source="cogatlas", data_dir=data_dir)
        for category, data_dict in zip(["task", "concept"], [tasks_dict, concepts_dict]):
            names_org, definitions_org = zip(*data_dict.items())

            # names = _preprocess_text(names_org)
            # definitions = _preprocess_text(definitions_org)
            names = names_org
            definitions = definitions_org

            names_emb = generator(names)
            definitions_emb = generator(definitions)
            combined_emb = names_emb * alpha + definitions_emb * (1 - alpha)

            names_lb = f"vocabulary-{source}_{category}-names_embedding-{model_name}"
            names_fn = op.join(voc_dir, f"{names_lb}.txt")
            names_emb_fn = op.join(voc_dir, f"{names_lb}.npy")

            definitions_lb = f"vocabulary-{source}_{category}-definitions_embedding-{model_name}"
            definitions_fn = op.join(voc_dir, f"{definitions_lb}.txt")
            definitions_emb_fn = op.join(voc_dir, f"{definitions_lb}.npy")

            category_lb = f"vocabulary-{source}_{category}-combined_embedding-{model_name}"
            combined_fn = op.join(voc_dir, f"{category_lb}.txt")
            combined_emb_fn = op.join(voc_dir, f"{category_lb}.npy")

            _write_vocabulary(names_org, names_fn, names_emb, names_emb_fn)
            _write_vocabulary(names_org, definitions_fn, definitions_emb, definitions_emb_fn)
            _write_vocabulary(names_org, combined_fn, combined_emb, combined_emb_fn)
    else:
        vocabulary = _get_vocabulary(source=source, data_dir=data_dir)
        vocabulary_emb = generator(vocabulary)

        vocabulary_fn = op.join(voc_dir, f"vocabulary-{source}.txt")
        vocabulary_emb_fn = op.join(voc_dir, f"vocabulary-{source}_embedding-{model_name}.npy")
        _write_vocabulary(vocabulary, vocabulary_fn, vocabulary_emb, vocabulary_emb_fn)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
