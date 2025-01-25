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

    if not op.exists(vocabulary_fn):
        with open(vocabulary_fn, "w") as f:
            for item in vocabulary:
                f.write("%s\n" % item)


def main(project_dir):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    source = "cogatlas"  # cogatlas, neurosynth
    max_length = 512

    model_id = "BrainGPT/BrainGPT-7B-v0.2"  # BrainGPT/BrainGPT-7B-v0.2, mistralai/Mistral-7B-v0.1
    model_name = model_id.split("/")[-1]
    generator = TextEmbedding(model_name=model_id, max_length=max_length)

    if source == "cogatlas":
        tasks_dict, concepts_dict = _get_vocabulary(source="cogatlas", data_dir=data_dir)
        for category, data_dict in zip(["task", "concept"], [tasks_dict, concepts_dict]):
            query, vocabulary = [], []
            for name, definition in data_dict.items():
                query.append(f"{name}: {definition}"[:max_length])
                vocabulary.append(name)

            vocabulary_emb = generator(query)

            vocabulary_fn = op.join(data_dir, f"vocabulary-{source}-{category}.txt")
            vocabulary_emb_fn = op.join(
                data_dir, f"vocabulary-{source}-{category}_embedding-{model_name}.npy"
            )
            _write_vocabulary(vocabulary, vocabulary_fn, vocabulary_emb, vocabulary_emb_fn)
    else:
        vocabulary = _get_vocabulary(source=source, data_dir=data_dir)
        vocabulary_emb = generator(vocabulary)

        vocabulary_fn = op.join(data_dir, f"vocabulary-{source}.txt")
        vocabulary_emb_fn = op.join(data_dir, f"vocabulary-{source}_embedding-{model_name}.npy")
        _write_vocabulary(vocabulary, vocabulary_fn, vocabulary_emb, vocabulary_emb_fn)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
