import argparse
import os
import os.path as op

import nimare
import numpy as np
import pandas as pd

from braindec.cogatlas import CognitiveAtlas
from braindec.embedding import TextEmbedding
from braindec.utils import _generate_counts


def _get_parser():
    parser = argparse.ArgumentParser(description="Run gradient-decoding workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    return parser


def _write_vocabulary(vocabulary, vocabulary_fn):
    with open(vocabulary_fn, "w") as f:
        for item in vocabulary:
            f.write("%s\n" % item)


def _get_counts_tfidf(data_df, vocabulary, text_column):
    stoplist = op.join(nimare.utils.get_resource_path(), "neurosynth_stoplist.txt")
    with open(stoplist, "r") as fo:
        stop_words = fo.read().splitlines()

    stop_words = stop_words + ["task", "paradigm"]

    # Remove stop words from vocabulary
    new_vocabulary = []
    for vocab in vocabulary:
        vocab_word = vocab.split()

        new_vocab_word = []
        for v in vocab_word:
            if v.lower() not in stop_words:
                new_vocab_word.append(v)

        new_vocab_word_str = " ".join(new_vocab_word).lower()
        new_vocabulary.append(new_vocab_word_str)

    vocabulary_mapping = dict(zip(vocabulary, new_vocabulary))

    # Remove duplicates
    new_vocabulary = list(set(new_vocabulary))
    count_df, tfidf_df = _generate_counts(
        data_df,
        vocabulary=new_vocabulary,
        stop_words=stop_words,
        text_column=text_column,
    )

    count_arr = []
    tfidf_arr = []
    for word in vocabulary:
        new_word = vocabulary_mapping[word]
        # Select column from tfidf_df where the column name is the word
        count_arr.append(count_df[new_word].values)
        tfidf_arr.append(tfidf_df[new_word].values)

    return np.array(count_arr), np.array(tfidf_arr)


def _annotate_dset(dset, vocabulary, data, prefix):
    vocabulary_names = [f"{prefix}__{word}" for word in vocabulary]

    annot_df = pd.DataFrame(
        index=dset.annotations["id"],
        columns=vocabulary_names,
        data=data,
    )

    annotations = dset.annotations.copy()
    annotations = pd.merge(annotations, annot_df, left_on="id", right_index=True)
    new_dset = dset.copy()
    new_dset.annotations = annotations
    return new_dset


def main(project_dir):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    voc_dir = op.join(data_dir, "vocabulary")
    source = "cogatlas"
    alpha = 0.5
    categories = ["task", "concept"]
    sub_categories = ["names", "definitions", "combined"]
    sections = ["abstract", "body"]
    os.makedirs(voc_dir, exist_ok=True)

    dset = nimare.dataset.Dataset.load(op.join(data_dir, "dset-pubmed_nimare.pkl"))

    cognitiveatlas = CognitiveAtlas(
        data_dir=data_dir,
        task_snapshot=op.join(data_dir, "cognitive_atlas", "task_snapshot-02-19-25.json"),
        concept_snapshot=op.join(data_dir, "cognitive_atlas", "concept_snapshot-02-19-25.json"),
    )

    model_ids = [
        "BrainGPT/BrainGPT-7B-v0.2",
        "mistralai/Mistral-7B-v0.1",
        "BrainGPT/BrainGPT-7B-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
    ]
    for model_id in model_ids:
        model_name = model_id.split("/")[-1]
        generator = TextEmbedding(model_name=model_id)

        for category in categories:
            if category == "task":
                names = cognitiveatlas.task_names
                definitions = cognitiveatlas.task_definitions
            elif category == "concept":
                names = cognitiveatlas.concept_names
                definitions = cognitiveatlas.concept_definitions
            else:
                raise ValueError(f"Invalid category: {category}")

            names_fn = op.join(voc_dir, f"vocabulary-{source}_{category}.txt")
            _write_vocabulary(names, names_fn)

            names_emb = generator(names)
            definitions_emb = generator(definitions)
            combined_emb = names_emb * alpha + definitions_emb * (1 - alpha)

            embeddings = [names_emb, definitions_emb, combined_emb]
            for sub_category, emb in zip(sub_categories, embeddings):
                lb = f"vocabulary-{source}_{category}-{sub_category}_embedding-{model_name}"
                emb_fn = op.join(voc_dir, f"{lb}.npy")
                np.save(emb_fn, emb)

            # Get counts and tfidf, only for one model
            if model_id == "BrainGPT/BrainGPT-7B-v0.2":
                for section in sections:
                    base_lb = f"vocabulary-{source}_{category}-names"
                    counts_fn = op.join(voc_dir, f"{base_lb}_section-{section}_counts.npy")
                    tfidf_fn = op.join(voc_dir, f"{base_lb}_section-{section}_tfidf.npy")

                    counts, tfidf = _get_counts_tfidf(dset.texts, names, section)
                    np.save(counts_fn, counts)
                    np.save(tfidf_fn, tfidf)

                    # Annotate dataset
                    for data, data_lb in zip([counts, tfidf], ["counts", "tfidf"]):
                        prefix = f"{source}-{category}_section-{section}_annot-{data_lb}"
                        dset = _annotate_dset(dset, names, data.T, prefix)

                dset.save(op.join(data_dir, "dset-pubmed_annotated_nimare.pkl"))


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
