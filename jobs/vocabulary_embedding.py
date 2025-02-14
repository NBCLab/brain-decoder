import argparse
import os.path as op
import re

import nimare
import numpy as np

from braindec.dataset import _get_vocabulary
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


def _preprocess_text(text):
    # Remove special characters from the text; e.g., "fMRI", "localizer", "task", "paradigm"
    # text = [
    #    t.replace("fMRI", "").replace("localizer", "").replace("task", "").replace("paradigm", "")
    #    for t in text
    # ]

    text = re.sub(r"^\S", lambda m: m.group().lower(), text)
    return text


def _get_counts_tfidf(data_df, vocabulary):
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
        text_column="body",
    )

    count_arr = []
    tfidf_arr = []
    for word in vocabulary:
        new_word = vocabulary_mapping[word]
        # Select column from tfidf_df where the column name is the word
        count_arr.append(count_df[new_word].values)
        tfidf_arr.append(tfidf_df[new_word].values)

    return np.array(count_arr), np.array(tfidf_arr)


def main(project_dir):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    voc_dir = op.join(data_dir, "vocabulary")
    source = "cogatlas"  # cogatlas, neurosynth
    alpha = 0.5
    categories = ["task", "concept"]
    sub_categories = ["names", "definitions", "combined"]
    sections = ["abstract", "body"]

    dset = nimare.dataset.Dataset.load(op.join(data_dir, "dset-pubmed_nimare.pkl"))

    model_id = "BrainGPT/BrainGPT-7B-v0.2"
    model_name = model_id.split("/")[-1]
    generator = TextEmbedding(model_name=model_id)  # , device="cpu"

    if source == "cogatlas":
        tasks_dict, concepts_dict = _get_vocabulary(source="cogatlas", data_dir=data_dir)
        for category, data_dict in zip(categories, [tasks_dict, concepts_dict]):
            names_org, definitions_org = zip(*data_dict.items())

            # names = _preprocess_text(names_org)
            # definitions = _preprocess_text(definitions_org)
            names = names_org
            definitions = definitions_org

            # Get counts and tfidf
            for section in sections:
                lb = f"vocabulary-{source}_{category}-names_embedding-{model_name}"
                counts_fn = op.join(voc_dir, f"{lb}_section-{section}_counts.npy")
                tfidf_fn = op.join(voc_dir, f"{lb}_section-{section}_tfidf.npy")

                counts, tfidf = _get_counts_tfidf(dset.texts, names)
                np.save(counts_fn, counts)
                np.save(tfidf_fn, tfidf)

            names_emb = generator(names)
            definitions_emb = generator(definitions)
            combined_emb = names_emb * alpha + definitions_emb * (1 - alpha)

            embeddings = [names_emb, definitions_emb, combined_emb]
            for sub_category, emb in zip(sub_categories, embeddings):
                lb = f"vocabulary-{source}_{category}-{sub_category}_embedding-{model_name}"
                emb_fn = op.join(voc_dir, f"{lb}.npy")
                names_fn = op.join(voc_dir, f"{lb}.txt")

                _write_vocabulary(names_org, names_fn)
                np.save(emb_fn, emb)

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
