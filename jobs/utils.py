import numpy as np


def _read_vocabulary(vocabulary_fn, vocabulary_emb_fn, vocabulary_prior_fn=None):
    with open(vocabulary_fn, "r") as f:
        vocabulary = [line.strip() for line in f]

    if vocabulary_prior_fn is not None:
        return vocabulary, np.load(vocabulary_emb_fn), np.load(vocabulary_prior_fn)
    else:
        return vocabulary, np.load(vocabulary_emb_fn)
