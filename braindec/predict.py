"""Predicts the output of the model on the test data."""

import os.path as op

import numpy as np
import pandas as pd
import torch
from nilearn.image import load_img
from scipy.special import expit, softmax

from braindec.embedding import ImageEmbedding
from braindec.model import build_model
from braindec.utils import _get_device, get_data_dir


def preprocess_image(image, standardize=False, data_dir=None):
    """
    Preprocess the image.

    Args:
        image: Images
    """
    nilearn_dir = get_data_dir(op.join(data_dir, "nilearn"))

    image_emb_gene = ImageEmbedding(standardize=standardize, data_dir=nilearn_dir)
    image_embedding_arr = image_emb_gene(image)

    return torch.from_numpy(image_embedding_arr).float()


def image_to_labels(
    image,
    model_path,
    vocabulary,
    vocabulary_emb,
    prior_probability,
    topk=10,
    standardize=False,
    logit_scale=None,
    data_dir=None,
    device=None,
):
    """
    Predict the label of an image.

    Args:
        image: Images

    Returns:
        Predicted label
    """
    if device is None:
        device = _get_device()

    image = load_img(image)
    image_input = preprocess_image(image, standardize=standardize, data_dir=data_dir).to(device)
    text_inputs = torch.from_numpy(vocabulary_emb).float().to(device)
    prior_probability = torch.from_numpy(prior_probability).float().to(device)

    # Normalize the embeddings
    text_inputs = text_inputs / (text_inputs.norm(dim=-1, keepdim=True) + 1e-8)
    image_input = image_input / (image_input.norm(dim=-1, keepdim=True) + 1e-8)

    # Calculate features
    model = build_model(model_path, device=device)
    with torch.no_grad():
        # image_features = model.encode_image(image_input)  # not normalized
        # text_features = model.encode_text(text_inputs)  # not normalized
        image_features, text_features = model(image_input, text_inputs)  # normalized

    # Get the scaling factor: inverse temperature
    logit_scale = model.logit_scale.item() if logit_scale is None else logit_scale

    # Pick the top topk most similar labels for the image
    # similarity = logit_scale * image_features @ text_features.T
    likelihood = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
    # Flatten the probability distribution, since image_features is a single image
    likelihood = likelihood.flatten()  # P(A|T)

    joint_probability = likelihood * prior_probability  # P(A|T) * P(A)
    total_probability = joint_probability.sum()  # P(T)
    posterior_probability = joint_probability / total_probability  # P(T|A) = P(A|T) * P(A) / P(T)

    # Calculate the strength of the evidence using the Bayes factor
    posterrior_odds = posterior_probability / (1 - posterior_probability)
    prior_odds = prior_probability / (1 - prior_probability)
    bayes_factor = posterrior_odds / prior_odds

    # Get the top k predictions
    values, indices = posterior_probability.topk(topk)
    selectivity = values.cpu().detach().numpy()
    indices = indices.cpu().detach().numpy()
    likelihood = likelihood.cpu().detach().numpy()
    joint_probability = joint_probability.cpu().detach().numpy()
    prior_probability = prior_probability.cpu().detach().numpy()
    bayes_factor = bayes_factor.cpu().detach().numpy()

    return pd.DataFrame(
        {
            "label": np.array(vocabulary)[indices],
            "likelihood": likelihood[indices],
            "prior_prob": prior_probability[indices],
            "joint_prob": joint_probability[indices],
            "posterior_prob": selectivity,
            "bayes_factor": bayes_factor[indices],
        }
    )
