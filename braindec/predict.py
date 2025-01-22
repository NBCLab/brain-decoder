"""Predicts the output of the model on the test data."""

import os.path as op

import numpy as np
import pandas as pd
import torch
from nilearn import datasets
from nilearn.image import load_img
from nilearn.maskers import MultiNiftiMapsMasker

from braindec.dataset import _get_vocabulary
from braindec.embedding import TextEmbedding
from braindec.fetcher import _fetch_vocabulary
from braindec.model import build_model
from braindec.utils import _get_device


def preprocess_image(image, data_dir=None):
    """
    Preprocess the image.

    Args:
        image: Images
    """
    # Preprocess the image
    difumo = datasets.fetch_atlas_difumo(
        dimension=512,
        resolution_mm=2,
        legacy_format=False,
        data_dir=data_dir,
    )
    masker_parc = MultiNiftiMapsMasker(maps_img=difumo.maps)
    return torch.from_numpy(masker_parc.fit_transform(image)).float()


def _preprocess_vocabulary(source, data_dir=None):
    generator = TextEmbedding()
    # vocabulary = _fetch_vocabulary(source=source, data_dir=data_dir)
    vocabulary = _get_vocabulary(source=source, data_dir=data_dir)
    vocabulary_emb = np.array([generator(f"{c}") for c in vocabulary])

    return vocabulary, torch.from_numpy(vocabulary_emb).float()


def image_to_labels(
    image,
    model_path,
    vocabulary_path=None,
    source="cogatlas",
    topk=10,
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
    image_input = preprocess_image(image, data_dir=data_dir).to(device)

    if vocabulary_path is None:
        vocabulary, text_inputs = _preprocess_vocabulary(source, data_dir=data_dir)
        text_inputs = text_inputs.to(device)
        vocabulary_path = op.join(data_dir, "vocabulary_tensor.pth")
        torch.save(text_inputs, vocabulary_path)
    else:
        # vocabulary = _fetch_vocabulary(source=source, data_dir=data_dir)
        vocabulary = _get_vocabulary(source=source, data_dir=data_dir)
        text_inputs = torch.load(vocabulary_path).to(device)

    # Calculate features
    model = build_model(model_path, device=device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top topk most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(topk)

    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{vocabulary[index]:>16s}: {100 * value.item():.2f}%")

    probability = values.cpu().detach().numpy()
    indices = indices.cpu().detach().numpy()
    # Print the result
    result_df = pd.DataFrame({"label": np.array(vocabulary)[indices], "probability": probability})

    return result_df
