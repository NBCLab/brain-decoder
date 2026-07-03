import numpy as np
import pandas as pd
import torch

from braindec import predict


class _FakeModel:
    logit_scale = torch.tensor(1.0)

    def __call__(self, image_input, text_inputs):
        return image_input, text_inputs


def _call_image_to_labels(return_posterior_probability=False):
    vocabulary = ["motor", "language"]
    vocabulary_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    prior = np.array([0.5, 0.5], dtype=np.float32)

    return predict.image_to_labels(
        image=object(),
        model_path="unused",
        vocabulary=vocabulary,
        vocabulary_emb=vocabulary_emb,
        prior_probability=prior,
        model=_FakeModel(),
        topk=2,
        return_posterior_probability=return_posterior_probability,
    )


def test_image_to_labels_returns_dataframe_by_default(monkeypatch):
    monkeypatch.setattr(predict, "_get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        predict,
        "preprocess_image",
        lambda image, **kwargs: torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    result = _call_image_to_labels()

    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0]["pred"] == "motor"


def test_image_to_labels_can_return_posterior_probability(monkeypatch):
    monkeypatch.setattr(predict, "_get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        predict,
        "preprocess_image",
        lambda image, **kwargs: torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    task_df, posterior_probability = _call_image_to_labels(return_posterior_probability=True)

    assert isinstance(task_df, pd.DataFrame)
    assert isinstance(posterior_probability, torch.Tensor)
