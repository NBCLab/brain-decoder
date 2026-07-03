import gradio as gr
import pandas as pd
import pytest

import app


def _fake_task_df():
    return pd.DataFrame(
        {
            "pred": ["motor"],
            "prob": [0.9],
            "similarity": [0.5],
            "likelihood": [0.5],
            "prior_prob": [0.5],
            "joint_prob": [0.5],
            "bayes_factor": [2.0],
        }
    )


def test_predict_rejects_none_input():
    with pytest.raises(gr.Error):
        app.predict(None, 10)


def test_predict_rejects_non_nifti_extension():
    with pytest.raises(gr.Error):
        app.predict("scan.txt", 10)


def test_predict_accepts_nii_gz_and_returns_formatted_dataframe(monkeypatch):
    app._RESOURCES.update(
        model=object(),
        device="cpu",
        image_embedder=object(),
        vocabulary=["motor"],
        vocabulary_emb=None,
        vocabulary_prior=None,
        data_dir="/tmp/data",
    )
    monkeypatch.setattr(
        "braindec.predict.image_to_labels", lambda *args, **kwargs: _fake_task_df()
    )

    result = app.predict("scan.nii.gz", 10)

    assert list(result.columns) == ["pred", "prob", "bayes_factor"]
    assert result.iloc[0]["pred"] == "motor"


def test_predict_wraps_prediction_errors(monkeypatch):
    app._RESOURCES.update(
        model=object(),
        device="cpu",
        image_embedder=object(),
        vocabulary=["motor"],
        vocabulary_emb=None,
        vocabulary_prior=None,
        data_dir="/tmp/data",
    )

    def _boom(*args, **kwargs):
        raise ValueError("bad atlas")

    monkeypatch.setattr("braindec.predict.image_to_labels", _boom)

    with pytest.raises(gr.Error):
        app.predict("scan.nii", 10)
