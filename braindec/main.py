import argparse
import os
import os.path as op
import shutil
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from braindec.loss import ClipLoss
from braindec.metrics import mix_match, recall_n
from braindec.model import CLIP, build_model, count_parameters
from braindec.plot import plot_matrix
from braindec.train import predict, train_clip_model
from braindec.utils import _get_device


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


def _initialize_clip_model(
    embedding_dim,
    output_dim,
    dropout,
    learning_rate,
    weight_decay,
    temperature,
    device,
    verbose=1,
):
    criterion = ClipLoss()
    is_clip_loss = criterion.__class__ == ClipLoss

    loss_specific_kwargs = {
        "logit_scale": temperature if is_clip_loss else np.log(temperature),
        "logit_bias": None if is_clip_loss else -temperature,
    }

    # Initialize model, loss function, and optimizer
    model = CLIP(
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        dropout=dropout,
        **loss_specific_kwargs,
    ).to(device)
    model.device = device

    # Get the number of parameters
    num_params = count_parameters(model)
    if verbose > 1:
        print(f"Total number of parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None

    return model, optimizer, criterion, scheduler


def _evaluate_clip_model(
    train_loader,
    val_loader,
    test_loader,
    best_model_fn,
    last_model_fn,
    device,
    plot_verbose=False,
):
    recall_fn = partial(recall_n, thresh=0.95, reduce_mean=True)

    # Predict on test set and evaluate
    metrics = []
    for loader_name, loader, weights_path in [
        ("train", train_loader, last_model_fn),
        ("val", val_loader, best_model_fn),
        ("test", test_loader, best_model_fn),
    ]:
        # model.load_state_dict(torch.load(weights_path, weights_only=True))
        model = build_model(weights_path, device=device)
        logit_scale = model.logit_scale.item()

        image_embeddings, text_embeddings = predict(model, loader)

        similarity = (logit_scale * image_embeddings @ text_embeddings.T).softmax(dim=-1).numpy()
        if plot_verbose:
            # Plot similarity matrices that should be diagonal
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
            gauss_similarity = (image_embeddings @ image_embeddings.T).numpy()
            plot_matrix(gauss_similarity[:100, :100], ax=axes[0], title="Gauss-to-Gauss")
            text_similarity = (text_embeddings @ text_embeddings.T).numpy()
            plot_matrix(text_similarity[:100, :100], ax=axes[1], title="Text-to-text")
            plot_matrix(similarity[:100, :100], ax=axes[2], title="Gauss-to-Text")
            fig.suptitle(f"Learnt similarities - {loader_name}")
            plt.tight_layout()
            plt.show()

        nq_perf = recall_fn(similarity, np.eye(len(similarity)), n_first=10)
        nq_perf_100 = recall_fn(similarity, np.eye(len(similarity)), n_first=100)
        nq_perf_mixmatch = 100 * mix_match(similarity)

        metrics.append([nq_perf, nq_perf_100, nq_perf_mixmatch])

    return metrics, ["recall@10", "recall@100", "mix&match"]


def _get_data_loader(emb_x, emb_y, batch_size, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(emb_x).float(), torch.from_numpy(emb_y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Main training loop
def main(project_dir, section="abstract", model_id="mistralai/Mistral-7B-v0.1", device=None):
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    output_dir = op.join(results_dir, "pubmed")
    os.makedirs(output_dir, exist_ok=True)
    verbose = 2

    device = _get_device() if device is None else device
    print(f"Using device: {device}")

    # Load embeddings
    model_name = model_id.split("/")[-1]  # Embedding model name
    img_emb = np.load(
        op.join(data_dir, "image", "image-normalized_coord-MKDA_embedding-DiFuMo.npy")
    )
    text_emb = np.load(
        op.join(data_dir, "text", f"text-normalized_section-{section}_embedding-{model_name}.npy")
    )

    model_fn = f"model-clip_section-{section}_embedding-{model_name}"
    best_model_fn = op.join(output_dir, f"{model_fn}_best.pth")
    best_model_index_fn = op.join(output_dir, f"{model_fn}_best-indices.npz")
    current_best_model_fn = op.join(output_dir, f"{model_fn}_current.pth")
    last_model_fn = op.join(output_dir, f"{model_fn}_last.pth")

    assert text_emb.shape[0] == img_emb.shape[0]

    # Hyperparameters taken from the Neurocontext paper
    batch_size = 128  # 128, 256, 512, 1024
    temperature = 10  # inverse temperature
    num_epochs = 50
    learning_rate = 5e-4
    weight_decay = 0.1
    dropout = 0.5

    # Other parameters
    val_size = 1000
    test_size = 1000
    sample_size = text_emb.shape[0]
    text_emb_dim = text_emb.shape[1]
    output_dim = img_emb.shape[1]  # number of region in difumo
    test_folds_to_run = np.inf  # np.inf
    val_folds_to_run = np.inf  # np.inf
    plot_verbose = False

    metrics = {
        "test_fold": [],
        "val_fold": [],
        "sample": [],
        "train": [],
        "val": [],
        "test": [],
    }
    losses = {
        "test_fold": [],
        "val_fold": [],
        "epoch": [],
        "train": [],
        "val": [],
    }

    # Split data into train, validation, and test sets
    test_k_fold = KFold(n_splits=sample_size // test_size)
    train_test_split = test_k_fold.split(text_emb)
    best_recall = 0
    for test_fold, (train_val_index, test_index) in enumerate(train_test_split):
        if test_fold >= test_folds_to_run:
            break

        test_loader = _get_data_loader(
            img_emb[test_index],
            text_emb[test_index],
            batch_size,
            shuffle=False,
        )

        val_k_fold = KFold(n_splits=len(train_val_index) // val_size)
        train_val_split = val_k_fold.split(text_emb[train_val_index])
        for val_fold, (train_index, val_index) in enumerate(train_val_split):
            if val_fold >= val_folds_to_run:
                break

            if verbose > 0:
                print(f"Test fold: {test_fold}, Val fold: {val_fold}")

            train_loader = _get_data_loader(
                img_emb[train_val_index][train_index],
                text_emb[train_val_index][train_index],
                batch_size,
                shuffle=True,
            )

            val_loader = _get_data_loader(
                img_emb[train_val_index][val_index],
                text_emb[train_val_index][val_index],
                batch_size,
                shuffle=False,
            )

            if verbose > 0:
                print("Initializing CLIP model")
            clip_model, optimizer, criterion, scheduler = _initialize_clip_model(
                text_emb_dim,
                output_dim,
                dropout,
                learning_rate,
                weight_decay,
                temperature,
                device,
                verbose=verbose,
            )

            if verbose > 0:
                print("Training CLIP model")
            train_losses, val_losses = train_clip_model(
                clip_model,
                criterion,
                optimizer,
                num_epochs,
                train_loader,
                val_loader,
                current_best_model_fn,
                last_model_fn,
                verbose=verbose,
                plot_verbose=plot_verbose,
            )
            final_num_epochs = len(train_losses)
            losses["test_fold"].extend([test_fold] * final_num_epochs)
            losses["val_fold"].extend([val_fold] * final_num_epochs)
            losses["epoch"].extend(list(range(final_num_epochs)))
            losses["train"].extend(train_losses)
            losses["val"].extend(val_losses)

            if verbose > 0:
                print("Evaluating CLIP model")
            (train_recalls, val_recalls, test_recalls), metric_names = _evaluate_clip_model(
                train_loader,
                val_loader,
                test_loader,
                current_best_model_fn,
                last_model_fn,
                device,
                plot_verbose=plot_verbose,
            )
            metrics["test_fold"].extend([test_fold] * len(metric_names))
            metrics["val_fold"].extend([val_fold] * len(metric_names))
            metrics["sample"].extend(metric_names)
            metrics["train"].extend(train_recalls)
            metrics["val"].extend(val_recalls)
            metrics["test"].extend(test_recalls)

            current_recall = test_recalls[0]  # Get the recall@10
            if verbose > 0:
                print(f"\tCurrent recall: {current_recall}, Best recall: {best_recall}")
            if current_recall > best_recall:
                if verbose > 0:
                    print(
                        f"Saving best model with recall: {current_recall}; "
                        f"test fold: {test_fold}, val fold: {val_fold}"
                    )
                best_recall = current_recall

                # Save current best model as the best model
                shutil.copy(current_best_model_fn, best_model_fn)  # Overwrite best model
                np.savez_compressed(
                    best_model_index_fn,
                    train=train_val_index[train_index],
                    val=train_val_index[val_index],
                    test=test_index,
                )

    if verbose > 0:
        print(f"Best recall: {best_recall}")

    # Save metrics
    metrics_fn = op.join(
        output_dir,
        f"model-clip_section-{section}_embedding-{model_name}_metrics.csv",
    )
    losses_fn = op.join(
        output_dir,
        f"model-clip_section-{section}_embedding-{model_name}_losses.csv",
    )
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_fn, index=False)
    losses_df = pd.DataFrame(losses)
    losses_df.to_csv(losses_fn, index=False)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
