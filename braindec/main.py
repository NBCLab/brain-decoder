import os
import os.path as op
import pickle
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from braindec.loss import ClipLoss
from braindec.metrics import mix_match, recall_n
from braindec.model import CLIP, count_parameters
from braindec.plot import plot_matrix
from braindec.train import predict, train_clip_model
from braindec.utils import _get_device


def _initialize_clip_model(
    embedding_dim,
    output_dim,
    dropout,
    learning_rate,
    weight_decay,
    device,
):
    criterion = ClipLoss()
    is_clip_loss = criterion.__class__ == ClipLoss

    loss_specific_kwargs = {
        "logit_scale": 10 if is_clip_loss else np.log(10),
        "logit_bias": None if is_clip_loss else -10,
    }

    # Initialize model, loss function, and optimizer
    model = CLIP(
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        dropout=dropout,
        **loss_specific_kwargs,
    ).to(device)

    # Get the number of parameters
    num_params = count_parameters(model)
    print(f"Total number of parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    # summary(model, (1, 28, 28))
    return model, optimizer, criterion, scheduler


def _evaluate_clip_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    best_model_fn,
    last_model_fn,
    device,
    plot_verbose=False,
):
    metrics = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
        "test": defaultdict(list),
    }
    recall_fn = partial(recall_n, thresh=0.95, reduce_mean=True)

    # Predict on test set and evaluate
    for loader_name, loader, weights_path in [
        ("train", train_loader, last_model_fn),
        ("validation", val_loader, best_model_fn),
        ("test", test_loader, best_model_fn),
    ]:
        model.load_state_dict(torch.load(weights_path))

        image_embeddings, text_embeddings = predict(model, loader, device=device)
        similarity = (image_embeddings @ text_embeddings.T).softmax(dim=1).numpy()
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

        random_perf = 10 / len(similarity)

        nq_perf = recall_fn(similarity, np.eye(len(similarity)), n_first=10)
        nq_perf_100 = recall_fn(similarity, np.eye(len(similarity)), n_first=100)
        nq_perf_all = recall_fn(similarity, np.eye(len(similarity)), n_first=len(similarity))

        metrics[loader_name]["recall@10"].append(nq_perf)
        metrics[loader_name]["recall@100"].append(nq_perf_100)
        metrics[loader_name]["mix_match"].append(100 * mix_match(similarity))

    return metrics


def _get_data_loader(emb_x, emb_y, batch_size, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(emb_x).float(), torch.from_numpy(emb_y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Main training loop
def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    output_dir = op.join(results_dir, "neurostore")
    os.makedirs(output_dir, exist_ok=True)

    best_model_fn = op.join(output_dir, "best_clip-model.pth")
    last_model_fn = op.join(output_dir, "last_clip-model.pth")

    device = _get_device()
    print(f"Using device: {device}")

    # Load dataset
    text_emb = np.load(op.join(data_dir, "text_embedding.npy"))
    img_emb = np.load(op.join(data_dir, "image_embedding.npy"))

    assert text_emb.shape[0] == img_emb.shape[0]

    # Hyperparameters
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 0.1
    dropout = 0.5

    # Other parameters
    val_size = 1000
    test_size = 1000
    sample_size = text_emb.shape[0]
    text_emb_dim = text_emb.shape[1]
    output_dim = img_emb.shape[1]  # number of region in difumo
    test_folds_to_run = 1
    val_folds_to_run = 1
    plot_verbose = True

    # Split data into train, validation, and test sets
    test_k_fold = KFold(n_splits=sample_size // test_size)
    train_test_split = test_k_fold.split(text_emb)
    for test_fold, (train_val_index, test_index) in enumerate(train_test_split):
        test_index = test_index[:test_size]  # Strict 1000 validation samples

        if test_fold >= test_folds_to_run:
            break

        test_loader = _get_data_loader(
            img_emb[test_index],
            text_emb[test_index],
            batch_size,
            shuffle=False,
        )

        val_k_fold = KFold(n_splits=sample_size // val_size)
        train_val_split = val_k_fold.split(text_emb[train_val_index])
        for val_fold, (train_index, val_index) in enumerate(train_val_split):

            if val_fold >= val_folds_to_run:
                break

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

            print("Initializing CLIP model")
            clip_model, optimizer, criterion, scheduler = _initialize_clip_model(
                text_emb_dim,
                output_dim,
                dropout,
                learning_rate,
                weight_decay,
                device,
            )

            print("Training CLIP model")
            clip_model, _, _ = train_clip_model(
                clip_model,
                criterion,
                optimizer,
                num_epochs,
                train_loader,
                val_loader,
                best_model_fn,
                last_model_fn,
                device,
                plot_verbose=plot_verbose,
            )

            print("Evaluating CLIP model")
            metrics = _evaluate_clip_model(
                clip_model,
                train_loader,
                val_loader,
                test_loader,
                best_model_fn,
                last_model_fn,
                device,
                plot_verbose=plot_verbose,
            )

    print(f"Metrics after {val_fold} folds")
    for loader_name in ["train", "validation", "test"]:
        print("=" * 10, loader_name, "=" * 10)
        for metric_name in ["recall@10", "recall@100", "mix_match"]:
            print(
                f"{metric_name}: {np.mean(metrics[loader_name][metric_name]):.3f}"
                f" +- {np.std(metrics[loader_name][metric_name]):.3f}"
            )


if __name__ == "__main__":
    main()
