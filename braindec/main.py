import os
import os.path as op
import pickle

import numpy as np
import pandas as pd
import torch

from braindec.dataset import MRIDataset, create_balanced_loaders, create_random_loaders
from braindec.loss import ClipLoss
from braindec.model import CLIP, count_parameters
from braindec.plot import _plot_training_history
from braindec.train import test, train, validate
from braindec.utils import _get_device


# Main training loop
def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    data_dir = op.join(project_dir, "data")
    out_dir = op.join(project_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    best_model_fn = op.join(out_dir, "best_model.pth")

    # Hyperparameters
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-4
    plot_verbose = True
    batch_size = 128
    lr = 1e-4
    weight_decay = 0.1
    dropout = 0.6
    num_epochs = 50
    output_dim = 444  # number of region in difumo
    embedding_dim = 512

    criterion = ClipLoss()
    is_clip_loss = criterion.__class__ == ClipLoss
    loss_specific_kwargs = {
        "logit_scale": 10 if is_clip_loss else np.log(10),
        "logit_bias": None if is_clip_loss else -10,
    }
    # criterion = nn.CrossEntropyLoss() # for hard labels
    # criterion = nn.BCELoss() # when using sigmoid in the output layer
    # criterion = nn.KLDivLoss(reduction="batchmean")  # for soft labels

    device = _get_device()
    print(f"Using device: {device}")

    # Create dataset
    dataset_fn = op.join(data_dir, "dataset.pkl")
    if op.exists(dataset_fn):
        with open(dataset_fn, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = MRIDataset(data_dir)
        with open(dataset_fn, "wb") as f:
            pickle.dump(dataset, f)

    print(f"Number of samples: {len(dataset)}")

    # Create data loaders for training, validation, and testing
    # train_loader, val_loader, test_loader = create_balanced_loaders(dataset, batch_size=batch_size)
    train_loader, val_loader, test_loader = create_random_loaders(dataset, batch_size=batch_size)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    # summary(model, (1, 28, 28))

    # Training loop
    best_val_loss = float("inf")
    patience = 10
    counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_fn)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Test the model
    # test_acc, test_roc_auc = test(model, test_loader, device)
    # print(f"Test Accuracy: {test_acc:.2f}%, Test ROC AUC: {test_roc_auc:.4f}")

    # Plot training history
    _plot_training_history(
        train_losses,
        val_losses,
        out_dir,
        # train_accs,
        # train_roc_aucs,
        # val_accs,
        # val_roc_aucs,
    )

    # Save results
    results_df = pd.DataFrame()
    results_df["train_loss"] = train_losses
    # results_df["train_acc"] = train_accs
    # results_df["train_roc_auc"] = train_roc_aucs
    results_df["val_loss"] = val_losses
    # results_df["val_acc"] = val_accs
    # results_df["val_roc_auc"] = val_roc_aucs
    results_df.to_csv(op.join(out_dir, "results.csv"), index=False)


if __name__ == "__main__":
    main()
