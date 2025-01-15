"""Plotting functions for braindec."""

import os.path as op

import matplotlib.pyplot as plt


def _plot_training_history(
    train_losses,
    val_losses,
    out_dir,
    train_accs=None,
    val_accs=None,
    train_roc_aucs=None,
    val_roc_aucs=None,
):
    # Plot training and validation losses
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 1, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    if train_accs is None or val_accs is None:
        return

    # Plot training and validation accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    if train_roc_aucs is None or val_roc_aucs is None:
        return
    plt.subplot(1, 3, 3)
    plt.plot(train_roc_aucs, label="Train ROC AUC")
    plt.plot(val_roc_aucs, label="Validation ROC AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.legend()
    plt.title("Training and Validation ROC AUC")

    plt.tight_layout()
    plt.savefig(op.join(out_dir, "training_history.png"))
    plt.show()
