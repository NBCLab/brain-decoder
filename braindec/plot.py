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


def plot_training(clip_train_loss, clip_val_loss, callback_outputs=None):
    fontsize = 14
    callback_kwargs = [
        {"ylabel": "Validation\nRecall@10", "color": "b", "ylim": [0, 1]},
        {
            "ylabel": "Diagonal Mean",
            "color": "b",
            "ylim": [1e-7, 1],
            "yscale": "log",
        },
        {
            "ylabel": "Non-diagonal Mean",
            "color": "b",
            "ylim": [1e-7, 1],
            "yscale": "log",
        },
        {"ylabel": "Logit scale", "color": "black"},
        {"ylabel": "Logit bias", "color": "black"},
    ]

    callback_outputs = callback_outputs if callback_outputs else []
    num_callbacks = len(callback_outputs[0]) if callback_outputs else 0
    num_epochs = len(clip_train_loss)
    fig, axes = plt.subplots(
        nrows=2 + num_callbacks, ncols=1, sharex=True, figsize=(10, 4 + num_callbacks * 2)
    )
    axes[0].plot(
        range(num_epochs),
        clip_train_loss,
        linestyle="-",
        markersize=3,
        color="r",
        label="CLIP - Training",
    )
    axes[0].plot(
        range(num_epochs),
        clip_val_loss,
        linestyle="-",
        markersize=3,
        color="b",
        label="CLIP - Validation",
    )
    axes[0].set_yscale("log")
    # axes[0].set_xticks([])
    # axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("CLIP Loss", fontsize=fontsize)
    axes[1].plot(
        range(num_epochs),
        clip_train_loss,
        linestyle="-",
        markersize=3,
        color="r",
        label="CLIP - Training",
    )
    axes[1].plot(
        range(num_epochs),
        clip_val_loss,
        linestyle="-",
        markersize=3,
        color="b",
        label="CLIP - Validation",
    )
    # axes[1].set_xticks(list(range(0, num_epochs, int(num_epochs / 10))))
    # axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("CLIP Loss", fontsize=fontsize)
    plt.legend()
    if callback_outputs:
        for index in range(num_callbacks):
            axes[2 + index].plot(
                range(num_epochs),
                [callback_outputs[epoch_index][index] for epoch_index in range(num_epochs)],
                linestyle="-",
                markersize=3,
                color=(
                    callback_kwargs[index]["color"] if "color" in callback_kwargs[index] else "r"
                ),
            )
            axes[2 + index].set_xticks(list(range(0, num_epochs, int(num_epochs / 10))))
            if "xlabel" in callback_kwargs[index]:
                axes[2 + index].set_xlabel(callback_kwargs[index]["xlabel"], fontsize=fontsize)
            if "ylabel" in callback_kwargs[index]:
                axes[2 + index].set_ylabel(callback_kwargs[index]["ylabel"], fontsize=fontsize)
            if "xlim" in callback_kwargs[index]:
                axes[2 + index].set_xlim(callback_kwargs[index]["xlim"])
            if "ylim" in callback_kwargs[index]:
                axes[2 + index].set_ylim(callback_kwargs[index]["ylim"])
            if "yscale" in callback_kwargs[index]:
                axes[2 + index].set_yscale(callback_kwargs[index]["yscale"])
    plt.tight_layout()
    plt.show()


def plot_matrix(array, ax=None, title=None, xlabel=None, ylabel=None, fontsize=16, **kwargs):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    kw = {
        "origin": "upper",
        "interpolation": "nearest",
        "aspect": "equal",  # (already the imshow default)
        **kwargs,
    }
    im = ax.imshow(array, **kw)
    ax.title.set_y(1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    return ax, im
