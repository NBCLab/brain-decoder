"""Plotting functions for braindec."""

import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from neuromaps import transforms
from neuromaps.datasets import fetch_fslr
from nilearn import datasets
from nilearn.plotting import plot_stat_map
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from surfplot import Plot

from braindec.utils import _vol_to_surf, _zero_medial_wall

CMAP = nilearn_cmaps["cold_hot"]


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


def plot_vol(
    nii_img_thr,
    threshold,
    out_file,
    mask_contours=None,
    coords=None,
    vmax=8,
    alpha=1,
    cmap=CMAP,
):
    template = datasets.load_mni152_template(resolution=1)

    display_modes = ["x", "y", "z"]
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    gs = GridSpec(2, 2, figure=fig)

    for dsp_i, display_mode in enumerate(display_modes):
        if display_mode == "z":
            ax = fig.add_subplot(gs[:, 1], aspect="equal")
            colorbar = True
        else:
            ax = fig.add_subplot(gs[dsp_i, 0], aspect="equal")
            colorbar = False

        if coords is not None:
            cut_coords = [coords[dsp_i]]
            if np.isnan(cut_coords):
                cut_coords = 1
        else:
            cut_coords = 1

        display = plot_stat_map(
            nii_img_thr,
            bg_img=template,
            black_bg=False,
            draw_cross=False,
            annotate=True,
            alpha=alpha,
            cmap=cmap,
            threshold=threshold,
            symmetric_cbar=True,
            colorbar=colorbar,
            display_mode=display_mode,
            cut_coords=cut_coords,
            vmax=vmax,
            axes=ax,
        )
        if mask_contours:
            display.add_contours(mask_contours, levels=[0.5], colors="black")

    fig.savefig(out_file, bbox_inches="tight", dpi=300)


def plot_surf(ibma_img_fn, out_file, mask_contours=None, vmax=8, cmap=CMAP):
    if isinstance(ibma_img_fn, str):
        map_lh, map_rh = _vol_to_surf(ibma_img_fn, return_hemis=True)
    elif isinstance(ibma_img_fn, tuple):
        map_lh, map_rh = ibma_img_fn
        if isinstance(ibma_img_fn[0], str):
            map_lh, map_rh, _ = _zero_medial_wall(map_lh, map_rh)

    surfaces = fetch_fslr(density="32k")
    lh, rh = surfaces["inflated"]
    sulc_lh, sulc_rh = surfaces["sulc"]

    p = Plot(surf_lh=lh, surf_rh=rh, layout="grid")
    p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)
    p.add_layer(
        {"left": map_lh, "right": map_rh}, cmap=cmap, cbar=False, color_range=(-vmax, vmax)
    )
    if mask_contours:
        mask_lh, mask_rh = transforms.mni152_to_fslr(mask_contours, fslr_density="32k")
        mask_lh, mask_rh = _zero_medial_wall(
            mask_lh,
            mask_rh,
            space="fsLR",
            density="32k",
        )
        mask_arr_lh = mask_lh.agg_data()
        mask_arr_rh = mask_rh.agg_data()
        countours_lh = np.zeros_like(mask_arr_lh)
        countours_lh[mask_arr_lh != 0] = 1
        countours_rh = np.zeros_like(mask_arr_rh)
        countours_rh[mask_arr_rh != 0] = 1

        colors = [(0, 0, 0, 0)]
        contour_cmap = ListedColormap(colors, "regions", N=1)
        line_cmap = ListedColormap(["black"], "regions", N=1)
        p.add_layer(
            {"left": countours_lh, "right": countours_rh},
            cmap=line_cmap,
            as_outline=True,
            cbar=False,
        )
        p.add_layer(
            {"left": countours_lh, "right": countours_rh},
            cmap=contour_cmap,
            cbar=False,
        )
    fig = p.build()
    fig.savefig(out_file, bbox_inches="tight", transparent=True, dpi=300)
