r"""
NiCLIP: Functional Brain Decoding Tutorial
===========================================

`NiCLIP <https://doi.org/10.1101/2025.06.14.659706>`_ is a contrastive
language–image pre-training (CLIP) model trained on ~23,000 neuroimaging
articles that maps brain activation patterns to cognitive task descriptions
from the `Cognitive Atlas <https://www.cognitiveatlas.org/>`_ ontology.

This tutorial walks through the main use cases:

1. **Group-level task decoding** — predict tasks, concepts, and cognitive
   process domains from a group-level activation map.
2. **Hierarchical decoding** — obtain predictions at three ontology levels
   (tasks → concepts → domains) using the noisy-OR propagation rule.
3. **Brain region characterization** — characterize anatomical ROIs
   without pre-computed meta-analytic maps.
4. **Subject-level decoding** — apply decoding to noisier single-subject maps.
5. **Custom vocabulary** — decode against a user-supplied task vocabulary.
6. **Latent space exploration** — visualize the shared image–text embedding
   space learned by NiCLIP.

.. note::

   **Run this tutorial on Google Colab**

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/jdkent/brain-decoder/blob/main/docs/auto_examples/02_niclip_demo.ipynb
      :alt: Open In Colab

   The first notebook cell installs :code:`braindec` and its dependencies
   automatically.  The full install takes a few minutes on a fresh Colab
   runtime; subsequent runs reuse the cached packages.
"""

# %%
# Section 0: Download example assets
# ------------------------------------
# NiCLIP ships a curated set of publishable assets on
# `OSF <https://osf.io/dsj56/>`_.  The ``example_prediction`` bundle
# contains the pre-trained CLIP model, reduced Cognitive Atlas vocabulary,
# pre-computed vocabulary embeddings, vocabulary prior, brain mask, and
# Cognitive Atlas ontology snapshots — everything required to run decoding.
#
# The download is skipped automatically for files that already exist locally.

import os
import os.path as op
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import requests

from braindec.fetcher import download_bundle, get_data_dir

work_dir = get_data_dir()
print(f"Working directory: {work_dir}")

downloaded = download_bundle("example_prediction", destination_root=work_dir)
print(f"Bundle contains {len(downloaded)} files")

# %%
# Construct paths to downloaded assets.
# These mirror the OSF folder structure that ``download_bundle`` preserves.

MODEL_NAME = "BrainGPT-7B-v0.2"
SECTION = "body"
SOURCE = "cogatlasred"
VOC_LABEL = f"vocabulary-{SOURCE}_task-combined_embedding-{MODEL_NAME}"

data_dir = op.join(work_dir, "data")
results_dir = op.join(work_dir, "results")
voc_dir = op.join(data_dir, "vocabulary")
cog_atlas_dir = op.join(data_dir, "cognitive_atlas")

model_fn = op.join(results_dir, "pubmed",
                   f"model-clip_section-{SECTION}_embedding-{MODEL_NAME}_best.pth")
vocabulary_fn = op.join(voc_dir, f"vocabulary-{SOURCE}_task.txt")
vocabulary_emb_fn = op.join(voc_dir, f"{VOC_LABEL}.npy")
vocabulary_prior_fn = op.join(voc_dir, f"{VOC_LABEL}_section-{SECTION}_prior.npy")
mask_fn = op.join(data_dir, "MNI152_2x2x2_brainmask.nii.gz")

for label, path in [
    ("model", model_fn),
    ("vocabulary", vocabulary_fn),
    ("vocabulary embeddings", vocabulary_emb_fn),
    ("vocabulary prior", vocabulary_prior_fn),
    ("brain mask", mask_fn),
    ("cognitive atlas", cog_atlas_dir),
]:
    status = "✓" if op.exists(path) else "✗ MISSING"
    print(f"  {status}  {label}: {path}")

# %%
# Download representative HCP group-level contrast maps from
# `NeuroVault <https://neurovault.org/collections/457/>`_ (public access,
# no account required).  We use three contrasts that span different cognitive
# domains to illustrate decoding across sections 1–6.

HCP_COLLECTION_ID = 457
HCP_MAPS = {
    "motor": "tfMRI_MOTOR_AVG_zstat1.nii.gz",
    "language": "tfMRI_LANGUAGE_STORY-MATH_zstat1.nii.gz",
    "emotion": "tfMRI_EMOTION_FACES-SHAPES_zstat1.nii.gz",
    "working_memory": "tfMRI_WM_2BK-0BK_zstat1.nii.gz",
}

hcp_dir = Path(data_dir) / "hcp" / "neurovault"
hcp_dir.mkdir(parents=True, exist_ok=True)

hcp_paths = {}
for domain, filename in HCP_MAPS.items():
    dest = hcp_dir / filename
    if not dest.exists():
        url = f"https://neurovault.org/media/images/{HCP_COLLECTION_ID}/{filename}"
        print(f"Downloading {domain} map …")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
    hcp_paths[domain] = str(dest)
    print(f"  {domain}: {dest}")

# %%
# Initialise shared resources that will be reused across sections.
# Building the model and image embedder once avoids repeated I/O and
# DiFuMo atlas downloads.

import torch
from braindec.cogatlas import CognitiveAtlas
from braindec.embedding import ImageEmbedding
from braindec.model import build_model
from braindec.utils import _get_device

device = _get_device()
print(f"Using device: {device}")

model = build_model(model_fn, device=device)

image_embedder = ImageEmbedding(
    standardize=False,
    nilearn_dir=op.join(data_dir, "nilearn"),
    space="MNI152",
)

with open(vocabulary_fn) as fh:
    vocabulary = [line.strip() for line in fh]
vocabulary_emb = np.load(vocabulary_emb_fn)
vocabulary_prior = np.load(vocabulary_prior_fn)

print(f"Vocabulary size: {len(vocabulary)} tasks")
print(f"Embedding shape: {vocabulary_emb.shape}")

# %%
# Section 1: Group-level task decoding
# ----------------------------------------
# The primary NiCLIP use case is *functional decoding*: given a brain
# activation map, retrieve the most likely cognitive tasks from the Cognitive
# Atlas.  NiCLIP computes posterior probabilities P(T|A) using Bayes'
# theorem over the CLIP cosine similarities.

import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

from braindec.predict import image_to_labels

motor_img = nib.load(hcp_paths["motor"])

task_df = image_to_labels(
    motor_img,
    model_path=model_fn,
    vocabulary=vocabulary,
    vocabulary_emb=vocabulary_emb,
    prior_probability=vocabulary_prior,
    topk=10,
    logit_scale=20.0,
    model=model,
    image_emb_gene=image_embedder,
    data_dir=data_dir,
)

print("Top-10 task predictions for HCP Motor (AVG) contrast:")
print(task_df.to_string(index=False))

# %%
# Visualise the input activation map alongside the top-5 task predictions.

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: activation map
display = plot_stat_map(
    motor_img,
    display_mode="z",
    cut_coords=5,
    colorbar=True,
    threshold=2.0,
    title="HCP Motor (AVG) z-stat",
    axes=axes[0],
)

# Right: posterior probability bar chart
top5 = task_df.head(5)
short_labels = [t[:40] + "…" if len(t) > 40 else t for t in top5["pred"]]
axes[1].barh(range(len(top5)), top5["prob"], color="steelblue")
axes[1].set_yticks(range(len(top5)))
axes[1].set_yticklabels(short_labels, fontsize=9)
axes[1].invert_yaxis()
axes[1].set_xlabel("Posterior probability P(T|A)")
axes[1].set_title("Top-5 task predictions")
axes[1].set_xlim(0, top5["prob"].max() * 1.2)
plt.tight_layout()
plt.show()

# %%
# Section 2: Hierarchical decoding
# -------------------------------------
# NiCLIP propagates task posteriors up the Cognitive Atlas ontology using
# a noisy-OR model to derive concept and cognitive process domain
# probabilities: P(C|A) and P(D|A).  This produces interpretations at
# three levels of specificity.

import json

from braindec.predict import image_to_labels_hierarchical

concept_to_process_fn = op.join(cog_atlas_dir, "concept_to_process.json")
with open(concept_to_process_fn) as fh:
    concept_to_process = json.load(fh)

reduced_tasks_df = pd.read_csv(op.join(cog_atlas_dir, "reduced_tasks.csv"))

cog_atlas = CognitiveAtlas(
    data_dir=data_dir,
    task_snapshot=op.join(cog_atlas_dir, "task_snapshot-02-19-25.json"),
    concept_snapshot=op.join(cog_atlas_dir, "concept_extended_snapshot-02-19-25.json"),
    concept_to_process=concept_to_process,
    reduced_tasks=reduced_tasks_df,
)

print(f"Cognitive Atlas: {len(cog_atlas.task_names)} tasks  |  "
      f"{len(cog_atlas.concept_names)} concepts  |  "
      f"{len(cog_atlas.process_names)} domains")

# %%

task_df_h, concept_df, domain_df = image_to_labels_hierarchical(
    motor_img,
    model_path=model_fn,
    vocabulary=vocabulary,
    vocabulary_emb=vocabulary_emb,
    prior_probability=vocabulary_prior,
    cognitiveatlas=cog_atlas,
    topk=5,
    logit_scale=20.0,
    model=model,
    image_emb_gene=image_embedder,
    data_dir=data_dir,
)

# %%
# Display predictions at all three ontology levels.

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
panels = [
    (task_df_h, "Tasks  P(T|A)", "prob"),
    (concept_df, "Concepts  P(C|A)", "prob"),
    (domain_df, "Domains  P(D|A)", "prob"),
]

for ax, (df, title, col) in zip(axes, panels):
    df_top = df.head(5)
    labels = [t[:35] + "…" if len(t) > 35 else t for t in df_top["pred"]]
    ax.barh(range(len(df_top)), df_top[col], color="steelblue")
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Posterior probability")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, df_top[col].max() * 1.3)

fig.suptitle("HCP Motor (AVG) — Hierarchical decoding", fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Section 3: Brain region characterization
# -------------------------------------------
# Instead of a task activation map, NiCLIP can decode *anatomical ROIs*
# directly — enabling functional characterisation of brain regions without
# requiring pre-computed meta-analytic maps.  This supports hypothesis
# generation about the cognitive roles of unstudied regions.
#
# We create binary ROI masks from the
# `AAL atlas <https://www.gin.cnrs.fr/en/tools/aal/>`_ (available via
# nilearn) and decode each region.

from nilearn import datasets, image as nli_image

aal = datasets.fetch_atlas_aal()
aal_img = nib.load(aal.maps)
aal_data = aal_img.get_fdata()

# Map AAL label names → image indices (1-based in the NIfTI data).
aal_label_to_idx = {name: int(idx) for name, idx in zip(aal.labels, aal.indices)}


def make_aal_roi(region_names):
    """Binary mask from one or more AAL region names."""
    mask = np.zeros(aal_data.shape, dtype=np.float32)
    for name in region_names:
        if name in aal_label_to_idx:
            mask[aal_data == aal_label_to_idx[name]] = 1.0
        else:
            print(f"  Warning: '{name}' not found in AAL atlas.")
    return nib.Nifti1Image(mask, aal_img.affine, aal_img.header)


# Six regions matching those in the NiCLIP paper.
ROI_SPECS = {
    "Amygdala": ["Amygdala_L", "Amygdala_R"],
    "Hippocampus": ["Hippocampus_L", "Hippocampus_R"],
    "Insula": ["Insula_L", "Insula_R"],
    "Striatum": ["Putamen_L", "Putamen_R", "Caudate_L", "Caudate_R"],
    "rTPJ": ["Angular_R", "SupraMarginal_R"],
    "vmPFC": ["Frontal_Med_Orb_L", "Frontal_Med_Orb_R"],
}

roi_images = {name: make_aal_roi(labels) for name, labels in ROI_SPECS.items()}

# %%
# Decode each ROI and collect the top task, concept, and domain.

roi_summary = []
for roi_name, roi_img in roi_images.items():
    t_df, c_df, d_df = image_to_labels_hierarchical(
        roi_img,
        model_path=model_fn,
        vocabulary=vocabulary,
        vocabulary_emb=vocabulary_emb,
        prior_probability=vocabulary_prior,
        cognitiveatlas=cog_atlas,
        topk=3,
        logit_scale=20.0,
        model=model,
        image_emb_gene=image_embedder,
        data_dir=data_dir,
    )
    roi_summary.append({
        "ROI": roi_name,
        "Top task": t_df.iloc[0]["pred"],
        "Task P(T|A)": f"{t_df.iloc[0]['prob']:.3f}",
        "Top concept": c_df.iloc[0]["pred"],
        "Top domain": d_df.iloc[0]["pred"],
    })

summary_df = pd.DataFrame(roi_summary)
print(summary_df.to_string(index=False))

# %%
# Visualise one ROI alongside its top prediction.

from nilearn.plotting import plot_roi

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

display = plot_roi(
    roi_images["Amygdala"],
    title="Amygdala (bilateral)",
    display_mode="ortho",
    cut_coords=(0, -4, -18),
    axes=axes[0],
    colorbar=False,
)

# Decode the amygdala with a finer top-k for the bar chart.
t_df, c_df, d_df = image_to_labels_hierarchical(
    roi_images["Amygdala"],
    model_path=model_fn,
    vocabulary=vocabulary,
    vocabulary_emb=vocabulary_emb,
    prior_probability=vocabulary_prior,
    cognitiveatlas=cog_atlas,
    topk=5,
    logit_scale=20.0,
    model=model,
    image_emb_gene=image_embedder,
    data_dir=data_dir,
)

labels = [t[:38] + "…" if len(t) > 38 else t for t in t_df["pred"]]
axes[1].barh(range(5), t_df["prob"], color="salmon")
axes[1].set_yticks(range(5))
axes[1].set_yticklabels(labels, fontsize=8)
axes[1].invert_yaxis()
axes[1].set_xlabel("P(T|A)")
axes[1].set_title("Top-5 task predictions for Amygdala ROI")

plt.tight_layout()
plt.show()

# %%
# Section 4: Subject-level decoding
# -------------------------------------
# NiCLIP can decode single-subject activation maps, though performance is
# lower than group-level due to higher noise.  Here we simulate a
# subject-level map by adding Gaussian noise to the group-level motor
# contrast, then compare predicted ranks to the clean result.
#
# In practice, you would supply your own subject-level t-stat or z-stat
# NIfTI image in place of the simulated map below.

motor_data = motor_img.get_fdata()
rng = np.random.default_rng(42)
noise_std = motor_data.std()
noisy_data = motor_data + rng.normal(scale=noise_std, size=motor_data.shape)
noisy_img = nib.Nifti1Image(noisy_data, motor_img.affine, motor_img.header)

task_df_noisy = image_to_labels(
    noisy_img,
    model_path=model_fn,
    vocabulary=vocabulary,
    vocabulary_emb=vocabulary_emb,
    prior_probability=vocabulary_prior,
    topk=10,
    logit_scale=20.0,
    model=model,
    image_emb_gene=image_embedder,
    data_dir=data_dir,
)

# %%
# Compare predictions from the clean group map vs. the simulated subject map.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (df, title, color) in zip(
    axes,
    [
        (task_df.head(5), "Group-level (clean)", "steelblue"),
        (task_df_noisy.head(5), "Subject-level (simulated noise)", "orange"),
    ],
):
    labels = [t[:40] + "…" if len(t) > 40 else t for t in df["pred"]]
    ax.barh(range(len(df)), df["prob"], color=color)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("P(T|A)")
    ax.set_title(title)

fig.suptitle("Motor decoding: group vs. subject-level noise", fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Section 5: Custom vocabulary decoding
# -----------------------------------------
# NiCLIP accepts any list of task names paired with their LLM-derived text
# embeddings.  This lets you decode against a domain-specific vocabulary
# instead of (or in addition to) the full Cognitive Atlas.
#
# **Two workflows:**
#
# *Workflow A — subset the existing vocabulary.*
# Select a subset of Cognitive Atlas tasks relevant to your study domain
# and decode with just those terms.  No additional embedding needed.
#
# *Workflow B — embed entirely new task names.*
# Use :class:`~braindec.embedding.TextEmbedding` with BrainGPT to embed
# custom task descriptions, then pass them directly to
# :func:`~braindec.predict.image_to_labels`.

# Workflow A: emotion-focused vocabulary subset
EMOTION_KEYWORDS = ["emotion", "fear", "affect", "face", "amygdala", "valence", "threat"]

custom_idx = [
    i for i, task in enumerate(vocabulary)
    if any(kw in task.lower() for kw in EMOTION_KEYWORDS)
]
custom_vocabulary = [vocabulary[i] for i in custom_idx]
custom_vocabulary_emb = vocabulary_emb[custom_idx]
custom_prior = vocabulary_prior[custom_idx]
# Re-normalise prior so probabilities sum to 1.
custom_prior = custom_prior / custom_prior.sum()

print(f"Custom emotion vocabulary: {len(custom_vocabulary)} tasks")
print("  " + "\n  ".join(custom_vocabulary[:8]))

# %%

emotion_img = nib.load(hcp_paths["emotion"])

task_df_custom = image_to_labels(
    emotion_img,
    model_path=model_fn,
    vocabulary=custom_vocabulary,
    vocabulary_emb=custom_vocabulary_emb,
    prior_probability=custom_prior,
    topk=min(8, len(custom_vocabulary)),
    logit_scale=20.0,
    model=model,
    image_emb_gene=image_embedder,
    data_dir=data_dir,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

display = plot_stat_map(
    emotion_img,
    display_mode="z",
    cut_coords=5,
    threshold=2.0,
    title="HCP Emotion (Faces vs Shapes)",
    axes=axes[0],
)

labels = [t[:38] + "…" if len(t) > 38 else t for t in task_df_custom["pred"]]
axes[1].barh(range(len(task_df_custom)), task_df_custom["prob"], color="mediumpurple")
axes[1].set_yticks(range(len(task_df_custom)))
axes[1].set_yticklabels(labels, fontsize=8)
axes[1].invert_yaxis()
axes[1].set_xlabel("P(T|A)")
axes[1].set_title("Emotion-focused vocabulary predictions")
plt.tight_layout()
plt.show()

# %%
# **Workflow B — embedding truly new task names (GPU required).**
#
# If you have task descriptions not present in the Cognitive Atlas, embed
# them with :class:`~braindec.embedding.TextEmbedding` and build a prior
# from uniform weights.  The code below is shown for reference; a GPU with
# ≥14 GB VRAM (e.g., A100) is needed to run BrainGPT-7B.

# .. code-block:: python
#
#     from braindec.embedding import TextEmbedding
#
#     my_tasks = [
#         "emotional conflict task",
#         "social exclusion paradigm",
#         "fear extinction training",
#     ]
#
#     text_embedder = TextEmbedding(
#         model_name="BrainGPT/BrainGPT-7B-v0.2",
#         batch_size=1,
#     )
#     my_vocabulary_emb = text_embedder(my_tasks)     # shape (n_tasks, embedding_dim)
#     my_prior = np.full(len(my_tasks), 1.0 / len(my_tasks))
#
#     task_df_new = image_to_labels(
#         emotion_img,
#         model_path=model_fn,
#         vocabulary=my_tasks,
#         vocabulary_emb=my_vocabulary_emb,
#         prior_probability=my_prior,
#         topk=len(my_tasks),
#         logit_scale=20.0,
#         model=model,
#         image_emb_gene=image_embedder,
#         data_dir=data_dir,
#     )

# %%
# Section 6: Latent space exploration
# ----------------------------------------
# NiCLIP learns a shared image–text embedding space.  Here we visualise:
#
# * The DiFuMo-512 parcellation atlas used to project activation maps.
# * Cosine similarity between embedded HCP contrasts (image–image).
# * Cosine similarity between embedded HCP contrasts and vocabulary
#   terms (image–text).

# %%
# **6a. DiFuMo-512 parcellation atlas**
#
# NiCLIP compresses each activation map to a 512-dimensional vector using
# the `DiFuMo atlas <https://parietal-inria.github.io/DiFuMo/>`_ before
# passing it through the CLIP image encoder.

from nilearn import datasets as nl_datasets
from nilearn.plotting import plot_roi

difumo_kwargs = dict(dimension=512, resolution_mm=2,
                     data_dir=op.join(data_dir, "nilearn"))
try:
    difumo = nl_datasets.fetch_atlas_difumo(legacy_format=False, **difumo_kwargs)
except TypeError:
    difumo = nl_datasets.fetch_atlas_difumo(**difumo_kwargs)

# Show a handful of DiFuMo components to illustrate the parcellation.
difumo_img = nib.load(difumo.maps)
n_components_to_show = 6
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for comp_i, ax in enumerate(axes.ravel()):
    comp_img = nli_image.index_img(difumo_img, comp_i)
    plot_roi(
        comp_img,
        display_mode="z",
        cut_coords=1,
        title=f"DiFuMo component {comp_i + 1}",
        axes=ax,
        colorbar=False,
    )
fig.suptitle("DiFuMo-512 atlas: first 6 components", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# **6b. Image–image cosine similarity across HCP contrasts**
#
# Embed all four downloaded HCP contrasts and measure how similar they are
# to each other in the shared CLIP latent space.  Semantically related
# contrasts (e.g., tasks in the same cognitive domain) should cluster.

from braindec.predict import preprocess_image

contrast_names = list(hcp_paths.keys())
image_embeddings = {}
for domain, img_path in hcp_paths.items():
    img = nib.load(img_path)
    img_emb = preprocess_image(
        img,
        data_dir=data_dir,
        image_emb_gene=image_embedder,
    )
    # Project through CLIP image encoder.
    with torch.no_grad():
        img_feat = model.encode_image(img_emb.to(device))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    image_embeddings[domain] = img_feat.cpu().numpy().squeeze()

# Compute pairwise cosine similarity.
n = len(contrast_names)
img_sim_matrix = np.zeros((n, n))
for i, d1 in enumerate(contrast_names):
    for j, d2 in enumerate(contrast_names):
        img_sim_matrix[i, j] = np.dot(image_embeddings[d1], image_embeddings[d2])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(img_sim_matrix, vmin=-1, vmax=1, cmap="RdYlBu_r")
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(contrast_names, rotation=30, ha="right")
ax.set_yticklabels(contrast_names)
plt.colorbar(im, ax=ax, label="Cosine similarity")
ax.set_title("Image–image similarity in CLIP latent space")
plt.tight_layout()
plt.show()

# %%
# **6c. Image–text similarity heatmap**
#
# Show how strongly each HCP contrast aligns with a curated set of
# vocabulary terms after projection through the CLIP encoders.  High
# similarity scores (warm colours) indicate that NiCLIP associates a
# contrast with a given cognitive task.

HIGHLIGHT_TASKS = [
    "motor fMRI task paradigm",
    "language processing fMRI task paradigm",
    "emotion processing fMRI task paradigm",
    "working memory fMRI task paradigm",
    "response inhibition",
    "mental rotation",
    "face recognition",
    "attention",
]

# Find indices of the highlight tasks in the vocabulary.
highlight_idx = []
highlight_found = []
for task in HIGHLIGHT_TASKS:
    if task in vocabulary:
        highlight_idx.append(vocabulary.index(task))
        highlight_found.append(task)
    else:
        # Fuzzy match: pick the vocabulary term with the most word overlap.
        query_words = set(task.lower().split())
        best_match = max(
            range(len(vocabulary)),
            key=lambda i: len(query_words & set(vocabulary[i].lower().split())),
        )
        highlight_idx.append(best_match)
        highlight_found.append(vocabulary[best_match])

# Text embeddings for the selected tasks (subset of precomputed array).
text_emb_subset = torch.from_numpy(vocabulary_emb[highlight_idx]).float().to(device)
text_emb_subset = text_emb_subset / (text_emb_subset.norm(dim=-1, keepdim=True) + 1e-8)

with torch.no_grad():
    text_feat_subset = model.encode_text(text_emb_subset)
    text_feat_subset = text_feat_subset / text_feat_subset.norm(dim=-1, keepdim=True)

text_feat_np = text_feat_subset.cpu().numpy()

# Build image × text similarity matrix.
img_text_sim = np.zeros((n, len(highlight_found)))
for i, domain in enumerate(contrast_names):
    img_text_sim[i] = text_feat_np @ image_embeddings[domain]

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(img_text_sim.T, aspect="auto", cmap="RdYlBu_r", vmin=-0.5, vmax=0.5)
ax.set_xticks(range(n))
ax.set_yticks(range(len(highlight_found)))
ax.set_xticklabels(contrast_names)
ax.set_yticklabels(
    [t[:45] + "…" if len(t) > 45 else t for t in highlight_found],
    fontsize=8,
)
plt.colorbar(im, ax=ax, label="Cosine similarity")
ax.set_title("Image–text CLIP similarity: HCP contrasts × selected vocabulary terms")
plt.tight_layout()
plt.show()

# %%
# **Summary**
#
# This tutorial demonstrated:
#
# * **Flat task decoding** (:func:`~braindec.predict.image_to_labels`) —
#   direct task posterior probabilities from a group-level map.
# * **Hierarchical decoding** (:func:`~braindec.predict.image_to_labels_hierarchical`) —
#   noisy-OR propagation to concept and domain levels.
# * **ROI characterization** — decoding anatomical binary masks to
#   characterise brain regions without meta-analytic maps.
# * **Subject-level decoding** — applying the same pipeline to noisier
#   single-subject maps (performance is lower; preprocessing choices matter).
# * **Custom vocabulary** — subsetting or replacing the Cognitive Atlas
#   vocabulary with domain-specific task lists.
# * **Latent space exploration** — inspecting the shared image–text
#   embedding space through cosine similarity matrices.
#
# Cite NiCLIP as:
#
# .. code-block:: text
#
#     Peraza et al. (2025). NiCLIP: Neuroimaging contrastive language-image
#     pretraining model for predicting text from brain activation images.
#     bioRxiv. https://doi.org/10.1101/2025.06.14.659706
