"""Predicts the output of the model on the test data."""

import argparse
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn._utils.niimg_conversions import check_same_fov
from nilearn.image import load_img, resample_to_img

from braindec.cogatlas import CognitiveAtlas
from braindec.embedding import ImageEmbedding
from braindec.model import build_model
from braindec.utils import _get_device, _read_vocabulary, get_data_dir


def preprocess_image(image, standardize=False, data_dir=None, space="MNI152", density=None):
    """
    Preprocess the image.

    Args:
        image: Images
    """
    data_dir = get_data_dir(data_dir)
    nilearn_dir = op.join(data_dir, "nilearn")

    image_emb_gene = ImageEmbedding(
        standardize=standardize,
        nilearn_dir=nilearn_dir,
        space=space,
        density=density,
    )
    image_embedding_arr = image_emb_gene(image)

    return torch.from_numpy(image_embedding_arr).float()


def image_to_labels(
    image,
    model_path,
    vocabulary,
    vocabulary_emb,
    prior_probability,
    topk=10,
    logit_scale=None,
    return_posterior_probability=False,
    device=None,
    **kwargs,
):
    """Predict the labels of an image using a pre-trained model."""
    if device is None:
        device = _get_device()

    if isinstance(image, str):
        image = load_img(image)

    image_input = preprocess_image(image, **kwargs).to(device)
    text_inputs = torch.from_numpy(vocabulary_emb).float().to(device)
    prior_probability = torch.from_numpy(prior_probability).float().to(device)

    # Normalize the embeddings
    text_inputs = text_inputs / (text_inputs.norm(dim=-1, keepdim=True) + 1e-8)
    image_input = image_input / (image_input.norm(dim=-1, keepdim=True) + 1e-8)

    # Calculate features
    model = build_model(model_path, device=device)
    with torch.no_grad():
        image_features, text_features = model(image_input, text_inputs)  # normalized

    # Get the scaling factor: inverse temperature
    logit_scale = model.logit_scale.item() if logit_scale is None else logit_scale

    # Pick the top topk most similar labels for the image
    # similarity = logit_scale * image_features @ text_features.T
    similarity = logit_scale * image_features @ text_features.T
    likelihood = similarity.softmax(dim=-1)
    # Flatten the probability distribution, since image_features is a single image
    similarity = similarity.flatten()
    likelihood = likelihood.flatten()  # P(A|T)

    joint_probability = likelihood * prior_probability  # P(A|T) * P(A)
    total_probability = joint_probability.sum()  # P(T)
    posterior_probability = joint_probability / total_probability  # P(T|A) = P(A|T) * P(A) / P(T)

    # Calculate the strength of the evidence using the Bayes factor
    posterrior_odds = posterior_probability / (1 - posterior_probability)
    prior_odds = prior_probability / (1 - prior_probability)
    bayes_factor = posterrior_odds / prior_odds

    # Get top tasks
    top_task_prob, top_indices = posterior_probability.topk(topk)
    top_indices = top_indices.cpu().detach().numpy()
    top_task_prob = top_task_prob.cpu().detach().numpy()

    similarity = similarity.cpu().detach().numpy()
    likelihood = likelihood.cpu().detach().numpy()
    joint_probability = joint_probability.cpu().detach().numpy()
    prior_probability = prior_probability.cpu().detach().numpy()
    bayes_factor = bayes_factor.cpu().detach().numpy()

    task_prob_df = pd.DataFrame(
        {
            "pred": np.array(vocabulary)[top_indices],
            "prob": top_task_prob,
            "similarity": similarity[top_indices],
            "likelihood": likelihood[top_indices],
            "prior_prob": prior_probability[top_indices],
            "joint_prob": joint_probability[top_indices],
            "bayes_factor": bayes_factor[top_indices],
        }
    )
    return task_prob_df, posterior_probability if return_posterior_probability else task_prob_df


def image_to_labels_hierarchical(
    image,
    model_path,
    vocabulary,
    vocabulary_emb,
    prior_probability,
    cognitiveatlas,
    topk=10,
    logit_scale=None,
    device=None,
    **kwargs,
):
    """Predict the label of an image."""
    task_prob_df, posterior_probability = image_to_labels(
        image,
        model_path,
        vocabulary,
        vocabulary_emb,
        prior_probability,
        topk=topk,
        logit_scale=logit_scale,
        return_posterior_probability=True,
        device=device,
        **kwargs,
    )

    # Get mapping and names from cognitive atlas object
    concept_to_task_idxs = cognitiveatlas.concept_to_task_idxs
    process_to_concept_idxs = cognitiveatlas.process_to_concept_idxs
    concept_names = cognitiveatlas.concept_names
    process_names = cognitiveatlas.process_names

    # Calculate P(C|A) = 1 - Prod(1 - P(T|A))
    concept_posterior_probability = torch.zeros(len(concept_names))  # Pre-allocate tensor
    for c_i in range(len(concept_names)):
        task_indices = concept_to_task_idxs[c_i]
        concept_probability = 1 - (1 - posterior_probability[task_indices]).prod()
        concept_posterior_probability[c_i] = concept_probability.cpu().detach()

    process_posterior_probability = torch.zeros(len(process_names))  # Pre-allocate tensor
    for p_i in range(len(process_names)):
        concept_indices = process_to_concept_idxs[p_i]
        process_probability = 1 - (1 - concept_posterior_probability[concept_indices]).prod()
        process_posterior_probability[p_i] = process_probability.cpu().detach()

    # Get the top k predictions
    top_concepts, top_concept_indices = concept_posterior_probability.topk(topk)
    top_concept_indices = top_concept_indices.cpu().detach().numpy()
    top_concepts = top_concepts.cpu().detach().numpy()

    top_processes, top_process_indices = torch.sort(process_posterior_probability, descending=True)
    top_process_indices = top_process_indices.cpu().detach().numpy()
    top_processes = top_processes.cpu().detach().numpy()

    process_prob_df = pd.DataFrame(
        {
            "pred": np.array(process_names)[top_process_indices],
            "prob": top_processes,
        }
    )
    concept_prob_df = pd.DataFrame(
        {
            "pred": np.array(concept_names)[top_concept_indices],
            "prob": top_concepts,
        }
    )
    return task_prob_df, concept_prob_df, process_prob_df


def _get_parser():
    parser = argparse.ArgumentParser(description="Run gradient-decoding workflow")
    parser.add_argument(
        "--image",
        dest="image",
        required=True,
        help="Path to image file (e.g., NIfTI format).",
    )
    parser.add_argument(
        "--model",
        dest="model",
        required=True,
        help="Path to the pre-trained CLIP model.",
    )
    parser.add_argument(
        "--vocabulary",
        dest="vocabulary",
        required=True,
        help="Path to the vocabulary file.",
    )
    parser.add_argument(
        "--vocabulary_emb",
        dest="vocabulary_emb",
        required=True,
        help="Path to the vocabulary embedding file.",
    )
    parser.add_argument(
        "--vocabulary_prior",
        dest="vocabulary_prior",
        required=True,
        help="Path to the vocabulary prior file.",
    )
    parser.add_argument(
        "--cognitiveatlas",
        dest="cognitiveatlas",
        required=False,
        help="Path to the cognitive atlas object file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hierarchical",
        dest="hierarchical",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--reduced",
        dest="reduced",
        required=False,
        default=True,
        help="Whether to use the reduced task set.",
    )
    parser.add_argument(
        "--mask",
        dest="mask",
        required=True,
        help="Path to the mask file.",
    )
    parser.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=10,
        help="Number of top predictions to return (default: 10).",
    )
    parser.add_argument(
        "--logit_scale",
        dest="logit_scale",
        type=float,
        default=10.0,
        help="Logit scale for temperature scaling (default: None).",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help="Device to use for computation (default: None). Possible values: cpu, mps, cuda.",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        required=False,
        help="Path to the output directory. Defaults to the same directory as the input image.",
    )
    return parser


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    image_fn = option.image
    model_fn = option.model
    vocabulary_fn = option.vocabulary
    vocabulary_emb_fn = option.vocabulary_emb
    vocabulary_prior_fn = option.vocabulary_prior
    cognitiveatlas_path = option.cognitiveatlas
    hierarchical = option.hierarchical
    reduced = option.reduced
    topk = option.topk
    logit_scale = option.logit_scale
    device = option.device
    mask_fn = option.mask
    output_dir = option.output_dir

    if hierarchical and cognitiveatlas_path is None:
        raise ValueError(
            "Cognitive atlas files are required for hierarchical decoding. "
            "Please provide the --cognitiveatlas path argument."
        )

    output_dir = op.dirname(image_fn) if output_dir is None else op.abspath(output_dir)
    image_name = op.basename(image_fn).split(".")[0]

    img = nib.load(image_fn)
    mask_img = nib.load(mask_fn)
    if not check_same_fov(img, reference_masker=mask_img):
        img = resample_to_img(img, mask_img)

    vocabulary, vocabulary_emb, vocabulary_prior = _read_vocabulary(
        vocabulary_fn,
        vocabulary_emb_fn,
        vocabulary_prior_fn,
    )

    if hierarchical:
        reduced_tasks_fn = op.join(cognitiveatlas_path, "reduced_tasks.csv")
        reduced_tasks_df = pd.read_csv(reduced_tasks_fn) if reduced else None

        cognitiveatlas = CognitiveAtlas(
            data_dir=cognitiveatlas_path,
            task_snapshot=op.join(cognitiveatlas_path, "task_snapshot-02-19-25.json"),
            concept_snapshot=op.join(
                cognitiveatlas_path, "concept_extended_snapshot-02-19-25.json"
            ),
            reduced_tasks=reduced_tasks_df,
        )

        task_prob_df, concept_prob_df, process_prob_df = image_to_labels_hierarchical(
            img,
            model_fn,
            vocabulary,
            vocabulary_emb,
            vocabulary_prior,
            cognitiveatlas,
            topk=topk,
            logit_scale=logit_scale,
            device=device,
        )

        task_prob_df.to_csv(op.join(output_dir, f"{image_name}_task_predictions.csv"), index=False)
        concept_prob_df.to_csv(
            op.join(output_dir, f"{image_name}_concept_predictions.csv"), index=False
        )
        process_prob_df.to_csv(
            op.join(output_dir, f"{image_name}_process_predictions.csv"), index=False
        )

    else:

        prob_df = image_to_labels(
            img,
            model_fn,
            vocabulary,
            vocabulary_emb,
            vocabulary_prior,
            topk=topk,
            logit_scale=logit_scale,
            return_posterior_probability=False,
            device=device,
        )

        prob_df.to_csv(op.join(output_dir, f"{image_name}_predictions.csv"), index=False)


if __name__ == "__main__":
    _main()
