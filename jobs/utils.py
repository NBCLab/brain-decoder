"""Shared helpers for analysis and decoding jobs."""

import argparse
import ast
import json
import os.path as op

import numpy as np
import pandas as pd

DEFAULT_MODEL_IDS = [
    "BrainGPT/BrainGPT-7B-v0.2",
    "mistralai/Mistral-7B-v0.1",
    "BrainGPT/BrainGPT-7B-v0.1",
    "meta-llama/Llama-2-7b-chat-hf",
]
DEFAULT_SECTIONS = ["abstract", "body"]


def _read_prior(prior_fn):
    if op.exists(prior_fn):
        return np.load(prior_fn)

    csv_prior_fn = f"{op.splitext(prior_fn)[0]}.csv"
    if not op.exists(csv_prior_fn):
        raise FileNotFoundError(f"Could not find prior file {prior_fn} or CSV fallback {csv_prior_fn}.")

    prior_df = pd.read_csv(csv_prior_fn)
    if "prior" in prior_df.columns:
        prior_values = prior_df["prior"].to_numpy()
    else:
        numeric_columns = prior_df.select_dtypes(include="number").columns
        if len(numeric_columns) == 0:
            raise ValueError(f"CSV prior fallback {csv_prior_fn} does not contain a numeric prior column.")
        prior_values = prior_df[numeric_columns[-1]].to_numpy()

    np.save(prior_fn, prior_values)
    return prior_values


def _read_vocabulary(vocabulary_fn, vocabulary_emb_fn, vocabulary_prior_fn=None):
    with open(vocabulary_fn, "r") as f:
        vocabulary = [line.strip() for line in f]

    if vocabulary_prior_fn is not None:
        return vocabulary, np.load(vocabulary_emb_fn), _read_prior(vocabulary_prior_fn)

    return vocabulary, np.load(vocabulary_emb_fn)


def str_to_bool(value):
    """Parse common string forms of booleans for argparse."""
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value from {value!r}.")


def get_default_project_dir():
    """Return the repository root based on the current jobs directory."""
    return op.abspath(op.join(op.dirname(__file__), ".."))


def resolve_project_paths(project_dir=None, data_dir=None, results_dir=None):
    """Resolve project-relative data and results directories."""
    project_dir = get_default_project_dir() if project_dir is None else op.abspath(project_dir)
    data_dir = op.join(project_dir, "data") if data_dir is None else op.abspath(data_dir)
    results_dir = op.join(project_dir, "results") if results_dir is None else op.abspath(results_dir)
    return project_dir, data_dir, results_dir


def get_source(reduced):
    """Return the ontology source label for a reduced/full vocabulary."""
    return "cogatlasred" if reduced else "cogatlas"


def get_model_name(model_id):
    """Extract the short model name used in filenames."""
    return model_id.split("/")[-1]


def strip_nii_suffix(path_or_name):
    """Return a filename stem while preserving inner dots in `.nii.gz` names."""
    filename = op.basename(path_or_name)
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return op.splitext(filename)[0]


def infer_prediction_label(path_or_name, delimiter="_", token_index=None):
    """Infer the prediction label used in output filenames from an image filename."""
    stem = strip_nii_suffix(path_or_name)
    if token_index is None:
        return stem

    parts = stem.split(delimiter)
    if token_index >= len(parts) or token_index < -len(parts):
        raise ValueError(
            f"Cannot extract token index {token_index} from {path_or_name!r} using delimiter {delimiter!r}."
        )
    return parts[token_index]


def parse_name_list(value):
    """Parse a CSV/JSON-ish field into a normalized list of label strings."""
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return []

        if normalized[0] in {"[", "("}:
            try:
                parsed = ast.literal_eval(normalized)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]

        for delimiter in (";", "|", ","):
            if delimiter in normalized:
                return [item.strip() for item in normalized.split(delimiter) if item.strip()]

        return [normalized]

    return [str(value).strip()]


def add_common_job_args(parser):
    """Add the shared CLI surface used by analysis jobs."""
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        default=get_default_project_dir(),
        help="Path to the repository root. Defaults to the parent of the jobs directory.",
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        default=None,
        help="Optional explicit path to the data directory. Defaults to <project_dir>/data.",
    )
    parser.add_argument(
        "--results_dir",
        dest="results_dir",
        default=None,
        help="Optional explicit path to the results directory. Defaults to <project_dir>/results.",
    )
    parser.add_argument(
        "--sections",
        dest="sections",
        nargs="+",
        default=list(DEFAULT_SECTIONS),
        help="Text sections to evaluate. Defaults to abstract and body.",
    )
    parser.add_argument(
        "--model_ids",
        dest="model_ids",
        nargs="+",
        default=list(DEFAULT_MODEL_IDS),
        help="One or more embedding model identifiers to evaluate.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help="Device to use for computation. Defaults to the braindec device helper.",
    )
    parser.add_argument(
        "--reduced",
        dest="reduced",
        type=str_to_bool,
        default=True,
        help="Whether to use the reduced Cognitive Atlas vocabulary. Defaults to true.",
    )
    parser.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=20,
        help="Number of predictions to keep per image.",
    )
    parser.add_argument(
        "--standardize",
        dest="standardize",
        type=str_to_bool,
        default=False,
        help="Whether to standardize images before embedding.",
    )
    parser.add_argument(
        "--logit_scale",
        dest="logit_scale",
        type=float,
        default=20.0,
        help="Override CLIP logit scale used during decoding.",
    )
    return parser


def build_cognitiveatlas(data_dir, reduced, concept_to_process_fn=None):
    """Construct a CognitiveAtlas object from local snapshots."""
    from braindec.cogatlas import CognitiveAtlas

    reduced_tasks_fn = op.join(data_dir, "cognitive_atlas", "reduced_tasks.csv")
    reduced_tasks_df = pd.read_csv(reduced_tasks_fn) if reduced else None

    concept_to_process = None
    if concept_to_process_fn is not None and op.exists(concept_to_process_fn):
        with open(concept_to_process_fn, "r") as file:
            concept_to_process = json.load(file)

    return CognitiveAtlas(
        data_dir=data_dir,
        task_snapshot=op.join(data_dir, "cognitive_atlas", "task_snapshot-02-19-25.json"),
        concept_snapshot=op.join(data_dir, "cognitive_atlas", "concept_extended_snapshot-02-19-25.json"),
        concept_to_process=concept_to_process,
        reduced_tasks=reduced_tasks_df,
    )


def load_decoding_resources(results_dir, voc_dir, source, category, sub_category, model_id, section):
    """Load model and vocabulary artifacts for a decoding run."""
    model_name = get_model_name(model_id)
    model_path = op.join(
        results_dir,
        "pubmed",
        f"model-clip_section-{section}_embedding-{model_name}_best.pth",
    )
    vocabulary_label = f"vocabulary-{source}_{category}-{sub_category}_embedding-{model_name}"
    vocabulary_fn = op.join(voc_dir, f"vocabulary-{source}_{category}.txt")
    vocabulary_emb_fn = op.join(voc_dir, f"{vocabulary_label}.npy")
    vocabulary_prior_fn = op.join(voc_dir, f"{vocabulary_label}_section-{section}_prior.npy")
    vocabulary, vocabulary_emb, vocabulary_prior = _read_vocabulary(
        vocabulary_fn,
        vocabulary_emb_fn,
        vocabulary_prior_fn,
    )

    return {
        "model_name": model_name,
        "model_path": model_path,
        "vocabulary_label": vocabulary_label,
        "vocabulary": vocabulary,
        "vocabulary_emb": vocabulary_emb,
        "vocabulary_prior": vocabulary_prior,
    }
