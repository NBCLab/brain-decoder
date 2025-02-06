"""Code to determine embeddings for text and images."""

import textwrap
from typing import List, Union

import numpy as np
import torch
from nilearn import datasets
from nilearn.image import concat_imgs
from nilearn.maskers import MultiNiftiMapsMasker
from nimare.dataset import Dataset
from nimare.meta.kernel import MKDAKernel
from peft import PeftConfig, PeftModel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from braindec.utils import _get_device


def _coordinates_to_image(dset: Dataset, kernel: str = "mkda"):
    if kernel == "mkda":
        kernel = MKDAKernel()
    else:
        raise ValueError(f"Kernel {kernel} not supported.")
    return kernel.transform(dset, return_type="image")


class TextEmbedding:
    def __init__(
        self,
        model_name: str = "BrainGPT/BrainGPT-7B-v0.2",
        max_length: int = None,
        device: str = None,
    ):
        """
        Initialize the embedding generator with specified model and parameters.

        Args:
            model_name: Name of the model to use. Supported models are:
                - "mistralai/Mistral-7B-v0.1"
                - "meta-llama/Llama-2-7b-chat-hf"
                - "BrainGPT/BrainGPT-7B-v0.1"
                - "BrainGPT/BrainGPT-7B-v0.2"
            max_length: Maximum token length for each chunk
            device: Device to use for computation. If None, the device is automatically selected.
        """
        self.device = _get_device() if device is None else device
        self.model_name = model_name

        if model_name == "mistralai/Mistral-7B-v0.1":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.max_length = 8192 if max_length is None else max_length

        elif model_name == "meta-llama/Llama-2-7b-chat-hf":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.max_length = 4096 if max_length is None else max_length

        elif model_name == "BrainGPT/BrainGPT-7B-v0.1":
            config = PeftConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

            self.model = PeftModel.from_pretrained(model, model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            self.max_length = 4096 if max_length is None else max_length

        elif model_name == "BrainGPT/BrainGPT-7B-v0.2":
            config = PeftConfig.from_pretrained(model_name)
            # The config file has path to the base model instead of the model name
            model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

            self.model = PeftModel.from_pretrained(model, model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            self.max_length = 8192 if max_length is None else max_length
        else:
            raise ValueError(f"Model name {model_name} not supported.")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks that respect the model's token limit.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks
        """
        # Use approximate characters per token ratio to estimate chunk size
        chars_per_token = 4
        chars_per_chunk = self.max_length * chars_per_token

        # Split into chunks while trying to respect sentence boundaries
        chunks = textwrap.wrap(
            text,
            width=chars_per_chunk,
            break_long_words=False,
            break_on_hyphens=False,
        )

        return chunks

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single chunk of text.

        Args:
            text: Input text chunk

        Returns:
            Numpy array containing the embedding
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        # padding=True

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.model_name.startswith("BrainGPT") or self.model_name.startswith("meta-llama"):
            # Use the mean of the last hidden state as the embedding
            embeddings = outputs.logits.mean(dim=1)
        else:
            # Use the mean of the last hidden state as the embedding
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy()[0]

    def process_large_text(self, text: str) -> np.ndarray:
        """
        Process large text by chunking and averaging embeddings.

        Args:
            text: Large input text

        Returns:
            Averaged embedding vector for the entire text
        """
        # Split text into chunks
        chunks = self.chunk_text(text)

        # Generate embeddings for each chunk
        chunk_embeddings = []
        for chunk in tqdm(chunks, desc="Processing chunks", leave=False):
            embedding = self.generate_embedding(chunk)
            chunk_embeddings.append(embedding)

        # Average the embeddings
        average_embedding = np.mean(chunk_embeddings, axis=0)

        return average_embedding

    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for input text(s).

        Args:
            text: Input text or list of texts

        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, str):
            return self.process_large_text(text)
        else:
            # Process multiple texts with progress bar
            embeddings = []
            for t in tqdm(text, desc="Processing texts"):
                embeddings.append(self.process_large_text(t))

            return np.stack(embeddings)


class ImageEmbedding:
    def __init__(
        self,
        standardize: bool = False,
        data_dir: str = None,
        atlas: str = "difumo",
        dimension: int = 512,
    ):
        """
        Initialize the image embedding generator with specified model.

        Args:
            model_name: Name of the DeiT model to use
        """
        self.data_dir = data_dir
        self.atlas = atlas
        self.dimension = dimension

        if atlas == "difumo":
            difumo = datasets.fetch_atlas_difumo(
                dimension=dimension,
                resolution_mm=2,
                legacy_format=False,
                data_dir=data_dir,
            )
            self.masker = MultiNiftiMapsMasker(maps_img=difumo.maps, standardize=standardize)
        else:
            # Implement other atlases
            raise ValueError(f"Atlas {atlas} not supported.")

    def generate_embedding(self, images) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image: Input image as a numpy array

        Returns:
            Numpy array containing the embedding
        """
        if isinstance(images, list):
            # Concat images to improve performance
            images = concat_imgs(images)

        return self.masker.fit_transform(images)

    def __call__(self, images) -> np.ndarray:
        """
        Generate embeddings for input images.

        Args:
            images: List of input images as numpy arrays

        Returns:
            Numpy array of embeddings
        """
        # Accept nifti and path to image as input
        return self.generate_embedding(images)


def get_word_relevance(document_embeddings, word_embeddings, threshold=0.5):
    """
    Calculate overall word importance combining multiple metrics

    Args:
        word_embeddings: dict of word embeddings
    """
    semantic_tf = document_embeddings @ word_embeddings.T
    semantic_count = np.sum(semantic_tf > threshold, axis=0)

    semantic_idf = TfidfTransformer().fit_transform(semantic_count)

    return semantic_tf, semantic_count, semantic_tf * semantic_idf
