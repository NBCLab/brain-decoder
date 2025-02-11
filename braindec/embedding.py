"""Code to determine embeddings for text and images."""

from typing import List, Union

import numpy as np
import torch
from nilearn import datasets
from nilearn.image import concat_imgs
from nilearn.maskers import MultiNiftiMapsMasker
from nimare.dataset import Dataset
from nimare.meta.kernel import MKDAKernel
from peft import PeftConfig, PeftModel
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
        batch_size: int = 1,
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
            batch_size: Batch size for processing (total number of papers to process at once)
            device: Device to use for computation. If None, the device is automatically selected.
        """
        self.device = _get_device() if device is None else device
        self.model_name = model_name
        self.batch_size = batch_size

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

        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def clear_device_cache(self):
        """Clear memory cache for the current device type."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            # MPS (Apple Silicon) garbage collection
            torch.mps.empty_cache()

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

        return sentence_embeddings

    def generate_embedding(self, token_embeddings, attention_mask) -> np.ndarray:
        """
        Generate embedding from token embeddings.

        Args:
            token_embeddings: Token embeddings
            attention_mask: Attention mask

        Returns:
            Numpy array containing the average embedding
        """
        embeddings = self.mean_pooling(token_embeddings, attention_mask)

        return embeddings.cpu().numpy()

    def get_token_embeddings(self, tokenized: dict) -> np.ndarray:
        """
        Get token embeddings for a single text chunk.

        Args:
            tokenized: Tokenized dictionary containing input_ids and attention_mask

        Returns:
            Numpy array containing the embedding
        """
        # Send to device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**tokenized, output_hidden_states=True)
            return outputs.hidden_states[-1]

    def chunk_text(self, texts: str) -> List[str]:
        """
        Split text into chunks that respect the model's token limit.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks (tokenized dictionaries)
        """
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Split into chunks of dictionaries
        chunks = []
        for i in range(0, input_ids.size(1), self.max_length):
            chunks.append(
                {
                    "input_ids": input_ids[:, i : i + self.max_length],
                    "attention_mask": attention_mask[:, i : i + self.max_length],
                }
            )

        return chunks

    def process_text(self, text: List[str]) -> np.ndarray:
        """
        Process text by chunking and averaging token embeddings.

        Args:
            text: Batch of texts

        Returns:
            Averaged embedding vector for the entire batch of texts
        """
        # Split text into chunks
        chunks = self.chunk_text(text)

        # Generate embeddings for each chunk
        token_embeddings = []
        attention_masks = []
        for chunk in tqdm(chunks, desc="Processing chunks", leave=False):
            token_embeddings.append(self.get_token_embeddings(chunk))
            attention_masks.append(chunk["attention_mask"].to(self.device))

        # Delete chunks to free up memory
        del chunks

        token_embeddings = torch.cat(token_embeddings, dim=1)
        attention_masks = torch.cat(attention_masks, dim=1)

        return self.generate_embedding(token_embeddings, attention_masks)

    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for input text(s).

        Args:
            text: Input text or list of texts

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            return self.process_text([texts])
        else:
            # Process multiple texts in batches
            embeddings = []
            for t in tqdm(range(0, len(texts), self.batch_size), desc="Processing batches"):
                embeddings.append(self.process_text(texts[t : t + self.batch_size]))

            return np.concatenate(embeddings)


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


def normalize_embeddings(embeddings):
    # Calculate L2 norm along last dimension
    # keepdims=True maintains the dimension for broadcasting
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)

    # Divide by norm to normalize
    # Added small epsilon to prevent division by zero
    return embeddings / (norms + 1e-8)
