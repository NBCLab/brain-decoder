"""Code to determine embeddings for text and images."""

import textwrap
from typing import List, Tuple, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from nilearn import datasets
from nilearn.maskers import MultiNiftiMapsMasker
from nimare.dataset import Dataset
from nimare.meta.kernel import MKDAKernel
from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from braindec.utils import _get_device


class TextEmbedding:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1", max_length: int = 512):
        """
        Initialize the embedding generator with specified model and parameters.

        Args:
            model_name: Name of the Mistral model to use
            max_length: Maximum token length for each chunk
        """
        self.device = _get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
        self.vocabulary_embeddings = None

    def set_vocabulary(self, vocabulary: List[str]):
        """
        Set the vocabulary and generate embeddings for each word.

        Args:
            vocabulary: List of words to use as vocabulary
        """
        # Generate embeddings for each vocabulary word
        self.vocabulary = vocabulary
        self.vocabulary_embeddings = []

        for word in vocabulary:
            embedding = self.generate_embedding(word)
            self.vocabulary_embeddings.append(embedding)

        self.vocabulary_embeddings = np.array(self.vocabulary_embeddings)

    def decode_embedding(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar words in the vocabulary for a given embedding.

        Args:
            embedding: Input embedding to decode
            top_k: Number of most similar words to return

        Returns:
            List of tuples containing (word, similarity_score)
        """
        if self.vocabulary_embeddings is None:
            raise ValueError("Vocabulary not set. Call set_vocabulary() first.")

        # Calculate cosine similarity between input embedding and vocabulary embeddings
        similarities = []
        for vocab_embedding in self.vocabulary_embeddings:
            similarity = 1 - cosine(embedding, vocab_embedding)
            similarities.append(similarity)

        # Convert similarities to probabilities using softmax
        similarities = torch.tensor(similarities)
        probabilities = torch.softmax(similarities, dim=0).numpy()

        # Get top-k similar words
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        results = [(self.vocabulary[idx], probabilities[idx]) for idx in top_indices]

        return results

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
            text, width=chars_per_chunk, break_long_words=False, break_on_hyphens=False
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
        for chunk in chunks:
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
            return np.stack([self.process_large_text(t) for t in text])


class ImageEmbedding:
    def __init__(self):
        """
        Initialize the image embedding generator with specified model.

        Args:
            model_name: Name of the DeiT model to use
        """
        difumo = datasets.fetch_atlas_difumo(dimension=512, resolution_mm=2, legacy_format=False)
        self.masker = MultiNiftiMapsMasker(maps_img=difumo.maps)

    def process_single_image(self, image, masker):
        """Process a single image using the provided masker

        Args:
            image: Single image to process
            masker: Initialized masker object with fit_transform method

        Returns:
            Transformed image embedding
        """
        return masker.fit_transform(image)

    def parallel_image_masking(self, image_output, masker, n_jobs=-1):
        """Apply masking transformation to multiple images in parallel

        Args:
            image_output: List/array of images to process
            masker: Initialized masker object (e.g. DiFuMo masker)
            n_jobs: Number of parallel jobs (-1 for all available cores)

        Returns:
            Array of transformed image embeddings
        """
        # Create progress bar
        with tqdm(total=len(image_output), desc="Processing images") as pbar:
            # Process images in parallel with progress updates
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.process_single_image)(img, masker)
                for img in tqdm(image_output, leave=False)
            )

        # Stack results into a single array
        return np.vstack(results)

    def generate_embedding(self, dset: Dataset) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image: Input image as a numpy array

        Returns:
            Numpy array containing the embedding
        """
        # Get images from coordinates
        kernel = MKDAKernel()
        self.images = kernel.transform(dset, return_type="image")

        return self.parallel_image_masking(self.images, self.masker)

    def __call__(self, dset: Dataset) -> np.ndarray:
        """
        Generate embeddings for input images.

        Args:
            images: List of input images as numpy arrays

        Returns:
            Numpy array of embeddings
        """
        return self.generate_embedding(dset)
