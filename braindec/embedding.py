"""Code to determine embeddings for text and images."""

import textwrap
from typing import List, Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import cosine
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
