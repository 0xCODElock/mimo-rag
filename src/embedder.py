"""TF-IDF based embedder and vector store."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunker import Chunk


class VectorStore:
    """
    In-memory vector store backed by TF-IDF representations.

    Stores:
        - chunks: the original Chunk objects
        - matrix: sparse TF-IDF matrix (n_chunks × vocab_size)
        - vectorizer: fitted TfidfVectorizer for query transformation
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self._is_fitted = False

    def add(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the store and refit the TF-IDF vectorizer.

        Args:
            chunks: New chunks to index.
        """
        self.chunks.extend(chunks)
        self._refit()

    def search(self, query: str, top_k: int = 4) -> List[tuple[Chunk, float]]:
        """
        Find the top-k most relevant chunks for a query.

        Args:
            query: User's question string.
            top_k: Number of results to return.

        Returns:
            List of (Chunk, similarity_score) tuples, sorted by relevance.
        """
        if not self._is_fitted:
            raise RuntimeError("Vector store is empty. Ingest documents first.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def save(self, path: str | Path) -> None:
        """Persist the vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "store.pkl", "wb") as f:
            pickle.dump(
                {"chunks": self.chunks, "vectorizer": self.vectorizer, "matrix": self.matrix},
                f,
            )
        print(f"[VectorStore] Saved {len(self.chunks)} chunks to {path}")

    def load(self, path: str | Path) -> None:
        """Load a previously saved vector store from disk."""
        path = Path(path)
        store_file = path / "store.pkl"
        if not store_file.exists():
            raise FileNotFoundError(f"No vector store found at {path}")
        with open(store_file, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        self._is_fitted = True
        print(f"[VectorStore] Loaded {len(self.chunks)} chunks from {path}")

    @property
    def size(self) -> int:
        return len(self.chunks)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _refit(self) -> None:
        """Refit the TF-IDF vectorizer on all current chunks."""
        texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
