"""Retriever — wraps the VectorStore for high-level query execution."""

from __future__ import annotations

from typing import List

from src.chunker import Chunk
from src.embedder import VectorStore


class Retriever:
    """
    High-level retrieval interface on top of VectorStore.

    Responsibilities:
        - Accept natural language queries
        - Return the most relevant chunks with scores
        - Filter out low-confidence results
    """

    MIN_SCORE_THRESHOLD = 0.01

    def __init__(self, vector_store: VectorStore, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Chunk]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: User's natural language question.

        Returns:
            List of relevant Chunk objects (filtered by score).
        """
        results = self.vector_store.search(query, top_k=self.top_k)
        relevant = [
            chunk for chunk, score in results
            if score >= self.MIN_SCORE_THRESHOLD
        ]
        return relevant

    def retrieve_with_scores(self, query: str) -> List[tuple[Chunk, float]]:
        """Retrieve chunks along with their similarity scores."""
        return self.vector_store.search(query, top_k=self.top_k)
