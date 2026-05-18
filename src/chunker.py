"""Recursive text splitter for creating document chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.document_loader import Document


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    text: str
    source: str
    chunk_index: int
    total_chunks: int
    doc_type: str = "unknown"

    @property
    def metadata(self) -> dict:
        return {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "doc_type": self.doc_type,
        }


class TextChunker:
    """
    Recursively splits documents into overlapping text chunks.

    Strategy:
        1. Try to split on double newlines (paragraphs) first.
        2. Fall back to single newlines, then sentences, then characters.
        3. Respect chunk_size while preserving context via overlap.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size:    Target maximum characters per chunk.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._separators = ["\n\n", "\n", ". ", " ", ""]

    def split(self, document: Document) -> List[Chunk]:
        """
        Split a Document into a list of Chunk objects.

        Args:
            document: Loaded Document to split.

        Returns:
            List of Chunk objects ready for embedding.
        """
        raw_chunks = self._recursive_split(document.content, self._separators)
        chunks = []
        for i, text in enumerate(raw_chunks):
            chunks.append(
                Chunk(
                    text=text,
                    source=document.source,
                    chunk_index=i,
                    total_chunks=len(raw_chunks),
                    doc_type=document.doc_type,
                )
            )
        return chunks

    def split_many(self, documents: List[Document]) -> List[Chunk]:
        """Split multiple documents into chunks."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split(doc))
        return all_chunks

    # ── Private helpers ──────────────────────────────────────────────────────

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the best available separator."""
        if not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()]

        separator = separators[0] if separators else ""
        next_separators = separators[1:] if len(separators) > 1 else [""]

        splits = text.split(separator) if separator else list(text)
        chunks: List[str] = []
        current = ""

        for split in splits:
            candidate = (current + separator + split).strip() if current else split.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    # Carry over overlap
                    overlap_start = max(0, len(current) - self.chunk_overlap)
                    current = current[overlap_start:] + (separator if separator else "") + split.strip()
                else:
                    # Single split too long — recurse with finer separator
                    sub = self._recursive_split(split.strip(), next_separators)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]
