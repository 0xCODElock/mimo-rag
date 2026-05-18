"""End-to-end RAG pipeline — the single entry point for external code."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.document_loader import DocumentLoader
from src.chunker import TextChunker
from src.embedder import VectorStore
from src.retriever import Retriever
from src.generator import MiMoGenerator, RAGResponse
from config import AppConfig, config as default_config


class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation workflow.

    Usage::

        pipeline = RAGPipeline(api_key="...")
        pipeline.ingest("report.pdf")
        pipeline.ingest("notes.md")
        response = pipeline.query("What is the conclusion?")
        print(response.answer)
    """

    def __init__(
        self,
        api_key: str,
        cfg: Optional[AppConfig] = None,
    ):
        self.cfg = cfg or default_config
        self.cfg.mimo.api_key = api_key

        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=self.cfg.rag.chunk_size,
            chunk_overlap=self.cfg.rag.chunk_overlap,
        )
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store, top_k=self.cfg.rag.top_k)
        self.generator = MiMoGenerator(
            api_key=self.cfg.mimo.api_key,
            api_base=self.cfg.mimo.api_base,
            model=self.cfg.mimo.model,
            max_tokens=self.cfg.mimo.max_tokens,
            temperature=self.cfg.mimo.temperature,
        )
        self._ingested_files: List[str] = []

    def ingest(self, path: str) -> int:
        """
        Load and index a document.

        Args:
            path: File path to the document (PDF / DOCX / MD / TXT).

        Returns:
            Number of chunks indexed.
        """
        doc = self.loader.load(path)
        chunks = self.chunker.split(doc)
        self.vector_store.add(chunks)
        self._ingested_files.append(str(path))
        print(f"[Pipeline] Ingested '{path}' → {len(chunks)} chunks")
        return len(chunks)

    def query(self, question: str) -> RAGResponse:
        """
        Answer a question using the indexed documents.

        Args:
            question: Natural language question.

        Returns:
            RAGResponse containing the answer and source citations.
        """
        if self.vector_store.size == 0:
            raise RuntimeError("No documents ingested. Call pipeline.ingest() first.")

        chunks = self.retriever.retrieve(question)

        if not chunks:
            return RAGResponse(
                answer="I couldn't find relevant information in the documents for your question.",
                sources=[],
                chunks_used=0,
                model=self.cfg.mimo.model,
            )

        return self.generator.generate(question, chunks)

    def save(self, path: Optional[str] = None) -> None:
        """Persist the current vector store to disk."""
        save_path = path or self.cfg.rag.vector_store_path
        self.vector_store.save(save_path)

    def load(self, path: Optional[str] = None) -> None:
        """Load a previously saved vector store."""
        load_path = path or self.cfg.rag.vector_store_path
        self.vector_store.load(load_path)

    @property
    def document_count(self) -> int:
        return len(self._ingested_files)

    @property
    def chunk_count(self) -> int:
        return self.vector_store.size
