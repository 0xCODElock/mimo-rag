"""Integration tests for the full RAG pipeline."""

import os
import tempfile
import pytest

from src.pipeline import RAGPipeline


SAMPLE_MARKDOWN = """
# Artificial Intelligence Overview

Artificial Intelligence (AI) refers to the simulation of human intelligence
processed by machines, especially computer systems.

## Machine Learning

Machine learning is a subset of AI that enables systems to learn and improve
from experience without being explicitly programmed.

## Deep Learning

Deep learning uses neural networks with many layers to analyze various factors
of data, achieving state-of-the-art results on tasks like image recognition,
natural language processing, and more.

## Applications

- Healthcare: disease diagnosis, drug discovery
- Finance: fraud detection, algorithmic trading
- Transportation: autonomous vehicles, route optimization
- Education: personalized learning, automated grading
"""


@pytest.fixture
def sample_doc(tmp_path):
    """Create a temporary markdown document for testing."""
    doc = tmp_path / "sample.md"
    doc.write_text(SAMPLE_MARKDOWN)
    return str(doc)


@pytest.fixture
def pipeline(sample_doc):
    """Initialize pipeline with a dummy API key (no real calls in unit tests)."""
    p = RAGPipeline(api_key="test_key_placeholder")
    p.ingest(sample_doc)
    return p


def test_pipeline_ingest_returns_chunk_count(sample_doc):
    p = RAGPipeline(api_key="test")
    count = p.ingest(sample_doc)
    assert count > 0


def test_pipeline_chunk_count_after_ingest(pipeline):
    assert pipeline.chunk_count > 0


def test_pipeline_document_count(pipeline):
    assert pipeline.document_count == 1


def test_pipeline_retriever_finds_relevant_chunks(pipeline):
    chunks = pipeline.retriever.retrieve("What is machine learning?")
    assert len(chunks) > 0
    # The most relevant chunk should contain machine learning content
    texts = " ".join(c.text for c in chunks).lower()
    assert "machine" in texts or "learning" in texts or "ai" in texts


def test_pipeline_multiple_ingest(tmp_path):
    doc1 = tmp_path / "doc1.md"
    doc2 = tmp_path / "doc2.md"
    doc1.write_text("Document one about Python programming language." * 10)
    doc2.write_text("Document two about deep neural networks." * 10)

    p = RAGPipeline(api_key="test")
    p.ingest(str(doc1))
    p.ingest(str(doc2))

    assert p.document_count == 2
    assert p.chunk_count > 0


def test_pipeline_save_and_load(pipeline, tmp_path):
    save_path = str(tmp_path / "store")
    pipeline.save(save_path)

    new_pipeline = RAGPipeline(api_key="test")
    new_pipeline.load(save_path)

    assert new_pipeline.chunk_count == pipeline.chunk_count


def test_pipeline_query_no_docs_raises():
    p = RAGPipeline(api_key="test")
    with pytest.raises(RuntimeError):
        p.query("anything")
