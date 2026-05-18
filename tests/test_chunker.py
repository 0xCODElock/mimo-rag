"""Unit tests for the TextChunker."""

import pytest
from src.document_loader import Document
from src.chunker import TextChunker


SAMPLE_DOC = Document(
    content="This is sentence one. " * 50 + "\n\nNew paragraph here. " * 20,
    source="test.txt",
    doc_type="txt",
)


def test_chunker_basic():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split(SAMPLE_DOC)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 300  # some tolerance


def test_chunker_metadata():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split(SAMPLE_DOC)
    assert chunks[0].source == "test.txt"
    assert chunks[0].doc_type == "txt"
    assert chunks[0].total_chunks == len(chunks)


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=200)


def test_chunker_short_text():
    doc = Document(content="Short text.", source="test.txt", doc_type="txt")
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."
