"""Unit tests for the Retriever and VectorStore."""

import pytest
from src.chunker import Chunk
from src.embedder import VectorStore
from src.retriever import Retriever


def make_chunks(texts):
    return [
        Chunk(text=t, source="test.txt", chunk_index=i, total_chunks=len(texts))
        for i, t in enumerate(texts)
    ]


TEXTS = [
    "Python is a popular programming language used for data science.",
    "Machine learning models require large amounts of training data.",
    "The Eiffel Tower is located in Paris, France.",
    "RAG combines retrieval with language model generation.",
]


def test_vector_store_search():
    store = VectorStore()
    store.add(make_chunks(TEXTS))
    results = store.search("machine learning data", top_k=2)
    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert top_score > 0


def test_vector_store_empty_raises():
    store = VectorStore()
    with pytest.raises(RuntimeError):
        store.search("anything")


def test_retriever_relevance():
    store = VectorStore()
    store.add(make_chunks(TEXTS))
    retriever = Retriever(store, top_k=2)
    chunks = retriever.retrieve("Paris Eiffel Tower")
    assert any("Eiffel" in c.text or "Paris" in c.text for c in chunks)
