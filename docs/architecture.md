# MiMo-RAG Architecture

## Overview

MiMo-RAG is built around three core stages: **ingestion**, **retrieval**, and **generation**. Each stage is implemented as an independent, testable module.

---

## Stage 1: Document Ingestion

### DocumentLoader (`src/document_loader.py`)

Handles multi-format document parsing:

| Format | Library | Notes |
|--------|---------|-------|
| PDF | `pypdf` | Multi-page text extraction |
| DOCX | `python-docx` | Paragraph-level extraction |
| Markdown | Built-in | UTF-8 read |
| TXT | Built-in | UTF-8 read |

### TextChunker (`src/chunker.py`)

Uses a **recursive character splitter** strategy:

1. Attempt to split on `\n\n` (paragraph boundaries)
2. Fall back to `\n` (line boundaries)
3. Fall back to `. ` (sentence boundaries)
4. Fall back to ` ` (word boundaries)
5. Last resort: character-level splitting

This preserves semantic coherence while respecting `chunk_size` limits. The `chunk_overlap` parameter ensures context continuity across adjacent chunks.

---

## Stage 2: Embedding & Retrieval

### VectorStore (`src/embedder.py`)

- Uses **TF-IDF** (Term Frequency–Inverse Document Frequency) vectorization
- `ngram_range=(1,2)` captures both unigrams and bigrams
- Stores sparse matrix for memory efficiency
- Supports `save()` / `load()` for persistence

### Retriever (`src/retriever.py`)

- Transforms the user query with the fitted TF-IDF vectorizer
- Computes **cosine similarity** between query vector and all chunk vectors
- Returns top-k chunks above a minimum score threshold

---

## Stage 3: Generation

### MiMoGenerator (`src/generator.py`)

- Uses the **OpenAI-compatible** MiMo API via the `openai` Python SDK
- Constructs a two-part prompt:
  1. **System prompt** — instructs the model to be grounded and cite sources
  2. **User prompt** — formatted context blocks + the user's question
- Returns a `RAGResponse` with the answer, sources, and metadata

---

## Pipeline Orchestration

### RAGPipeline (`src/pipeline.py`)

The `RAGPipeline` class wires all components together:

```
RAGPipeline.ingest(path)
    └── DocumentLoader.load(path)      → Document
    └── TextChunker.split(document)    → List[Chunk]
    └── VectorStore.add(chunks)        → updates TF-IDF index

RAGPipeline.query(question)
    └── Retriever.retrieve(question)   → List[Chunk]
    └── MiMoGenerator.generate(...)   → RAGResponse
```

---

## Design Decisions

### Why TF-IDF instead of dense embeddings?

- **Zero external dependencies** — no need for a separate embedding API or model download
- **Fast and deterministic** — no GPU required, instant indexing
- **Sufficient for focused document Q&A** — outperforms dense embeddings on keyword-heavy queries
- **Easily swappable** — the `VectorStore` interface can be replaced with a dense embedding store (FAISS, ChromaDB) without changing the rest of the pipeline

### Why OpenAI SDK for MiMo?

MiMo's API follows the OpenAI chat completion standard, so using the `openai` SDK requires only changing `base_url` and `model`. This maximizes compatibility and reduces maintenance burden.
