# 📄 MiMo-RAG — Retrieval-Augmented Generation with MiMo API

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MiMo API](https://img.shields.io/badge/Powered%20by-MiMo%20V2.5-orange)](https://100t.xiaomimimo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you chat with your own documents (PDF, Word, Markdown) using the **MiMo API** as the language model backbone.

---

## ✨ Features

- 📂 **Multi-format document ingestion** — PDF, DOCX, and Markdown support out of the box
- ✂️ **Intelligent chunking** — Recursive character-based splitting with configurable overlap
- 🔍 **Semantic retrieval** — TF-IDF + cosine similarity for fast, accurate context lookup
- 🤖 **MiMo-powered answers** — Sends retrieved context + question to MiMo API for grounded responses
- 🌐 **Streamlit web UI** — Clean chat interface, drag-and-drop upload, source citation
- ⚡ **REST API** — FastAPI endpoints for programmatic access
- 💾 **Vector store persistence** — Save and reload your indexed documents

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MiMo-RAG Pipeline                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [1] DOCUMENT INGESTION                                  │
│      PDF / DOCX / Markdown                               │
│           │                                              │
│           ▼                                              │
│      DocumentLoader ──► TextChunker                      │
│      (extract text)     (split into ~500-token chunks)   │
│           │                                              │
│  [2] INDEXING                                            │
│           ▼                                              │
│      Embedder ──────────► VectorStore                    │
│      (TF-IDF vectors)     (persisted index)              │
│           │                                              │
│  [3] RETRIEVAL & GENERATION                              │
│           ▼                                              │
│      Query ──► Retriever ──► top-k chunks                │
│                                   │                      │
│                                   ▼                      │
│                           PromptBuilder                   │
│                                   │                      │
│                                   ▼                      │
│                          MiMo API (LLM)                  │
│                                   │                      │
│                                   ▼                      │
│                            Final Answer                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/0xCODElock/mimo-rag.git
cd mimo-rag
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and set your MiMo API key
```

```env
MIMO_API_KEY=your_api_key_here
MIMO_API_BASE=https://api.xiaomimimo.com/v1
MIMO_MODEL=MiMo-V2.5
```

### 3. Run the Web UI

```bash
streamlit run app.py
```

### 4. Or Use the REST API

```bash
uvicorn api:app --reload
```

---

## 💻 Usage

### Python SDK

```python
from src.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(api_key="your_mimo_api_key")

# Ingest documents
pipeline.ingest("docs/research_paper.pdf")
pipeline.ingest("docs/manual.docx")

# Ask questions
response = pipeline.query("What are the key findings?")
print(response.answer)
print(response.sources)   # Which chunks were used
```

### REST API

```bash
# Upload a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the main points"}'
```

---

## 📁 Project Structure

```
mimo-rag/
├── src/
│   ├── document_loader.py   # PDF, DOCX, MD ingestion
│   ├── chunker.py           # Recursive text splitting
│   ├── embedder.py          # TF-IDF vectorization
│   ├── retriever.py         # Cosine similarity search
│   ├── generator.py         # MiMo API client & prompt builder
│   └── pipeline.py          # End-to-end orchestration
├── app.py                   # Streamlit web interface
├── api.py                   # FastAPI REST endpoints
├── config.py                # Configuration management
├── examples/
│   └── example_usage.py     # Usage examples
├── docs/
│   └── architecture.md      # Detailed architecture notes
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | `500` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `4` | Number of chunks to retrieve |
| `MIMO_MODEL` | `MiMo-V2.5` | MiMo model to use |
| `MAX_TOKENS` | `1024` | Max response tokens |
| `TEMPERATURE` | `0.7` | Response creativity |

### Supported Models (2026)

| Model | Params | Context | Notes |
|-------|--------|---------|-------|
| `MiMo-V2.5` | 310B MoE (15B active) | 1M tokens | Latest open-source, **recommended** |
| `MiMo-V2.5-Pro` | ~1.02T | 256K tokens | Proprietary, highest capability |
| `MiMo-V2-Flash` | 309B | 256K tokens | Fast inference |
| `MiMo-V2-Pro` | 1T | 256K tokens | Proprietary |
| `MiMo-7B-RL` | 7B | 32K tokens | Lightweight, open-weight |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📜 License

MIT © 2026 [0xCODElock](https://github.com/0xCODElock)
