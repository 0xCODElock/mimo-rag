# 📄 MiMo-RAG — Retrieval-Augmented Generation with MiMo API

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MiMo API](https://img.shields.io/badge/Powered%20by-MiMo%20V2.5-orange)](https://100t.xiaomimimo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![Token Usage](https://img.shields.io/badge/Est.%20Token%20Usage-50M%2B%2Fmonth-red)](docs/token_usage.md)

A production-ready **Retrieval-Augmented Generation (RAG)** system designed for **high-throughput, multi-user document intelligence** powered by the **MiMo API**. Built to handle enterprise-scale document collections with sustained, heavy API usage.

---

## ✨ Features

- 📂 **Multi-format document ingestion** — PDF, DOCX, and Markdown support out of the box
- ✂️ **Intelligent chunking** — Recursive character-based splitting with configurable overlap
- 🔍 **Semantic retrieval** — TF-IDF + cosine similarity for fast, accurate context lookup
- 🤖 **MiMo-powered answers** — Full context + question sent to MiMo API per query
- 📦 **Batch summarization** — Summarize hundreds of documents in parallel, generating one API call per document
- 🧠 **Multi-turn conversation memory** — Maintains full conversation history per session (grows token usage per turn)
- 🌐 **Streamlit web UI** — Multi-user capable, each user session = independent API call stream
- ⚡ **REST API** — FastAPI endpoints designed for concurrent request handling
- 📊 **Token usage estimator** — Built-in utility to forecast monthly token consumption

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
│  [3] RETRIEVAL & GENERATION  (per user, per query)       │
│           ▼                                              │
│      Query ──► Retriever ──► top-k chunks                │
│                                   │                      │
│                                   ▼                      │
│                     ConversationMemory (full history)    │
│                                   │                      │
│                                   ▼                      │
│                          MiMo API Call                   │
│                    [system + history + context + query]  │
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

### 4. Batch Summarize Documents

```bash
python -m src.batch_processor --input ./docs/ --output ./summaries/
```

---

## 💻 Usage

### Single Query

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(api_key="your_mimo_api_key")
pipeline.ingest("docs/research_paper.pdf")

response = pipeline.query("What are the key findings?")
print(response.answer)
```

### Multi-Turn Conversation (High Token Usage)

```python
from src.conversation import ConversationSession

session = ConversationSession(api_key="your_mimo_api_key")
session.load_documents(["report_q1.pdf", "report_q2.pdf", "report_q3.pdf"])

# Each turn appends full history → token count grows per message
print(session.chat("Summarize Q1 performance"))
print(session.chat("How does that compare to Q2?"))   # +prev turn tokens
print(session.chat("What caused the Q3 drop?"))       # +prev 2 turns tokens
print(f"Total tokens used: {session.total_tokens_used}")
```

### Batch Document Summarization

```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor(api_key="your_mimo_api_key")

# 1 MiMo API call per document
results = processor.summarize_all("./document_library/")  # e.g. 200 docs
print(f"Processed {len(results)} documents")
print(f"Total tokens consumed: {processor.total_tokens}")
```

---

## 📊 Estimated Token Usage

This project is designed for **production-scale deployment** with continuous, high-volume API consumption. Below is the projected monthly token usage based on realistic usage scenarios.

> Full breakdown → [`docs/token_usage.md`](docs/token_usage.md)

### Scenario A — Small Team (10 users)

| Activity | Volume | Tokens/Call | Monthly Total |
|---|---|---|---|
| RAG queries (multi-turn, avg 5 turns/session) | 50 sessions/day | ~3,500 | **5.25M** |
| Batch doc summarization | 100 docs/week | ~1,200 | **480K** |
| New document ingestion + reindex | 20 docs/day | ~800 | **480K** |
| **Monthly Total** | | | **~6.2M tokens** |

### Scenario B — Production (100 users)

| Activity | Volume | Tokens/Call | Monthly Total |
|---|---|---|---|
| RAG queries (multi-turn, avg 8 turns/session) | 500 sessions/day | ~6,000 | **90M** |
| Automated batch summarization pipeline | 1,000 docs/week | ~1,200 | **4.8M** |
| Background re-indexing & embeddings | continuous | ~500 | **3M** |
| **Monthly Total** | | | **~98M tokens** |

### Why Token Usage Is High

1. **Multi-turn memory** — every conversation appends the full history to each new API call. A 10-turn session sends up to 10× the tokens of a single query.
2. **Batch summarization** — each document triggers an independent, full-context API call
3. **Concurrent users** — multiple sessions run simultaneously, each with their own token stream
4. **1M context window (MiMo-V2.5)** — enables loading entire documents at once for deep analysis tasks, maximizing tokens per call

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
| `MAX_HISTORY_TURNS` | `20` | Conversation turns kept in memory |
| `BATCH_CONCURRENCY` | `5` | Parallel API calls in batch mode |

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
