# Contributing to MiMo-RAG

Thank you for your interest in contributing! This document outlines how to get started.

---

## Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/your-username/mimo-rag.git
   cd mimo-rag
   ```
3. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Set up your `.env`**:
   ```bash
   cp .env.example .env
   # Add your MIMO_API_KEY
   ```

---

## Development Guidelines

- Follow **PEP 8** style conventions
- Add **type hints** to all new functions
- Write **docstrings** for all public classes and methods
- Add or update **tests** in `tests/` for any new functionality
- Keep pull requests focused — one feature or fix per PR

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Areas Open for Contribution

| Area | Description |
|------|-------------|
| 🔌 **Embeddings** | Add dense embedding support (FAISS, ChromaDB) |
| 📄 **Loaders** | Add support for HTML, CSV, Excel formats |
| 🌐 **UI** | Improve the Streamlit interface |
| ⚡ **Performance** | Async batch processing, caching |
| 🧪 **Tests** | Improve test coverage |
| 📖 **Docs** | Tutorials, usage examples |

---

## Submitting a Pull Request

1. Make sure all tests pass: `pytest tests/ -v`
2. Describe your changes clearly in the PR description
3. Reference any related issues

---

## Code of Conduct

Be respectful, constructive, and collaborative. All contributors are expected to follow basic open-source etiquette.
