"""
Example usage of the MiMo-RAG pipeline.

Run:
    python examples/example_usage.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import RAGPipeline


def main():
    api_key = os.getenv("MIMO_API_KEY")
    if not api_key:
        print("ERROR: Set MIMO_API_KEY in your .env file")
        return

    print("🚀 Initializing MiMo-RAG pipeline...")
    pipeline = RAGPipeline(api_key=api_key)

    # ── Ingest documents ──────────────────────────────────────────────────
    # pipeline.ingest("path/to/your_document.pdf")
    # pipeline.ingest("path/to/your_notes.md")

    # For demo: create a tiny in-memory document
    import tempfile, os
    sample = """
    # Introduction to Retrieval-Augmented Generation

    Retrieval-Augmented Generation (RAG) is a framework that enhances large language
    models by retrieving relevant information from external knowledge bases before
    generating responses.

    ## Key Benefits
    - Reduces hallucination by grounding answers in real documents
    - Enables knowledge updates without retraining the model
    - Provides source citations for transparency

    ## How It Works
    1. User submits a query
    2. System retrieves top-k relevant document chunks
    3. Retrieved chunks are prepended to the prompt
    4. LLM generates a grounded, cited answer
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(sample)
        tmp_path = f.name

    try:
        pipeline.ingest(tmp_path)

        # ── Query ─────────────────────────────────────────────────────────
        questions = [
            "What is RAG and what are its key benefits?",
            "How does the retrieval step work?",
        ]

        for question in questions:
            print(f"\n❓ {question}")
            response = pipeline.query(question)
            print(f"💬 {response.answer}")
            print(f"📎 Sources: {response.sources}")
            print(f"🔢 Chunks used: {response.chunks_used}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
