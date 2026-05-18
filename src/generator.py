"""MiMo API client and answer generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from src.chunker import Chunk


SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions based on the provided context.
If the context does not contain enough information to answer the question, say so honestly.
Always cite which part of the documents you used."""

CONTEXT_TEMPLATE = """Use the following document excerpts to answer the question.

{context}

---
Question: {question}
Answer:"""


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    answer: str
    sources: List[str]
    chunks_used: int
    model: str


class MiMoGenerator:
    """
    Generates answers by sending retrieved context to the MiMo API.

    The MiMo API follows the OpenAI-compatible chat completion interface,
    so we use the ``openai`` SDK with a custom ``base_url``.

    Supported models (as of 2026):
        - MiMo-V2.5        (310B MoE, multimodal, 1M context — recommended)
        - MiMo-V2.5-Pro    (proprietary, higher capability)
        - MiMo-V2-Flash    (309B, fast inference)
        - MiMo-V2-Pro      (1T params, proprietary)
        - MiMo-7B-RL       (open-weight, lightweight)
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.xiaomimimo.com/v1",
        model: str = "MiMo-V2.5",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def generate(self, question: str, chunks: List[Chunk]) -> RAGResponse:
        """
        Generate a grounded answer using retrieved chunks.

        Args:
            question: The user's question.
            chunks:   Retrieved context chunks.

        Returns:
            RAGResponse with the answer, sources, and metadata.
        """
        context = self._build_context(chunks)
        prompt = CONTEXT_TEMPLATE.format(context=context, question=question)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        answer = response.choices[0].message.content.strip()
        sources = list({c.source for c in chunks})

        return RAGResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            model=self.model,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_context(self, chunks: List[Chunk]) -> str:
        """Format chunks into a readable context block."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Excerpt {i} — {chunk.source}]\n{chunk.text}"
            )
        return "\n\n".join(parts)
