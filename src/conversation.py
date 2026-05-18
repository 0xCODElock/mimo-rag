"""Multi-turn conversation session with persistent memory.

Token profile:
    Each turn appends the full conversation history to the next API call.
    Token usage grows linearly with the number of turns:

        Turn N cost ≈ sum(tokens for turns 1..N-1) + context_tokens + query_tokens

    A 10-turn session on a medium-sized document set can consume
    60,000–100,000 tokens total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI

from src.pipeline import RAGPipeline
from src.generator import RAGResponse


@dataclass
class Turn:
    """A single conversation turn."""
    role: str        # 'user' or 'assistant'
    content: str
    tokens: int = 0


class ConversationSession:
    """
    Stateful multi-turn conversation over indexed documents.

    Unlike a single-shot RAG query, ConversationSession preserves the full
    message history and injects it into every subsequent API call. This means
    token usage **compounds with each turn** — making it one of the most
    token-intensive usage patterns in this system.

    Example token growth over 10 turns (medium document set):
        Turn  1:  ~2,000 tokens
        Turn  3:  ~6,500 tokens
        Turn  5:  ~12,000 tokens
        Turn 10:  ~28,000 tokens
        ─────────────────────────
        Session total: ~80,000+ tokens
    """

    SYSTEM_PROMPT = (
        "You are a knowledgeable assistant that answers questions based on "
        "provided document context. Always maintain continuity with the "
        "conversation history. Reference previous answers when relevant."
    )

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.xiaomimimo.com/v1",
        model: str = "MiMo-V2.5",
        max_tokens: int = 1024,
        max_history_turns: int = 20,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_history_turns = max_history_turns
        self.client = OpenAI(api_key=api_key, base_url=api_base)

        self.pipeline = RAGPipeline(api_key=api_key)
        self.history: List[Turn] = []
        self.total_tokens_used: int = 0

    def load_documents(self, paths: List[str]) -> None:
        """Ingest documents into the session's RAG index."""
        for path in paths:
            self.pipeline.ingest(path)
        print(f"[Session] Loaded {len(paths)} documents, {self.pipeline.chunk_count} chunks indexed.")

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response, maintaining full conversation history.

        Token cost increases each turn as history accumulates.

        Args:
            user_message: The user's current message.

        Returns:
            Assistant's response string.
        """
        # Retrieve relevant context from documents
        chunks = self.pipeline.retriever.retrieve(user_message)
        context_text = "\n\n".join(
            f"[Doc excerpt {i+1}]\n{c.text}" for i, c in enumerate(chunks)
        )

        # Build full message list: system + history + context + new query
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Inject conversation history (this is what makes tokens compound)
        for turn in self.history[-self.max_history_turns:]:
            messages.append({"role": turn.role, "content": turn.content})

        # Add current context + question
        user_content = (
            f"Relevant document context:\n{context_text}\n\n"
            f"Question: {user_message}"
        ) if context_text else user_message

        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        self.total_tokens_used += tokens

        # Save to history
        self.history.append(Turn(role="user", content=user_message, tokens=0))
        self.history.append(Turn(role="assistant", content=answer, tokens=tokens))

        return answer

    def reset(self) -> None:
        """Clear conversation history (start a new session)."""
        self.history.clear()
        self.total_tokens_used = 0

    @property
    def turn_count(self) -> int:
        return len(self.history) // 2
