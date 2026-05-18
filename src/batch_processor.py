"""Batch document summarization — one MiMo API call per document."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from src.document_loader import DocumentLoader
from src.chunker import TextChunker


@dataclass
class SummaryResult:
    """Result of summarizing a single document."""
    source: str
    summary: str
    tokens_used: int
    processing_time_s: float
    success: bool
    error: Optional[str] = None


class BatchProcessor:
    """
    Processes an entire directory of documents, generating one MiMo API
    call per document. Designed for high-throughput, token-intensive workloads.

    Token profile:
        - Each document generates an independent API call with full context.
        - For 200 documents averaging 1,200 tokens/call: ~240,000 tokens/run.
        - Weekly automated runs = millions of tokens per month.
    """

    SUMMARY_PROMPT = (
        "You are an expert document analyst. Provide a comprehensive, structured "
        "summary of the following document. Include: main topics, key findings or "
        "conclusions, important data points, and any action items or recommendations.\n\n"
        "Document:\n{content}\n\nSummary:"
    )

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.xiaomimimo.com/v1",
        model: str = "MiMo-V2.5",
        max_tokens: int = 1500,
        concurrency: int = 5,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=2000, chunk_overlap=100)

        self._results: List[SummaryResult] = []
        self.total_tokens: int = 0

    def summarize_all(
        self,
        directory: str,
        output_dir: Optional[str] = None,
    ) -> List[SummaryResult]:
        """
        Summarize every supported document in a directory.

        Args:
            directory:  Path to folder containing documents.
            output_dir: If provided, save summaries as .txt files here.

        Returns:
            List of SummaryResult objects.
        """
        doc_paths = self._discover_documents(directory)
        print(f"[BatchProcessor] Found {len(doc_paths)} documents to summarize.")

        self._results = []
        self.total_tokens = 0

        for i, path in enumerate(doc_paths, 1):
            print(f"[{i}/{len(doc_paths)}] Summarizing: {Path(path).name}")
            result = self._summarize_one(path)
            self._results.append(result)
            self.total_tokens += result.tokens_used

            if output_dir and result.success:
                self._save_summary(result, output_dir)

            # Brief pause to respect rate limits
            time.sleep(0.3)

        self._print_report()
        return self._results

    def summarize_file(self, path: str) -> SummaryResult:
        """Summarize a single document file."""
        return self._summarize_one(path)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _summarize_one(self, path: str) -> SummaryResult:
        start = time.time()
        try:
            doc = self.loader.load(path)
            # Use full content (up to model context limit)
            content = doc.content[:50000]  # ~40K tokens — well within MiMo-V2.5's 1M window
            prompt = self.SUMMARY_PROMPT.format(content=content)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
            )

            summary = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0

            return SummaryResult(
                source=str(path),
                summary=summary,
                tokens_used=tokens,
                processing_time_s=round(time.time() - start, 2),
                success=True,
            )
        except Exception as e:
            return SummaryResult(
                source=str(path),
                summary="",
                tokens_used=0,
                processing_time_s=round(time.time() - start, 2),
                success=False,
                error=str(e),
            )

    def _discover_documents(self, directory: str) -> List[str]:
        extensions = DocumentLoader.SUPPORTED_EXTENSIONS
        paths = []
        for ext in extensions:
            paths.extend(Path(directory).rglob(f"*{ext}"))
        return sorted(str(p) for p in paths)

    def _save_summary(self, result: SummaryResult, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(result.source).stem
        (out / f"{stem}_summary.txt").write_text(result.summary, encoding="utf-8")

    def _print_report(self) -> None:
        successful = sum(1 for r in self._results if r.success)
        failed = len(self._results) - successful
        avg_time = (
            sum(r.processing_time_s for r in self._results) / len(self._results)
            if self._results else 0
        )
        print("\n" + "=" * 50)
        print(f"  Batch Summary Report")
        print("=" * 50)
        print(f"  Documents processed : {len(self._results)}")
        print(f"  Successful          : {successful}")
        print(f"  Failed              : {failed}")
        print(f"  Total tokens used   : {self.total_tokens:,}")
        print(f"  Avg time/doc        : {avg_time:.1f}s")
        print("=" * 50)
