"""Document loader — supports PDF, DOCX, and Markdown."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    """Represents a loaded document with its text and metadata."""
    content: str
    source: str
    doc_type: str
    page_count: int = 1


class DocumentLoader:
    """
    Multi-format document loader.

    Supported formats:
        - PDF  (.pdf)
        - Word (.docx)
        - Markdown (.md, .markdown)
        - Plain text (.txt)
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}

    def load(self, path: str | Path) -> Document:
        """
        Load a document from the given file path.

        Args:
            path: Path to the document file.

        Returns:
            Document object containing extracted text and metadata.

        Raises:
            ValueError: If the file format is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        loaders = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".doc": self._load_docx,
            ".md": self._load_text,
            ".markdown": self._load_text,
            ".txt": self._load_text,
        }

        return loaders[ext](path)

    def load_many(self, paths: List[str | Path]) -> List[Document]:
        """Load multiple documents at once."""
        docs = []
        for p in paths:
            try:
                docs.append(self.load(p))
            except Exception as e:
                print(f"[WARNING] Could not load {p}: {e}")
        return docs

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> Document:
        """Extract text from a PDF file using pypdf."""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        content = "\n\n".join(pages)
        return Document(
            content=content,
            source=str(path),
            doc_type="pdf",
            page_count=len(reader.pages),
        )

    def _load_docx(self, path: Path) -> Document:
        """Extract text from a Word document using python-docx."""
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        content = "\n\n".join(paragraphs)
        return Document(
            content=content,
            source=str(path),
            doc_type="docx",
            page_count=1,
        )

    def _load_text(self, path: Path) -> Document:
        """Load a plain text or Markdown file."""
        content = path.read_text(encoding="utf-8", errors="replace")
        return Document(
            content=content,
            source=str(path),
            doc_type=path.suffix.lstrip("."),
            page_count=1,
        )
