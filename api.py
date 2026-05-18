"""FastAPI REST interface for MiMo-RAG."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import RAGPipeline

app = FastAPI(
    title="MiMo-RAG API",
    description="Retrieval-Augmented Generation powered by MiMo API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline (initialized on first use)
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        api_key = os.getenv("MIMO_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="MIMO_API_KEY is not configured")
        _pipeline = RAGPipeline(api_key=api_key)
    return _pipeline


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    model: str


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    total_chunks: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "documents": pipeline.document_count,
        "chunks": pipeline.chunk_count,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF, DOCX, MD, TXT)."""
    pipeline = get_pipeline()
    suffix = Path(file.filename).suffix

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        n = pipeline.ingest(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        filename=file.filename,
        chunks_indexed=n,
        total_chunks=pipeline.chunk_count,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question against the indexed documents."""
    pipeline = get_pipeline()
    if pipeline.chunk_count == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")
    try:
        response = pipeline.query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        chunks_used=response.chunks_used,
        model=response.model,
    )
