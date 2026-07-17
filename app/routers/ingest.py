"""
Endpoints for loading documents into the vector store.

Supports plain text upload for now. Extend this with PDF/docx loaders
(e.g. pypdf, python-docx) as your knowledge base grows.
"""

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from app.core.chunking import chunk_text
from app.core.vectorstore import add_documents

router = APIRouter()


class TextIngestRequest(BaseModel):
    source: str
    content: str


@router.post("/text")
def ingest_text(payload: TextIngestRequest):
    """Ingest raw text directly (useful for quick testing)."""
    chunks = chunk_text(payload.content)
    added = add_documents(chunks, source=payload.source)
    return {"source": payload.source, "chunks_added": added}


@router.post("/file")
async def ingest_file(file: UploadFile = File(...), source: str = Form(None)):
    """Ingest a plain .txt file. Add PDF/docx parsing here as needed."""
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")

    chunks = chunk_text(text)
    added = add_documents(chunks, source=source or file.filename)
    return {"source": source or file.filename, "chunks_added": added}
