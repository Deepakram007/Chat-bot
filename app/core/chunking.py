"""
Simple fixed-size chunking with overlap.

Good enough to start. If retrieval quality is poor later, this is the
first thing worth revisiting -- try semantic or sentence-aware chunking.
"""

from typing import List

from app.core.config import settings


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
