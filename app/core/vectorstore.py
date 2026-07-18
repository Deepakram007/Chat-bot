"""
Thin wrapper around a persistent Chroma collection.

Kept deliberately simple (no LangChain) so you can see exactly what's
happening at each step -- swap in LangChain/LlamaIndex later once you
understand this.
"""

import datetime
import uuid
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings

_embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
_collection = _client.get_or_create_collection(name=settings.COLLECTION_NAME)


def embed(texts: List[str]) -> List[List[float]]:
    return _embedder.encode(texts, convert_to_numpy=True).tolist()


def add_documents(chunks: List[str], source: str) -> int:
    """Embed and store a list of text chunks. Returns number of chunks added."""
    if not chunks:
        return 0

    # Generate a timestamp and append it to the source name to keep repeated uploads unique
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_source = f"{source}_{timestamp_str}"

    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = embed(chunks)
    metadatas = [
        {"source": unique_source, "chunk_index": i, "timestamp": timestamp_str}
        for i in range(len(chunks))
    ]

    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def query(question: str, top_k: int = None) -> List[Dict]:
    """Return the top_k most relevant chunks for a question."""
    top_k = top_k or settings.TOP_K
    query_embedding = embed([question])[0]

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    hits = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "source": meta.get("source"), "score": distance})
    return hits
