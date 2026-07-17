from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.vectorstore import query as vector_query
from app.core.llm import generate_answer

router = APIRouter()


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None
    top_k: Optional[int] = None


@router.post("")
def chat(payload: ChatRequest):
    hits = vector_query(payload.question, top_k=payload.top_k)

    if not hits:
        return {
            "answer": "I don't have any relevant documents to answer that yet. "
            "Try ingesting some content first via /ingest.",
            "sources": [],
        }

    history = [m.model_dump() for m in (payload.history or [])]
    answer = generate_answer(payload.question, hits, history=history)

    return {
        "answer": answer,
        "sources": [{"source": h["source"], "score": h["score"]} for h in hits],
    }
