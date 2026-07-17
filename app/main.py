"""
RAG Chatbot - FastAPI entrypoint

Run with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, ingest

app = FastAPI(
    title="RAG Chatbot API",
    description="A retrieval-augmented chatbot backed by Chroma + Claude API",
    version="0.1.0",
)

# Allow the frontend (React/Streamlit) to call this API during development.
# Lock this down to your actual frontend origin before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG chatbot API is running"}
