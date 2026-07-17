"""
Central config. Loads secrets from environment variables (.env file locally).
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    # Where Chroma persists its vector store on disk
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "knowledge_base")

    # Embedding model (runs locally via sentence-transformers, no API cost)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "4"))


settings = Settings()

if not settings.ANTHROPIC_API_KEY:
    print(
        "[WARNING] ANTHROPIC_API_KEY is not set. "
        "Create a .env file (see .env.example) before calling /chat."
    )
