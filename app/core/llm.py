"""
Wrapper around the NVIDIA NIM API (OpenAI-compatible) for the generation step.

Quota-saving/cost-saving optimisations applied:
  1. Context chunks trimmed to MAX_CHARS_PER_CHUNK (avoids giant context windows)
  2. Only the last HISTORY_TURNS pairs of messages are sent (not full history)
  3. max_tokens capped to avoid runaway long answers
  4. Graceful error handling so the server never crashes on quota exhaustion
"""

from typing import List, Dict

import openai
from openai import OpenAI

from app.core.config import settings

def get_client():
    # Use a dummy key if empty to prevent library validation errors on startup
    api_key = settings.NVIDIA_API_KEY or "PLACEHOLDER"
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

# ── Quota-saving knobs ────────────────────────────────────────────────────────
# How many characters to keep from each retrieved chunk.
# 400 chars ≈ 100 tokens. 4 chunks × 100 tokens = 400 tokens saved vs default.
MAX_CHARS_PER_CHUNK: int = 400

# Only send this many recent user+assistant turn pairs to the model.
# Each extra history turn costs tokens every request.
HISTORY_TURNS: int = 3

# Cap the answer length. Prevents accidental long responses burning extra tokens.
MAX_OUTPUT_TOKENS: int = 512
# ─────────────────────────────────────────────────────────────────────────────

# Kept deliberately short — system prompt is sent on EVERY request.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "If the answer isn't in the context, say you don't know. Be concise."
)


def build_context_block(hits: List[Dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        # Trim each chunk to MAX_CHARS_PER_CHUNK to control token cost
        snippet = hit["text"][:MAX_CHARS_PER_CHUNK]
        if len(hit["text"]) > MAX_CHARS_PER_CHUNK:
            snippet += "…"
        blocks.append(f"[Source {i}: {hit['source']}]\n{snippet}")
    return "\n\n".join(blocks)


def generate_answer(question: str, hits: List[Dict], history: List[Dict] = None) -> str:
    context = build_context_block(hits)

    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    # Keep only the last HISTORY_TURNS pairs (user + assistant = 2 messages per turn)
    trimmed_history = (history or [])[-HISTORY_TURNS * 2:]

    # Format the chat messages for OpenAI/NVIDIA API
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in trimmed_history:
        role = "assistant" if msg["role"] == "assistant" else "user"
        messages.append({"role": role, "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    if not settings.NVIDIA_API_KEY:
        return (
            "⚠️ NVIDIA API key is invalid or not configured. "
            "Please check the NVIDIA_API_KEY in your .env file."
        )

    try:
        _client = get_client()
        response = _client.chat.completions.create(
            model=settings.NVIDIA_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        return (
            "⚠️ NVIDIA API key is invalid or not configured. "
            "Please check the NVIDIA_API_KEY in your .env file."
        )
    except openai.RateLimitError:
        return (
            "⚠️ The NVIDIA AI model is currently rate-limited (free credits exhausted or rate limit hit). "
            "Please wait a few minutes and try again."
        )
    except openai.OpenAIError as e:
        return f"⚠️ An error occurred with the AI provider: {str(e)}"
