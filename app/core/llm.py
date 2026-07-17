"""
Wrapper around the Anthropic API for the generation step.
"""

from typing import List, Dict

import anthropic

from app.core.config import settings

_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
provided context. If the answer is not contained in the context, say you don't \
know rather than guessing. Keep answers concise and cite which source each \
piece of information came from when possible."""


def build_context_block(hits: List[Dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[Source {i}: {hit['source']}]\n{hit['text']}")
    return "\n\n".join(blocks)


def generate_answer(question: str, hits: List[Dict], history: List[Dict] = None) -> str:
    context = build_context_block(hits)

    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})

    response = _client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text
