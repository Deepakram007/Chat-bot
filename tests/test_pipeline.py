"""
Minimal smoke tests + a tiny "eval set" pattern.

Run with:
    pytest tests/

Expand eval_cases with real questions/expected-keyword pairs from your
own knowledge base once you've ingested real documents -- this is the
piece that lets you claim "measured retrieval/answer quality" on your
resume instead of just "built a chatbot".
"""

from app.core.chunking import chunk_text


def test_chunking_basic():
    text = "a" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=120)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_chunking_empty():
    assert chunk_text("") == []


# --- Example eval scaffold (requires ingested data + API key to run live) ---
eval_cases = [
    # {"question": "What is our refund policy?", "expect_keyword": "30 days"},
]


def test_eval_placeholder():
    """Fill in eval_cases above with real Q/A pairs once you have data."""
    assert isinstance(eval_cases, list)
