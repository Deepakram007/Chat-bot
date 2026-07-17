# RAG Chatbot

A retrieval-augmented chatbot: it answers questions grounded in a set of
documents you provide, rather than relying purely on the LLM's training
data. Built with FastAPI, Chroma (vector store), sentence-transformers
(embeddings), and the Anthropic API (generation).

## Architecture

```
User question
     │
     ▼
Embed question (sentence-transformers)
     │
     ▼
Vector search over Chroma  ──►  top-k relevant chunks
     │
     ▼
Build prompt: question + retrieved context
     │
     ▼
Claude API generates grounded answer
     │
     ▼
Response + source citations
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # then add your ANTHROPIC_API_KEY
```

## Run the backend

```bash
uvicorn app.main:app --reload
```

API docs (auto-generated): http://localhost:8000/docs

## Run the demo frontend (optional, quick testing)

```bash
streamlit run frontend/streamlit_app.py
```

## Usage

**1. Ingest some content**

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"source": "policy.txt", "content": "Refunds are accepted within 30 days of purchase."}'
```

Or upload a `.txt` file:

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@your_doc.txt" -F "source=your_doc.txt"
```

**2. Ask a question**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

## Project structure

```
app/
  main.py              FastAPI app + router registration
  core/
    config.py          Settings, loaded from .env
    chunking.py         Text splitting
    vectorstore.py       Chroma embed/store/query logic
    llm.py              Claude API call + prompt construction
  routers/
    ingest.py           /ingest endpoints
    chat.py             /chat endpoint
frontend/
  streamlit_app.py      Quick demo UI
tests/
  test_pipeline.py      Chunking tests + eval scaffold
```

## Next steps / ways to extend this

- **PDF/docx ingestion** — add `pypdf` / `python-docx` loaders in `ingest.py`
- **Reranking** — add a cross-encoder rerank step after initial retrieval for
  better precision
- **Streaming responses** — use Claude's streaming API + FastAPI
  `StreamingResponse` for a more production-feel UX
- **Evaluation** — fill in `eval_cases` in `tests/test_pipeline.py` with real
  question/answer pairs from your knowledge base to measure retrieval and
  answer quality, not just "it seems to work"
- **Auth + rate limiting** — needed before any real deployment
- **React frontend** — replace the Streamlit demo for something more
  presentable in a portfolio
- **Deploy** — Render/Fly.io/Railway (backend) + Vercel (frontend)

## Notes

- Embeddings run locally via `sentence-transformers` (`all-MiniLM-L6-v2`),
  so ingestion doesn't cost API calls. Only the generation step
  (`app/core/llm.py`) hits the Claude API.
- Chroma persists to disk at `data/chroma_db` by default, so your knowledge
  base survives restarts.
