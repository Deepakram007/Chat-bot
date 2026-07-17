"""
Quick chat UI for local testing / demo purposes.

Run with:
    streamlit run frontend/streamlit_app.py

(Make sure the FastAPI backend is running first: uvicorn app.main:app --reload)

This is meant as a fast way to demo your backend. Swap it out for the
React app once the pipeline is solid -- that's the stronger resume piece.
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("💬 RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Add knowledge")
    source_name = st.text_input("Source name", value="manual-note")
    text_input = st.text_area("Paste text to ingest")
    if st.button("Ingest text") and text_input.strip():
        resp = requests.post(
            f"{API_URL}/ingest/text",
            json={"source": source_name, "content": text_input},
        )
        st.success(resp.json())

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask something...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    resp = requests.post(
        f"{API_URL}/chat",
        json={"question": question, "history": st.session_state.history[:-1]},
    )
    data = resp.json()
    answer = data.get("answer", "Error getting response.")
    sources = data.get("sources", [])

    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- {s['source']} (distance: {s['score']:.3f})")

    st.session_state.history.append({"role": "assistant", "content": answer})
