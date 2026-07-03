"""
app.py
------
Streamlit front end for the RAG-Powered AI Study Assistant.
Talks to the FastAPI backend for ingestion and question answering.
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="centered")
st.title("📚 RAG-Powered AI Study Assistant")
st.caption("Ask questions about your course PDFs and notes — answers come with cited sources.")

with st.sidebar:
    st.header("Course Materials")
    st.write("Drop PDFs into `data/course_pdfs/`, then click below to index them.")
    if st.button("🔄 Re-index course materials"):
        with st.spinner("Ingesting and embedding PDFs..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/ingest", timeout=300)
                resp.raise_for_status()
                st.success(f"Indexed {resp.json()['chunks_indexed']} chunks.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about your course material...")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("View sources"):
                for src in turn["sources"]:
                    st.markdown(f"**{src['label']}** — {src['file']}, p.{src['page']}")
                    st.caption(src["excerpt"] + "...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ask", json={"question": question}, timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander("View sources"):
                        for src in data["sources"]:
                            st.markdown(f"**{src['label']}** — {src['file']}, p.{src['page']}")
                            st.caption(src["excerpt"] + "...")
                st.session_state.history.append(
                    {"role": "assistant", "content": data["answer"], "sources": data.get("sources")}
                )
            except Exception as e:
                st.error(f"Error: {e}")
