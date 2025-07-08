import os
import json
import time
from typing import List, Dict

import requests
import streamlit as st
from PyPDF2 import PdfReader

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BACKEND_URL = os.getenv("SENTIO_BACKEND_URL", "http://sentio-api-free.westeurope.azurecontainer.io:8000")

st.set_page_config(page_title="Sentio RAG UI", page_icon="📚", layout="centered")

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def read_file_content(file) -> str:
    """Return text content from an uploaded file (supports txt / pdf)."""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    # Fallback treat as text
    return file.read().decode("utf-8", errors="ignore")


def embed_document(doc_id: str, content: str, metadata: Dict) -> Dict:
    """Call backend /embed endpoint."""
    payload = {
        "id": doc_id,
        "content": content,
        "metadata": metadata,
    }
    url = f"{BACKEND_URL}/embed"
    response = requests.post(url, json=payload, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"/embed failed ({response.status_code}): {response.text}")
    return response.json()


def ask_question(question: str, top_k: int = 3, temperature: float = 0.7) -> Dict:
    payload = {
        "question": question,
        "history": [],
        "top_k": top_k,
        "temperature": temperature,
    }
    url = f"{BACKEND_URL}/chat"
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"/chat failed ({response.status_code}): {response.text}")
    return response.json()

# -----------------------------------------------------------------------------
# UI Layout
# -----------------------------------------------------------------------------

tabs = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])

# ------------------ Tab 1: Upload Documents ------------------
with tabs[0]:
    st.header("Upload documents to your knowledge base")
    uploaded_files = st.file_uploader(
        "Choose files (PDF or TXT)", accept_multiple_files=True, type=["pdf", "txt"], key="file_uploader"
    )

    if uploaded_files:
        if st.button("Ingest Documents", key="ingest_btn"):
            progress = st.progress(0.0, text="Starting ingestion...")
            total = len(uploaded_files)
            success, failed = 0, 0
            for idx, file in enumerate(uploaded_files, start=1):
                try:
                    content = read_file_content(file)
                    doc_id = f"{file.name}-{int(time.time())}"
                    metadata = {"file_name": file.name}
                    embed_document(doc_id, content, metadata)
                    success += 1
                except Exception as e:
                    st.error(f"❌ {file.name}: {e}")
                    failed += 1
                progress.progress(idx / total, text=f"Processed {idx}/{total} file(s)...")
            progress.empty()
            st.success(f"✅ Ingestion completed: {success} success, {failed} failed.")

# ------------------ Tab 2: Ask Questions ------------------
with tabs[1]:
    st.header("Ask questions")
    question = st.text_input("Your question", key="question_input")
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Top K", min_value=1, max_value=10, value=3, step=1)
    with col2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

    if st.button("Ask", key="ask_btn") and question:
        with st.spinner("Generating answer..."):
            try:
                response = ask_question(question, top_k, temperature)
                st.markdown(f"### Answer\n{response['answer']}")
                st.markdown("---")
                st.markdown("#### Sources")
                for src in response.get("sources", []):
                    score = src.get("score", 0.0)
                    source_label = src.get("source", "unknown")
                    st.markdown(f"* **{source_label}** (score: {score:.2f})")
            except Exception as e:
                st.error(str(e)) 