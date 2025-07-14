import os
import json
import time
import re
from typing import List, Dict
import uuid

import requests
import streamlit as st
from PyPDF2 import PdfReader

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BACKEND_URL = os.getenv("SENTIO_BACKEND_URL", "http://localhost:8000")
MAX_CHUNK_SIZE = 45000  # A bit less than the API limit for safety
MAX_TOKENS_PER_DOC = 8000  # Approximate token limit for Jina API

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


def clean_text(text: str) -> str:
    """Cleans text from invalid characters and normalizes it."""
    # Remove invisible control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Replace multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Truncate text if it's too long (approximate token estimate)
    words = text.split()
    if len(words) > MAX_TOKENS_PER_DOC:
        text = ' '.join(words[:MAX_TOKENS_PER_DOC]) + '...'
    return text.strip()


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """Split text into chunks smaller than max_size."""
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    # Split by paragraphs for more natural boundaries
    paragraphs = text.split("\n\n")
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If the paragraph itself is too large, split it by sentences
        if len(paragraph) > max_size:
            sentences = paragraph.replace(". ", ".\n").split("\n")
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 2 <= max_size:
                    if current_chunk:
                        current_chunk += "\n\n" if sentence else ""
                    current_chunk += sentence
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence
        # Otherwise, add the whole paragraph if it fits
        elif len(current_chunk) + len(paragraph) + 2 <= max_size:
            if current_chunk:
                current_chunk += "\n\n" if paragraph else ""
            current_chunk += paragraph
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def clear_collection() -> Dict:
    """Call backend /clear endpoint to clear Qdrant collection."""
    url = f"{BACKEND_URL}/clear"
    response = requests.post(url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"/clear failed ({response.status_code}): {response.text}")
    return response.json()


def embed_document(doc_id: str, content: str, metadata: Dict) -> Dict:
    """Call backend /embed endpoint."""
    # Clean the text before processing
    content = clean_text(content)
    
    # Split large documents into chunks
    chunks = chunk_text(content)
    
    if len(chunks) > 1:
        # If the document was split, add chunk number to ID and metadata
        results = []
        for i, chunk in enumerate(chunks):
            # Generate UUID for each chunk
            chunk_id = str(uuid.uuid4())
            chunk_metadata = metadata.copy()
            chunk_metadata["part"] = i + 1
            chunk_metadata["total_parts"] = len(chunks)
            chunk_metadata["original_filename"] = doc_id
            
            payload = {
                "id": chunk_id,
                "content": chunk,
                "metadata": chunk_metadata,
            }
            url = f"{BACKEND_URL}/embed"
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code != 200:
                raise RuntimeError(f"/embed failed ({response.status_code}): {response.text}")
            results.append(response.json())
        return results[0]  # Return the result of the first part for compatibility
    else:
        # If the document does not require splitting, send as usual
        # Generate UUID for the document
        uuid_id = str(uuid.uuid4())
        payload = {
            "id": uuid_id,
            "content": content,
            "metadata": metadata,
        }
        url = f"{BACKEND_URL}/embed"
        response = requests.post(url, json=payload, timeout=300)
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
    
    # File uploader 
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
                    # Use filename only for metadata
                    metadata = {"file_name": file.name, "source": file.name}
                    embed_document(file.name, content, metadata)
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
        top_k = st.slider("Top K", min_value=1, max_value=20, value=3, step=1)
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