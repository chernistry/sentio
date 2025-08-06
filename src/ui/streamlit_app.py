"""Streamlit UI for Sentio RAG System (LangGraph Version)

This module provides a user interface for the Sentio RAG system, allowing users to:
- Upload documents to the knowledge base
- Ask questions to the RAG system
- View evaluation metrics
"""

import logging
import os
import re
import socket
import uuid
from urllib.parse import urlparse
from typing import Dict, Any

import requests
import streamlit as st
from PyPDF2 import PdfReader

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _resolve_backend_url() -> str:
    """Resolve backend URL with localhost fallback."""
    url = os.getenv("SENTIO_BACKEND_URL", "http://localhost:8000")
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    try:
        socket.gethostbyname(host)
    except Exception:
        logger.warning("Backend host %s unreachable, falling back to localhost", host)
        url = "http://localhost:8000"
    return url


BACKEND_URL = _resolve_backend_url()
logger.info("Using backend %s", BACKEND_URL)
MAX_CHUNK_SIZE = 45000  # A bit less than the API limit for safety
MAX_TOKENS_PER_DOC = 8000  # Approximate token limit for embedding API

# Add health check after defining BACKEND_URL
try:
    import requests
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    response.raise_for_status()
    logger.info("Backend health check successful")
except Exception as e:
    logger.error(f"Backend health check failed: {e}")
    # You can add a Streamlit warning here if needed, but for now, just log it

# Page configuration
st.set_page_config(
    page_title="Sentio RAG UI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

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
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # Replace multiple spaces and newlines
    text = re.sub(r"\s+", " ", text)
    # Truncate text if it's too long (approximate token estimate)
    words = text.split()
    if len(words) > MAX_TOKENS_PER_DOC:
        text = " ".join(words[:MAX_TOKENS_PER_DOC]) + "..."
    return text.strip()


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
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


def clear_collection() -> dict:
    """Call backend /clear endpoint to clear vector store collection."""
    url = f"{BACKEND_URL}/clear"
    try:
        response = requests.post(url, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error("Failed to clear collection", exc_info=True)
        st.error(f"Failed to clear collection: {e}")
        return {"status": "error", "message": str(e)}


def embed_document(doc_id: str, content: str, metadata: dict) -> dict:
    """Call backend /embed endpoint to ingest a document."""
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
            try:
                url = f"{BACKEND_URL}/embed"
                response = requests.post(url, json=payload, timeout=300)
                response.raise_for_status()
                results.append(response.json())
            except Exception as e:
                logger.error("Chunk %s failed to embed", i + 1, exc_info=True)
                st.error(f"Failed to embed chunk {i+1}/{len(chunks)}: {e}")
                continue
        return {"status": "success", "chunks": len(chunks), "successful": len(results)}
    # If the document does not require splitting, send as usual
    # Generate UUID for the document
    uuid_id = str(uuid.uuid4())
    payload = {
        "id": uuid_id,
        "content": content,
        "metadata": metadata,
    }
    try:
        url = f"{BACKEND_URL}/embed"
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error("Document ingestion failed", exc_info=True)
        st.error(f"Failed to embed document: {e}")
        return {"status": "error", "message": str(e)}


def ask_question(question: str, top_k: int = 3, temperature: float = 0.7) -> dict:
    """Send a question to the RAG system."""
    payload = {
        "question": question,
        "history": [],
        "top_k": top_k,
        "temperature": temperature,
    }
    try:
        url = f"{BACKEND_URL}/chat"
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to process question: {e}")
        return {"status": "error", "message": str(e)}





# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Sentio RAG")
    st.caption(f"Backend: {BACKEND_URL}")

    # Health check functionality has been moved to separate module
    # To enable health checks, import and call: health_check.render_health_panel(BACKEND_URL)

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
tabs = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])

# ------------------ Tab 1: Upload Documents ------------------
with tabs[0]:
    st.header("Upload documents to your knowledge base")

    # Add a button to clear the collection
    if st.button("🗑️ Clear Knowledge Base", key="clear_btn"):
        with st.spinner("Clearing knowledge base..."):
            result = clear_collection()
            if result.get("status") == "success":
                st.success(f"✅ {result.get('message', 'Knowledge base cleared successfully')}")
            else:
                st.error(f"❌ Failed to clear knowledge base: {result.get('message', 'Unknown error')}")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files (PDF or TXT)", accept_multiple_files=True, type=["pdf", "txt"], key="file_uploader"
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", key="ingest_btn"):
            progress = st.progress(0.0, text="Starting ingestion...")
            total = len(uploaded_files)
            success, failed = 0, 0

            for idx, file in enumerate(uploaded_files, start=1):
                try:
                    with st.spinner(f"Processing {file.name}..."):
                        content = read_file_content(file)
                        # Use filename only for metadata
                        metadata = {"file_name": file.name, "source": file.name}
                        result = embed_document(file.name, content, metadata)

                        if result.get("status") == "success" or "chunks_created" in result:
                            success += 1
                            chunk_info = f" ({result.get('chunks_created', 1)} chunks)" if "chunks_created" in result else ""
                            st.success(f"✅ {file.name}{chunk_info}")
                        else:
                            failed += 1
                            st.error(f"❌ {file.name}: {result.get('message', 'Unknown error')}")
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

    if st.button("🔍 Ask", key="ask_btn") and question:
        with st.spinner("Generating answer..."):
            try:
                response = ask_question(question, top_k, temperature)

                if "answer" in response:
                    st.markdown(f"### Answer\n{response['answer']}")

                    # Display sources if available
                    if response.get("sources"):
                        st.markdown("---")
                        st.markdown("#### Sources")
                        for src in response["sources"]:
                            score = src.get("score", 0.0)
                            source_label = src.get("source", "unknown")
                            st.markdown(f"* **{source_label}** (score: {score:.2f})")
                            with st.expander("Show content"):
                                st.markdown(src.get("text", ""))
                    else:
                        st.info("No sources provided")
                else:
                    st.error(f"Error: {response.get('message', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to get answer: {e}")

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Sentio RAG System | Built with LangGraph"
    "</div>",
    unsafe_allow_html=True
)
