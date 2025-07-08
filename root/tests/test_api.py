import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the parent directory to sys.path to import the app module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_search_endpoint():
    """Test the search endpoint with a simple query."""
    response = client.get("/search?query=test&limit=5")
    assert response.status_code == 200
    assert "results" in response.json()
    
def test_chat_endpoint():
    """Test the chat endpoint with a simple query."""
    payload = {
        "query": "What is this system about?",
        "conversation_id": "test-conversation",
        "stream": False
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert "answer" in response.json()
    
def test_ingest_endpoint():
    """Test the document ingestion endpoint."""
    payload = {
        "document_id": "test-document",
        "content": "This is a test document content.",
        "metadata": {
            "title": "Test Document",
            "author": "Test Author"
        }
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 202  # Accepted for async processing 