"""
Tests for API endpoints.

This module contains tests for the primary API endpoints such as
/chat, /embed, and /health. It relies on the `api_client` fixture
from `conftest.py` to provide a mocked application instance.
"""
import os
import sys
import asyncio

import httpx
import pytest

# Add project root to path to allow importing from `root`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ==== API Endpoint Tests ====

pytestmark_api = pytest.mark.api

@pytest.mark.asyncio
@pytestmark_api
async def test_health_check(api_client: httpx.AsyncClient):
    """
    Tests that the /health endpoint is reachable and returns a healthy status.
    """
    response = await api_client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "healthy"
    assert "qdrant" in json_response["services"]


@pytest.mark.asyncio
@pytestmark_api
async def test_embed_endpoint_valid(api_client: httpx.AsyncClient):
    """
    Tests the /embed endpoint with a valid payload.
    """
    payload = {"content": "This is a test document for embedding."}
    response = await api_client.post("/embed", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert "document_id" in json_response
    assert json_response["chunks_upserted"] > 0


@pytest.mark.asyncio
@pytestmark_api
async def test_embed_endpoint_invalid(api_client: httpx.AsyncClient):
    """
    Tests the /embed endpoint with an invalid payload (e.g., empty content).
    """
    payload = {"content": ""}  # Invalid content
    response = await api_client.post("/embed", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.asyncio
@pytestmark_api
async def test_chat_endpoint_valid(api_client: httpx.AsyncClient):
    """
    Tests the /chat endpoint with a valid question.
    Asserts the response structure is correct.
    """
    # First, embed a document to ensure the RAG pipeline has context
    # Use a document that the mock embedding function in conftest recognizes
    await api_client.post(
        "/embed", json={"content": "qdrant is a vector database"}
    )
    # Give it a moment to process.
    await asyncio.sleep(1)

    # Use a query that the mock embedding function recognizes
    payload = {"question": "what is a vector database"}
    response = await api_client.post("/chat", json=payload)

    assert response.status_code == 200
    json_response = response.json()
    assert "answer" in json_response
    assert "sources" in json_response
    assert isinstance(json_response["answer"], str)
    assert isinstance(json_response["sources"], list)
    # Check that the mock response from conftest is returned
    assert "the answer is Qdrant" in json_response["answer"] 