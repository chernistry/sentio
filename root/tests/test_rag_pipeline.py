"""
End-to-end tests for the full RAG pipeline.
"""
import asyncio
from typing import List
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient
from _pytest.monkeypatch import MonkeyPatch

# Although we are patching EmbeddingModel, we might need it for type hints
from root.src.core.tasks.embeddings import EmbeddingModel
from root.app import app

pytestmark = pytest.mark.e2e


@pytest.fixture
def mock_embedding_model(monkeypatch: MonkeyPatch, embed_model: EmbeddingModel) -> MagicMock:
    """
    Mocks the EmbeddingModel to return deterministic vectors for offline testing.
    This prevents calls to external APIs (e.g., Jina) during the E2E test.
    """
    # Create a fixed vector of the correct dimension
    mock_vector = [0.1] * embed_model.dimension

    # The mock needs to be an async function
    async def mock_embed_single(text: str) -> List[float]:
        # Return a slightly different vector based on text length for variety
        return mock_vector[:-1] + [len(text) / 1000.0]

    async def mock_embed_many(texts: List[str]) -> List[List[float]]:
        return [await mock_embed_single(text) for text in texts]

    # Patch the methods on the class itself. This ensures that any instance
    # of EmbeddingModel created by the pipeline will use these mocked methods.
    # The patch is applied where the class is looked up, which is in the pipeline module.
    monkeypatch.setattr(
        "root.src.core.pipeline.EmbeddingModel.embed_async_single",
        mock_embed_single
    )
    monkeypatch.setattr(
        "root.src.core.pipeline.EmbeddingModel.embed_async_many",
        mock_embed_many
    )
    # Also patch where it's used directly in app.py for the /embed endpoint
    monkeypatch.setattr(
        "root.app.get_query_embedding",
        mock_embed_single
    )

@pytest.mark.asyncio
async def test_rag_pipeline_full_flow(
    api_client: AsyncClient,
    sample_docs: List[str],
    mock_embedding_model: MagicMock,
):
    """
    Tests the full RAG pipeline from ingestion to chat response.

    1. Ingests a set of sample documents via the /embed endpoint.
    2. Asks a question related to the ingested content via the /chat endpoint.
    3. Verifies that the answer contains relevant keywords from the source docs.
    """
    # 1. Ingest documents
    # The /embed endpoint may process docs one by one
    for doc in sample_docs:
        response = await api_client.post("/embed", json={"content": doc})
        assert response.status_code == 200

    # Give the system a moment to process the ingested documents
    await asyncio.sleep(2)

    # 2. Ask a question that should be answerable from the ingested docs
    question = "What is a vector database?"
    
    # One of the sample docs is: "Qdrant is a vector database used for similarity search."
    # We expect the answer to contain "Qdrant".
    expected_keyword = "Qdrant"

    # 3. Get chat response
    response = await api_client.post("/chat", json={"question": question})
    assert response.status_code == 200
    
    json_response = response.json()
    answer = json_response.get("answer", "").lower()
    sources = json_response.get("sources", [])

    # 4. Verify the answer
    assert expected_keyword.lower() in answer

    # And verify that one of the sources contains the keyword
    source_texts = " ".join(s.get("text", "") for s in sources).lower()
    assert expected_keyword.lower() in source_texts 