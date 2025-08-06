"""Tests for Jina embeddings provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.embeddings.providers.jina import JinaEmbeddings
from src.core.models.document import Document


@pytest.fixture
def mock_httpx_client():
    """Mock httpx async client."""
    client = AsyncMock()
    
    # Mock successful embedding response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3] * 128,  # 384 dimensions
                "index": 0
            },
            {
                "object": "embedding", 
                "embedding": [0.4, 0.5, 0.6] * 128,
                "index": 1
            }
        ],
        "model": "jina-embeddings-v3",
        "usage": {
            "total_tokens": 50,
            "prompt_tokens": 50
        }
    }
    
    client.post.return_value = mock_response
    return client


@pytest.fixture
def jina_embeddings(mock_httpx_client):
    """Create JinaEmbeddings with mocked client."""
    with patch('httpx.AsyncClient', return_value=mock_httpx_client):
        embeddings = JinaEmbeddings(
            api_key="test-key",
            model="jina-embeddings-v3"
        )
        embeddings._client = mock_httpx_client
        return embeddings


@pytest.mark.asyncio
class TestJinaEmbeddings:
    """Test Jina embeddings functionality."""

    async def test_embed_documents_success(self, jina_embeddings, mock_httpx_client):
        """Test successful document embedding."""
        documents = [
            Document(text="First document", metadata={"source": "doc1.pdf"}),
            Document(text="Second document", metadata={"source": "doc2.pdf"})
        ]

        embeddings = await jina_embeddings.embed_documents(documents)

        # Verify API call
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        
        assert "https://api.jina.ai/v1/embeddings" in call_args[0][0]
        request_data = call_args.kwargs["json"]
        assert request_data["model"] == "jina-embeddings-v3"
        assert len(request_data["input"]) == 2
        assert "First document" in request_data["input"]
        assert "Second document" in request_data["input"]

        # Verify embeddings
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384

    async def test_embed_query_success(self, jina_embeddings, mock_httpx_client):
        """Test successful query embedding."""
        query = "What is machine learning?"

        embedding = await jina_embeddings.embed_query(query)

        # Verify API call
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        
        request_data = call_args.kwargs["json"]
        assert request_data["input"] == [query]
        assert request_data["task"] == "retrieval.query"  # Different task for queries

        # Verify embedding
        assert len(embedding) == 384

    async def test_embed_documents_batch_processing(self, jina_embeddings, mock_httpx_client):
        """Test batch processing of large document sets."""
        # Create many documents to test batching
        documents = [
            Document(text=f"Document {i}", metadata={"source": f"doc{i}.pdf"})
            for i in range(100)
        ]

        embeddings = await jina_embeddings.embed_documents(documents)

        # Should handle batching (multiple API calls if batch size exceeded)
        assert len(embeddings) == 100
        assert all(len(emb) == 384 for emb in embeddings)

    async def test_embed_documents_api_error(self, jina_embeddings, mock_httpx_client):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid input",
                "type": "invalid_request_error"
            }
        }
        mock_httpx_client.post.return_value = mock_response

        documents = [Document(text="Test document", metadata={})]

        with pytest.raises(Exception, match="Invalid input"):
            await jina_embeddings.embed_documents(documents)

    async def test_embed_documents_rate_limit(self, jina_embeddings, mock_httpx_client):
        """Test handling of rate limit errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        }
        mock_httpx_client.post.return_value = mock_response

        documents = [Document(text="Test document", metadata={})]

        with pytest.raises(Exception, match="Rate limit exceeded"):
            await jina_embeddings.embed_documents(documents)

    async def test_embed_documents_network_error(self, jina_embeddings, mock_httpx_client):
        """Test handling of network errors."""
        mock_httpx_client.post.side_effect = Exception("Connection failed")

        documents = [Document(text="Test document", metadata={})]

        with pytest.raises(Exception, match="Connection failed"):
            await jina_embeddings.embed_documents(documents)

    async def test_embed_empty_documents(self, jina_embeddings):
        """Test embedding empty document list."""
        embeddings = await jina_embeddings.embed_documents([])
        assert embeddings == []

    async def test_embed_documents_with_metadata(self, jina_embeddings, mock_httpx_client):
        """Test that metadata is preserved but not sent to API."""
        documents = [
            Document(
                text="Document with metadata",
                metadata={"source": "test.pdf", "page": 1, "category": "technical"}
            )
        ]

        await jina_embeddings.embed_documents(documents)

        # Verify only text was sent to API, not metadata
        call_args = mock_httpx_client.post.call_args
        request_data = call_args.kwargs["json"]
        assert request_data["input"] == ["Document with metadata"]

    async def test_different_models(self, mock_httpx_client):
        """Test different Jina embedding models."""
        models_to_test = [
            "jina-embeddings-v2-base-en",
            "jina-embeddings-v3",
            "jina-clip-v1"
        ]

        for model in models_to_test:
            with patch('httpx.AsyncClient', return_value=mock_httpx_client):
                embeddings = JinaEmbeddings(
                    api_key="test-key",
                    model=model
                )
                embeddings._client = mock_httpx_client

                await embeddings.embed_query("Test query")

                # Verify correct model was used
                call_args = mock_httpx_client.post.call_args
                request_data = call_args.kwargs["json"]
                assert request_data["model"] == model

    async def test_health_check_success(self, jina_embeddings, mock_httpx_client):
        """Test successful health check."""
        # Mock successful embedding response for health check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 384, "index": 0}]
        }
        mock_httpx_client.post.return_value = mock_response

        is_healthy = await jina_embeddings.health_check()

        assert is_healthy is True
        mock_httpx_client.post.assert_called_once()

    async def test_health_check_failure(self, jina_embeddings, mock_httpx_client):
        """Test health check failure."""
        mock_httpx_client.post.side_effect = Exception("API unavailable")

        is_healthy = await jina_embeddings.health_check()

        assert is_healthy is False

    async def test_dimension_property(self, jina_embeddings):
        """Test dimension property returns correct value."""
        # Default dimension for jina-embeddings-v3
        assert jina_embeddings.dimension == 1024

    async def test_custom_task_types(self, jina_embeddings, mock_httpx_client):
        """Test different task types for embeddings."""
        # Test document task
        documents = [Document(text="Test document", metadata={})]
        await jina_embeddings.embed_documents(documents)
        
        call_args = mock_httpx_client.post.call_args
        request_data = call_args.kwargs["json"]
        assert request_data.get("task") == "retrieval.passage"

        # Reset mock
        mock_httpx_client.reset_mock()

        # Test query task
        await jina_embeddings.embed_query("Test query")
        
        call_args = mock_httpx_client.post.call_args
        request_data = call_args.kwargs["json"]
        assert request_data.get("task") == "retrieval.query"

    async def test_request_headers(self, jina_embeddings, mock_httpx_client):
        """Test that correct headers are sent."""
        await jina_embeddings.embed_query("Test query")

        call_args = mock_httpx_client.post.call_args
        headers = call_args.kwargs["headers"]
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    async def test_concurrent_requests(self, jina_embeddings, mock_httpx_client):
        """Test concurrent embedding requests."""
        import asyncio

        documents_sets = [
            [Document(text=f"Document set {i} doc {j}", metadata={})
             for j in range(5)]
            for i in range(3)
        ]

        # Run concurrent embedding requests
        tasks = [
            jina_embeddings.embed_documents(docs)
            for docs in documents_sets
        ]

        results = await asyncio.gather(*tasks)

        # All requests should complete
        assert len(results) == 3
        assert all(len(result) == 5 for result in results)
        assert mock_httpx_client.post.call_count == 3

    async def test_text_preprocessing(self, jina_embeddings, mock_httpx_client):
        """Test text preprocessing before embedding."""
        documents = [
            Document(text="  Text with extra whitespace  \n\n", metadata={}),
            Document(text="", metadata={}),  # Empty text
            Document(text="Normal text", metadata={})
        ]

        await jina_embeddings.embed_documents(documents)

        call_args = mock_httpx_client.post.call_args
        request_data = call_args.kwargs["json"]
        
        # Should handle preprocessing (exact behavior depends on implementation)
        assert len(request_data["input"]) <= 3  # May filter empty texts
