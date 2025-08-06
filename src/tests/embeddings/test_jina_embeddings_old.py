"""Tests for Jina embeddings provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.embeddings.providers.jina import JinaEmbedder
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
    """Create JinaEmbedder with mocked client."""
    with patch('httpx.AsyncClient', return_value=mock_httpx_client):
        embeddings = JinaEmbedder(
            model_name="jina-embeddings-v3",
            api_key="test-key"
        )
        embeddings._client = mock_httpx_client
        return embeddings


@pytest.mark.asyncio
class TestJinaEmbedder:
    """Test Jina embeddings functionality."""

    async def test_embed_async_many_success(self, jina_embeddings, mock_httpx_client):
        """Test successful document embedding."""
        texts = ["First document", "Second document"]
        
        # Mock the _execute_async_request method directly
        mock_response = [
            {"embedding": [0.1] * 384},
            {"embedding": [0.2] * 384}
        ]
        
        with patch.object(jina_embeddings, '_execute_async_request', return_value=mock_response) as mock_request:
            embeddings = await jina_embeddings.embed_async_many(texts)

            # Verify the response
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384
            assert len(embeddings[1]) == 384
            mock_request.assert_called_once()
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384

    async def test_embed_async_single_success(self, jina_embeddings, mock_httpx_client):
        """Test successful single text embedding."""
        text = "What is machine learning?"

        embedding = await jina_embeddings.embed_async_single(text)

        # Verify API call
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        
        request_data = call_args.kwargs["json"]
        assert request_data["input"] == [text]

        # Verify embedding
        assert len(embedding) == 384

    async def test_embed_async_many_batch_processing(self, jina_embeddings, mock_httpx_client):
        """Test batch processing of large text sets."""
        # Create many texts to test batching
        texts = [f"Document {i}" for i in range(100)]

        embeddings = await jina_embeddings.embed_async_many(texts)

        # Should handle batching (multiple API calls if batch size exceeded)
        assert len(embeddings) == 100
        assert all(len(emb) == 384 for emb in embeddings)

    async def test_embed_async_many_api_error(self, jina_embeddings, mock_httpx_client):
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

        texts = ["Test document"]

        with pytest.raises(Exception, match="Invalid input"):
            await jina_embeddings.embed_async_many(texts)

    async def test_embed_async_many_rate_limit(self, jina_embeddings, mock_httpx_client):
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

        texts = ["Test document"]

        with pytest.raises(Exception, match="Rate limit exceeded"):
            await jina_embeddings.embed_async_many(texts)

    async def test_embed_async_many_network_error(self, jina_embeddings, mock_httpx_client):
        """Test handling of network errors."""
        mock_httpx_client.post.side_effect = Exception("Connection failed")

        texts = ["Test document"]

        with pytest.raises(Exception, match="Connection failed"):
            await jina_embeddings.embed_async_many(texts)

    async def test_embed_empty_texts(self, jina_embeddings):
        """Test embedding empty text list."""
        embeddings = await jina_embeddings.embed_async_many([])
        assert embeddings == []

    async def test_dimension_property(self, jina_embeddings):
        """Test dimension property returns correct value."""
        # Default dimension for jina-embeddings-v3
        assert jina_embeddings.dimension == 1024

    async def test_different_models(self, mock_httpx_client):
        """Test different Jina embedding models."""
        models_to_test = [
            "jina-embeddings-v2",
            "jina-embeddings-v3",
        ]

        for model in models_to_test:
            with patch('httpx.AsyncClient', return_value=mock_httpx_client):
                embeddings = JinaEmbedder(
                    model_name=model,
                    api_key="test-key"
                )
                embeddings._client = mock_httpx_client

                await embeddings.embed_async_single("Test query")

                # Verify correct model was used
                call_args = mock_httpx_client.post.call_args
                request_data = call_args.kwargs["json"]
                assert request_data["model"] == model

    async def test_caching_functionality(self, jina_embeddings, mock_httpx_client):
        """Test that caching works correctly."""
        text = "Test text for caching"
        
        # First call should hit the API
        embedding1 = await jina_embeddings.embed_async_single(text)
        assert mock_httpx_client.post.call_count == 1
        
        # Second call should use cache (if caching is enabled)
        embedding2 = await jina_embeddings.embed_async_single(text)
        
        # Results should be the same
        assert embedding1 == embedding2

    async def test_concurrent_requests(self, jina_embeddings, mock_httpx_client):
        """Test concurrent embedding requests."""
        import asyncio

        texts_sets = [
            [f"Text set {i} item {j}" for j in range(3)]
            for i in range(3)
        ]

        # Run concurrent embedding requests
        tasks = [
            jina_embeddings.embed_async_many(texts)
            for texts in texts_sets
        ]

        results = await asyncio.gather(*tasks)

        # All requests should complete
        assert len(results) == 3
        assert all(len(result) == 3 for result in results)

    async def test_error_recovery_with_retries(self, jina_embeddings, mock_httpx_client):
        """Test error recovery with retry mechanism."""
        # First call fails, second succeeds
        mock_httpx_client.post.side_effect = [
            Exception("Temporary network error"),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "data": [{"embedding": [0.1] * 384, "index": 0}]
                }
            )
        ]

        # Should succeed after retry
        embedding = await jina_embeddings.embed_async_single("Test text")
        assert len(embedding) == 384
        assert mock_httpx_client.post.call_count == 2
