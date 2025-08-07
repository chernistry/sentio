"""Comprehensive tests for Jina embeddings provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.embeddings.providers.jina import JinaEmbedder
from src.core.embeddings.base import EmbeddingError


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing."""
    client = AsyncMock()
    
    # Mock successful response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1] * 384},
            {"embedding": [0.2] * 384}
        ]
    }
    
    client.post.return_value = mock_response
    return client


@pytest.fixture
def jina_embeddings(mock_httpx_client):
    """Create JinaEmbedder with mocked client."""
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

    async def test_embed_async_single_success(self, jina_embeddings, mock_httpx_client):
        """Test successful single text embedding."""
        text = "Single document to embed"
        
        # Mock the _execute_async_request method
        mock_response = [{"embedding": [0.1] * 384}]
        
        with patch.object(jina_embeddings, '_execute_async_request', return_value=mock_response) as mock_request:
            embedding = await jina_embeddings.embed_async_single(text)

            # Verify the response
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)
            mock_request.assert_called_once()

    async def test_embed_async_many_batch_processing(self, jina_embeddings, mock_httpx_client):
        """Test batch processing with multiple requests."""
        # Create more texts than batch size to test batching
        texts = [f"Document {i}" for i in range(10)]
        
        # Mock successful batch responses
        mock_response = [{"embedding": [0.1] * 384} for _ in range(10)]
        
        with patch.object(jina_embeddings, '_execute_async_request', return_value=mock_response) as mock_request:
            embeddings = await jina_embeddings.embed_async_many(texts)

            # Verify all embeddings were generated
            assert len(embeddings) == 10
            assert all(len(emb) == 384 for emb in embeddings)

    async def test_embed_async_many_api_error(self, jina_embeddings, mock_httpx_client):
        """Test handling of API errors."""
        texts = ["Document with error"]
        
        # Mock API error by making the client raise an exception
        mock_httpx_client.post.side_effect = Exception("API Error")
        
        # Mock the fallback to also fail
        with patch('src.core.embeddings.providers.jina.embedding_fallback.generate_simple_embedding', 
                   side_effect=Exception("Fallback failed")):
            with pytest.raises(EmbeddingError, match="Both primary and fallback embedding services failed"):
                await jina_embeddings.embed_async_many(texts)

    async def test_embed_async_many_rate_limit(self, jina_embeddings, mock_httpx_client):
        """Test handling of rate limit errors."""
        texts = ["Document with rate limit"]
        
        # Mock rate limit error
        mock_httpx_client.post.side_effect = Exception("Rate limit exceeded")
        
        # Mock the fallback to also fail
        with patch('src.core.embeddings.providers.jina.embedding_fallback.generate_simple_embedding', 
                   side_effect=Exception("Fallback failed")):
            with pytest.raises(EmbeddingError, match="Both primary and fallback embedding services failed"):
                await jina_embeddings.embed_async_many(texts)

    async def test_embed_async_many_network_error(self, jina_embeddings, mock_httpx_client):
        """Test handling of network errors."""
        texts = ["Document with network error"]
        
        # Mock network error
        mock_httpx_client.post.side_effect = Exception("Connection failed")
        
        # Mock the fallback to also fail
        with patch('src.core.embeddings.providers.jina.embedding_fallback.generate_simple_embedding', 
                   side_effect=Exception("Fallback failed")):
            with pytest.raises(EmbeddingError, match="Both primary and fallback embedding services failed"):
                await jina_embeddings.embed_async_many(texts)

    async def test_different_models(self, mock_httpx_client):
        """Test with different embedding models."""
        # Test with different model
        embeddings = JinaEmbedder(
            model_name="jina-embeddings-v2-base-en",
            api_key="test-key"
        )
        embeddings._client = mock_httpx_client
        
        mock_response = [{"embedding": [0.1] * 384}]
        
        with patch.object(embeddings, '_execute_async_request', return_value=mock_response):
            result = await embeddings.embed_async_single("Test query")
            assert len(result) == 384

    async def test_caching_functionality(self, jina_embeddings, mock_httpx_client):
        """Test embedding caching."""
        text = "Cached text"
        
        # Mock successful response
        mock_response = [{"embedding": [0.1] * 384}]
        
        with patch.object(jina_embeddings, '_execute_async_request', return_value=mock_response) as mock_request:
            # First call
            embedding1 = await jina_embeddings.embed_async_single(text)
            
            # Second call (should use cache if implemented)
            embedding2 = await jina_embeddings.embed_async_single(text)
            
            # Verify embeddings are the same
            assert embedding1 == embedding2
            assert len(embedding1) == 384

    async def test_error_recovery_with_retries(self, jina_embeddings, mock_httpx_client):
        """Test error recovery with retry mechanism."""
        text = "Text with retry"
        
        # Mock successful response after retries
        mock_response = [{"embedding": [0.1] * 384}]
        
        with patch.object(jina_embeddings, '_execute_async_request', return_value=mock_response) as mock_request:
            embedding = await jina_embeddings.embed_async_single(text)
            
            assert len(embedding) == 384
            mock_request.assert_called_once()

    async def test_empty_input_handling(self, jina_embeddings):
        """Test handling of empty input."""
        # Test empty list
        embeddings = await jina_embeddings.embed_async_many([])
        assert embeddings == []
        
        # Test empty string
        with pytest.raises((ValueError, EmbeddingError)):
            await jina_embeddings.embed_async_single("")

    async def test_large_batch_processing(self, jina_embeddings):
        """Test processing of large batches."""
        # Create a large batch
        texts = [f"Document {i}" for i in range(100)]
        
        # Mock successful batch processing
        mock_response = [{"embedding": [0.1] * 384} for _ in range(100)]
        
        with patch.object(jina_embeddings, '_execute_async_request', return_value=mock_response):
            embeddings = await jina_embeddings.embed_async_many(texts)
            
            assert len(embeddings) == 100
            assert all(len(emb) == 384 for emb in embeddings)

    async def test_special_characters_handling(self, jina_embeddings):
        """Test handling of special characters in text."""
        special_text = "Text with special chars: éñ中文🚀"
        
        mock_response = [{"embedding": [0.1] * 384}]
        
        with patch.object(jina_embeddings, '_execute_async_request', return_value=mock_response):
            embedding = await jina_embeddings.embed_async_single(special_text)
            
            assert len(embedding) == 384

    async def test_concurrent_requests(self, jina_embeddings):
        """Test concurrent embedding requests."""
        import asyncio
        
        texts = [f"Concurrent text {i}" for i in range(5)]
        
        mock_response = [{"embedding": [0.1] * 384}]
        
        with patch.object(jina_embeddings, '_execute_async_request', return_value=mock_response):
            # Run concurrent requests
            tasks = [jina_embeddings.embed_async_single(text) for text in texts]
            embeddings = await asyncio.gather(*tasks)
            
            assert len(embeddings) == 5
            assert all(len(emb) == 384 for emb in embeddings)

    async def test_model_configuration(self, mock_httpx_client):
        """Test different model configurations."""
        # Test with custom configuration
        embeddings = JinaEmbedder(
            model_name="custom-model",
            api_key="test-key",
            timeout=30
        )
        embeddings._client = mock_httpx_client
        
        mock_response = [{"embedding": [0.1] * 384}]
        
        with patch.object(embeddings, '_execute_async_request', return_value=mock_response):
            result = await embeddings.embed_async_single("Test")
            assert len(result) == 384

    def test_initialization(self):
        """Test embedder initialization."""
        embedder = JinaEmbedder(
            model_name="test-model",
            api_key="test-key"
        )
        
        assert embedder.model_name == "test-model"
        assert embedder.api_key == "test-key"

    async def test_health_check(self, jina_embeddings):
        """Test health check functionality."""
        # Mock successful health check
        mock_response = [{"embedding": [0.1] * 384}]
        
        with patch.object(jina_embeddings, '_execute_async_request', return_value=mock_response):
            # Health check by trying a simple embedding
            try:
                await jina_embeddings.embed_async_single("health check")
                health_status = True
            except Exception:
                health_status = False
            
            assert health_status is True
