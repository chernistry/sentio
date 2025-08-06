"""Tests for Jina reranker."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.rerankers.jina_reranker import JinaReranker
from src.core.models.document import Document


@pytest.fixture
def mock_httpx_client():
    """Mock httpx async client."""
    client = AsyncMock()
    
    # Mock successful reranking response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "index": 1,
                "relevance_score": 0.95,
                "document": {
                    "text": "Second document content"
                }
            },
            {
                "index": 0,
                "relevance_score": 0.87,
                "document": {
                    "text": "First document content"
                }
            },
            {
                "index": 2,
                "relevance_score": 0.72,
                "document": {
                    "text": "Third document content"
                }
            }
        ],
        "usage": {
            "total_tokens": 150
        }
    }
    
    client.post.return_value = mock_response
    return client


@pytest.fixture
def jina_reranker(mock_httpx_client):
    """Create JinaReranker with mocked client."""
    with patch('httpx.AsyncClient', return_value=mock_httpx_client):
        reranker = JinaReranker(
            api_key="test-key",
            model="jina-reranker-v2-base-multilingual"
        )
        reranker._client = mock_httpx_client
        return reranker


@pytest.fixture
def sample_documents():
    """Sample documents for reranking."""
    return [
        Document(
            id="doc1",
            text="First document content about machine learning",
            metadata={"source": "doc1.pdf", "original_score": 0.8}
        ),
        Document(
            id="doc2", 
            text="Second document content about artificial intelligence",
            metadata={"source": "doc2.pdf", "original_score": 0.9}
        ),
        Document(
            id="doc3",
            text="Third document content about data science",
            metadata={"source": "doc3.pdf", "original_score": 0.7}
        )
    ]


@pytest.mark.asyncio
class TestJinaReranker:
    """Test Jina reranker functionality."""

    async def test_rerank_success(self, jina_reranker, mock_httpx_client, sample_documents):
        """Test successful document reranking."""
        query = "What is machine learning?"
        
        reranked_docs = await jina_reranker.rerank(query, sample_documents, top_k=3)
        
        # Verify API call
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        
        assert "https://api.jina.ai/v1/rerank" in call_args[0][0]
        request_data = call_args.kwargs["json"]
        assert request_data["model"] == "jina-reranker-v2-base-multilingual"
        assert request_data["query"] == query
        assert len(request_data["documents"]) == 3
        
        # Verify reranked results
        assert len(reranked_docs) == 3
        # Should be reordered by relevance score (doc2, doc1, doc3 based on mock)
        assert reranked_docs[0].id == "doc2"  # Highest relevance (0.95)
        assert reranked_docs[1].id == "doc1"  # Second highest (0.87)
        assert reranked_docs[2].id == "doc3"  # Lowest (0.72)
        
        # Should have rerank scores in metadata
        for doc in reranked_docs:
            assert "rerank_score" in doc.metadata

    async def test_rerank_with_top_k_limit(self, jina_reranker, sample_documents):
        """Test reranking with top_k limit."""
        query = "Test query"
        
        reranked_docs = await jina_reranker.rerank(query, sample_documents, top_k=2)
        
        # Should return only top 2 documents
        assert len(reranked_docs) == 2
        assert reranked_docs[0].id == "doc2"  # Highest score
        assert reranked_docs[1].id == "doc1"  # Second highest

    async def test_rerank_empty_documents(self, jina_reranker):
        """Test reranking with empty document list."""
        query = "Test query"
        
        reranked_docs = await jina_reranker.rerank(query, [], top_k=5)
        
        assert len(reranked_docs) == 0

    async def test_rerank_single_document(self, jina_reranker, mock_httpx_client):
        """Test reranking with single document."""
        query = "Test query"
        single_doc = [Document(
            id="single",
            text="Single document content",
            metadata={"source": "single.pdf"}
        )]
        
        # Mock response for single document
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "index": 0,
                    "relevance_score": 0.85,
                    "document": {"text": "Single document content"}
                }
            ]
        }
        mock_httpx_client.post.return_value = mock_response
        
        reranked_docs = await jina_reranker.rerank(query, single_doc, top_k=5)
        
        assert len(reranked_docs) == 1
        assert reranked_docs[0].id == "single"
        assert reranked_docs[0].metadata["rerank_score"] == 0.85

    async def test_rerank_api_error(self, jina_reranker, mock_httpx_client, sample_documents):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid request format",
                "type": "invalid_request_error"
            }
        }
        mock_httpx_client.post.return_value = mock_response
        
        with pytest.raises(Exception, match="Invalid request format"):
            await jina_reranker.rerank("test query", sample_documents, top_k=3)

    async def test_rerank_rate_limit(self, jina_reranker, mock_httpx_client, sample_documents):
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
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await jina_reranker.rerank("test query", sample_documents, top_k=3)

    async def test_rerank_network_error(self, jina_reranker, mock_httpx_client, sample_documents):
        """Test handling of network errors."""
        mock_httpx_client.post.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            await jina_reranker.rerank("test query", sample_documents, top_k=3)

    async def test_metadata_preservation(self, jina_reranker, sample_documents):
        """Test that original metadata is preserved."""
        query = "Test query"
        
        reranked_docs = await jina_reranker.rerank(query, sample_documents, top_k=3)
        
        for doc in reranked_docs:
            # Should preserve original metadata
            assert "source" in doc.metadata
            assert "original_score" in doc.metadata
            # Should add rerank score
            assert "rerank_score" in doc.metadata

    async def test_different_models(self, mock_httpx_client):
        """Test different Jina reranker models."""
        models_to_test = [
            "jina-reranker-v1-base-en",
            "jina-reranker-v2-base-multilingual",
            "jina-colbert-v1-en"
        ]
        
        for model in models_to_test:
            with patch('httpx.AsyncClient', return_value=mock_httpx_client):
                reranker = JinaReranker(
                    api_key="test-key",
                    model=model
                )
                reranker._client = mock_httpx_client
                
                await reranker.rerank("test query", [
                    Document(id="test", text="test content", metadata={})
                ], top_k=1)
                
                # Verify correct model was used
                call_args = mock_httpx_client.post.call_args
                request_data = call_args.kwargs["json"]
                assert request_data["model"] == model

    async def test_batch_processing(self, jina_reranker, mock_httpx_client):
        """Test batch processing of large document sets."""
        # Create many documents to test batching
        large_doc_set = [
            Document(
                id=f"doc_{i}",
                text=f"Document {i} content",
                metadata={"source": f"doc{i}.pdf"}
            )
            for i in range(100)
        ]
        
        # Mock response for large batch
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "index": i,
                    "relevance_score": 0.9 - (i * 0.001),  # Decreasing scores
                    "document": {"text": f"Document {i} content"}
                }
                for i in range(100)
            ]
        }
        mock_httpx_client.post.return_value = mock_response
        
        reranked_docs = await jina_reranker.rerank("test query", large_doc_set, top_k=50)
        
        # Should handle large batches and return top_k
        assert len(reranked_docs) == 50
        # Should be ordered by relevance score
        scores = [doc.metadata["rerank_score"] for doc in reranked_docs]
        assert scores == sorted(scores, reverse=True)

    async def test_health_check_success(self, jina_reranker, mock_httpx_client):
        """Test successful health check."""
        # Mock successful reranking response for health check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.8, "document": {"text": "test"}}
            ]
        }
        mock_httpx_client.post.return_value = mock_response
        
        is_healthy = await jina_reranker.health_check()
        
        assert is_healthy is True
        mock_httpx_client.post.assert_called_once()

    async def test_health_check_failure(self, jina_reranker, mock_httpx_client):
        """Test health check failure."""
        mock_httpx_client.post.side_effect = Exception("API unavailable")
        
        is_healthy = await jina_reranker.health_check()
        
        assert is_healthy is False

    async def test_request_headers(self, jina_reranker, mock_httpx_client, sample_documents):
        """Test that correct headers are sent."""
        await jina_reranker.rerank("test query", sample_documents, top_k=3)
        
        call_args = mock_httpx_client.post.call_args
        headers = call_args.kwargs["headers"]
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    async def test_query_preprocessing(self, jina_reranker, mock_httpx_client, sample_documents):
        """Test query preprocessing before reranking."""
        query = "  What is machine learning?  \n\n"
        
        await jina_reranker.rerank(query, sample_documents, top_k=3)
        
        call_args = mock_httpx_client.post.call_args
        request_data = call_args.kwargs["json"]
        
        # Query should be cleaned (exact preprocessing depends on implementation)
        sent_query = request_data["query"]
        assert isinstance(sent_query, str)
        assert len(sent_query.strip()) > 0

    async def test_document_text_extraction(self, jina_reranker, mock_httpx_client):
        """Test that document text is properly extracted for API."""
        docs_with_metadata = [
            Document(
                id="doc1",
                text="Main document text",
                metadata={
                    "content": "Alternative content",
                    "title": "Document Title",
                    "source": "doc1.pdf"
                }
            )
        ]
        
        await jina_reranker.rerank("test query", docs_with_metadata, top_k=1)
        
        call_args = mock_httpx_client.post.call_args
        request_data = call_args.kwargs["json"]
        
        # Should send the document text, not metadata
        assert request_data["documents"][0]["text"] == "Main document text"

    async def test_concurrent_reranking(self, jina_reranker):
        """Test concurrent reranking requests."""
        import asyncio
        
        document_sets = [
            [Document(id=f"set{i}_doc{j}", text=f"Set {i} doc {j}", metadata={})
             for j in range(3)]
            for i in range(3)
        ]
        
        # Run concurrent reranking requests
        tasks = [
            jina_reranker.rerank(f"query {i}", docs, top_k=2)
            for i, docs in enumerate(document_sets)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All requests should complete
        assert len(results) == 3
        assert all(len(result) <= 2 for result in results)

    async def test_score_normalization(self, jina_reranker, sample_documents):
        """Test that rerank scores are properly normalized."""
        reranked_docs = await jina_reranker.rerank("test query", sample_documents, top_k=3)
        
        for doc in reranked_docs:
            rerank_score = doc.metadata["rerank_score"]
            # Scores should be between 0 and 1 (or whatever range is expected)
            assert 0 <= rerank_score <= 1
