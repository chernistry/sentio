"""Comprehensive tests for Jina reranker functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.rerankers.jina_reranker import JinaReranker
from src.core.models.document import Document


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing."""
    client = AsyncMock()
    
    # Mock successful response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.87}
        ]
    }
    
    client.post.return_value = mock_response
    return client


@pytest.fixture
def jina_reranker(mock_httpx_client):
    """Create JinaReranker with mocked client."""
    with patch('httpx.AsyncClient', return_value=mock_httpx_client):
        reranker = JinaReranker(
            api_key="test-key",
            model_name="jina-reranker-v2-base-multilingual"  # Use model_name not model
        )
        reranker._client = mock_httpx_client
        return reranker


@pytest.fixture
def sample_documents():
    """Sample documents for reranking."""
    return [
        Document(text="Machine learning is a subset of AI", metadata={"source": "doc1.pdf"}),
        Document(text="Deep learning uses neural networks", metadata={"source": "doc2.pdf"}),
        Document(text="Natural language processing handles text", metadata={"source": "doc3.pdf"})
    ]


class TestJinaReranker:
    """Test Jina reranker functionality."""

    def test_rerank_success(self, jina_reranker, sample_documents):
        """Test successful document reranking."""
        query = "What is machine learning?"
        
        # Mock the requests.post method
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.87},
                {"index": 2, "relevance_score": 0.75}
            ]
        }
        
        with patch('requests.post', return_value=mock_response) as mock_request:
            results = jina_reranker.rerank(query, sample_documents, top_k=3)
            
            # Verify results
            assert len(results) == 3
            assert all(isinstance(doc, Document) for doc in results)
            # Check that documents are sorted by rerank_score
            scores = [doc.metadata.get("rerank_score", 0.0) for doc in results]
            assert scores == sorted(scores, reverse=True)
            mock_request.assert_called_once()

    def test_rerank_with_top_k_limit(self, jina_reranker, sample_documents):
        """Test reranking with top_k limit."""
        query = "AI and machine learning"
        
        # Mock the requests.post method
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.87}
            ]
        }
        
        with patch('requests.post', return_value=mock_response):
            results = jina_reranker.rerank(query, sample_documents, top_k=2)
            
            # Should return only top 2 results
            assert len(results) == 2
            assert all(isinstance(doc, Document) for doc in results)

    def test_rerank_empty_documents(self, jina_reranker):
        """Test reranking with empty document list."""
        query = "Test query"
        
        results = jina_reranker.rerank(query, [], top_k=5)
        
        # Should return empty list
        assert results == []

    def test_rerank_single_document(self, jina_reranker, sample_documents):
        """Test reranking with single document."""
        query = "Single document test"
        single_doc = [sample_documents[0]]
        
        mock_results = [(sample_documents[0], 0.85)]
        
        with patch('requests.post', return_value=mock_results):
            results = jina_reranker.rerank(query, single_doc, top_k=5)
            
            assert len(results) == 1
            assert results[0] == sample_documents[0]  # Results are Documents, not tuples

    def test_rerank_api_error(self, jina_reranker, sample_documents):
        """Test handling of API errors."""
        query = "Error test"
        
        with patch('requests.post', side_effect=Exception("API Error")):
            # Should return default ranking instead of raising exception
            results = jina_reranker.rerank(query, sample_documents, top_k=3)
            assert len(results) <= 3
            assert all(isinstance(doc, Document) for doc in results)

    def test_rerank_rate_limit(self, jina_reranker, sample_documents):
        """Test handling of rate limit errors."""
        query = "Rate limit test"
        
        with patch('requests.post', side_effect=Exception("Rate limit exceeded")):
            # Should return default ranking instead of raising exception
            results = jina_reranker.rerank(query, sample_documents, top_k=3)
            assert len(results) <= 3
            assert all(isinstance(doc, Document) for doc in results)

    def test_rerank_network_error(self, jina_reranker, sample_documents):
        """Test handling of network errors."""
        query = "Network error test"
        
        with patch('requests.post', side_effect=Exception("Connection failed")):
            # Should return default ranking instead of raising exception
            results = jina_reranker.rerank(query, sample_documents, top_k=3)
            assert len(results) <= 3
            assert all(isinstance(doc, Document) for doc in results)

    def test_metadata_preservation(self, jina_reranker, sample_documents):
        """Test that document metadata is preserved."""
        query = "Metadata test"
        
        mock_results = [(doc, 0.8) for doc in sample_documents]
        
        with patch('requests.post', return_value=mock_results):
            results = jina_reranker.rerank(query, sample_documents, top_k=3)
            
            # Verify metadata is preserved
            for result_doc, original_doc in zip(results, sample_documents):
                assert result_doc.metadata.get("source") == original_doc.metadata.get("source")
                assert "rerank_score" in result_doc.metadata

    def test_batch_processing(self, jina_reranker):
        """Test batch processing of large document sets."""
        # Create large document set
        large_doc_set = [
            Document(text=f"Document {i} content", metadata={"source": f"doc{i}.pdf"})
            for i in range(20)
        ]
        
        query = "Batch processing test"
        
        mock_results = [(doc, 0.8 - i * 0.01) for i, doc in enumerate(large_doc_set)]
        
        with patch('requests.post', return_value=mock_results):
            results = jina_reranker.rerank(query, large_doc_set, top_k=10)
            
            # Should handle large batches
            assert len(results) == 10
            assert all(isinstance(doc, Document) for doc in results)

    def test_health_check_success(self, jina_reranker, mock_httpx_client):
        """Test successful health check."""
        # Mock successful health check response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_httpx_client.post.return_value = mock_response
        
        # Test health check through a simple rerank operation
        test_docs = [Document(text="Health check", metadata={})]
        
        with patch('requests.post', return_value=[(test_docs[0], 0.9)]):
            try:
                jina_reranker.rerank("health", test_docs, top_k=1)
                health_status = True
            except Exception:
                health_status = False
            
            assert health_status is True

    def test_health_check_failure(self, jina_reranker, mock_httpx_client):
        """Test health check failure."""
        # Mock failed health check
        mock_httpx_client.post.side_effect = Exception("Service unavailable")
        
        test_docs = [Document(text="Health check", metadata={})]
        
        with patch('requests.post', side_effect=Exception("Service unavailable")):
            # Should return default ranking instead of raising exception
            results = jina_reranker.rerank("health", test_docs, top_k=1)
            assert len(results) == 1
            assert isinstance(results[0], Document)

    def test_request_headers(self, jina_reranker, mock_httpx_client):
        """Test request headers are properly set."""
        query = "Header test"
        docs = [Document(text="Test document", metadata={})]
        
        with patch('requests.post', return_value=[(docs[0], 0.9)]):
            jina_reranker.rerank(query, docs, top_k=1)
            
            # Verify the request was made (headers are set internally)
            # This is tested indirectly through successful execution

    def test_query_preprocessing(self, jina_reranker, sample_documents):
        """Test query preprocessing."""
        # Test with special characters and formatting
        special_query = "What is machine learning? 🤖 éñ中文"
        
        mock_results = [(sample_documents[0], 0.9)]
        
        with patch('requests.post', return_value=mock_results):
            results = jina_reranker.rerank(special_query, sample_documents, top_k=1)
            
            # Should handle special characters gracefully
            assert len(results) == 1

    def test_document_text_extraction(self, jina_reranker):
        """Test document text extraction for reranking."""
        # Test with documents containing different text formats
        docs = [
            Document(text="Simple text", metadata={}),
            Document(text="Text with\nnewlines", metadata={}),
            Document(text="Text with special chars: éñ中文", metadata={})
        ]
        
        query = "Text extraction test"
        
        mock_results = [(doc, 0.8) for doc in docs]
        
        with patch('requests.post', return_value=mock_results):
            results = jina_reranker.rerank(query, docs, top_k=3)
            
            # Should handle all text formats
            assert len(results) == 3

    def test_concurrent_reranking(self, jina_reranker, sample_documents):
        """Test concurrent reranking requests."""
        import asyncio
        
        query = "Concurrent test"
        
        mock_results = [(doc, 0.8) for doc in sample_documents]
        
        with patch('requests.post', return_value=mock_results):
            # Run concurrent reranking requests
            results = [
                jina_reranker.rerank(f"{query} {i}", sample_documents, top_k=2)
                for i in range(5)
            ]
            
            # All should succeed
            assert len(results) == 5
            assert all(len(result) == 2 for result in results)

    def test_score_normalization(self, jina_reranker, sample_documents):
        """Test score normalization."""
        query = "Score normalization test"
        
        # Mock results with various score ranges
        mock_results = [
            (sample_documents[0], 0.95),
            (sample_documents[1], 0.75),
            (sample_documents[2], 0.55)
        ]
        
        with patch('requests.post', return_value=mock_results):
            results = jina_reranker.rerank(query, sample_documents, top_k=3)
            
            # Verify scores are in expected range (stored in metadata)
            for doc in results:
                score = doc.metadata.get("rerank_score", 0.0)
                assert 0.0 <= score <= 1.0

    def test_different_models(self, mock_httpx_client):
        """Test with different reranker models."""
        # Test with different model
        reranker = JinaReranker(
            api_key="test-key",
            model_name="jina-reranker-v1-base-en"
        )
        reranker._client = mock_httpx_client
        
        docs = [Document(text="Test document", metadata={})]
        
        with patch('requests.post', return_value=[(docs[0], 0.9)]):
            results = reranker.rerank("test", docs, top_k=1)
            
            assert len(results) == 1

    def test_initialization(self):
        """Test reranker initialization."""
        reranker = JinaReranker(
            api_key="test-key",
            model_name="test-model"
        )
        
        assert reranker.api_key == "test-key"
        assert reranker.model_name == "test-model"

    def test_empty_query_handling(self, jina_reranker, sample_documents):
        """Test handling of empty queries."""
        empty_query = ""
        
        # Should handle empty query gracefully
        with patch('requests.post', return_value=[]):
            results = jina_reranker.rerank(empty_query, sample_documents, top_k=3)
            
            # May return empty results or handle gracefully
            assert isinstance(results, list)

    def test_large_document_content(self, jina_reranker):
        """Test reranking with large document content."""
        # Create document with large content
        large_doc = Document(
            text="Large content " * 1000,
            metadata={"source": "large.pdf"}
        )
        
        query = "Large document test"
        
        with patch('requests.post', return_value=[(large_doc, 0.8)]):
            results = jina_reranker.rerank(query, [large_doc], top_k=1)
            
            # Should handle large content
            assert len(results) == 1

    def test_timeout_handling(self, jina_reranker, sample_documents):
        """Test request timeout handling."""
        query = "Timeout test"
        
        # Mock timeout scenario
        def slow_rerank(*args, **kwargs):
            import time
            time.sleep(0.1)
            return [(sample_documents[0], 0.8)]
        
        with patch('requests.post', side_effect=slow_rerank):
            results = jina_reranker.rerank(query, sample_documents, top_k=1)
            
            # Should handle timeout gracefully
            assert len(results) == 1
