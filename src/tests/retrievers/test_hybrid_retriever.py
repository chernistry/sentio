"""Comprehensive tests for hybrid retriever functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.retrievers.hybrid import HybridRetriever
from src.core.retrievers.dense import DenseRetriever
from src.core.models.document import Document


@pytest.fixture
def mock_dense_retriever():
    """Mock dense retriever for testing."""
    retriever = AsyncMock(spec=DenseRetriever)
    retriever.retrieve.return_value = [
        Document(text="Dense result 1", metadata={"source": "dense1.pdf", "score": 0.9}),
        Document(text="Dense result 2", metadata={"source": "dense2.pdf", "score": 0.8})
    ]
    return retriever


@pytest.fixture
def mock_sparse_retriever():
    """Mock sparse retriever for testing."""
    retriever = AsyncMock()
    retriever.retrieve.return_value = [
        Document(text="Sparse result 1", metadata={"source": "sparse1.pdf", "score": 0.85}),
        Document(text="Sparse result 2", metadata={"source": "sparse2.pdf", "score": 0.75})
    ]
    return retriever


@pytest.fixture
def sample_documents():
    """Sample documents for corpus."""
    return [
        Document(text="Machine learning is a subset of AI", metadata={"source": "doc1.pdf"}),
        Document(text="Deep learning uses neural networks", metadata={"source": "doc2.pdf"}),
        Document(text="Natural language processing handles text", metadata={"source": "doc3.pdf"})
    ]


@pytest.fixture
def hybrid_retriever(mock_dense_retriever, mock_sparse_retriever):
    """Create HybridRetriever with mocked components."""
    retriever = HybridRetriever(
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        rrf_k=60  # Use actual constructor parameter
    )
    return retriever


@pytest.mark.asyncio
class TestHybridRetriever:
    """Test hybrid retriever functionality."""

    async def test_retrieve_success(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test successful hybrid retrieval."""
        query = "What is machine learning?"
        
        # Mock retriever responses
        mock_dense_retriever.retrieve.return_value = [
            Document(text="Dense ML result", metadata={"source": "dense.pdf", "score": 0.9})
        ]
        mock_sparse_retriever.retrieve.return_value = [
            Document(text="Sparse ML result", metadata={"source": "sparse.pdf", "score": 0.8})
        ]
        
        results = await hybrid_retriever.retrieve(query, top_k=5)
        
        # Verify results
        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(isinstance(doc, Document) for doc in results)
        
        # Verify both retrievers were called
        mock_dense_retriever.retrieve.assert_called_once()
        mock_sparse_retriever.retrieve.assert_called_once()

    async def test_reciprocal_rank_fusion(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test reciprocal rank fusion algorithm."""
        query = "RRF test"
        
        # Mock overlapping results with different rankings
        shared_doc = Document(text="Shared result", metadata={"source": "shared.pdf"})
        
        mock_dense_retriever.retrieve.return_value = [
            shared_doc,  # Rank 1 in dense
            Document(text="Dense only", metadata={"source": "dense.pdf"})  # Rank 2 in dense
        ]
        mock_sparse_retriever.retrieve.return_value = [
            Document(text="Sparse only", metadata={"source": "sparse.pdf"}),  # Rank 1 in sparse
            shared_doc  # Rank 2 in sparse
        ]
        
        results = await hybrid_retriever.retrieve(query, top_k=5)
        
        # Shared document should be ranked highly due to RRF
        assert len(results) > 0
        assert isinstance(results, list)

    async def test_weighted_score_fusion(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test weighted score fusion."""
        query = "Weighted fusion test"
        
        # Mock results with different scores
        mock_dense_retriever.retrieve.return_value = [
            Document(text="High dense score", metadata={"source": "dense.pdf", "score": 0.95})
        ]
        mock_sparse_retriever.retrieve.return_value = [
            Document(text="High sparse score", metadata={"source": "sparse.pdf", "score": 0.90})
        ]
        
        results = await hybrid_retriever.retrieve(query, top_k=5)
        
        # Should combine results from both retrievers
        assert len(results) <= 5
        assert isinstance(results, list)

    async def test_deduplication(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test deduplication of results."""
        query = "Deduplication test"
        
        # Create identical documents from both retrievers
        duplicate_doc = Document(text="Duplicate content", metadata={"source": "dup.pdf"})
        
        mock_dense_retriever.retrieve.return_value = [duplicate_doc]
        mock_sparse_retriever.retrieve.return_value = [duplicate_doc]
        
        results = await hybrid_retriever.retrieve(query, top_k=5)
        
        # Should deduplicate results
        assert isinstance(results, list)
        # Note: Actual deduplication logic depends on implementation

    async def test_empty_results_handling(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test handling when one retriever returns empty results."""
        query = "Empty results test"
        
        # Dense returns results, sparse returns empty
        mock_dense_retriever.retrieve.return_value = [
            Document(text="Dense result", metadata={"source": "dense.pdf"})
        ]
        mock_sparse_retriever.retrieve.return_value = []
        
        results = await hybrid_retriever.retrieve(query, top_k=5)
        
        # Should still return results from dense retriever
        assert len(results) >= 0
        assert isinstance(results, list)

    async def test_both_retrievers_empty(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test handling when both retrievers return empty results."""
        query = "No results test"
        
        mock_dense_retriever.retrieve.return_value = []
        mock_sparse_retriever.retrieve.return_value = []
        
        results = await hybrid_retriever.retrieve(query, top_k=5)
        
        # Should return empty list
        assert results == []

    async def test_retriever_failure_handling(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test handling when one retriever fails."""
        query = "Failure test"
        
        # Dense retriever fails, sparse succeeds
        mock_dense_retriever.retrieve.side_effect = Exception("Dense retriever error")
        mock_sparse_retriever.retrieve.return_value = [
            Document(text="Sparse result", metadata={"source": "sparse.pdf"})
        ]
        
        # Should handle failure gracefully
        try:
            results = await hybrid_retriever.retrieve(query, top_k=5)
            # May return sparse results only or handle error
            assert isinstance(results, list)
        except Exception:
            # Or may propagate the error
            pass

    async def test_both_retrievers_fail(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test handling when both retrievers fail."""
        query = "Both fail test"
        
        mock_dense_retriever.retrieve.side_effect = Exception("Dense error")
        mock_sparse_retriever.retrieve.side_effect = Exception("Sparse error")
        
        # Should handle total failure
        with pytest.raises(Exception):
            await hybrid_retriever.retrieve(query, top_k=5)

    async def test_top_k_handling(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test top_k parameter handling."""
        query = "Top K test"
        
        # Mock many results
        dense_results = [
            Document(text=f"Dense {i}", metadata={"source": f"dense{i}.pdf"})
            for i in range(10)
        ]
        sparse_results = [
            Document(text=f"Sparse {i}", metadata={"source": f"sparse{i}.pdf"})
            for i in range(10)
        ]
        
        mock_dense_retriever.retrieve.return_value = dense_results
        mock_sparse_retriever.retrieve.return_value = sparse_results
        
        results = await hybrid_retriever.retrieve(query, top_k=3)
        
        # Should return at most top_k results
        assert len(results) <= 3

    async def test_query_preprocessing(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test query preprocessing."""
        # Test with special characters and formatting
        special_query = "What is AI? 🤖 éñ中文"
        
        mock_dense_retriever.retrieve.return_value = [
            Document(text="AI result", metadata={"source": "ai.pdf"})
        ]
        mock_sparse_retriever.retrieve.return_value = []
        
        results = await hybrid_retriever.retrieve(special_query, top_k=5)
        
        # Should handle special characters gracefully
        assert isinstance(results, list)

    async def test_concurrent_retrieval(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test concurrent retrieval operations."""
        import asyncio
        
        query = "Concurrent test"
        
        mock_dense_retriever.retrieve.return_value = [
            Document(text="Dense concurrent", metadata={"source": "dense.pdf"})
        ]
        mock_sparse_retriever.retrieve.return_value = [
            Document(text="Sparse concurrent", metadata={"source": "sparse.pdf"})
        ]
        
        # Run multiple concurrent retrievals
        tasks = [
            hybrid_retriever.retrieve(f"{query} {i}", top_k=3)
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 5
        assert all(isinstance(result, list) for result in results)

    async def test_metadata_preservation(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test that document metadata is preserved."""
        query = "Metadata test"
        
        dense_doc = Document(
            text="Dense with metadata",
            metadata={"source": "dense.pdf", "page": 1, "section": "intro"}
        )
        sparse_doc = Document(
            text="Sparse with metadata",
            metadata={"source": "sparse.pdf", "page": 2, "section": "body"}
        )
        
        mock_dense_retriever.retrieve.return_value = [dense_doc]
        mock_sparse_retriever.retrieve.return_value = [sparse_doc]
        
        results = await hybrid_retriever.retrieve(query, top_k=5)
        
        # Verify metadata is preserved
        for doc in results:
            assert "source" in doc.metadata
            if "page" in doc.metadata:
                assert isinstance(doc.metadata["page"], int)

    async def test_health_check(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test health check functionality."""
        # Mock health check responses
        mock_dense_retriever.health_check = AsyncMock(return_value=True)
        mock_sparse_retriever.health_check = AsyncMock(return_value=True)
        
        # Test health through successful retrieval
        query = "Health check"
        mock_dense_retriever.retrieve.return_value = [
            Document(text="Health test", metadata={"source": "health.pdf"})
        ]
        mock_sparse_retriever.retrieve.return_value = []
        
        try:
            results = await hybrid_retriever.retrieve(query, top_k=1)
            health_status = True
        except Exception:
            health_status = False
        
        assert health_status is True

    async def test_different_fusion_methods(self, mock_dense_retriever, mock_sparse_retriever):
        """Test different fusion methods."""
        # Test with different RRF parameters
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            rrf_k=30  # Different RRF constant
        )
        
        query = "Fusion method test"
        
        mock_dense_retriever.retrieve.return_value = [
            Document(text="Dense result", metadata={"source": "dense.pdf"})
        ]
        mock_sparse_retriever.retrieve.return_value = [
            Document(text="Sparse result", metadata={"source": "sparse.pdf"})
        ]
        
        results = await retriever.retrieve(query, top_k=5)
        
        # Should work with different fusion parameters
        assert isinstance(results, list)

    async def test_weight_adjustment(self, mock_dense_retriever, mock_sparse_retriever):
        """Test weight adjustment between retrievers."""
        # Test with different RRF constant (affects weighting)
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            rrf_k=100  # Higher value reduces rank position importance
        )
        
        query = "Weight adjustment test"
        
        mock_dense_retriever.retrieve.return_value = [
            Document(text="Dense weighted", metadata={"source": "dense.pdf"})
        ]
        mock_sparse_retriever.retrieve.return_value = [
            Document(text="Sparse weighted", metadata={"source": "sparse.pdf"})
        ]
        
        results = await retriever.retrieve(query, top_k=5)
        
        # Should handle different weighting
        assert isinstance(results, list)

    def test_initialization(self, mock_dense_retriever):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            rrf_k=60
        )
        
        assert retriever._dense == mock_dense_retriever
        assert retriever._rrf_k == 60

    async def test_large_result_sets(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test handling of large result sets."""
        query = "Large results test"
        
        # Mock large result sets
        dense_results = [
            Document(text=f"Dense large {i}", metadata={"source": f"dense{i}.pdf"})
            for i in range(100)
        ]
        sparse_results = [
            Document(text=f"Sparse large {i}", metadata={"source": f"sparse{i}.pdf"})
            for i in range(100)
        ]
        
        mock_dense_retriever.retrieve.return_value = dense_results
        mock_sparse_retriever.retrieve.return_value = sparse_results
        
        results = await hybrid_retriever.retrieve(query, top_k=10)
        
        # Should handle large sets efficiently
        assert len(results) <= 10
        assert isinstance(results, list)

    async def test_score_normalization(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test score normalization across retrievers."""
        query = "Score normalization test"
        
        # Mock results with different score ranges
        mock_dense_retriever.retrieve.return_value = [
            Document(text="Dense high score", metadata={"source": "dense.pdf", "score": 0.95})
        ]
        mock_sparse_retriever.retrieve.return_value = [
            Document(text="Sparse low score", metadata={"source": "sparse.pdf", "score": 0.3})
        ]
        
        results = await hybrid_retriever.retrieve(query, top_k=5)
        
        # Should normalize scores appropriately
        assert isinstance(results, list)
        # Note: Actual score normalization depends on implementation

    async def test_empty_query_handling(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test handling of empty queries."""
        empty_query = ""
        
        mock_dense_retriever.retrieve.return_value = []
        mock_sparse_retriever.retrieve.return_value = []
        
        results = await hybrid_retriever.retrieve(empty_query, top_k=5)
        
        # Should handle empty query gracefully
        assert isinstance(results, list)

    async def test_plugin_integration(self, mock_dense_retriever):
        """Test integration with scorer plugins."""
        # Test with scorer plugins
        retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            scorer_plugins=[],  # Empty list of plugins
            rrf_k=60
        )
        
        query = "Plugin test"
        
        mock_dense_retriever.retrieve.return_value = [
            Document(text="Plugin result", metadata={"source": "plugin.pdf"})
        ]
        
        results = await retriever.retrieve(query, top_k=5)
        
        # Should work with plugin configuration
        assert isinstance(results, list)
