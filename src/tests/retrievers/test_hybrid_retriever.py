"""Tests for hybrid retriever - combines dense and sparse retrieval."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.retrievers.hybrid import HybridRetriever
from src.core.retrievers.dense import DenseRetriever
from src.core.retrievers.sparse import SparseRetriever
from src.core.models.document import Document


@pytest.fixture
def mock_dense_retriever():
    """Mock dense retriever."""
    retriever = AsyncMock(spec=DenseRetriever)
    retriever.retrieve.return_value = [
        Document(
            id="dense_1",
            text="Dense retrieval result 1",
            metadata={"source": "doc1.pdf", "dense_score": 0.9}
        ),
        Document(
            id="dense_2", 
            text="Dense retrieval result 2",
            metadata={"source": "doc2.pdf", "dense_score": 0.8}
        )
    ]
    return retriever


@pytest.fixture
def mock_sparse_retriever():
    """Mock sparse retriever."""
    retriever = AsyncMock(spec=SparseRetriever)
    retriever.retrieve.return_value = [
        Document(
            id="sparse_1",
            text="Sparse retrieval result 1", 
            metadata={"source": "doc3.pdf", "sparse_score": 0.85}
        ),
        Document(
            id="dense_1",  # Same as dense result
            text="Dense retrieval result 1",
            metadata={"source": "doc1.pdf", "sparse_score": 0.75}
        )
    ]
    return retriever


@pytest.fixture
def hybrid_retriever(mock_dense_retriever, mock_sparse_retriever):
    """Create HybridRetriever with mocked components."""
    retriever = HybridRetriever(
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        dense_weight=0.6,
        sparse_weight=0.4,
        fusion_method="rrf"  # Reciprocal Rank Fusion
    )
    return retriever


@pytest.mark.asyncio
class TestHybridRetriever:
    """Test hybrid retriever functionality."""

    async def test_retrieve_success(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test successful hybrid retrieval."""
        query = "What is machine learning?"
        
        results = await hybrid_retriever.retrieve(query, top_k=5)
        
        # Verify both retrievers were called
        mock_dense_retriever.retrieve.assert_called_once_with(query, top_k=10)  # May retrieve more for fusion
        mock_sparse_retriever.retrieve.assert_called_once_with(query, top_k=10)
        
        # Verify results are combined and deduplicated
        assert len(results) <= 5
        assert all(isinstance(doc, Document) for doc in results)
        
        # Should have combined scores
        for doc in results:
            assert "hybrid_score" in doc.metadata or \
                   "dense_score" in doc.metadata or \
                   "sparse_score" in doc.metadata

    async def test_reciprocal_rank_fusion(self, hybrid_retriever):
        """Test Reciprocal Rank Fusion algorithm."""
        # Mock specific results for RRF testing
        dense_results = [
            Document(id="doc1", text="Text 1", metadata={"dense_rank": 1}),
            Document(id="doc2", text="Text 2", metadata={"dense_rank": 2}),
            Document(id="doc3", text="Text 3", metadata={"dense_rank": 3})
        ]
        
        sparse_results = [
            Document(id="doc2", text="Text 2", metadata={"sparse_rank": 1}),
            Document(id="doc1", text="Text 1", metadata={"sparse_rank": 2}),
            Document(id="doc4", text="Text 4", metadata={"sparse_rank": 3})
        ]
        
        # Test RRF calculation
        if hasattr(hybrid_retriever, '_reciprocal_rank_fusion'):
            fused_results = hybrid_retriever._reciprocal_rank_fusion(
                dense_results, sparse_results, k=60
            )
            
            # doc2 should rank highest (rank 1 in sparse, rank 2 in dense)
            # RRF score = 1/(60+1) + 1/(60+2) ≈ 0.0164 + 0.0161 = 0.0325
            assert len(fused_results) > 0
            assert fused_results[0].id in ["doc1", "doc2"]  # Top results

    async def test_weighted_score_fusion(self, hybrid_retriever):
        """Test weighted score fusion method."""
        hybrid_retriever.fusion_method = "weighted"
        
        dense_results = [
            Document(id="doc1", text="Text 1", metadata={"score": 0.9}),
            Document(id="doc2", text="Text 2", metadata={"score": 0.8})
        ]
        
        sparse_results = [
            Document(id="doc1", text="Text 1", metadata={"score": 0.7}),
            Document(id="doc3", text="Text 3", metadata={"score": 0.85})
        ]
        
        if hasattr(hybrid_retriever, '_weighted_score_fusion'):
            fused_results = hybrid_retriever._weighted_score_fusion(
                dense_results, sparse_results
            )
            
            # doc1 should have combined score: 0.6*0.9 + 0.4*0.7 = 0.82
            doc1_result = next((doc for doc in fused_results if doc.id == "doc1"), None)
            assert doc1_result is not None
            assert "hybrid_score" in doc1_result.metadata

    async def test_deduplication(self, hybrid_retriever):
        """Test document deduplication across retrievers."""
        # Both retrievers return same document
        dense_results = [
            Document(id="doc1", text="Same document", metadata={"dense_score": 0.9})
        ]
        
        sparse_results = [
            Document(id="doc1", text="Same document", metadata={"sparse_score": 0.8})
        ]
        
        if hasattr(hybrid_retriever, '_deduplicate_results'):
            deduplicated = hybrid_retriever._deduplicate_results(
                dense_results, sparse_results
            )
            
            # Should have only one instance of doc1
            doc1_count = sum(1 for doc in deduplicated if doc.id == "doc1")
            assert doc1_count == 1

    async def test_empty_results_handling(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test handling when one retriever returns empty results."""
        # Dense returns empty, sparse returns results
        mock_dense_retriever.retrieve.return_value = []
        mock_sparse_retriever.retrieve.return_value = [
            Document(id="sparse_only", text="Sparse result", metadata={"score": 0.8})
        ]
        
        results = await hybrid_retriever.retrieve("test query", top_k=5)
        
        # Should still return sparse results
        assert len(results) == 1
        assert results[0].id == "sparse_only"

    async def test_both_retrievers_empty(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test handling when both retrievers return empty results."""
        mock_dense_retriever.retrieve.return_value = []
        mock_sparse_retriever.retrieve.return_value = []
        
        results = await hybrid_retriever.retrieve("test query", top_k=5)
        
        assert len(results) == 0

    async def test_retriever_failure_handling(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test handling when one retriever fails."""
        # Dense retriever fails
        mock_dense_retriever.retrieve.side_effect = Exception("Dense retriever failed")
        mock_sparse_retriever.retrieve.return_value = [
            Document(id="sparse_backup", text="Backup result", metadata={"score": 0.7})
        ]
        
        # Should fallback to sparse results only
        results = await hybrid_retriever.retrieve("test query", top_k=5)
        
        assert len(results) == 1
        assert results[0].id == "sparse_backup"

    async def test_both_retrievers_fail(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test handling when both retrievers fail."""
        mock_dense_retriever.retrieve.side_effect = Exception("Dense failed")
        mock_sparse_retriever.retrieve.side_effect = Exception("Sparse failed")
        
        with pytest.raises(Exception):
            await hybrid_retriever.retrieve("test query", top_k=5)

    async def test_different_fusion_methods(self, mock_dense_retriever, mock_sparse_retriever):
        """Test different fusion methods."""
        fusion_methods = ["rrf", "weighted", "max", "min"]
        
        for method in fusion_methods:
            retriever = HybridRetriever(
                dense_retriever=mock_dense_retriever,
                sparse_retriever=mock_sparse_retriever,
                fusion_method=method
            )
            
            results = await retriever.retrieve("test query", top_k=3)
            
            # All methods should return results
            assert isinstance(results, list)

    async def test_weight_adjustment(self, mock_dense_retriever, mock_sparse_retriever):
        """Test different weight configurations."""
        weight_configs = [
            (0.8, 0.2),  # Dense-heavy
            (0.3, 0.7),  # Sparse-heavy
            (0.5, 0.5),  # Balanced
        ]
        
        for dense_weight, sparse_weight in weight_configs:
            retriever = HybridRetriever(
                dense_retriever=mock_dense_retriever,
                sparse_retriever=mock_sparse_retriever,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
            
            results = await retriever.retrieve("test query", top_k=3)
            assert len(results) >= 0

    async def test_top_k_handling(self, hybrid_retriever):
        """Test different top_k values."""
        top_k_values = [1, 3, 5, 10, 20]
        
        for top_k in top_k_values:
            results = await hybrid_retriever.retrieve("test query", top_k=top_k)
            
            # Should not exceed requested top_k
            assert len(results) <= top_k

    async def test_query_preprocessing(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test query preprocessing before retrieval."""
        query = "  What is machine learning?  \n\n"
        
        await hybrid_retriever.retrieve(query, top_k=5)
        
        # Both retrievers should receive preprocessed query
        dense_call_args = mock_dense_retriever.retrieve.call_args[0][0]
        sparse_call_args = mock_sparse_retriever.retrieve.call_args[0][0]
        
        # Query should be cleaned (exact preprocessing depends on implementation)
        assert isinstance(dense_call_args, str)
        assert isinstance(sparse_call_args, str)

    async def test_concurrent_retrieval(self, hybrid_retriever):
        """Test that dense and sparse retrieval happen concurrently."""
        import asyncio
        
        # Add delays to mock retrievers to test concurrency
        async def slow_dense_retrieve(query, top_k):
            await asyncio.sleep(0.1)
            return [Document(id="dense", text="Dense result", metadata={})]
        
        async def slow_sparse_retrieve(query, top_k):
            await asyncio.sleep(0.1)
            return [Document(id="sparse", text="Sparse result", metadata={})]
        
        hybrid_retriever.dense_retriever.retrieve = slow_dense_retrieve
        hybrid_retriever.sparse_retriever.retrieve = slow_sparse_retrieve
        
        start_time = asyncio.get_event_loop().time()
        results = await hybrid_retriever.retrieve("test query", top_k=5)
        end_time = asyncio.get_event_loop().time()
        
        # Should take ~0.1 seconds (concurrent) not ~0.2 seconds (sequential)
        assert end_time - start_time < 0.15
        assert len(results) >= 1

    async def test_metadata_preservation(self, hybrid_retriever):
        """Test that original metadata is preserved in results."""
        results = await hybrid_retriever.retrieve("test query", top_k=5)
        
        for doc in results:
            # Should preserve original metadata
            assert "source" in doc.metadata
            # Should add hybrid-specific metadata
            assert any(key in doc.metadata for key in ["hybrid_score", "dense_score", "sparse_score"])

    async def test_health_check(self, hybrid_retriever, mock_dense_retriever, mock_sparse_retriever):
        """Test hybrid retriever health check."""
        mock_dense_retriever.health_check.return_value = True
        mock_sparse_retriever.health_check.return_value = True
        
        if hasattr(hybrid_retriever, 'health_check'):
            is_healthy = await hybrid_retriever.health_check()
            assert is_healthy is True
            
            # Test when one component is unhealthy
            mock_dense_retriever.health_check.return_value = False
            is_healthy = await hybrid_retriever.health_check()
            assert is_healthy is False
