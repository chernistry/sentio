"""Tests for Qdrant vector store."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.vector_store.qdrant_store import QdrantStore
from src.core.models.document import Document


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = MagicMock()
    
    # Mock search response
    mock_search_result = MagicMock()
    mock_search_result.id = "doc1"
    mock_search_result.score = 0.95
    mock_search_result.payload = {
        "text": "Test document content",
        "metadata": {"source": "test.pdf", "page": 1}
    }
    
    client.search.return_value = [mock_search_result]
    client.count.return_value.count = 100
    client.get_collection.return_value = MagicMock(status="green")
    
    return client


@pytest.fixture
def qdrant_store(mock_qdrant_client):
    """Create QdrantStore with mocked client."""
    with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client), \
         patch.object(QdrantStore, '_bootstrap_collection'):
        store = QdrantStore(
            url="http://localhost:6333",
            collection_name="test_collection",
            vector_size=384
        )
        store._client = mock_qdrant_client
        return store


@pytest.mark.asyncio
class TestQdrantStore:
    """Test QdrantStore functionality."""

    async def test_search_success(self, qdrant_store, mock_qdrant_client):
        """Test successful vector search."""
        query_vector = [0.1, 0.2, 0.3] * 128  # 384 dimensions
        
        results = await qdrant_store.search(
            query_vector=query_vector,
            top_k=5,
            filter_dict=None
        )

        # Verify client was called correctly
        mock_qdrant_client.search.assert_called_once()
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["query_vector"] == query_vector
        assert call_args.kwargs["limit"] == 5

        # Verify results
        assert len(results) == 1
        assert results[0].id == "doc1"
        assert results[0].text == "Test document content"
        assert results[0].metadata["source"] == "test.pdf"

    async def test_search_with_filter(self, qdrant_store, mock_qdrant_client):
        """Test vector search with metadata filter."""
        query_vector = [0.1] * 384
        filter_dict = {"source": "specific.pdf"}

        await qdrant_store.search(
            query_vector=query_vector,
            top_k=3,
            filter_dict=filter_dict
        )

        # Verify filter was applied
        call_args = mock_qdrant_client.search.call_args
        assert "query_filter" in call_args.kwargs
        # Filter structure depends on implementation

    async def test_search_empty_results(self, qdrant_store, mock_qdrant_client):
        """Test search with no results."""
        mock_qdrant_client.search.return_value = []
        
        results = await qdrant_store.search(
            query_vector=[0.1] * 384,
            top_k=5,
            filter_dict=None
        )

        assert len(results) == 0

    async def test_add_documents_success(self, qdrant_store, mock_qdrant_client):
        """Test successful document addition."""
        documents = [
            Document(
                id="doc1",
                text="First document",
                metadata={"source": "doc1.pdf"}
            ),
            Document(
                id="doc2", 
                text="Second document",
                metadata={"source": "doc2.pdf"}
            )
        ]
        
        embeddings = [
            [0.1] * 384,
            [0.2] * 384
        ]

        await qdrant_store.add_documents(documents, embeddings)

        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args
        
        assert call_args.kwargs["collection_name"] == "test_collection"
        points = call_args.kwargs["points"]
        assert len(points) == 2

    async def test_add_documents_mismatched_lengths(self, qdrant_store):
        """Test error when documents and embeddings lengths don't match."""
        documents = [Document(id="doc1", text="Test", metadata={})]
        embeddings = [[0.1] * 384, [0.2] * 384]  # More embeddings than documents

        with pytest.raises(ValueError, match="Number of documents.*embeddings"):
            await qdrant_store.add_documents(documents, embeddings)

    async def test_delete_documents(self, qdrant_store, mock_qdrant_client):
        """Test document deletion."""
        document_ids = ["doc1", "doc2", "doc3"]

        await qdrant_store.delete_documents(document_ids)

        # Verify delete was called
        mock_qdrant_client.delete.assert_called_once()
        call_args = mock_qdrant_client.delete.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"

    async def test_get_collection_info(self, qdrant_store, mock_qdrant_client):
        """Test getting collection information."""
        info = await qdrant_store.get_collection_info()

        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")
        assert "status" in info

    async def test_health_check_healthy(self, qdrant_store, mock_qdrant_client):
        """Test health check when Qdrant is healthy."""
        mock_qdrant_client.get_collections.return_value = MagicMock()

        is_healthy = await qdrant_store.health_check()

        assert is_healthy is True
        mock_qdrant_client.get_collections.assert_called_once()

    async def test_health_check_unhealthy(self, qdrant_store, mock_qdrant_client):
        """Test health check when Qdrant is unhealthy."""
        mock_qdrant_client.get_collections.side_effect = Exception("Connection failed")

        is_healthy = await qdrant_store.health_check()

        assert is_healthy is False

    async def test_create_collection(self, qdrant_store, mock_qdrant_client):
        """Test collection creation."""
        await qdrant_store.create_collection(
            collection_name="new_collection",
            vector_size=512
        )

        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "new_collection"

    async def test_delete_collection(self, qdrant_store, mock_qdrant_client):
        """Test collection deletion."""
        await qdrant_store.delete_collection("test_collection")

        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")

    async def test_collection_exists(self, qdrant_store, mock_qdrant_client):
        """Test checking if collection exists."""
        # Collection exists
        mock_qdrant_client.get_collection.return_value = MagicMock()
        exists = await qdrant_store.collection_exists("test_collection")
        assert exists is True

        # Collection doesn't exist
        mock_qdrant_client.get_collection.side_effect = Exception("Not found")
        exists = await qdrant_store.collection_exists("nonexistent_collection")
        assert exists is False

    async def test_get_document_count(self, qdrant_store, mock_qdrant_client):
        """Test getting document count."""
        mock_qdrant_client.count.return_value.count = 150

        count = await qdrant_store.get_document_count()

        assert count == 150
        mock_qdrant_client.count.assert_called_once_with("test_collection")

    async def test_batch_operations(self, qdrant_store, mock_qdrant_client):
        """Test batch operations for large document sets."""
        # Create large number of documents
        documents = [
            Document(
                id=f"doc{i}",
                text=f"Document {i}",
                metadata={"source": f"doc{i}.pdf"}
            )
            for i in range(1000)
        ]
        embeddings = [[0.1] * 384 for _ in range(1000)]

        await qdrant_store.add_documents(documents, embeddings)

        # Should handle batching internally
        assert mock_qdrant_client.upsert.call_count >= 1

    async def test_connection_retry(self, qdrant_store, mock_qdrant_client):
        """Test connection retry logic."""
        # First call fails, second succeeds
        mock_qdrant_client.search.side_effect = [
            Exception("Connection timeout"),
            [MagicMock(id="doc1", score=0.9, payload={"text": "Test"})]
        ]

        # If retry logic is implemented
        if hasattr(qdrant_store, '_retry_on_connection_error'):
            results = await qdrant_store.search([0.1] * 384, top_k=5)
            assert len(results) == 1
        else:
            with pytest.raises(Exception, match="Connection timeout"):
                await qdrant_store.search([0.1] * 384, top_k=5)

    async def test_vector_dimension_validation(self, qdrant_store):
        """Test vector dimension validation."""
        # Wrong dimension vector
        wrong_vector = [0.1] * 256  # Should be 384

        with pytest.raises(ValueError, match="Vector dimension"):
            await qdrant_store.search(wrong_vector, top_k=5)

    async def test_concurrent_operations(self, qdrant_store, mock_qdrant_client):
        """Test concurrent search operations."""
        import asyncio

        query_vectors = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384
        ]

        # Run concurrent searches
        tasks = [
            qdrant_store.search(vector, top_k=5)
            for vector in query_vectors
        ]

        results = await asyncio.gather(*tasks)

        # All searches should complete
        assert len(results) == 3
        assert mock_qdrant_client.search.call_count == 3

    async def test_metadata_filtering_complex(self, qdrant_store, mock_qdrant_client):
        """Test complex metadata filtering."""
        complex_filter = {
            "source": {"$in": ["doc1.pdf", "doc2.pdf"]},
            "page": {"$gte": 5},
            "category": "technical"
        }

        await qdrant_store.search(
            query_vector=[0.1] * 384,
            top_k=10,
            filter_dict=complex_filter
        )

        # Verify complex filter was processed
        mock_qdrant_client.search.assert_called_once()
        call_args = mock_qdrant_client.search.call_args
        assert "query_filter" in call_args.kwargs
