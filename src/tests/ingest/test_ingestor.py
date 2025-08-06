"""Tests for the DocumentIngestor class.

These tests verify the functionality of the document ingestion pipeline.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.ingest.ingest import DocumentIngestor
from src.core.models.document import Document


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        Document(
            text="This is a test document 1.",
            metadata={"source": "test1.txt"}
        ),
        Document(
            text="This is a test document 2.",
            metadata={"source": "test2.txt"}
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Fixture providing sample document chunks for testing."""
    return [
        Document(
            text="This is a test chunk 1.",
            metadata={"source": "test1.txt", "chunk_index": 0}
        ),
        Document(
            text="This is a test chunk 2.",
            metadata={"source": "test1.txt", "chunk_index": 1}
        ),
        Document(
            text="This is a test chunk 3.",
            metadata={"source": "test2.txt", "chunk_index": 0}
        ),
    ]


@pytest.fixture
def sample_embeddings():
    """Fixture providing sample embeddings for testing."""
    return {
        "doc-id-1": [0.1, 0.2, 0.3],
        "doc-id-2": [0.4, 0.5, 0.6],
        "doc-id-3": [0.7, 0.8, 0.9],
    }


@pytest.fixture
def mock_chunker():
    """Fixture providing a mock TextChunker."""
    chunker = MagicMock()
    chunker.split.return_value = [
        Document(
            text="This is a test chunk 1.",
            metadata={"source": "test1.txt", "chunk_index": 0}
        ),
        Document(
            text="This is a test chunk 2.",
            metadata={"source": "test1.txt", "chunk_index": 1}
        ),
        Document(
            text="This is a test chunk 3.",
            metadata={"source": "test2.txt", "chunk_index": 0}
        ),
    ]
    return chunker


@pytest.fixture
def mock_embedder():
    """Fixture providing a mock Embedder."""
    embedder = MagicMock()
    embedder.dimension = 3
    # Use AsyncMock for asynchronous methods
    embedder.embed_async_many = AsyncMock(return_value=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ])
    return embedder


@pytest.fixture
def mock_vector_store():
    """Fixture providing a mock VectorStore."""
    vector_store = MagicMock()
    vector_store._client = MagicMock()
    vector_store.health_check = MagicMock(return_value=True)
    # Make add_embeddings async
    vector_store.add_embeddings = AsyncMock()
    return vector_store


@pytest.mark.asyncio
async def test_ingestor_initialization():
    """Test DocumentIngestor initialization."""
    ingestor = DocumentIngestor(
        collection_name="test_collection",
        chunk_size=256,
        chunk_overlap=32,
        chunking_strategy="recursive",
        embedder_name="jina",
        vector_store_name="qdrant",
    )

    assert ingestor.collection_name == "test_collection"
    assert ingestor.chunk_size == 256
    assert ingestor.chunk_overlap == 32
    assert ingestor.chunking_strategy == "recursive"
    assert ingestor.embedder_name == "jina"
    assert ingestor.vector_store_name == "qdrant"

    # Components should be None before initialization
    assert ingestor.chunker is None
    assert ingestor.embedder is None
    assert ingestor.vector_store is None


@pytest.mark.asyncio
async def test_load_documents_from_directory():
    """Test loading documents from a directory."""
    ingestor = DocumentIngestor()

    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_file1 = Path(temp_dir) / "test1.txt"
        test_file2 = Path(temp_dir) / "test2.txt"

        test_file1.write_text("This is test file 1.")
        test_file2.write_text("This is test file 2.")

        # Load documents
        documents = ingestor._load_documents_from_directory(Path(temp_dir))

        # Check results
        assert len(documents) == 2
        assert any(d.text == "This is test file 1." for d in documents)
        assert any(d.text == "This is test file 2." for d in documents)

        # Check stats
        assert ingestor.stats["documents_processed"] == 2
        assert ingestor.stats["bytes_processed"] > 0


@pytest.mark.asyncio
async def test_ingest_documents_pipeline(
    mock_chunker, mock_embedder, mock_vector_store, sample_documents
):
    """Test the complete document ingestion pipeline."""
    ingestor = DocumentIngestor()

    # Set mocked components
    ingestor.chunker = mock_chunker
    ingestor.embedder = mock_embedder
    ingestor.vector_store = mock_vector_store

    # Mock _load_documents_from_directory
    ingestor._load_documents_from_directory = MagicMock(return_value=sample_documents)

    # Manually set the documents_processed value since we're mocking the method that sets it
    ingestor._stats["documents_processed"] = len(sample_documents)

    # Run ingestion
    with tempfile.TemporaryDirectory() as temp_dir:
        stats = await ingestor.ingest_documents(temp_dir)

    # Verify components were called
    ingestor._load_documents_from_directory.assert_called_once()
    mock_chunker.split.assert_called_once_with(sample_documents)
    mock_embedder.embed_async_many.assert_called_once()
    mock_vector_store.add_embeddings.assert_called_once()

    # Check stats
    assert stats["documents_processed"] == 2
    assert stats["chunks_created"] == 3
    assert stats["embeddings_generated"] == 3
