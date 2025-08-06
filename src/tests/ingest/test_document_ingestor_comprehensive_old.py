"""Comprehensive tests for DocumentIngestor - addressing 0% coverage."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.ingest.ingest import DocumentIngestor
from src.core.models.document import Document


@pytest.fixture
def mock_chunker():
    """Mock text chunker."""
    chunker = MagicMock()
    chunker.split.return_value = [
        Document(
            text="This is chunk 1.",
            metadata={"source": "test1.txt", "chunk_index": 0},
            id="chunk-1"
        ),
        Document(
            text="This is chunk 2.",
            metadata={"source": "test1.txt", "chunk_index": 1},
            id="chunk-2"
        )
    ]
    return chunker


@pytest.fixture
def mock_embedder():
    """Mock embedder."""
    embedder = AsyncMock()
    embedder.embed_async_many.return_value = [
        [0.1, 0.2, 0.3] * 128,  # 384 dimensions
        [0.4, 0.5, 0.6] * 128
    ]
    embedder.dimension = 384
    return embedder


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = AsyncMock()
    store.add_embeddings = AsyncMock()
    store.health_check.return_value = True
    return store


@pytest.fixture
def document_ingestor():
    """Create DocumentIngestor instance."""
    return DocumentIngestor(
        collection_name="test_collection",
        chunk_size=512,
        chunk_overlap=64,
        chunking_strategy="recursive"
    )


@pytest.mark.asyncio
class TestDocumentIngestor:
    """Test DocumentIngestor functionality."""

    async def test_initialization(self, document_ingestor):
        """Test DocumentIngestor initialization."""
        assert document_ingestor.collection_name == "test_collection"
        assert document_ingestor.chunk_size == 512
        assert document_ingestor.chunk_overlap == 64
        assert document_ingestor.chunking_strategy == "recursive"
        assert document_ingestor._stats["documents_processed"] == 0

    async def test_initialize_components(self, document_ingestor):
        """Test component initialization."""
        with patch('src.core.ingest.ingest.TextChunker') as mock_chunker_class, \
             patch('src.core.ingest.ingest.get_embedder') as mock_get_embedder, \
             patch('src.core.ingest.ingest.get_vector_store') as mock_get_vector_store:
            
            mock_chunker_class.create.return_value = AsyncMock()
            mock_get_embedder.return_value = AsyncMock()
            mock_get_vector_store.return_value = AsyncMock()

            await document_ingestor.initialize()

            # Verify components were initialized
            assert document_ingestor.chunker is not None
            assert document_ingestor.embedder is not None
            assert document_ingestor.vector_store is not None

    async def test_load_documents_from_directory(self, document_ingestor):
        """Test loading documents from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test1.txt").write_text("This is test file 1.")
            (temp_path / "test2.md").write_text("# Test File 2\nThis is markdown.")
            (temp_path / "test3.pdf").write_text("PDF content")  # Mock PDF
            
            # Create subdirectory with file
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            (sub_dir / "test4.txt").write_text("Subdirectory file.")

            documents = document_ingestor._load_documents_from_directory(temp_path)

            # Should load all text files recursively
            assert len(documents) >= 3  # At least txt, md, and subdir file
            
            # Check document properties
            for doc in documents:
                assert isinstance(doc, Document)
                assert len(doc.text) > 0
                assert "source" in doc.metadata
                assert doc.id is not None

    async def test_load_single_file(self, document_ingestor):
        """Test loading a single file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("This is a test file content.")
            temp_file_path = temp_file.name

        try:
            documents = document_ingestor._load_documents_from_directory(Path(temp_file_path))
            
            assert len(documents) == 1
            assert documents[0].text == "This is a test file content."
            assert documents[0].metadata["source"] == temp_file_path
        finally:
            Path(temp_file_path).unlink()

    async def test_chunk_documents(self, document_ingestor, mock_chunker):
        """Test document chunking."""
        document_ingestor.chunker = mock_chunker
        
        documents = [
            Document(text="Long document content that needs chunking.", metadata={"source": "test.txt"})
        ]

        chunks = document_ingestor._chunk_documents(documents)

        # Verify chunker was called
        mock_chunker.split.assert_called_once_with(documents)
        
        # Verify chunks
        assert len(chunks) == 2
        assert chunks[0].text == "This is chunk 1."
        assert chunks[1].text == "This is chunk 2."

    async def test_generate_embeddings(self, document_ingestor, mock_embedder):
        """Test embedding generation."""
        document_ingestor.embedder = mock_embedder
        
        chunks = [
            Document(text="Chunk 1", metadata={}, id="chunk-1"),
            Document(text="Chunk 2", metadata={}, id="chunk-2")
        ]

        embeddings = await document_ingestor._generate_embeddings(chunks)

        # Verify embedder was called
        mock_embedder.embed_async_many.assert_called_once_with(["Chunk 1", "Chunk 2"])
        
        # Verify embeddings mapping
        assert len(embeddings) == 2
        assert "chunk-1" in embeddings
        assert "chunk-2" in embeddings
        assert len(embeddings["chunk-1"]) == 384

    async def test_store_chunks_with_embeddings(self, document_ingestor, mock_vector_store):
        """Test storing chunks with embeddings."""
        document_ingestor.vector_store = mock_vector_store
        
        chunks = [
            Document(text="Chunk 1", metadata={"source": "test.txt"}, id="chunk-1"),
            Document(text="Chunk 2", metadata={"source": "test.txt"}, id="chunk-2")
        ]
        
        embeddings = {
            "chunk-1": [0.1, 0.2, 0.3] * 128,
            "chunk-2": [0.4, 0.5, 0.6] * 128
        }

        await document_ingestor._store_chunks_with_embeddings(chunks, embeddings)

        # Verify vector store was called
        mock_vector_store.add_embeddings.assert_called_once()
        call_args = mock_vector_store.add_embeddings.call_args
        
        assert len(call_args.kwargs["texts"]) == 2
        assert len(call_args.kwargs["embeddings"]) == 2
        assert len(call_args.kwargs["metadatas"]) == 2
        assert len(call_args.kwargs["ids"]) == 2

    async def test_ingest_documents_full_pipeline(self, document_ingestor, mock_chunker, mock_embedder, mock_vector_store):
        """Test complete document ingestion pipeline."""
        # Set up mocked components
        document_ingestor.chunker = mock_chunker
        document_ingestor.embedder = mock_embedder
        document_ingestor.vector_store = mock_vector_store
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.txt").write_text("This is a test document.")

            # Mock _load_documents_from_directory to avoid file system dependency
            with patch.object(document_ingestor, '_load_documents_from_directory') as mock_load:
                mock_load.return_value = [
                    Document(text="Test document content", metadata={"source": "test.txt"})
                ]

                stats = await document_ingestor.ingest_documents(temp_path)

                # Verify all components were called
                mock_load.assert_called_once()
                mock_chunker.split.assert_called_once()
                mock_embedder.embed_async_many.assert_called_once()
                mock_vector_store.add_embeddings.assert_called_once()

                # Verify stats
                assert stats["documents_processed"] == 1
                assert stats["chunks_created"] == 2
                assert stats["embeddings_generated"] == 2
                assert stats["processing_time"] > 0

    async def test_ingest_documents_error_handling(self, document_ingestor, mock_chunker, mock_embedder, mock_vector_store):
        """Test error handling during ingestion."""
        document_ingestor.chunker = mock_chunker
        document_ingestor.embedder = mock_embedder
        document_ingestor.vector_store = mock_vector_store
        
        # Make embedder fail
        mock_embedder.embed_async_many.side_effect = Exception("Embedding failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.txt").write_text("Test content")

            with patch.object(document_ingestor, '_load_documents_from_directory') as mock_load:
                mock_load.return_value = [
                    Document(text="Test document", metadata={"source": "test.txt"})
                ]

                with pytest.raises(Exception, match="Document ingestion failed"):
                    await document_ingestor.ingest_documents(temp_path)

    async def test_health_check(self, document_ingestor, mock_vector_store, mock_embedder):
        """Test health check functionality."""
        document_ingestor.vector_store = mock_vector_store
        document_ingestor.embedder = mock_embedder
        
        # Mock health checks
        mock_vector_store.health_check.return_value = True
        mock_embedder.health_check = AsyncMock(return_value=True)

        health = await document_ingestor.health_check()

        assert health["status"] == "healthy"
        assert health["components"]["vector_store"] is True
        assert health["components"]["embedder"] is True

    async def test_health_check_unhealthy(self, document_ingestor, mock_vector_store, mock_embedder):
        """Test health check when components are unhealthy."""
        document_ingestor.vector_store = mock_vector_store
        document_ingestor.embedder = mock_embedder
        
        # Make vector store unhealthy
        mock_vector_store.health_check.return_value = False
        mock_embedder.health_check = AsyncMock(return_value=True)

        health = await document_ingestor.health_check()

        assert health["status"] == "unhealthy"
        assert health["components"]["vector_store"] is False
        assert health["components"]["embedder"] is True

    async def test_get_stats(self, document_ingestor):
        """Test statistics retrieval."""
        # Update some stats
        document_ingestor._stats["documents_processed"] = 10
        document_ingestor._stats["chunks_created"] = 50
        document_ingestor._stats["embeddings_generated"] = 50

        stats = document_ingestor.stats

        assert stats["documents_processed"] == 10
        assert stats["chunks_created"] == 50
        assert stats["embeddings_generated"] == 50

    async def test_supported_file_types(self, document_ingestor):
        """Test that supported file types are handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files of different types
            (temp_path / "test.txt").write_text("Text file")
            (temp_path / "test.md").write_text("# Markdown file")
            (temp_path / "test.py").write_text("# Python file")
            (temp_path / "test.json").write_text('{"key": "value"}')
            (temp_path / "test.yaml").write_text("key: value")
            (temp_path / "test.csv").write_text("col1,col2\nval1,val2")
            
            # Create unsupported file
            (temp_path / "test.bin").write_bytes(b'\x00\x01\x02')

            documents = document_ingestor._load_documents_from_directory(temp_path)

            # Should load supported text files, skip binary
            supported_extensions = {'.txt', '.md', '.py', '.json', '.yaml', '.csv'}
            loaded_extensions = {Path(doc.metadata["source"]).suffix for doc in documents}
            
            # All loaded files should be supported types
            assert loaded_extensions.issubset(supported_extensions)
            assert '.bin' not in loaded_extensions

    async def test_concurrent_processing(self, document_ingestor, mock_chunker, mock_embedder, mock_vector_store):
        """Test concurrent document processing."""
        document_ingestor.chunker = mock_chunker
        document_ingestor.embedder = mock_embedder
        document_ingestor.vector_store = mock_vector_store

        # Create multiple temporary directories
        temp_dirs = []
        for i in range(3):
            temp_dir = tempfile.mkdtemp()
            temp_dirs.append(temp_dir)
            Path(temp_dir, f"test{i}.txt").write_text(f"Test document {i}")

        try:
            # Process multiple directories concurrently
            tasks = [
                document_ingestor.ingest_documents(Path(temp_dir))
                for temp_dir in temp_dirs
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should complete successfully
            assert len(results) == 3
            assert all(not isinstance(r, Exception) for r in results)

        finally:
            # Cleanup
            for temp_dir in temp_dirs:
                import shutil
                shutil.rmtree(temp_dir)

    async def test_large_document_handling(self, document_ingestor, mock_chunker, mock_embedder, mock_vector_store):
        """Test handling of large documents."""
        document_ingestor.chunker = mock_chunker
        document_ingestor.embedder = mock_embedder
        document_ingestor.vector_store = mock_vector_store

        # Create large document
        large_content = "This is a large document. " * 1000  # ~25KB
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "large.txt").write_text(large_content)

            with patch.object(document_ingestor, '_load_documents_from_directory') as mock_load:
                mock_load.return_value = [
                    Document(text=large_content, metadata={"source": "large.txt"})
                ]

                stats = await document_ingestor.ingest_documents(temp_path)

                # Should handle large documents without issues
                assert stats["documents_processed"] == 1
                assert stats["chunks_created"] == 2  # From mock
                assert stats["processing_time"] > 0

    async def test_cleanup_on_failure(self, document_ingestor, mock_chunker, mock_embedder, mock_vector_store):
        """Test cleanup when ingestion fails."""
        document_ingestor.chunker = mock_chunker
        document_ingestor.embedder = mock_embedder
        document_ingestor.vector_store = mock_vector_store

        # Make vector store fail
        mock_vector_store.add_embeddings.side_effect = Exception("Storage failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.txt").write_text("Test content")

            with patch.object(document_ingestor, '_load_documents_from_directory') as mock_load:
                mock_load.return_value = [
                    Document(text="Test document", metadata={"source": "test.txt"})
                ]

                with pytest.raises(Exception, match="Document ingestion failed"):
                    await document_ingestor.ingest_documents(temp_path)

                # Stats should reflect the failure
                assert document_ingestor._stats["errors"] > 0
