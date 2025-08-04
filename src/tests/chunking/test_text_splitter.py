
import pytest

from src.core.chunking.text_splitter import ChunkingStrategy, TextChunker
from src.core.models.document import Document

# Sample text for testing
SAMPLE_TEXT = (
    "This is the first sentence. This is the second sentence. "
    "Here comes the third sentence. The fourth sentence is here. "
    "Finally, the fifth sentence."
)

@pytest.fixture
def sample_doc() -> Document:
    """Provides a sample document for testing."""
    return Document(text=SAMPLE_TEXT, metadata={"source": "test.txt"})

@pytest.mark.asyncio
async def test_create_recursive_chunker():
    """Verify that a recursive chunker can be created."""
    chunker = await TextChunker.create(
        strategy="recursive",
        chunk_size=50,
        chunk_overlap=10,
    )
    assert chunker.strategy == ChunkingStrategy.RECURSIVE
    assert chunker.chunk_size == 50
    assert chunker.chunk_overlap == 10

@pytest.mark.asyncio
async def test_create_fixed_chunker():
    """Verify that a fixed-size chunker can be created."""
    chunker = await TextChunker.create(
        strategy="fixed",
        chunk_size=100,
        chunk_overlap=20,
    )
    assert chunker.strategy == ChunkingStrategy.FIXED
    assert chunker.chunk_size == 100
    assert chunker.chunk_overlap == 20

@pytest.mark.asyncio
async def test_recursive_splitting(sample_doc: Document):
    """Test the recursive splitting strategy."""
    chunker = await TextChunker.create("recursive", 80, 20)
    chunks = chunker.split([sample_doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.text) <= 80
        assert chunk.metadata["parent_id"] == sample_doc.id

    # Check that there is some overlap between sequential chunks (heuristic)
    # Validate overlap at word level (last 3 words of previous == first 3 words of next)
    tail_words = chunks[0].text.split()[-3:]
    head_words = chunks[1].text.split()[:3]
    assert tail_words == head_words

@pytest.mark.asyncio
async def test_fixed_splitting(sample_doc: Document):
    """Test the fixed-size splitting strategy."""
    chunker = await TextChunker.create("fixed", 60, 15)
    chunks = chunker.split([sample_doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.text) <= 60

@pytest.mark.asyncio
async def test_empty_document():
    """Test that an empty document produces no chunks."""
    chunker = await TextChunker.create("recursive", 100, 20)
    with pytest.raises(ValueError):
        _ = Document(text="", metadata={"source": "empty.txt"})

@pytest.mark.asyncio
async def test_stats_tracking(sample_doc: Document):
    """Verify that statistics are correctly tracked."""
    chunker = await TextChunker.create("recursive", 100, 20)

    # Check initial stats
    stats = chunker.stats
    assert stats["documents_processed"] == 0
    assert stats["total_chunks"] == 0

    # Process a document
    chunks = chunker.split([sample_doc])

    # Check updated stats
    stats = chunker.stats
    assert stats["documents_processed"] == 1
    assert stats["total_chunks"] == len(chunks)
    assert stats["avg_chunk_size"] > 0

    # Reset stats
    chunker.reset_stats()
    stats = chunker.stats
    assert stats["documents_processed"] == 0
    assert stats["total_chunks"] == 0
