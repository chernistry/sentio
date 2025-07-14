"""
Unit tests for the TextChunker.
"""
import pytest
from llama_index.core.schema import Document

from root.src.core.tasks.chunking import TextChunker

pytestmark = pytest.mark.unit


def test_chunker_short_text_is_one_chunk(chunker: TextChunker):
    """
    Tests that text shorter than the chunk size results in a single chunk.
    This test passed before and is a good basic check.
    """
    short_text = "This is a sentence that is shorter than the chunk size."
    document = Document(text=short_text)
    chunks = chunker.split([document])

    assert len(chunks) == 1
    assert chunks[0].text == short_text


def test_long_text_is_split(chunker: TextChunker):
    """
    Tests that a text significantly longer than the chunk size is split
    into more than one chunk.
    """
    # Create text that is guaranteed to be longer than the default chunk size (512)
    long_text = "word " * 500
    document = Document(text=long_text)
    chunks = chunker.split([document])

    assert len(chunks) > 1, "Long text should be split into multiple chunks"


def test_chunks_are_within_size_limits():
    """
    Tests that all generated chunks are within the allowed size limit.
    The chunker might create chunks up to 1.5x the configured chunk_size
    due to internal validation and splitting logic.
    """
    text = " ".join([f"word{i:02d}" for i in range(500)])
    chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    
    # The effective max size can be larger due to the validation logic
    effective_max_size = chunker.max_chunk_size or (chunker.chunk_size * 1.5)

    document = Document(text=text)
    chunks = chunker.split([document])

    assert len(chunks) > 0, "Chunking should produce at least one chunk"

    for i, chunk in enumerate(chunks):
        assert len(chunk.text) <= effective_max_size, \
            f"Chunk {i} with size {len(chunk.text)} exceeds the effective max size of {effective_max_size}" 