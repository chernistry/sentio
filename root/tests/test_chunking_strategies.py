import pytest
from llama_index.core.schema import Document

from root.src.core.tasks.chunking import (
    SentenceChunker,
    SemanticChunker,
    FixedChunker,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text,expected", [("Hello, world!", ["Hello", ",", "world", "!"])])
def test_smart_tokenizer_basic(text: str, expected: list[str]):
    """Ensure SentenceChunker._smart_tokenizer keeps punctuation as tokens."""
    tokens = SentenceChunker._smart_tokenizer(text)
    assert tokens == expected, "Tokenization should preserve punctuation boundaries"


def test_semantic_chunker_produces_metadata():
    """SemanticChunker must label each chunk with correct metadata."""
    sample_text = (
        "Paragraph one.\n\n"
        "Paragraph two.\n\n"
        "- Item one\n"
        "- Item two\n"
    )
    chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
    nodes = chunker.chunk_text(sample_text)

    # Ensure we get at least one chunk – depending on heuristics the
    # component may decide the entire text is already below the size
    # budget.  The important part is that chunk metadata is correct.
    assert len(nodes) >= 1

    for node in nodes:
        meta = node.metadata or {}
        assert meta.get("chunk_type") == "semantic", "chunk_type metadata missing or incorrect"
        # Ensure chunk text is not empty
        assert node.text.strip(), "Chunk text should not be empty"


def test_fixed_chunker_overlap():
    """FixedChunker must create overlapping chunks respecting chunk_overlap."""
    words = [f"w{i:02d}" for i in range(120)]  # ~120 words
    text = " ".join(words)

    chunk_size = 50  # characters
    overlap = 10     # characters
    chunker = FixedChunker(chunk_size=chunk_size, chunk_overlap=overlap)

    nodes = chunker.chunk_text(text)
    # Each chunk should be <= chunk_size
    for node in nodes:
        assert len(node.text) <= chunk_size, "Chunk exceeds configured size limit"

    # Adjacent chunks should share at least one word when overlap > 0
    if overlap:
        for first, second in zip(nodes, nodes[1:]):
            first_words = first.text.split()
            second_words = second.text.split()
            shared = set(first_words) & set(second_words)
            assert shared, "Expected overlapping words between consecutive chunks" 