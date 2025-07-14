import pytest
from llama_index.core.schema import Document

from root.src.core.tasks.chunking import TextChunker, ChunkingStrategy

pytestmark = pytest.mark.unit


def test_text_chunker_split_code_and_table():
    """TextChunker should split documents containing code blocks and tables."""
    sample_text = (
        "Here is a Python function::\n\n"
        "```python\n"
        "def foo():\n"
        "    return 42\n"
        "```\n\n"
        "| Col1 | Col2 |\n"
        "|------|------|\n"
        "|  A   |  B   |\n"
    )
    doc = Document(text=sample_text, metadata={"source": "unit_test"})

    chunker = TextChunker(
        strategy=ChunkingStrategy.HYBRID,
        chunk_size=64,
        chunk_overlap=8,
        min_chunk_size=10,
    )

    nodes = chunker.split([doc])

    # At least one valid chunk must be produced.
    assert nodes, "Expected non-empty chunk list"

    # All nodes should have essential metadata and non-blank text.
    for node in nodes:
        md = node.metadata or {}
        assert node.text.strip(), "Chunk text is unexpectedly empty"
        assert md.get("chunking_strategy") in {s.value for s in ChunkingStrategy}
        assert md.get("document_id") == doc.doc_id

    # Stats should accurately reflect processing.
    stats = chunker.get_stats()
    assert stats["documents_processed"] == 1
    assert stats["total_chunks"] == len(nodes) 