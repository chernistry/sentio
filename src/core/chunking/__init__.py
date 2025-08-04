"""Text chunking utilities for document processing.

This module provides tools for splitting documents into manageable chunks
for processing by LLMs and vector databases.
"""

from src.core.chunking.text_splitter import (
    ChunkingError,
    ChunkingStrategy,
    TextChunker,
)

__all__ = [
    "ChunkingError",
    "ChunkingStrategy",
    "TextChunker",
]
