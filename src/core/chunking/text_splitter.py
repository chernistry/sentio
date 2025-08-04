"""Streamlined, LangChain-driven text chunking.

This module provides a simplified TextChunker that relies directly on
production-ready splitters from the LangChain ecosystem. It abstracts away
the complexities of chunking behind a clean, configurable interface.
"""

import logging
from enum import Enum
from typing import Any

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.core.models.document import Document

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Defines the available chunking strategies."""

    RECURSIVE = "recursive"
    FIXED = "fixed"


class ChunkingError(Exception):
    """Custom exception for errors during the chunking process."""


class TextChunker:
    """A unified text chunker powered by LangChain splitters.

    This class provides a simple `split` method to chunk documents
    based on a selected strategy, size, and overlap.
    """

    __slots__ = (
        "_splitter",
        "_stats",
        "chunk_overlap",
        "chunk_size",
        "strategy",
    )

    def __init__(
        self,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """Initialize the TextChunker with a specific strategy.

        Args:
            strategy: The chunking strategy to use (e.g., RECURSIVE).
            chunk_size: The target size of each chunk.
            chunk_overlap: The amount of overlap between consecutive chunks.
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Instantiate the appropriate LangChain splitter
        if self.strategy == ChunkingStrategy.FIXED:
            self._splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        else:  # Default to RECURSIVE
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )

        self._stats: dict[str, Any] = self._reset_stats_dict()

    @classmethod
    async def create(
        cls,
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        **kwargs: Any,  # Absorb unused kwargs like min/max chunk size
    ) -> "TextChunker":
        """Asynchronously create and configure a TextChunker instance.

        This factory method translates string-based strategy from settings
        into a configured TextChunker.

        Args:
            strategy: The chunking strategy name (e.g., "recursive").
            chunk_size: Target size for chunks.
            chunk_overlap: Overlap between chunks.
            **kwargs: Additional unused arguments.

        Returns:
            A new instance of TextChunker.
        """
        try:
            strategy_enum = ChunkingStrategy(strategy.lower())
        except ValueError:
            logger.warning(
                f"Invalid chunking strategy '{strategy}'. "
                "Defaulting to 'recursive'."
            )
            strategy_enum = ChunkingStrategy.RECURSIVE

        return cls(
            strategy=strategy_enum,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split a list of documents into chunks using the configured strategy.

        Args:
            documents: A list of `Document` objects to be split.

        Returns:
            A list of new `Document` objects representing the chunks.

        Raises:
            ChunkingError: If the underlying LangChain splitter fails.
        """
        if not documents:
            return []

        self._stats["documents_processed"] += len(documents)

        # 1. Convert our internal Document to LangChain's format,
        #    preserving the original document ID in metadata.
        lc_docs = []
        for doc in documents:
            metadata = doc.metadata.copy()
            metadata["parent_id"] = doc.id
            lc_docs.append(
                LangChainDocument(page_content=doc.text, metadata=metadata)
            )

        # 2. Use the LangChain splitter to do the work.
        try:
            split_lc_docs = self._splitter.split_documents(lc_docs)
        except Exception as e:
            logger.error(f"LangChain splitter failed: {e}")
            raise ChunkingError(f"Chunking failed due to: {e}") from e

        # 3. Convert back to our internal Document format.
        final_chunks = [
            Document(text=d.page_content, metadata=d.metadata)
            for d in split_lc_docs
        ]

        self._stats["total_chunks"] += len(final_chunks)
        if final_chunks:
            total_size = sum(len(c.text) for c in final_chunks)
            self._stats["avg_chunk_size"] = total_size / len(final_chunks)

        logger.info(
            f"Processed {len(documents)} docs into {len(final_chunks)} chunks "
            f"via {self.strategy.value} strategy."
        )
        return final_chunks

    @staticmethod
    def _reset_stats_dict() -> dict[str, Any]:
        """Return a clean dictionary for tracking statistics."""
        return {
            "documents_processed": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0.0,
        }

    def reset_stats(self) -> None:
        """Reset all internal performance statistics."""
        self._stats = self._reset_stats_dict()
        logger.info("Chunking statistics have been reset.")

    @property
    def stats(self) -> dict[str, Any]:
        """Get a copy of the current chunking statistics."""
        return self._stats.copy()

    def __repr__(self) -> str:
        """Provide a human-readable representation of the chunker."""
        return (
            f"TextChunker(strategy={self.strategy.value}, "
            f"size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
