from __future__ import annotations

"""Base interfaces for rerankers in the RAG pipeline."""

import abc
from typing import Any, Protocol

from src.core.models.document import Document

__all__ = ["Reranker", "RerankerProtocol", "RerankingResult"]


class RerankerProtocol(Protocol):
    """Protocol defining the reranker interface."""

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int = 5,
        **kwargs: Any
    ) -> list[Document]:
        """Rerank documents based on their relevance to the query.
        
        Args:
            query: The user query string
            docs: List of documents to rerank
            top_k: Number of top documents to return
            **kwargs: Additional keyword arguments for specific implementations
            
        Returns:
            A reordered list of documents, limited to top_k
        """
        ...

    async def rerank_async(
        self,
        query: str,
        docs: list[Document],
        top_k: int = 5,
        **kwargs: Any
    ) -> list[Document]:
        """Async version of rerank method."""
        ...


class RerankingResult:
    """Container for reranking results with additional metadata."""

    def __init__(
        self,
        documents: list[Document],
        original_documents: list[Document] | None = None,
        metadata: dict[str, Any] | None = None
    ):
        """Initialize a reranking result.
        
        Args:
            documents: Reranked documents
            original_documents: Original documents before reranking
            metadata: Additional metadata about the reranking process
        """
        self.documents = documents
        self.original_documents = original_documents or []
        self.metadata = metadata or {}

    @property
    def top_document(self) -> Document | None:
        """Return the highest-ranked document or None if empty."""
        return self.documents[0] if self.documents else None

    def __len__(self) -> int:
        """Return the number of documents in the result."""
        return len(self.documents)

    def __getitem__(self, idx: int) -> Document:
        """Access a document by index."""
        return self.documents[idx]

    def __iter__(self):
        """Iterate over documents."""
        return iter(self.documents)


class Reranker(abc.ABC):
    """Abstract base class for document rerankers."""

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int = 5,
        **kwargs: Any
    ) -> list[Document]:
        """Rerank documents based on their relevance to the query.
        
        Args:
            query: The user query string
            docs: List of documents to rerank
            top_k: Number of top documents to return
            **kwargs: Additional keyword arguments for specific implementations
            
        Returns:
            A reordered list of documents, limited to top_k
        """
        raise NotImplementedError

    async def rerank_async(
        self,
        query: str,
        docs: list[Document],
        top_k: int = 5,
        **kwargs: Any
    ) -> list[Document]:
        """Async wrapper for rerank method.
        
        Args:
            query: The user query string
            docs: List of documents to rerank
            top_k: Number of top documents to return
            **kwargs: Additional keyword arguments
            
        Returns:
            A reordered list of documents, limited to top_k
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.rerank(query, docs, top_k, **kwargs)
        )
