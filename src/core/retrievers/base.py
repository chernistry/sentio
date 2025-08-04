from __future__ import annotations

"""Common interface for document retrievers."""

import abc
from typing import Protocol

from src.core.models.document import Document

__all__ = ["BaseRetriever", "ScorerPlugin"]


class ScorerPlugin(Protocol):
    """Protocol for pluggable scoring components."""

    def score(self, query: str, docs: list[Document]) -> list[float]:
        """Score the documents for the query.
        
        Args:
            query: The query string.
            docs: List of documents to score.
            
        Returns:
            List of scores corresponding to each document.
        """
        ...


class BaseRetriever(abc.ABC):
    """Abstract base class for document retrievers."""

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Return top_k relevant documents."""
        raise NotImplementedError

    async def retrieve_async(self, query: str, top_k: int = 10) -> list[Document]:
        """Async wrapper for :py:meth:`retrieve` running in executor."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.retrieve(query, top_k))
