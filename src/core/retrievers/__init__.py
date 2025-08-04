from __future__ import annotations

"""Retriever package public surface."""

from importlib import import_module
from typing import Any, List, Optional

__all__ = [
    "ScorerPlugin",
    "get_retriever",
    "get_scorer"
]

from .base import ScorerPlugin


def get_retriever(kind: str, **kwargs: Any):
    """Return retriever instance by *kind* identifier.

    Args:
        kind: Lower-case identifier (``"dense"``, ``"hybrid"``, ``"bm25"``, ``"pyserini"``).
        **kwargs: Forwarded to concrete constructor.
        
    Returns:
        An initialized retriever instance
        
    Raises:
        ValueError: If the specified retriever kind is not supported
    """
    kind = kind.lower()
    if kind == "dense":
        module = import_module("src.core.retrievers.dense")
        return module.DenseRetriever(**kwargs)
    if kind == "hybrid":
        module = import_module("src.core.retrievers.hybrid")
        # Extract top_k from kwargs as it's not a parameter for HybridRetriever constructor
        kwargs.pop("top_k", 10)
        # In a test environment, mock objects might be passed directly
        if "dense_retriever" not in kwargs:
            # For actual implementation, the dense retriever should be created externally
            # and passed explicitly as the tests expect the factory function to be mocked
            raise ValueError(
                "For hybrid retriever, 'dense_retriever' must be provided explicitly"
            )
        return module.HybridRetriever(**kwargs)
    if kind in ("bm25", "sparse"):
        module = import_module("src.core.retrievers.sparse")
        return module.BM25Retriever(**kwargs)
    if kind in ("pyserini", "lucene"):
        module = import_module("src.core.retrievers.sparse")
        return module.PyseriniBM25Retriever(**kwargs)

    raise ValueError(f"Unknown retriever kind: {kind}")


def get_scorer(kind: str, **kwargs: Any) -> ScorerPlugin:
    """Return a scorer plugin by kind identifier.
    
    Args:
        kind: Lower-case identifier for scorer type
        **kwargs: Forwarded to concrete constructor
        
    Returns:
        An instantiated ScorerPlugin
    
    Raises:
        ValueError: If the specified kind is not supported
    """
    module = import_module("src.core.retrievers.scorers")

    kind = kind.lower()
    if kind == "keyword":
        return module.KeywordMatchScorer(**kwargs)
    if kind in ("recency", "time"):
        return module.RecencyScorer(**kwargs)
    if kind in ("semantic", "similarity"):
        return module.SemanticSimilarityScorer(**kwargs)

    raise ValueError(f"Unknown scorer kind: {kind}")
