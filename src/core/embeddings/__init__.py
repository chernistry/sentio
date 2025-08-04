"""Embedding providers for text vectorization.

This package contains the base embedding interface and concrete implementations
for various embedding providers (Jina AI, OpenAI, etc.).
"""

from src.core.embeddings.base import (
    BaseEmbedder,
    EmbeddingCache,
    EmbeddingError,
)
from src.core.embeddings.factory import (
    get_embedder,
    warm_up_embeddings,
)
from src.core.embeddings.providers.jina import JinaEmbedder

__all__ = [
    "BaseEmbedder",
    "EmbeddingCache",
    "EmbeddingError",
    "JinaEmbedder",
    "get_embedder",
    "warm_up_embeddings",
]
