"""Core embeddings package.

This package contains the embeddings adapter and provider implementations.
"""

# Re-export the BaseEmbeddingModel and get_embedding_model for backward compatibility
from .embeddings_adapter import BaseEmbeddingModel, get_embedding_model 