from __future__ import annotations

"""Vector store layer abstraction for Sentio.

Currently exposes a thin wrapper around Qdrant Cloud.  In future this
module may grow to support additional providers (e.g. Weaviate, Milvus).
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover – import-time only
    from .async_qdrant_store import AsyncQdrantStore  # noqa: F401
    from .qdrant_store import QdrantStore  # noqa: F401


def get_vector_store(name: str, async_mode: bool = False, **kwargs: Any):
    """Factory returning the requested vector store implementation.

    Args:
        name: Lower-case identifier (e.g. ``"qdrant"``).
        async_mode: Whether to return async implementation.
        **kwargs: Forwarded to the concrete class constructor.

    Returns:
        Instantiated vector store client.

    Raises:
        ValueError: If *name* is unknown or the backend is not installed.
    """
    name = name.lower()
    if name == "qdrant":
        if async_mode:
            module = import_module("src.core.vector_store.async_qdrant_store")
            return module.AsyncQdrantStore(**kwargs)
        module = import_module("src.core.vector_store.qdrant_store")
        return module.QdrantStore(**kwargs)

    raise ValueError(f"Unknown vector store backend: {name}")


async def get_async_vector_store(name: str, **kwargs: Any):
    """Factory returning async vector store with automatic initialization.
    
    Args:
        name: Lower-case identifier (e.g. ``"qdrant"``).
        **kwargs: Forwarded to the concrete class constructor.
        
    Returns:
        Initialized async vector store client.
        
    Raises:
        ValueError: If *name* is unknown or the backend is not installed.
    """
    store = get_vector_store(name, async_mode=True, **kwargs)
    await store.initialize()
    return store
