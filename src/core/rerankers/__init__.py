from __future__ import annotations

"""Reranker package providing quality-aware reordering of retrieved docs."""

from importlib import import_module
from typing import Any

__all__ = ["get_reranker"]


def get_reranker(kind: str | None = None, **kwargs: Any):
    """Return a reranker instance.

    Args:
        kind: Lower-case identifier for reranker type (default: "jina")
        **kwargs: Additional keyword arguments passed to the reranker constructor

    Returns:
        An initialized reranker instance

    Raises:
        ValueError: If the specified reranker kind is not supported
    """
    kind = (kind or "jina").lower()

    if kind in {"jina", "jina-reranker"}:
        module = import_module("src.core.rerankers.jina_reranker")
        return module.JinaReranker(**kwargs)

    raise ValueError(f"Unknown reranker kind: {kind}")
