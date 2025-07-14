"""BGE cross-encoder reranker.

Provides a lightweight wrapper around the *BAAI/bge-reranker-base* model so
that downstream code can use a single-stage high-quality reranker when the
full cascade implemented in *cross_encoder.py* is unnecessary.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from plugins.interface import SentioPlugin

from sentence_transformers import CrossEncoder

__all__ = ["BGEReranker"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BGE_RERANK_MODEL = "BAAI/bge-reranker-base"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BGEReranker:  # noqa: D101 – concise wrapper
    """Single-stage reranker based on the BGE cross-encoder model.

    Example:
        >>> reranker = BGEReranker()
        >>> ranked_docs = reranker.rerank(query, docs)
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        """Initialise the BGE reranker.

        Args:
            model_name: Hugging Face model name or local path.  If *None*, the
                *BGE_RERANK_MODEL* environment variable is used and finally
                falls back to :data:`DEFAULT_BGE_RERANK_MODEL`.
            device: Execution device (``cpu``/``cuda``).  Defaults to the
                automatic selection performed by *sentence-transformers*.
        """

        self.model_name: str = model_name or os.getenv("BGE_RERANK_MODEL", DEFAULT_BGE_RERANK_MODEL)
        self.model = CrossEncoder(self.model_name, device=device)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        *,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return *top_k* docs ranked by relevance to *query*.

        Args:
            query: Search query string.
            docs: List of document dicts, each expected to contain a ``text``
                field.  Additional keys are preserved.
            top_k: Number of documents to return.
        """
        if not docs:
            return []

        pairs = [(query, d.get("text", "")) for d in docs]
        scores = self.model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)

        return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_k]


class BGERerankerPlugin(SentioPlugin):
    """Plugin wrapper for BGE reranker."""

    name = "bge_reranker"
    plugin_type = "reranker"

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.reranker = BGEReranker(model_name=model_name, device=device)

    def register(self, pipeline: Any) -> None:
        pipeline.reranker = self.reranker


def get_plugin() -> SentioPlugin:
    """Return plugin instance."""
    return BGERerankerPlugin()
