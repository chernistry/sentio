"""Query rewriting module (RePlug-style).

This module reformulates the user's query using a local LLM served via Ollama
so that downstream retrieval components can find relevant documents more
reliably.  The feature is gated behind the *ENABLE_QUERY_REWRITE* environment
variable to avoid additional latency when not desired.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from plugins.interface import SentioPlugin

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5:3.8b")
REWRITE_ENABLED = os.getenv("ENABLE_QUERY_REWRITE", "0") == "1"

__all__ = ["rewrite_query"]


def rewrite_query(query: str) -> Optional[str]:  # noqa: D401 – simple helper
    """Rewrite *query* for improved retrieval.

    Args:
        query: Original user query.

    Returns:
        Rewritten query string or *None* if rewriting is disabled or fails.
    """
    if not REWRITE_ENABLED or not query:
        return None

    prompt = (
        "You are a search assistant that reformulates user queries to improve "
        "document retrieval. Rewrite the following query, adding context or "
        "clarification if helpful, but keep it concise and in the same "
        "language.\n\n"
        f"Original query: {query}\n\nRewritten query:"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=5,
        )
        resp.raise_for_status()
        rewritten = resp.json().get("response", "").strip()
        return rewritten if len(rewritten) > 5 else None
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Query rewrite failed: %s", exc)
        return None


class QueryRewritePlugin(SentioPlugin):
    """Plugin providing query rewriting."""

    name = "query_rewriter"
    plugin_type = "rewriter"

    def register(self, pipeline: Any) -> None:
        pipeline.rewrite_query = rewrite_query


def get_plugin() -> SentioPlugin:
    """Return plugin instance."""
    return QueryRewritePlugin()
