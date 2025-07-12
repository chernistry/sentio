from __future__ import annotations

"""High-level **rerank orchestration task**.

This thin wrapper chooses the concrete reranker provider according to the
``RERANKER_PROVIDER`` environment variable (or an explicit *provider*
argument) and delegates the heavy lifting to the selected implementation.

Why live in ``core/``?
    •  Keeps the public API *very* stable and independent from provider churn.
    •  Avoids accidental import-time dependencies on heavy ML libraries for
       components that do not require them.

The module purposefully contains *zero* ML logic – all intelligence is
implemented by provider classes under
:pydata:`root.src.core.rerankers.providers`.
"""

import os
from typing import Any, Dict, List

from root.src.core.rerankers.rerank_adapter import RerankAdapter

__all__ = ["RerankTask"]


class RerankTask:  # noqa: D101 – single-responsibility façade
    """Thin façade delegating rerank calls to the chosen provider."""

    def __init__(self, provider: str | None = None, /, **provider_kwargs: Any) -> None:
        """Initialise the orchestration task.

        Parameters
        ----------
        provider:
            Optional string overriding the provider selection.  If *None*, the
            value of the ``RERANKER_PROVIDER`` environment variable is used
            (default: ``"local"``).
        **provider_kwargs:
            Arbitrary keyword arguments forwarded verbatim to the provider
            constructor.  This allows for dependency injection in unit tests
            or advanced configuration without polluting the public API.
        """
        self._reranker = RerankAdapter.create(provider, **provider_kwargs)

    # ------------------------------------------------------------------
    # Public API – mirrors provider
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Return *top_k* most relevant documents for *query*.

        This method simply forwards the call to the underlying provider.  Any
        additional keyword arguments are forwarded unchanged.
        """
        return self._reranker.rerank(query, docs, top_k=top_k, **kwargs)
