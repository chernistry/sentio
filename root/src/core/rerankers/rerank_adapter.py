from __future__ import annotations

"""Factory adapter for selecting and creating reranker provider instances.

This module implements a very small **Adapter / Factory** that hides the
instantiation details of concrete reranker providers and exposes a unified
creation API.  Down-stream orchestration code (see ``core.rerank``) can pick
providers purely by *name* – often coming from an environment variable –
while this module takes care of importing the correct class and returning a
ready-to-use instance.

Design goals
------------
1.  **Minimal public surface** – only the :py:meth:`create` class-method.
2.  **Lazy imports** – heavy ML libraries are imported *only* when the
    associated provider is requested.
3.  **Extensibility** – adding a new provider is a one-liner in
    :data:`_PROVIDER_MAPPING`.
4.  **Type-safety** – 100 % type hints; enforced 88-char line length.
"""

import importlib
import os
from typing import Any, Dict, Final, Mapping, MutableMapping

# Mapping of provider *key* → ``"package.module.ClassName"``
_PROVIDER_MAPPING: Final[Mapping[str, str]] = {
    "local": "root.src.core.rerankers.providers.cross_encoder.CrossEncoderReranker",
    "jina": "root.src.core.rerankers.providers.jina_reranker.JinaReranker",
    # "multipass" handled explicitly inside :pyclass:`RerankAdapter` as it may
    # require combining multiple providers.
}


class RerankAdapter:  # noqa: D101 – concise utility
    """Factory/adapter that returns a concrete reranker instance.

    Example
    -------
    >>> from root.src.core.rerankers.rerank_adapter import RerankAdapter
    >>> reranker = RerankAdapter.create("local")
    >>> reranker.rerank(query="foo", docs=[{"text": "bar"}])
    """

    @classmethod
    def create(cls, provider: str | None = None, /, **kwargs: Any) -> Any:  # noqa: ANN401
        """Return an *initialised* reranker according to *provider*.

        Parameters
        ----------
        provider:
            String identifier of the reranker provider.  If *None*, the value
            of the ``RERANKER_PROVIDER`` environment variable is used,
            defaulting to ``"local"``.
        **kwargs:
            Additional keyword arguments forwarded verbatim to the provider's
            constructor.

        Returns
        -------
        Any
            A fully-initialised reranker instance exposing a ``.rerank``
            method with the signature ``(query: str, docs: list[dict[str, Any]],
            *, top_k: int = 5, **kw) -> list[dict[str, Any]]``.
        """
        provider_name: str = (provider or os.getenv("RERANKER_PROVIDER", "local")).lower()

        if provider_name == "multipass":  # ──► composite provider
            return cls._create_multipass_reranker(**kwargs)

        try:
            dotted_path: str = _PROVIDER_MAPPING[provider_name]
        except KeyError as exc:  # pragma: no cover – misconfiguration
            supported = ", ".join(sorted(_PROVIDER_MAPPING))
            raise ValueError(
                f"Unsupported RERANKER_PROVIDER '{provider_name}'. "
                f"Supported values: {supported}."
            ) from exc

        module_path, _, class_name = dotted_path.rpartition(".")
        module = importlib.import_module(module_path)
        provider_cls = getattr(module, class_name)

        return provider_cls(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_multipass_reranker(**kwargs: Any) -> Any:  # noqa: ANN401
        """Create a *multipass* reranker that falls back to Jina if local fails.

        Current implementation is extremely naive – it simply instantiates the
        local cross-encoder reranker and signals *inside* the provider to fall
        back transparently.  A more sophisticated approach could chain
        providers explicitly here.
        """
        from root.src.core.rerankers.providers.cross_encoder import (
            CrossEncoderReranker,
        )

        # Keep kwargs unchanged; the internal logic of ``CrossEncoderReranker``
        # already honours environment variables for secondary reranker (Jina).
        return CrossEncoderReranker(**kwargs)
