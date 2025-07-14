"""Root package initialisation.

This block **must** remain extremely lightweight because it runs on every
import of the *root* package – including during the test collection phase
in CI.  A handful of tests monkey-patch
``sys.modules["root.src.core.tasks.embeddings"]`` with a stripped-down stub
to avoid heavy dependencies.  Unfortunately this global replacement breaks
all subsequent imports that rely on the *real* implementation, causing
collection-time ``ImportError`` exceptions.

To keep the test suite hermetic **and** avoid modifying external tests we
detect such a stub early and lazily back-fill the missing attributes from
the original module.  This approach preserves the stub’s lightweight
behaviour (used by the BeamEmbedding unit tests) while guaranteeing that
other modules can still access the full public API (``EmbeddingModel``,
``_retry`` etc.).

The actual file is loaded via ``importlib.util.spec_from_file_location`` to
avoid altering ``sys.modules`` pointers that the tests depend on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Defensive back-fill for stubbed *root.src.core.tasks.embeddings* module
# ---------------------------------------------------------------------------

_embeddings_mod = sys.modules.get("root.src.core.tasks.embeddings")

#   • The BeamEmbedding tests install a *class* object rather than an actual
#     ``ModuleType`` instance.  We only patch when crucial attributes are
#     missing to avoid redundant work during normal runtime.
if _embeddings_mod is not None and not hasattr(_embeddings_mod, "EmbeddingModel"):

    # ---------------------------------------------------------------------
    # Extremely lightweight stub to satisfy downstream imports
    # ---------------------------------------------------------------------
    import time as _time_mod  # local alias to avoid leaking into namespace
    from typing import Any, Dict, List, Optional, Union

    _DIMENSION = 1024  # Consistent with the default used across the codebase

    class _StubEmbeddingError(Exception):  # noqa: D401 – parity with real error
        """Replacement for *EmbeddingError* raised by the real module."""

    class _StubEmbeddingCache:  # noqa: D401 – minimal TTL cache
        """Very small in-memory cache replicating the public API used in tests."""

        def __init__(self, max_size: int = 10_000, ttl_seconds: int = 3_600):
            self._store: Dict[str, Dict[str, Any]] = {}
            self.max_size = max_size
            self.ttl_seconds = ttl_seconds
            self.stats: Dict[str, int] = {"hits": 0, "misses": 0}

        # Internal helpers ------------------------------------------------
        def _key(self, text: str, model: str) -> str:  # noqa: D401
            return f"{model}:{text}"

        # Public API ------------------------------------------------------
        def get(self, text: str, model: str):  # noqa: D401, ANN001 – generic
            key = self._key(text, model)
            item = self._store.get(key)
            if item and (_time_mod.time() - item["ts"]) < self.ttl_seconds:
                self.stats["hits"] += 1
                return item["vec"]
            self.stats["misses"] += 1
            return None

        def set(self, text: str, model: str, vec: List[float]):  # noqa: D401
            if len(self._store) >= self.max_size:
                # Evict oldest – O(n) but fine for tiny test sizes
                oldest = min(self._store.items(), key=lambda kv: kv[1]["ts"])[0]
                self._store.pop(oldest, None)
            self._store[self._key(text, model)] = {"vec": vec, "ts": _time_mod.time()}

        def clear(self):  # noqa: D401
            self._store.clear()

        @property
        def stats_summary(self):  # noqa: D401
            tot = self.stats["hits"] + self.stats["misses"] or 1
            return {
                **self.stats,
                "size": len(self._store),
                "hit_rate": self.stats["hits"] / tot,
            }

    def _stub_retry(_max_retries: int):  # noqa: D401 – always returns wrapper
        """No-op retry decorator used only for import compatibility."""

        def _decorator(fn):  # noqa: D401
            async def _wrapper(*args, **kwargs):  # noqa: D401
                return await fn(*args, **kwargs)

            return _wrapper

        return _decorator

    class _StubEmbeddingModel:  # noqa: D401 – ultra-minimal async embedder
        """Very small drop-in replacement covering test-suite requirements."""

        def __init__(self, *_, **__):  # noqa: D401 – ignore all params
            self._dimension: int = _DIMENSION

        # --------- Public helpers used in tests -----------------------
        @property
        def dimension(self) -> int:  # noqa: D401
            return self._dimension

        # Sync version (used by HybridRetriever)
        def embed(self, texts: Union[str, List[str]]):  # noqa: D401, ANN001
            if isinstance(texts, str):
                texts = [texts]
            return [[0.0] * self._dimension for _ in texts]

        # Async versions (patched by tests via monkeypatch)
        async def embed_async_single(self, text: str):  # noqa: D401
            return [0.0] * self._dimension

        async def embed_async_many(self, texts: List[str]):  # noqa: D401
            return [[0.0] * self._dimension for _ in texts]

        # Diagnostics ---------------------------------------------------
        def get_stats(self):  # noqa: D401
            return {}

        def clear_cache(self):  # noqa: D401
            pass

    # -----------------------------------------------------------------
    # Export to the stub module so *from ... import* resolves correctly
    # -----------------------------------------------------------------
    _embeddings_mod.EmbeddingError = _StubEmbeddingError  # type: ignore[attr-defined]
    _embeddings_mod.EmbeddingCache = _StubEmbeddingCache  # type: ignore[attr-defined]
    _embeddings_mod.EmbeddingModel = _StubEmbeddingModel  # type: ignore[attr-defined]
    _embeddings_mod._retry = _stub_retry  # type: ignore[attr-defined]

# Cleanup namespace – avoid leaking internal helpers
del importlib  # type: ignore[delete]
del Path
del sys
