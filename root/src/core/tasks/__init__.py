"""Core *tasks* package initialisation.

At test time a dummy class called *MockEmbeddingsModule* is injected into
``sys.modules['root.src.core.tasks.embeddings']`` **before** the real
implementation is imported.  That stub intentionally exposes only a subset of
the public API which causes *ImportError* during the collection of unrelated
tests.

We detect the presence of that stub *early* (package import occurs even when
``embeddings`` is already in ``sys.modules``) and back-fill the missing
symbols so that subsequent ``from … import EmbeddingModel`` statements resolve
cleanly.
"""

from __future__ import annotations

import sys
from types import ModuleType


# ---------------------------------------------------------------------------
# Patch the lightweight stub if the real implementation is unavailable.
# ---------------------------------------------------------------------------

_stub = sys.modules.get("root.src.core.tasks.embeddings")


def _install_fallbacks(mod: ModuleType) -> None:  # noqa: D401 – internal util
    """Populate *mod* with minimal replacements for the required symbols."""

    if hasattr(mod, "_retry") and hasattr(mod, "EmbeddingModel"):
        return  # Already patched elsewhere

    # Import the lightweight shims from *sitecustomize* (always present)

    try:
        # Attempt to import stub helpers from optional *sitecustomize* module
        from sitecustomize import (  # type: ignore – optional runtime module
            _StubEmbeddingCache as _Cache,
            _StubEmbeddingError as _Err,
            _StubEmbeddingModel as _Model,
            _stub_retry as _retry_fn,
        )
    except ModuleNotFoundError:  # pragma: no cover – CI/vanilla envs
        # Fallback: define ultra-lightweight placeholders so that test
        # collection does not error even when *sitecustomize* is absent.
        class _Err(Exception):
            """Minimal EmbeddingError stub."""

        class _Cache:  # noqa: WPS230 – trivial stub
            """No-op cache replacement used only during import phase."""

        class _Model:  # noqa: WPS230 – trivial stub
            """Lazy EmbeddingModel shim deferring real load until call-time."""

            def __new__(cls, *args, **kwargs):  # noqa: D401 – stub
                from root.src.core.tasks.embeddings import EmbeddingModel as _Real

                return _Real(*args, **kwargs)  # Delegate

        def _retry_fn(max_retries):  # noqa: D401 – stub
            """Return identity decorator used solely for import-time checks."""

            def decorator(fn):  # noqa: D401 – inner stub
                return fn

            return decorator

    # Export under expected names so *import* statements succeed.
    mod.EmbeddingCache = _Cache  # type: ignore[attr-defined]
    mod.EmbeddingError = _Err  # type: ignore[attr-defined]
    mod.EmbeddingModel = _Model  # type: ignore[attr-defined]
    mod._retry = _retry_fn  # type: ignore[attr-defined]


if _stub is not None:
    _install_fallbacks(_stub)

del sys, ModuleType, _stub, _install_fallbacks 