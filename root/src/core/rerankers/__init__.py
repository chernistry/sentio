from .providers.cross_encoder import CrossEncoderReranker

# Jina API-based reranker (high quality, requires key)
# Updated path after moving provider implementations under *providers/*
# Re-export Jina provider at package root for legacy imports
from .providers.jina_reranker import JinaReranker

# Create an explicit submodule attribute so that ``import \("root.src.core.rerankers.jina_reranker"``
# works even with introspection utilities that bypass ``sys.modules`` lookup.
import sys as _sys
from importlib import import_module as _import_mod

if __name__ + ".jina_reranker" not in _sys.modules:
    _sys.modules[__name__ + ".jina_reranker"] = _import_mod(
        ".providers.jina_reranker", package=__name__
    )

# Expose as attribute on the package for hasattr/getattr look-ups.
setattr(
    _sys.modules[__name__],
    "jina_reranker",
    _sys.modules[__name__ + ".jina_reranker"],
)

__all__ = ["CrossEncoderReranker", "JinaReranker"] 