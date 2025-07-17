from __future__ import annotations

"""Universal Embedding Adapter for Sentio.

This module provides a thin factory around multiple embedding back-ends so the
rest of the application can stay completely agnostic to the concrete provider.
The active provider is selected via the ``EMBEDDING_PROVIDER`` environment
variable (falls back to *jina*) and the model, if applicable, via
``EMBEDDING_MODEL``.

Adding support for a new provider is as easy as dropping a new implementation
into ``core/embeddings/providers`` and registering it in
``_PROVIDER_REGISTRY`` below – no changes elsewhere in the codebase are
required.
"""

import importlib
import os
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Union, List, Optional

from root.src.core.tasks.embeddings import BaseEmbeddingModel, EmbeddingError
from root.src.utils.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_PROVIDERS_DIR = Path(__file__).with_name("providers")


def _load_module_from_path(module_path: Path) -> ModuleType:  # pragma: no cover
    """Dynamically import a module by *absolute* file path.

    Args:
        module_path: Full path to "*.py" file.

    Returns:
        The imported module object.
    """
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for module at {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


# ---------------------------------------------------------------------------
# Registry – maps provider key ➜ (module_path, class_name)
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: Dict[str, Dict[str, Union[str, Path]]] = {
    # Cloud Jina AI – built-in implementation inside *embeddings.py*
    "jina": {
        "class": "JinaEmbedding",
        # Updated path to reflect pluralised file naming
        "path": _PROVIDERS_DIR / "jina_embeddings.py",
    },
    # Local HuggingFace / Sentence-Transformers
    "sentence": {
        "class": "SentenceTransformerEmbedding",
        "path": _PROVIDERS_DIR / "sentence_embeddings.py",
    },
    "sentence_transformers": {
        "class": "SentenceTransformerEmbedding",
        "path": _PROVIDERS_DIR / "sentence_embeddings.py",
    },
    # Local Ollama instance
    "ollama": {
        "class": "OllamaEmbedding",
        "path": _PROVIDERS_DIR / "ollama_embeddings.py",
    },
    "beam": {
        "class": "BeamEmbedding",
        "path": _PROVIDERS_DIR / "beam_embeddings.py",
    },
}

# Cached provider *classes* to avoid reloading the same module multiple times
# which would break runtime monkey-patching in unit-tests (each reload creates
# a *new* class object, so patches applied to the old one are lost).
_PROVIDER_CLASS_CACHE: dict[str, type[BaseEmbeddingModel]] = {}

# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_embedding_model(provider: str | None = None, **kwargs: Any) -> BaseEmbeddingModel:
    """Instantiate and return the selected embedding model.

    Args:
        provider: Provider key, e.g. ``"jina"`` or ``"ollama"``. Defaults to
            ``EMBEDDING_PROVIDER`` env var or *jina*.
        **kwargs: Arbitrary keyword arguments forwarded to the concrete
            provider constructor.

    Raises:
        EmbeddingError: If the provider is unknown or loading fails.
    """

    provider_key = (provider or settings.embedding_provider).lower()

    # ------------------------------------------------------------------
    # Graceful fallback – when running locally without a valid Beam config the
    # *beam* Python package may throw during import (handled via stubs), yet the
    # provider key can still be "beam" coming from env/defaults. We only degrade
    # to the lightweight CPU implementation when BOTH conditions are true:
    #   • We're not inside a Beam Cloud container (BeamRuntime.is_remote() is False)
    #   • No remote embedding endpoint has been configured via
    #     BEAM_EMBEDDING_BASE_CLOUD_URL or settings.BEAM_EMBEDDING_BASE_CLOUD_URL
    # Otherwise we keep provider="beam" so that HTTP fallback in *BeamEmbedding*
    # can leverage the remote endpoint.
    # ------------------------------------------------------------------
    if provider_key == "beam":
        _should_downgrade: bool = False
        try:
            from root.src.integrations.beam.runtime import BeamRuntime  # local import

            if not BeamRuntime.is_remote():
                from root.src.utils.settings import settings as _settings  # inline import

                if not _settings.BEAM_EMBEDDING_BASE_CLOUD_URL and not os.getenv("BEAM_EMBEDDING_BASE_CLOUD_URL", "") and not os.getenv("BEAM_EMBEDDING_BASE_LOCAL_URL", ""):
                    _should_downgrade = True
        except Exception as e:  # pragma: no cover – very defensive
            import os
            
            logger.warning(f"Error checking Beam runtime: {e}")
            if not os.getenv("BEAM_EMBEDDING_BASE_CLOUD_URL", "") and not os.getenv("BEAM_EMBEDDING_BASE_LOCAL_URL", ""):
                _should_downgrade = True

        if _should_downgrade:
            logger.error(
                "Beam provider selected but neither running inside Beam nor "
                "BEAM_EMBEDDING_BASE_CLOUD_URL is configured. Aborting – please set "
                "the remote endpoint or switch provider.",
            )
            raise EmbeddingError(
                "BEAM_EMBEDDING_BASE_CLOUD_URL not set and not running inside Beam",
            )

    # Fast-path: return cached class to maintain a single class object across
    # the entire test run (critical for monkey-patching reliability).
    if provider_key in _PROVIDER_CLASS_CACHE:
        model_cls = _PROVIDER_CLASS_CACHE[provider_key]
        try:
            return model_cls(**kwargs)  # type: ignore[return-value]
        except Exception as e:
            logger.error(f"Failed to instantiate cached provider '{provider_key}': {e}")
            # Remove from cache to allow retry with fresh import
            _PROVIDER_CLASS_CACHE.pop(provider_key, None)
            raise EmbeddingError(f"Failed to instantiate provider '{provider_key}': {e}") from e

    cfg = _PROVIDER_REGISTRY.get(provider_key)
    if cfg is None:
        # Fallback to jina if provider not found
        logger.warning(f"Unknown embedding provider '{provider_key}' - falling back to jina")
        provider_key = "jina"
        cfg = _PROVIDER_REGISTRY.get(provider_key)
        if cfg is None:
            raise EmbeddingError(
                f"Unknown embedding provider '{provider_key}' – add it to _PROVIDER_REGISTRY."
            )

    # All providers (including default 'jina') are loaded dynamically from file
    # path registered above.

    path = cfg["path"]
    if not path.exists():
        raise EmbeddingError(f"Provider module not found: {path}")

    try:
        module = _load_module_from_path(path)
        class_name = cfg["class"]
        model_cls = getattr(module, class_name)
        if not issubclass(model_cls, BaseEmbeddingModel):  # type: ignore[arg-type]
            raise EmbeddingError(
                f"{class_name} is not a subclass of BaseEmbeddingModel (provider '{provider_key}')"
            )

        # Cache for subsequent calls BEFORE instantiation to guarantee that
        # tests patching class attributes keep working regardless of how many
        # times *get_embedding_model* is invoked.
        _PROVIDER_CLASS_CACHE[provider_key] = model_cls  # type: ignore[assignment]

        # Expose in legacy module for convenient import paths & patching.
        legacy_mod = importlib.import_module("root.src.core.embeddings")
        setattr(legacy_mod, class_name, model_cls)

        logger.info("✓ Loaded embedding provider '%s' from %s", provider_key, path.name)
        return model_cls(**kwargs)  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover – defensive
        # Remove half-loaded entry on failure to avoid stale cache
        _PROVIDER_CLASS_CACHE.pop(provider_key, None)
        raise EmbeddingError(f"Failed to load provider '{provider_key}': {exc}") from exc


# Extend BaseEmbeddingModel with embed_sync method
def _add_embed_sync_to_base():
    """
    Добавляет метод embed_sync в BaseEmbeddingModel, если он отсутствует
    """
    if not hasattr(BaseEmbeddingModel, 'embed_sync'):
        def embed_sync(self, text: str) -> List[float]:
            """
            Синхронная версия embed_async_single для использования в синхронном контексте.
            Базовая реализация, которая использует синхронную обертку вокруг асинхронного метода.
            Провайдеры могут переопределить этот метод для более эффективной реализации.
            
            Args:
                text: Текст для эмбеддинга
                
            Returns:
                List[float]: Вектор эмбеддинга
            """
            import asyncio
            
            # Проверяем кэш
            cached = self._check_cache(text)
            if cached is not None:
                self._update_stats(hit=True)
                return cached
                
            try:
                # Используем синхронную обертку вокруг асинхронного метода
                try:
                    # Пробуем получить текущий event loop
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # Если loop не запущен, создаем новый
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                # Запускаем асинхронный метод синхронно
                embedding = loop.run_until_complete(self.embed_async_single(text))
                return embedding
            except Exception as e:
                logger.error(f"Failed to get embedding synchronously: {e}")
                # В случае ошибки возвращаем нулевой вектор соответствующей размерности
                return [0.0] * self.dimension
                
        # Добавляем метод в класс
        setattr(BaseEmbeddingModel, 'embed_sync', embed_sync)
        logger.debug("Added embed_sync method to BaseEmbeddingModel")


# Добавляем метод embed_sync при импорте модуля
_add_embed_sync_to_base()
