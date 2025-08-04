"""Factory for creating embedding models.

This module provides a unified way to instantiate embedding models based on configuration,
abstracting away the specific provider details from the rest of the application.
"""

import importlib
import logging
from pathlib import Path
from typing import Any

from src.core.embeddings.base import BaseEmbedder, EmbeddingError
from src.utils.settings import settings

logger = logging.getLogger(__name__)

# Directory containing provider implementations
_PROVIDERS_DIR = Path(__file__).parent / "providers"

# Registry mapping provider keys to implementation details
_PROVIDER_REGISTRY: dict[str, dict[str, str]] = {
    # Jina AI
    "jina": {
        "class": "JinaEmbedder",
        "module": "src.core.embeddings.providers.jina",
    },
}

# Cache of loaded provider classes
_PROVIDER_CLASS_CACHE: dict[str, type[BaseEmbedder]] = {}


def _load_module(module_path: str) -> Any:
    """Dynamically import a module.
    
    Args:
        module_path: Dot-separated path to the module.
        
    Returns:
        The imported module.
        
    Raises:
        ImportError: If module cannot be loaded.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        raise ImportError(f"Failed to load embedding provider module: {e}") from e


def get_embedder(
    provider: str | None = None,
    model_name: str | None = None,
    **kwargs: Any
) -> BaseEmbedder:
    """Get an instance of the configured embedding model.
    
    Args:
        provider: Provider key (e.g., 'jina'). If None, uses settings.embedder_name.
        model_name: Model name to use. If None, uses settings.embedding_model.
        **kwargs: Additional configuration passed to the embedding model constructor.
        
    Returns:
        Configured embedding model instance.
        
    Raises:
        EmbeddingError: If provider is unknown or instantiation fails.
    """
    provider_key = (provider or settings.embedder_name).lower()
    model = model_name or settings.embedding_model

    # Use cached class if available
    if provider_key in _PROVIDER_CLASS_CACHE:
        model_cls = _PROVIDER_CLASS_CACHE[provider_key]
        try:
            return model_cls(model_name=model, **kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate cached provider '{provider_key}': {e}")
            # Remove from cache to allow retry with fresh import
            _PROVIDER_CLASS_CACHE.pop(provider_key, None)
            raise EmbeddingError(f"Failed to instantiate provider '{provider_key}': {e}") from e

    # Look up provider in registry
    provider_info = _PROVIDER_REGISTRY.get(provider_key)
    if not provider_info:
        # Default to Jina if provider not found
        logger.warning(f"Unknown embedding provider '{provider_key}' - falling back to jina")
        provider_key = "jina"
        provider_info = _PROVIDER_REGISTRY.get(provider_key)
        if not provider_info:
            raise EmbeddingError(f"Provider '{provider_key}' not found in registry")

    # Load the module and get the class
    try:
        module = _load_module(provider_info["module"])
        model_cls = getattr(module, provider_info["class"])

        # Verify it's a BaseEmbedder subclass
        if not issubclass(model_cls, BaseEmbedder):
            raise EmbeddingError(
                f"{provider_info['class']} is not a BaseEmbedder subclass"
            )

        # Cache for subsequent calls
        _PROVIDER_CLASS_CACHE[provider_key] = model_cls

        logger.info(f"Loaded embedding provider '{provider_key}' ({model})")
        return model_cls(model_name=model, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load embedding provider '{provider_key}': {e}")
        raise EmbeddingError(f"Failed to load provider '{provider_key}': {e}") from e


async def warm_up_embeddings() -> bool:
    """Initialize and warm up the default embedding model.
    
    This function is intended to be called during application startup
    to ensure the embedding model is ready to serve requests.
    
    Returns:
        True if warm-up was successful, False otherwise.
    """
    try:
        embedder = get_embedder()
        success = await embedder.warm_up()
        return success
    except Exception as e:
        logger.error(f"Failed to warm up embedding model: {e}")
        return False
