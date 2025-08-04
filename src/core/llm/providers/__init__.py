"""LLM provider adapters.

This package contains adapters for different LLM providers.
"""

from typing import Any, Dict, Optional, Type

from .base import BaseLLMProvider

__all__ = ["BaseLLMProvider", "get_provider"]

_PROVIDERS: dict[str, type[BaseLLMProvider]] = {}


def register_provider(name: str):
    """Register a provider class."""
    def decorator(cls: type[BaseLLMProvider]) -> type[BaseLLMProvider]:
        _PROVIDERS[name] = cls
        return cls
    return decorator


def get_provider(name: str, **kwargs: Any) -> BaseLLMProvider:
    """Get a provider instance by name.
    
    Args:
        name: Provider name
        **kwargs: Additional provider-specific arguments
        
    Returns:
        A provider instance
    """
    if name not in _PROVIDERS:
        # Import built-in providers
        from . import openai  # noqa: F401

        # Check again after imports
        if name not in _PROVIDERS:
            raise ValueError(f"Unknown provider: {name}")

    return _PROVIDERS[name](**kwargs)
