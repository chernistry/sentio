"""Generic OpenAI-compatible chat adapter.

This module provides a thin wrapper around any LLM endpoint that implements
(OpenAI) `/chat/completions` semantics (OpenRouter, Groq, Together, etc.).
It supports both standard JSON responses and streaming event-source format
(`data: ...\n\n`).

Usage
-----
>>> from src.core.llm.chat_adapter import ChatAdapter
>>> adapter = ChatAdapter()
>>> resp = await adapter.chat_completion({"messages": [...], "stream": False})

The active endpoint, model and API key are taken from :pydata:`utils.settings`.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from src.core.llm.providers import get_provider
from src.utils.settings import settings

logger = logging.getLogger(__name__)

__all__ = ["ChatAdapter", "chat_completion"]


class ChatAdapter:
    """OpenAI-compatible chat client with optional streaming."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        *,
        enable_header_stealth: bool | None = None,
    ) -> None:
        """Initialize the chat adapter.
        
        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
            provider: Provider name (default: from settings)
            enable_header_stealth: Whether to use stealth headers
        """
        self.provider_name = provider or settings.llm_provider

        # Create provider instance
        self._provider = get_provider(
            self.provider_name,
            base_url=base_url,
            api_key=api_key,
            model=model,
            enable_header_stealth=enable_header_stealth,
        )

        logger.debug("Initialized ChatAdapter with provider=%s", self.provider_name)

    async def chat_completion(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any] | AsyncGenerator[str, None]:
        """Generate a chat completion.
        
        Args:
            payload: Chat completion payload
            
        Returns:
            Chat completion response or streaming generator
        """
        return await self._provider.chat_completion(payload)

    async def close(self) -> None:
        """Close the provider."""
        await self._provider.close()


# Convenience module-level facade -------------------------------------------

_adapter: ChatAdapter | None = None


async def chat_completion(
    payload: dict[str, Any],
) -> dict[str, Any] | AsyncGenerator[str, None]:
    """Convenience facade that re-uses a singleton ``ChatAdapter``."""
    global _adapter
    if _adapter is None:
        _adapter = ChatAdapter()
    return await _adapter.chat_completion(payload)
