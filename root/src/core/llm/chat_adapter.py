from __future__ import annotations

"""Generic OpenAI-compatible chat adapter.

This module provides a thin wrapper around any LLM endpoint that implements
(OpenAI) `/chat/completions` semantics (OpenRouter, Groq, Together, etc.).
It supports both standard JSON responses and streaming event-source format
(`data: ...\n\n`).

Usage
-----
>>> from root.src.core.chat.chat_adapter import ChatAdapter
>>> adapter = ChatAdapter()
>>> resp = await adapter.chat_completion({"messages": [...], "stream": False})

The active endpoint, model and API key are taken from :pydata:`utils.settings`.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Union, Optional

import httpx
from httpx import HTTPError

from root.src.utils.settings import settings

import random
import string
from functools import lru_cache

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential  # type: ignore

# Optional dependency – generates realistic User-Agent strings for header stealthing.
try:
    from faker import Faker  # type: ignore
except Exception:  # pragma: no cover – optional dep not installed in CI
    Faker = None  # type: ignore

try:
    import instructor  # noqa: WPS433 – optional heavy dep
    import openai  # type: ignore
except Exception:  # pragma: no cover – optional
    instructor = None  # type: ignore
    openai = None  # type: ignore

try:
    from litellm import model_info  # type: ignore
except Exception:  # pragma: no cover
    model_info = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = ["ChatAdapter", "chat_completion"]


# ---------------------------------------------------------------------------
# Helper utilities – kept module-private
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_faker() -> "Faker | None":  # noqa: D401 – internal helper
    if Faker is None:
        return None
    try:
        return Faker()
    except Exception:  # pragma: no cover – very defensive
        return None


def _random_suffix(length: int = 4) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


class ChatAdapter:
    """OpenAI-compatible chat client with optional streaming and stealth headers."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        enable_header_stealth: bool | None = None,
    ) -> None:
        self.base_url = (base_url or settings.chat_llm_base_url).rstrip("/")
        self.api_key = api_key or settings.chat_llm_api_key
        self.model = model or settings.chat_llm_model
        self.enable_header_stealth = (
            enable_header_stealth
            if enable_header_stealth is not None
            else True  # default ON for production resilience
        )

        if not self.api_key:
            raise RuntimeError("CHAT_LLM_API_KEY is required")

        # Single client instance per adapter – timeout can be overridden upstream.
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=None)
        
        # Beam provider (lazy-loaded)
        self._beam_provider = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        payload: Dict[str, Any],
        *,
        max_retries: int = 3,
        response_model: Any | None = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Forward payload with optional structured *response_model* parsing.

        If *response_model* (Pydantic) is given **and** the *instructor* package
        is available, the call will automatically switch to OpenAI JSON mode and
        return the validated object.  Otherwise returns raw OpenAI-compatible
        JSON.
        """
        # Check if we should use Beam provider
        if settings.chat_provider == "beam":
            return await self._beam_chat_completion(payload, response_model=response_model)

        stream = bool(payload.get("stream", False))
        if stream and response_model is not None:
            raise ValueError("Cannot use streaming with structured JSON mode")

        payload.setdefault("model", self.model)
        if "max_tokens" not in payload:
            payload["max_tokens"] = self._safe_max_tokens(payload["model"], payload.get("messages", []))

        headers = self._build_headers()

        if response_model is not None and instructor is not None and openai is not None:
            # Bypass retries at HTTP layer – rely on OpenAI client built-in
            client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, default_headers=headers)
            patched = instructor.patch(client, mode=instructor.Mode.JSON)
            result = await patched.chat.completions.create(
                model=payload["model"],
                messages=payload["messages"],
                temperature=payload.get("temperature", 0.0),
                max_tokens=payload["max_tokens"],
                response_model=response_model,
            )
            return result  # type: ignore[return-value]

        # ---- Fallback → plain HTTPX with retries ---- #
        retryer = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            reraise=True,
        )
        async for attempt in retryer:  # pragma: no cover – runtime loop
            with attempt:
                resp = await self._client.post("/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()

        if not stream:
            return resp.json()

        async def _gen() -> AsyncGenerator[str, None]:
            try:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield line
            finally:
                await resp.aclose()

        return _gen()

    async def close(self) -> None:  # pragma: no cover – simple cleanup
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _beam_chat_completion(
        self, 
        payload: Dict[str, Any],
        *,
        response_model: Any | None = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Use Beam provider for chat completion.

        Args:
            payload: Chat completion payload.
            response_model: Optional response model for structured output.

        Returns:
            Chat completion response or streaming generator.
        """
        if response_model is not None:
            raise ValueError("Structured response models not supported with Beam provider")
            
        # Lazy-load Beam provider
        if self._beam_provider is None:
            # Import here to avoid circular imports
            from root.src.core.llm.providers.beam_chat import BeamChatProvider
            self._beam_provider = BeamChatProvider()
            
        return await self._beam_provider.chat_completion(payload)

    def _build_headers(self) -> Dict[str, str]:
        """Return HTTP headers including optional stealth randomisation."""
        base_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Only apply stealth heuristics for OpenRouter (reduces accidental blocking).
        if not self.enable_header_stealth or "openrouter" not in self.base_url:
            return base_headers

        # Config-driven header pools (fallback to defaults)
        referers = [settings.openrouter_referer]
        titles = [settings.openrouter_title]

        referer_hdr = random.choice(referers)
        title_hdr = f"{random.choice(titles)}-{_random_suffix()}"

        faker = _get_faker()
        if faker is not None:
            ua_hdr = faker.user_agent()
        else:
            ua_hdr = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )

        return {
            **base_headers,
            "HTTP-Referer": referer_hdr,
            "X-Title": title_hdr,
            "User-Agent": ua_hdr,
        }

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _safe_max_tokens(self, model_name: str, messages: list[dict], fallback: int = 2048) -> int:  # noqa: D401
        """Compute conservative *max_tokens* based on context window (80 % rule)."""
        ctx_window = 0
        if model_info is not None:
            try:
                ctx_window = model_info(model_name).max_tokens or 0
            except Exception:
                ctx_window = 0
        if not ctx_window:
            return fallback
        try:
            import tiktoken  # local import to avoid hard dep

            enc = tiktoken.encoding_for_model(model_name.split("/")[-1])
            used = sum(len(enc.encode(m.get("content", ""))) for m in messages) + 200
            remain = int((ctx_window - used) * 0.8)
            return max(256, remain)
        except Exception:
            return fallback


# Convenience module-level facade -------------------------------------------

_adapter: ChatAdapter | None = None


async def chat_completion(
    payload: Dict[str, Any],
) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
    """Convenience facade that re-uses a singleton ``ChatAdapter``."""
    global _adapter  # noqa: PLW0603 – module-level cache is fine here
    if _adapter is None:
        _adapter = ChatAdapter()
    return await _adapter.chat_completion(payload)
