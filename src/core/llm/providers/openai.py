"""OpenAI-compatible chat provider."""

import logging
import random
import string
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any

import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from src.core.llm.providers import register_provider
from src.core.llm.providers.base import BaseLLMProvider
from src.utils.settings import settings

# Optional dependency – generates realistic User-Agent strings for header stealthing.
try:
    from faker import Faker
except ImportError:
    Faker = None

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_faker() -> "Faker | None":
    """Get a Faker instance for generating User-Agent strings."""
    if Faker is None:
        return None
    try:
        return Faker()
    except Exception:
        return None


def _random_suffix(length: int = 4) -> str:
    """Generate a random alphanumeric suffix."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible chat provider.
    
    This provider works with any API that implements the OpenAI chat completion
    interface, including OpenAI, Azure OpenAI, and many other providers.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        enable_header_stealth: bool | None = None,
    ) -> None:
        """Initialize the OpenAI provider.
        
        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
            enable_header_stealth: Whether to use stealth headers
        """
        self.base_url = (base_url or settings.chat_llm_base_url).rstrip("/")
        self.api_key = api_key or settings.chat_llm_api_key
        self.model = model or settings.chat_llm_model
        self.enable_header_stealth = (
            enable_header_stealth
            if enable_header_stealth is not None
            else True  # default ON for production resilience
        )

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Single client instance per provider
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

        logger.debug(
            "Initialized OpenAI provider with base_url=%s, model=%s",
            self.base_url,
            self.model
        )

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
        stream = bool(payload.get("stream", False))

        # Set defaults
        payload.setdefault("model", self.model)
        payload.setdefault("max_tokens", 1024)

        headers = self._build_headers()

        # Use tenacity for retries
        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            reraise=True,
        )

        async for attempt in retryer:
            with attempt:
                resp = await self._client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers
                )
                resp.raise_for_status()

        if not stream:
            return resp.json()

        async def _stream_generator() -> AsyncGenerator[str, None]:
            try:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield line
            finally:
                await resp.aclose()

        return _stream_generator()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the API request."""
        base_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Only apply stealth heuristics for OpenRouter (reduces accidental blocking)
        if not self.enable_header_stealth or "openrouter" not in self.base_url:
            return base_headers

        # Generate stealth headers
        referer_hdr = "https://sentio.ai/"
        title_hdr = f"Sentio-{_random_suffix()}"

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
