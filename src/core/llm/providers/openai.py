"""OpenAI-compatible chat provider."""

import logging
import random
import string
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any, List

import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from src.core.llm.providers import register_provider
from src.core.llm.providers.base import BaseLLMProvider
from src.core.models.document import Document
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
                    headers=headers,
                )
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:  # type: ignore[name-defined]
                    # OpenRouter occasionally varies between '/api/v1' and '/v1'.
                    # If we get 404 and the base URL contains '/api/', try a
                    # fallback request without the '/api' segment.
                    if (
                        exc.response is not None
                        and exc.response.status_code == 404
                        and "openrouter.ai" in self.base_url
                        and "/api/" in self.base_url
                    ):
                        alt_base = self.base_url.replace("/api", "", 1)
                        async with httpx.AsyncClient(base_url=alt_base, timeout=60.0) as alt_client:
                            alt_resp = await alt_client.post(
                                "/chat/completions",
                                json=payload,
                                headers=headers,
                            )
                            alt_resp.raise_for_status()
                            resp = alt_resp
                    else:
                        raise

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

    async def generate_response(
        self,
        query: str,
        documents: List[Document],
        history: List[dict] = None
    ) -> str:
        """Generate a response using the OpenAI API.
        
        Args:
            query: The user's query
            documents: List of relevant documents
            history: Conversation history
            
        Returns:
            Generated response text
        """
        if history is None:
            history = []
            
        messages = self._build_messages(query, documents, history)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        response = await self.chat_completion(payload)
        
        if isinstance(response, dict):
            return response["choices"][0]["message"]["content"]
        else:
            # Handle streaming response
            content = ""
            async for chunk in response:
                content += chunk
            return content

    def _build_messages(
        self,
        query: str,
        documents: List[Document],
        history: List[dict]
    ) -> List[dict]:
        """Build messages for the chat completion.
        
        Args:
            query: The user's query
            documents: List of relevant documents
            history: Conversation history
            
        Returns:
            List of messages for the API
        """
        messages = []
        
        # Add system message with context
        if documents:
            context = "\n\n".join([doc.text for doc in documents])
            system_message = f"""You are a helpful assistant. Use the following context to answer the user's question:

Context:
{context}

Please provide a helpful and accurate response based on the context provided."""
            messages.append({"role": "system", "content": system_message})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        
        # Add conversation history
        messages.extend(history)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages

    async def health_check(self) -> bool:
        """Check if the OpenAI API is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            
            response = await self.chat_completion(payload)
            return isinstance(response, dict) and "choices" in response
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model(self.model)
            return len(encoder.encode(text))
        except ImportError:
            # Fallback to rough estimation
            return len(text.split()) * 1.3  # Rough approximation
        except Exception:
            # Fallback to word count
            return len(text.split())

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the API request."""
        base_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
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
            # Standard header plus OpenRouter's documented variant
            "Referer": referer_hdr,
            "HTTP-Referer": referer_hdr,
            "X-Title": title_hdr,
            "User-Agent": ua_hdr,
        }
