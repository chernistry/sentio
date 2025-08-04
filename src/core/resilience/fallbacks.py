"""Fallback mechanisms for graceful degradation.

Provides fallback strategies when primary services are unavailable.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FallbackManager:
    """Manages fallback strategies for service failures.
    
    Provides cached responses, default values, and alternative service routing.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path(".fallback_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._cached_responses: dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached responses from disk."""
        cache_file = self.cache_dir / "responses.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    self._cached_responses = json.load(f)
                logger.info(f"Loaded {len(self._cached_responses)} cached responses")
            except Exception as e:
                logger.warning(f"Failed to load response cache: {e}")

    def _save_cache(self):
        """Save cached responses to disk."""
        cache_file = self.cache_dir / "responses.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(self._cached_responses, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save response cache: {e}")

    def cache_response(self, key: str, response: Any, ttl_seconds: int = 3600):
        """Cache a successful response for fallback use.
        
        Args:
            key: Cache key (usually query hash)
            response: Response to cache
            ttl_seconds: Time to live in seconds
        """
        import time

        cache_entry = {
            "response": response,
            "timestamp": time.time(),
            "ttl": ttl_seconds,
        }
        self._cached_responses[key] = cache_entry
        self._save_cache()

    def get_cached_response(self, key: str) -> Any | None:
        """Get cached response if still valid.
        
        Args:
            key: Cache key to lookup
            
        Returns:
            Cached response if valid, None otherwise
        """
        import time

        if key not in self._cached_responses:
            return None

        entry = self._cached_responses[key]
        if time.time() - entry["timestamp"] > entry["ttl"]:
            # Expired, remove from cache
            del self._cached_responses[key]
            self._save_cache()
            return None

        return entry["response"]

    def generate_cache_key(self, query: str, params: dict | None = None) -> str:
        """Generate consistent cache key from query and parameters."""
        content = query
        if params:
            content += json.dumps(params, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    async def with_fallback(
        self,
        primary_func: Callable[..., T],
        fallback_func: Callable[..., T] | None = None,
        cache_key: str | None = None,
        default_response: T | None = None,
        *args,
        **kwargs,
    ) -> T:
        """Execute function with fallback strategies.
        
        Args:
            primary_func: Primary function to execute
            fallback_func: Optional fallback function
            cache_key: Optional cache key for response caching
            default_response: Default response if all else fails
            *args: Arguments for functions
            **kwargs: Keyword arguments for functions
            
        Returns:
            Result from primary function, fallback, cache, or default
            
        Raises:
            Exception: If all fallback strategies fail
        """
        # Try primary function
        try:
            result = await primary_func(*args, **kwargs)

            # Cache successful response
            if cache_key:
                self.cache_response(cache_key, result)

            return result
        except Exception as e:
            logger.warning(f"Primary function failed: {e}")

        # Try fallback function
        if fallback_func:
            try:
                result = await fallback_func(*args, **kwargs)
                logger.info("Used fallback function successfully")
                return result
            except Exception as e:
                logger.warning(f"Fallback function failed: {e}")

        # Try cached response
        if cache_key:
            cached = self.get_cached_response(cache_key)
            if cached is not None:
                logger.info("Using cached response as fallback")
                return cached

        # Use default response
        if default_response is not None:
            logger.info("Using default response as fallback")
            return default_response

        # All fallbacks failed
        raise Exception("All fallback strategies exhausted")


class EmbeddingFallback:
    """Fallback strategies for embedding generation."""

    def __init__(self):
        self.simple_embeddings = {}  # Simple keyword-based embeddings

    async def generate_simple_embedding(self, text: str, dimension: int = 384) -> list[float]:
        """Generate simple embedding based on text characteristics.
        
        This is a last-resort fallback when all embedding services fail.
        """
        import hashlib
        import math

        # Generate deterministic embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Convert hash to numbers and normalize
        embedding = []
        for i in range(dimension):
            # Use different parts of hash for each dimension
            hash_part = text_hash[(i * 2) % len(text_hash):(i * 2 + 2) % len(text_hash)]
            if len(hash_part) < 2:
                hash_part = text_hash[0:2]

            # Convert to float between -1 and 1
            value = int(hash_part, 16) / 255.0 * 2 - 1
            embedding.append(value)

        # Add some text characteristics
        if len(embedding) > 10:
            embedding[0] = len(text) / 1000.0  # Text length signal
            embedding[1] = text.count(" ") / len(text) if text else 0  # Word density
            embedding[2] = sum(1 for c in text if c.isupper()) / len(text) if text else 0  # Uppercase ratio

        # Normalize to unit vector
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding


class LLMFallback:
    """Fallback strategies for LLM generation."""

    def __init__(self):
        self.template_responses = {
            "default": "I'm sorry, but I'm currently unable to process your request due to a temporary service issue. Please try again later.",
            "search": "I found some information related to your query, but I'm unable to provide a detailed response at the moment. Please try again later.",
            "error": "An error occurred while processing your request. Please check your input and try again.",
        }

    async def generate_fallback_response(
        self,
        query: str,
        context_docs: list[str] | None = None,
        response_type: str = "default",
    ) -> str:
        """Generate fallback response when LLM services are unavailable.
        
        Args:
            query: User query
            context_docs: Available context documents
            response_type: Type of fallback response
            
        Returns:
            Fallback response string
        """
        base_response = self.template_responses.get(response_type, self.template_responses["default"])

        # If we have context documents, try to provide some basic info
        if context_docs and response_type == "search":
            doc_count = len(context_docs)
            return f"I found {doc_count} relevant document{'s' if doc_count != 1 else ''} related to your query about '{query[:50]}...', but I'm currently unable to provide a detailed analysis. Please try again later."

        return base_response


# Global fallback manager instance
fallback_manager = FallbackManager()
embedding_fallback = EmbeddingFallback()
llm_fallback = LLMFallback()
