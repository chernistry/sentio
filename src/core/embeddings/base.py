"""Base classes for embedding models.

This module defines the core interface for embedding models and provides
a caching mechanism to improve efficiency.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

logger = logging.getLogger(__name__)

# Type for cache values
T = TypeVar("T")


class EmbeddingError(Exception):
    """Exception raised for embedding-related errors."""


class EmbeddingCache(Generic[T]):
    """LFU cache implementation for embedding vectors with TTL support.
    
    This cache uses a least-frequently-used eviction policy and supports
    time-to-live expiry for cache entries.
    """

    def __init__(
        self,
        max_size: int = 10_000,
        ttl: int = 3600,
    ) -> None:
        """Initialize the cache.
        
        Args:
            max_size: Maximum number of items in cache.
            ttl: Time-to-live in seconds.
        """
        self._cache: dict[str, dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> T | None:
        """Get item from cache if it exists and hasn't expired.
        
        Args:
            key: The cache key.
            
        Returns:
            The cached value or None if not found or expired.
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check if expired
        if self._ttl > 0 and time.time() - entry["timestamp"] > self._ttl:
            # Remove expired item
            del self._cache[key]
            return None

        # Update access count
        entry["access_count"] += 1
        return cast("T", entry["value"])

    def put(self, key: str, value: T) -> None:
        """Add item to cache, evicting least frequently used item if needed.
        
        Args:
            key: The cache key.
            value: The value to cache.
        """
        # Evict least frequently used items if needed
        if len(self._cache) >= self._max_size and key not in self._cache:
            self._evict_lfu()

        # Store with timestamp and initial access count
        self._cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "access_count": 1
        }

    def _evict_lfu(self) -> None:
        """Remove least frequently used item from cache."""
        if not self._cache:
            return

        # Find entry with lowest access count
        min_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k]["access_count"]
        )
        del self._cache[min_key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Get current size of cache."""
        return len(self._cache)


def _retry(max_attempts: int = 3):
    """Retry decorator for embedding operations.
    
    Args:
        max_attempts: Maximum number of retry attempts.
        
    Returns:
        Decorated function that will retry on failure.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempts = 0
            last_error = None

            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    last_error = e
                    if attempts < max_attempts:
                        wait_time = 0.5 * (2 ** attempts)  # Exponential backoff
                        logger.warning(
                            f"Embedding attempt {attempts} failed: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} embedding attempts failed.")

            # Re-raise the last error if all attempts failed
            raise last_error  # type: ignore

        return wrapper
    return decorator


class BaseEmbedder(ABC):
    """Abstract base class for all embedding models.
    
    This class defines the interface that all embedding implementations must follow
    and provides common caching functionality.
    """

    def __init__(
        self,
        model_name: str,
        cache_enabled: bool = True,
        cache_size: int = 10_000,
        cache_ttl: int = 3600,
        **kwargs: Any
    ) -> None:
        """Initialize the base embedder.
        
        Args:
            model_name: Name of the embedding model.
            cache_enabled: Whether to enable caching.
            cache_size: Maximum cache size.
            cache_ttl: Cache time-to-live in seconds.
            **kwargs: Additional keyword arguments passed to subclasses.
        """
        self.model_name = model_name
        self._cache_enabled = cache_enabled
        self._dimension: int | None = None

        # Initialize cache if enabled
        self._cache = EmbeddingCache[list[float]](max_size=cache_size, ttl=cache_ttl) if cache_enabled else None

        # Statistics tracking
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_time": 0.0,
        }

    @property
    def dimension(self) -> int:
        """Get the embedding dimension for this model.
        
        Returns:
            The dimension of the embedding vectors.
        
        Raises:
            EmbeddingError: If the dimension is not known.
        """
        if self._dimension is None:
            # Lazily compute dimension on first access
            self._dimension = self._get_embedding_dimension()

        return self._dimension

    @abstractmethod
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension for this model.
        
        This method should be implemented by subclasses to determine the
        dimension of the embedding vectors produced by the model.
        
        Returns:
            The dimension of the embedding vectors.
        
        Raises:
            EmbeddingError: If the dimension cannot be determined.
        """

    def _check_cache(self, text: str) -> list[float] | None:
        """Check if embedding for text is in cache.
        
        Args:
            text: The text to check.
            
        Returns:
            The cached embedding or None if not found.
        """
        if not self._cache_enabled or not self._cache:
            return None

        # Normalize key text by trimming whitespace
        key = text.strip()
        return self._cache.get(key)

    def _store_cache(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache.
        
        Args:
            text: The input text.
            embedding: The embedding vector.
        """
        if not self._cache_enabled or not self._cache:
            return

        # Normalize key text by trimming whitespace
        key = text.strip()
        self._cache.put(key, embedding)

    def _update_stats(self, hit: bool = False, error: bool = False, duration: float = 0.0) -> None:
        """Update usage statistics.
        
        Args:
            hit: Whether this was a cache hit.
            error: Whether an error occurred.
            duration: Operation duration in seconds.
        """
        self._stats["total_requests"] += 1
        if hit:
            self._stats["cache_hits"] += 1
        if error:
            self._stats["errors"] += 1
        self._stats["total_time"] += duration

    @property
    def stats(self) -> dict[str, Any]:
        """Get current usage statistics.
        
        Returns:
            Dictionary with usage statistics.
        """
        stats = self._stats.copy()
        # Add derived metrics
        total = stats["total_requests"]
        if total > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total
            stats["avg_time"] = stats["total_time"] / total
        if self._cache_enabled and self._cache:
            stats["cache_size"] = self._cache.size
        return stats

    def reset_stats(self) -> None:
        """Reset all usage statistics."""
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_time": 0.0,
        }

    @abstractmethod
    async def embed_async_single(self, text: str) -> list[float]:
        """Get embedding for a single text asynchronously.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector.
            
        Raises:
            EmbeddingError: If embedding fails.
        """

    @abstractmethod
    async def embed_async_many(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts asynchronously.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            EmbeddingError: If embedding fails.
        """

    def embed_sync(self, text: str) -> list[float]:
        """Get embedding for a single text synchronously.
        
        This default implementation wraps the async method with a new event loop.
        Subclasses may override with more efficient implementations.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector.
            
        Raises:
            EmbeddingError: If embedding fails.
        """
        start_time = time.time()

        # Check cache first
        cached = self._check_cache(text)
        if cached is not None:
            self._update_stats(hit=True, duration=time.time() - start_time)
            return cached

        try:
            # Create a new event loop to run the async method
            loop = asyncio.new_event_loop()
            try:
                embedding = loop.run_until_complete(self.embed_async_single(text))
            finally:
                loop.close()

            # Cache the result
            self._store_cache(text, embedding)

            self._update_stats(duration=time.time() - start_time)
            return embedding
        except Exception as e:
            self._update_stats(error=True, duration=time.time() - start_time)
            logger.error(f"Error getting embedding: {e}")
            raise EmbeddingError(f"Failed to embed text: {e}") from e

    def embed_many_sync(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts synchronously.
        
        This default implementation wraps the async method with a new event loop.
        Subclasses may override with more efficient implementations.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            EmbeddingError: If embedding fails.
        """
        start_time = time.time()

        try:
            # Create a new event loop to run the async method
            loop = asyncio.new_event_loop()
            try:
                embeddings = loop.run_until_complete(self.embed_async_many(texts))
            finally:
                loop.close()

            self._update_stats(duration=time.time() - start_time)
            return embeddings
        except Exception as e:
            self._update_stats(error=True, duration=time.time() - start_time)
            logger.error(f"Error getting batch embeddings: {e}")
            raise EmbeddingError(f"Failed to embed texts: {e}") from e

    async def warm_up(self, sample_texts: list[str] | None = None) -> bool:
        """Warm up the embedding model by running test queries.
        
        Args:
            sample_texts: Optional list of texts to use for warming up.
                If not provided, a default sample will be used.
                
        Returns:
            True if warm-up succeeded, False otherwise.
        """
        samples = sample_texts or [
            "This is a test sentence for warming up the embedding model.",
            "Second test sentence with different content to ensure cache doesn't interfere."
        ]

        logger.info(f"Warming up {self.model_name} embedding model...")
        try:
            # Time the warm-up
            start = time.time()
            await self.embed_async_many(samples)
            duration = time.time() - start

            # Get dimension to ensure it's cached
            _ = self.dimension

            logger.info(f"✓ {self.model_name} warm-up complete ({duration:.2f}s)")
            return True
        except Exception as e:
            logger.error(f"✗ {self.model_name} warm-up failed: {e}")
            return False

    async def close(self) -> None:
        """Close any resources held by the embedding model.
        
        Subclasses should override this if they maintain persistent resources.
        """
