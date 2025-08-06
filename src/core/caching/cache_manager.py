"""Unified cache manager with multiple backend support.

This module provides a high-level cache interface that can
use Redis, memory cache, or both in a multi-tier configuration.
"""

import logging
import os
from enum import Enum
from typing import Any

from .memory_cache import MemoryCache
from .redis_cache import RedisCache, RedisCacheError

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    MULTI_TIER = "multi_tier"


class CacheManager:
    """Unified cache manager with multi-tier support.
    
    Supports:
    - Memory-only caching (fast, limited capacity)
    - Redis-only caching (persistent, scalable)
    - Multi-tier caching (memory L1, Redis L2)
    
    Multi-tier mode uses memory cache as L1 (fast access) and
    Redis as L2 (persistent, larger capacity).
    """

    def __init__(
        self,
        backend: CacheBackend = CacheBackend.MEMORY,
        redis_url: str | None = None,
        memory_max_size: int = 1000,
        memory_ttl: float | None = 3600,
        redis_ttl: int = 3600,
        enable_fallback: bool = True,
    ):
        """Initialize cache manager.
        
        Args:
            backend: Cache backend type
            redis_url: Redis URL (required for Redis/multi-tier)
            memory_max_size: Memory cache max size
            memory_ttl: Memory cache default TTL
            redis_ttl: Redis cache default TTL
            enable_fallback: Enable fallback between tiers
        """
        self.backend = backend
        self.enable_fallback = enable_fallback

        # Initialize memory cache
        self.memory_cache = MemoryCache(
            max_size=memory_max_size,
            default_ttl=memory_ttl,
        )

        # Initialize Redis cache if needed
        self.redis_cache: RedisCache | None = None
        if backend in (CacheBackend.REDIS, CacheBackend.MULTI_TIER):
            if not redis_url:
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

            try:
                self.redis_cache = RedisCache(
                    redis_url=redis_url,
                    default_ttl=redis_ttl,
                )
                logger.info(f"Cache manager initialized with {backend.value} backend")
            except RedisCacheError as e:
                if backend == CacheBackend.REDIS:
                    # Redis-only mode, re-raise error
                    raise
                # Multi-tier mode, fall back to memory only
                logger.warning(f"Redis unavailable, falling back to memory: {e}")
                self.backend = CacheBackend.MEMORY
                self.redis_cache = None

    async def get(self, key: str) -> Any | None:
        """Get value from cache.
        
        In multi-tier mode, checks L1 (memory) first, then L2 (Redis).
        If found in L2, promotes to L1.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if self.backend == CacheBackend.MEMORY:
            return self.memory_cache.get(key)

        if self.backend == CacheBackend.REDIS:
            if self.redis_cache:
                return await self.redis_cache.get(key)
            if self.enable_fallback:
                return self.memory_cache.get(key)
            return None

        if self.backend == CacheBackend.MULTI_TIER:
            # Check L1 (memory) first
            value = self.memory_cache.get(key)
            if value is not None:
                return value

            # Check L2 (Redis)
            if self.redis_cache:
                try:
                    value = await self.redis_cache.get(key)
                    if value is not None:
                        # Promote to L1
                        self.memory_cache.set(key, value)
                        return value
                except Exception as e:
                    logger.warning(f"Redis get failed, key={key}: {e}")

            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
        tier: str | None = None,
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live
            tier: Specific tier to write to ('memory', 'redis', or None for all)
            
        Returns:
            True if successful
        """
        success = True

        if self.backend == CacheBackend.MEMORY:
            return self.memory_cache.set(key, value, ttl=ttl)

        if self.backend == CacheBackend.REDIS:
            if self.redis_cache:
                ttl_int = int(ttl) if ttl else None
                return await self.redis_cache.set(key, value, ttl_int)
            if self.enable_fallback:
                return self.memory_cache.set(key, value, ttl=ttl)
            return False

        if self.backend == CacheBackend.MULTI_TIER:
            # Write to both tiers unless specific tier requested
            if tier != "redis":
                success &= self.memory_cache.set(key, value, ttl=ttl)

            if tier != "memory" and self.redis_cache:
                try:
                    ttl_int = int(ttl) if ttl else None
                    success &= await self.redis_cache.set(key, value, ttl_int)
                except Exception as e:
                    logger.warning(f"Redis set failed, key={key}: {e}")
                    success = False

            return success

    async def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted from any tier
        """
        deleted = False

        if self.backend == CacheBackend.MEMORY:
            return self.memory_cache.delete(key)

        if self.backend == CacheBackend.REDIS:
            if self.redis_cache:
                return await self.redis_cache.delete(key)
            if self.enable_fallback:
                return self.memory_cache.delete(key)
            return False

        if self.backend == CacheBackend.MULTI_TIER:
            # Delete from both tiers
            deleted |= self.memory_cache.delete(key)

            if self.redis_cache:
                try:
                    deleted |= await self.redis_cache.delete(key)
                except Exception as e:
                    logger.warning(f"Redis delete failed, key={key}: {e}")

            return deleted

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists
        """
        if self.backend == CacheBackend.MEMORY:
            return self.memory_cache.exists(key)

        if self.backend == CacheBackend.REDIS:
            if self.redis_cache:
                return await self.redis_cache.exists(key)
            if self.enable_fallback:
                return self.memory_cache.exists(key)
            return False

        if self.backend == CacheBackend.MULTI_TIER:
            # Check L1 first
            if self.memory_cache.exists(key):
                return True

            # Check L2
            if self.redis_cache:
                try:
                    return await self.redis_cache.exists(key)
                except Exception as e:
                    logger.warning(f"Redis exists failed, key={key}: {e}")

            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern.
        
        Args:
            pattern: Key pattern
            
        Returns:
            Number of keys deleted
        """
        total_deleted = 0

        if self.backend == CacheBackend.MEMORY:
            return self.memory_cache.clear_pattern(pattern)

        if self.backend == CacheBackend.REDIS:
            if self.redis_cache:
                return await self.redis_cache.clear_pattern(pattern)
            if self.enable_fallback:
                return self.memory_cache.clear_pattern(pattern)
            return 0

        if self.backend == CacheBackend.MULTI_TIER:
            # Clear from both tiers
            total_deleted += self.memory_cache.clear_pattern(pattern)

            if self.redis_cache:
                try:
                    total_deleted += await self.redis_cache.clear_pattern(pattern)
                except Exception as e:
                    logger.warning(f"Redis clear pattern failed, pattern={pattern}: {e}")

            return total_deleted

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "backend": self.backend.value,
            "memory": self.memory_cache.get_stats(),
        }

        if self.redis_cache:
            try:
                stats["redis"] = await self.redis_cache.get_stats()
            except Exception as e:
                stats["redis"] = {"error": str(e)}

        return stats

    async def close(self):
        """Close cache connections."""
        if self.redis_cache:
            await self.redis_cache.close()

    # Convenience methods for specific data types

    async def get_embedding_cache(self, text: str) -> list[float] | None:
        """Get cached embedding for text."""
        if self.backend == CacheBackend.MEMORY:
            return self.memory_cache.get_embedding_cache(text)
        if self.redis_cache:
            return await self.redis_cache.get_embedding_cache(text)
        return None

    async def set_embedding_cache(
        self,
        text: str,
        embedding: list[float],
        ttl: float | None = 3600
    ) -> bool:
        """Cache embedding for text."""
        if self.backend == CacheBackend.MEMORY:
            return self.memory_cache.set_embedding_cache(text, embedding, ttl)
        if self.redis_cache:
            ttl_int = int(ttl) if ttl else 3600
            return await self.redis_cache.set_embedding_cache(text, embedding, ttl_int)
        return False

    async def get_query_cache(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Get cached query result."""
        if self.backend == CacheBackend.MEMORY:
            return self.memory_cache.get_query_cache(query, params)
        if self.redis_cache:
            return await self.redis_cache.get_query_cache(query, params)
        return None

    async def set_query_cache(
        self,
        query: str,
        params: dict[str, Any],
        result: dict[str, Any],
        ttl: float | None = 1800
    ) -> bool:
        """Cache query result."""
        if self.backend == CacheBackend.MEMORY:
            return self.memory_cache.set_query_cache(query, params, result, ttl)
        if self.redis_cache:
            ttl_int = int(ttl) if ttl else 1800
            return await self.redis_cache.set_query_cache(query, params, result, ttl_int)
        return False


# Global cache manager instance
_cache_manager: CacheManager | None = None


def get_cache_manager(
    backend: CacheBackend | None = None,
    redis_url: str | None = None,
    **kwargs
) -> CacheManager:
    """Get or create global cache manager instance.
    
    Args:
        backend: Cache backend type
        redis_url: Redis URL
        **kwargs: Additional arguments for CacheManager
        
    Returns:
        CacheManager instance
    """
    global _cache_manager

    if _cache_manager is None:
        # Determine backend from environment if not specified
        if backend is None:
            backend_str = os.getenv("CACHE_BACKEND", "memory").lower()
            try:
                backend = CacheBackend(backend_str)
            except ValueError:
                logger.warning(f"Invalid cache backend '{backend_str}', using memory")
                backend = CacheBackend.MEMORY

        _cache_manager = CacheManager(
            backend=backend,
            redis_url=redis_url,
            **kwargs
        )

    return _cache_manager
