"""Caching layer for performance optimization.

This package provides comprehensive caching mechanisms for embeddings,
query results, and vector operations with multiple backend support.
"""

from .cache_manager import CacheManager, get_cache_manager
from .memory_cache import MemoryCache
from .redis_cache import RedisCache, RedisCacheError
from .strategies import CacheStrategy, LRUStrategy, TTLStrategy

__all__ = [
    "CacheManager",
    "CacheStrategy",
    "LRUStrategy",
    "MemoryCache",
    "RedisCache",
    "RedisCacheError",
    "TTLStrategy",
    "get_cache_manager",
]
