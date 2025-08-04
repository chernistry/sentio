"""In-memory caching implementation with LRU eviction.

This module provides a fast, thread-safe in-memory cache with
configurable size limits and TTL support.
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheItem:
    """Cache item with metadata."""
    value: Any
    timestamp: float
    ttl: float | None = None
    access_count: int = 0
    last_access: float = 0.0

    def is_expired(self, current_time: float | None = None) -> bool:
        """Check if item is expired."""
        if self.ttl is None:
            return False
        current_time = current_time or time.time()
        return current_time - self.timestamp > self.ttl


class MemoryCache:
    """Thread-safe in-memory cache with LRU eviction and TTL support.
    
    Features:
    - LRU (Least Recently Used) eviction
    - TTL (Time To Live) support
    - Thread-safe operations
    - Memory usage tracking
    - Hit/miss statistics
    - Pattern-based deletion
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float | None = None,
        cleanup_interval: float = 300,  # 5 minutes
    ):
        """Initialize memory cache.
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default TTL in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "expirations": 0,
        }
        self._last_cleanup = time.time()

        logger.info(f"Memory cache initialized with max_size={max_size}")

    def _cleanup_expired(self, current_time: float | None = None) -> int:
        """Remove expired items from cache."""
        current_time = current_time or time.time()

        # Only cleanup if enough time has passed
        if current_time - self._last_cleanup < self.cleanup_interval:
            return 0

        expired_keys = []
        for key, item in self._cache.items():
            if item.is_expired(current_time):
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            self._stats["expirations"] += 1

        self._last_cleanup = current_time
        return len(expired_keys)

    def _evict_lru(self) -> int:
        """Evict least recently used items to make room."""
        evicted = 0
        while len(self._cache) >= self.max_size:
            # Remove least recently used item (first in OrderedDict)
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            evicted += 1
        return evicted

    def get(self, key: str) -> Any | None:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            current_time = time.time()

            # Cleanup expired items occasionally
            self._cleanup_expired(current_time)

            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            item = self._cache[key]

            # Check if expired
            if item.is_expired(current_time):
                del self._cache[key]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Update access stats
            item.access_count += 1
            item.last_access = current_time

            self._stats["hits"] += 1
            return item.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        with self._lock:
            current_time = time.time()

            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl

            # Create cache item
            item = CacheItem(
                value=value,
                timestamp=current_time,
                ttl=ttl,
                access_count=1,
                last_access=current_time,
            )

            # If key already exists, update it
            if key in self._cache:
                self._cache[key] = item
                self._cache.move_to_end(key)
            else:
                # Add new item
                self._cache[key] = item

                # Evict if necessary
                if len(self._cache) > self.max_size:
                    self._evict_lru()

            self._stats["sets"] += 1
            return True

    def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["deletes"] += 1
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists and not expired
        """
        with self._lock:
            if key not in self._cache:
                return False

            item = self._cache[key]
            if item.is_expired():
                del self._cache[key]
                self._stats["expirations"] += 1
                return False

            return True

    def clear(self) -> int:
        """Clear all items from cache.
        
        Returns:
            Number of items cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern.
        
        Args:
            pattern: Key pattern (supports * wildcard)
            
        Returns:
            Number of keys deleted
        """
        import fnmatch

        with self._lock:
            matching_keys = [
                key for key in self._cache.keys()
                if fnmatch.fnmatch(key, pattern)
            ]

            for key in matching_keys:
                del self._cache[key]

            return len(matching_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            current_time = time.time()

            # Count expired items
            expired_count = sum(
                1 for item in self._cache.values()
                if item.is_expired(current_time)
            )

            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "expired_items": expired_count,
                "hit_rate": hit_rate,
                "memory_usage_estimate": self._estimate_memory_usage(),
                **self._stats,
            }

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimate - this is not precise but gives an idea
        import sys

        total_size = 0
        for key, item in self._cache.items():
            total_size += sys.getsizeof(key)
            total_size += sys.getsizeof(item.value)
            total_size += sys.getsizeof(item)

        return total_size

    def get_items_info(self) -> list[dict[str, Any]]:
        """Get information about cached items.
        
        Returns:
            List of item information dictionaries
        """
        with self._lock:
            current_time = time.time()
            items_info = []

            for key, item in self._cache.items():
                age = current_time - item.timestamp
                time_to_expire = None
                if item.ttl is not None:
                    time_to_expire = max(0, item.ttl - age)

                items_info.append({
                    "key": key,
                    "age_seconds": age,
                    "ttl_seconds": item.ttl,
                    "time_to_expire": time_to_expire,
                    "access_count": item.access_count,
                    "last_access": current_time - item.last_access,
                    "expired": item.is_expired(current_time),
                })

            return items_info

    # Convenience methods for specific data types

    def get_embedding_cache(self, text: str) -> list[float] | None:
        """Get cached embedding for text."""
        key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"
        return self.get(key)

    def set_embedding_cache(
        self,
        text: str,
        embedding: list[float],
        ttl: float | None = 3600
    ) -> bool:
        """Cache embedding for text."""
        key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"
        return self.set(key, embedding, ttl)

    def get_query_cache(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Get cached query result."""
        query_hash = hashlib.sha256(
            (query + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()
        key = f"query:{query_hash}"
        return self.get(key)

    def set_query_cache(
        self,
        query: str,
        params: dict[str, Any],
        result: dict[str, Any],
        ttl: float | None = 1800
    ) -> bool:
        """Cache query result."""
        query_hash = hashlib.sha256(
            (query + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()
        key = f"query:{query_hash}"
        return self.set(key, result, ttl)
