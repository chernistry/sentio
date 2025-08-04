"""Redis-based caching implementation for high-performance caching.

This module provides Redis caching for embeddings, query results,
and other application data with automatic serialization and compression.
"""

import hashlib
import json
import logging
import pickle
import time
import zlib
from dataclasses import dataclass
from typing import Any

try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = logging.getLogger(__name__)


class RedisCacheError(Exception):
    """Exception raised for Redis cache operations."""


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    ttl: int | None = None
    compressed: bool = False
    serialization: str = "pickle"


class RedisCache:
    """High-performance Redis cache with compression and smart serialization.
    
    Features:
    - Automatic compression for large values
    - Multiple serialization formats (pickle, json)
    - TTL (time-to-live) support
    - Batch operations
    - Connection pooling
    - Circuit breaker pattern
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "sentio:",
        compression_threshold: int = 1000,  # bytes
        default_ttl: int = 3600,  # seconds
        max_connections: int = 20,
        health_check_interval: int = 30,
    ):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys
            compression_threshold: Compress values larger than this
            default_ttl: Default TTL in seconds
            max_connections: Maximum Redis connections
            health_check_interval: Health check interval in seconds
        """
        if not HAS_REDIS:
            raise RedisCacheError("redis package not installed")

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.compression_threshold = compression_threshold
        self.default_ttl = default_ttl
        self.max_connections = max_connections
        self.health_check_interval = health_check_interval

        self._client: redis.Redis | None = None
        self._connection_pool = None
        self._healthy = False
        self._last_health_check = 0

        logger.info(f"Redis cache initialized with URL: {redis_url}")

    async def _ensure_connected(self):
        """Ensure Redis connection is established."""
        if self._client is None:
            try:
                self._connection_pool = redis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=self.max_connections,
                    retry_on_timeout=True,
                    health_check_interval=self.health_check_interval,
                )
                self._client = redis.Redis(connection_pool=self._connection_pool)

                # Test connection
                await self._client.ping()
                self._healthy = True
                self._last_health_check = time.time()

                logger.info("Redis connection established")

            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._healthy = False
                raise RedisCacheError(f"Redis connection failed: {e}")

    async def _health_check(self):
        """Perform health check if needed."""
        current_time = time.time()
        if current_time - self._last_health_check > self.health_check_interval:
            try:
                if self._client:
                    await self._client.ping()
                self._healthy = True
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
                self._healthy = False
            finally:
                self._last_health_check = current_time

    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{self.key_prefix}{key}"

    def _serialize_value(self, value: Any) -> tuple[bytes, str, bool]:
        """Serialize value with optimal format selection.
        
        Returns:
            (serialized_data, format, compressed)
        """
        # Try JSON first for simple types
        if isinstance(value, (str, int, float, bool, list, dict)) and not isinstance(value, (bytes, bytearray)):
            try:
                serialized = json.dumps(value).encode("utf-8")
                format_type = "json"
            except (TypeError, ValueError):
                # Fall back to pickle
                serialized = pickle.dumps(value)
                format_type = "pickle"
        else:
            # Use pickle for complex objects
            serialized = pickle.dumps(value)
            format_type = "pickle"

        # Compress if large enough
        compressed = False
        if len(serialized) > self.compression_threshold:
            try:
                compressed_data = zlib.compress(serialized, level=6)
                if len(compressed_data) < len(serialized):
                    serialized = compressed_data
                    compressed = True
            except Exception as e:
                logger.warning(f"Compression failed: {e}")

        return serialized, format_type, compressed

    def _deserialize_value(self, data: bytes, format_type: str, compressed: bool) -> Any:
        """Deserialize cached value."""
        try:
            # Decompress if needed
            if compressed:
                data = zlib.decompress(data)

            # Deserialize based on format
            if format_type == "json":
                return json.loads(data.decode("utf-8"))
            return pickle.loads(data)

        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise RedisCacheError(f"Failed to deserialize cached value: {e}")

    async def get(self, key: str) -> Any | None:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        await self._ensure_connected()
        await self._health_check()

        if not self._healthy:
            logger.warning("Redis unhealthy, cache miss")
            return None

        try:
            full_key = self._make_key(key)
            data = await self._client.hgetall(full_key)

            if not data:
                return None

            # Check TTL
            if data.get(b"ttl"):
                ttl = int(data[b"ttl"])
                timestamp = float(data[b"timestamp"])
                if time.time() - timestamp > ttl:
                    # Expired, delete and return None
                    await self._client.delete(full_key)
                    return None

            # Deserialize
            value_data = data[b"value"]
            format_type = data[b"format"].decode("utf-8")
            compressed = data[b"compressed"] == b"1"

            return self._deserialize_value(value_data, format_type, compressed)

        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        await self._ensure_connected()
        await self._health_check()

        if not self._healthy:
            logger.warning("Redis unhealthy, cache set failed")
            return False

        try:
            full_key = self._make_key(key)

            # Serialize value
            serialized_data, format_type, compressed = self._serialize_value(value)

            # Prepare cache entry
            cache_data = {
                "value": serialized_data,
                "format": format_type,
                "compressed": "1" if compressed else "0",
                "timestamp": str(time.time()),
            }

            if ttl is not None:
                cache_data["ttl"] = str(ttl)
            elif self.default_ttl:
                cache_data["ttl"] = str(self.default_ttl)

            # Store in Redis
            await self._client.hset(full_key, mapping=cache_data)

            # Set expiration if TTL specified
            if ttl is not None:
                await self._client.expire(full_key, ttl)
            elif self.default_ttl:
                await self._client.expire(full_key, self.default_ttl)

            return True

        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
        """
        await self._ensure_connected()
        await self._health_check()

        if not self._healthy:
            return False

        try:
            full_key = self._make_key(key)
            result = await self._client.delete(full_key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists
        """
        await self._ensure_connected()
        await self._health_check()

        if not self._healthy:
            return False

        try:
            full_key = self._make_key(key)
            return await self._client.exists(full_key) > 0
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "embeddings:*")
            
        Returns:
            Number of keys deleted
        """
        await self._ensure_connected()
        await self._health_check()

        if not self._healthy:
            return 0

        try:
            full_pattern = self._make_key(pattern)
            keys = []
            async for key in self._client.scan_iter(match=full_pattern):
                keys.append(key)

            if keys:
                return await self._client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Redis clear pattern error for {pattern}: {e}")
            return 0

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        await self._ensure_connected()
        await self._health_check()

        stats = {
            "healthy": self._healthy,
            "connected": self._client is not None,
        }

        if self._healthy and self._client:
            try:
                info = await self._client.info()
                stats.update({
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                })

                # Calculate hit rate
                hits = stats["keyspace_hits"]
                misses = stats["keyspace_misses"]
                if hits + misses > 0:
                    stats["hit_rate"] = hits / (hits + misses)
                else:
                    stats["hit_rate"] = 0.0

            except Exception as e:
                logger.error(f"Failed to get Redis stats: {e}")
                stats["error"] = str(e)

        return stats

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._connection_pool:
            await self._connection_pool.disconnect()
            self._connection_pool = None
        logger.info("Redis connection closed")

    # Convenience methods for specific data types

    async def get_embedding_cache(self, text: str) -> list[float] | None:
        """Get cached embedding for text."""
        key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"
        return await self.get(key)

    async def set_embedding_cache(
        self,
        text: str,
        embedding: list[float],
        ttl: int = 3600
    ) -> bool:
        """Cache embedding for text."""
        key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"
        return await self.set(key, embedding, ttl)

    async def get_query_cache(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Get cached query result."""
        query_hash = hashlib.sha256(
            (query + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()
        key = f"query:{query_hash}"
        return await self.get(key)

    async def set_query_cache(
        self,
        query: str,
        params: dict[str, Any],
        result: dict[str, Any],
        ttl: int = 1800
    ) -> bool:
        """Cache query result."""
        query_hash = hashlib.sha256(
            (query + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()
        key = f"query:{query_hash}"
        return await self.set(key, result, ttl)
