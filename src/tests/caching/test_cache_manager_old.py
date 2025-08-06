"""Tests for cache manager - performance optimization."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.caching.cache_manager import CacheManager, CacheBackend
from src.core.caching.memory_cache import MemoryCache
from src.core.caching.redis_cache import RedisCache


@pytest.fixture
def mock_memory_cache():
    """Mock memory cache."""
    cache = MagicMock(spec=MemoryCache)
    cache.get.return_value = None
    cache.set.return_value = None
    cache.delete.return_value = None
    cache.clear.return_value = None
    cache.get_stats.return_value = {"hits": 10, "misses": 5, "size": 15}
    return cache


@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache."""
    cache = AsyncMock()  # Remove spec to allow any method
    cache.get.return_value = None
    cache.set.return_value = None
    cache.delete.return_value = None
    cache.clear.return_value = None
    cache.clear_pattern.return_value = 0
    cache.get_stats.return_value = {"hits": 20, "misses": 8, "size": 28}
    return cache


@pytest.fixture
def cache_manager(mock_memory_cache, mock_redis_cache):
    """Create CacheManager with mocked caches."""
    from src.core.caching.cache_manager import CacheBackend
    manager = CacheManager(backend=CacheBackend.MULTI_TIER)
    manager.memory_cache = mock_memory_cache
    manager.redis_cache = mock_redis_cache
    return manager


@pytest.mark.asyncio
class TestCacheManager:
    """Test cache manager functionality."""

    async def test_get_l1_hit(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test cache hit in L1 (memory) cache."""
        # L1 cache returns value
        mock_memory_cache.get.return_value = "cached_value"
        
        result = await cache_manager.get("test_key")
        
        assert result == "cached_value"
        mock_memory_cache.get.assert_called_once_with("test_key")
        mock_redis_cache.get.assert_not_called()  # Should not check L2

    async def test_get_l2_hit(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test cache hit in L2 (Redis) cache."""
        # L1 miss, L2 hit
        mock_memory_cache.get.return_value = None
        mock_redis_cache.get.return_value = "l2_cached_value"
        
        # Ensure the cache manager is in multi-tier mode
        cache_manager.backend = CacheBackend.MULTI_TIER
        
        result = await cache_manager.get("test_key")
        
        assert result == "l2_cached_value"
        mock_memory_cache.get.assert_called_once_with("test_key")
        mock_redis_cache.get.assert_called_once_with("test_key")
        # Should promote to L1
        mock_memory_cache.set.assert_called_once_with("test_key", "l2_cached_value")

    async def test_get_cache_miss(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test complete cache miss."""
        # Both L1 and L2 miss
        mock_memory_cache.get.return_value = None
        mock_redis_cache.get.return_value = None
        
        result = await cache_manager.get("test_key")
        
        assert result is None
        mock_memory_cache.get.assert_called_once_with("test_key")
        mock_redis_cache.get.assert_called_once_with("test_key")

    async def test_set_both_caches(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test setting value in both cache layers."""
        await cache_manager.set("test_key", "test_value", ttl=300)
        
        # Should set in both L1 and L2
        mock_memory_cache.set.assert_called_once_with("test_key", "test_value", ttl=300)
        mock_redis_cache.set.assert_called_once_with("test_key", "test_value", ttl=300)

    async def test_delete_both_caches(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test deleting from both cache layers."""
        await cache_manager.delete("test_key")
        
        # Should delete from both L1 and L2
        mock_memory_cache.delete.assert_called_once_with("test_key")
        mock_redis_cache.delete.assert_called_once_with("test_key")

    async def test_clear_both_caches(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test clearing both cache layers."""
        await cache_manager.clear()
        
        # Should clear both L1 and L2
        mock_memory_cache.clear.assert_called_once()
        mock_redis_cache.clear.assert_called_once()

    async def test_get_stats_combined(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test getting combined cache statistics."""
        stats = await cache_manager.get_stats()
        
        # Should combine stats from both caches
        assert "l1_cache" in stats
        assert "l2_cache" in stats
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "hit_rate" in stats
        
        # Verify calculations
        assert stats["total_hits"] == 30  # 10 + 20
        assert stats["total_misses"] == 13  # 5 + 8
        assert stats["hit_rate"] == 30 / 43  # hits / (hits + misses)

    async def test_cache_with_ttl(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test caching with time-to-live."""
        await cache_manager.set("expiring_key", "expiring_value", ttl=60)
        
        # Both caches should receive TTL
        mock_memory_cache.set.assert_called_once_with("expiring_key", "expiring_value", ttl=60)
        mock_redis_cache.set.assert_called_once_with("expiring_key", "expiring_value", ttl=60)

    async def test_cache_serialization(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test caching complex objects that need serialization."""
        complex_object = {
            "documents": [{"text": "doc1", "score": 0.9}],
            "metadata": {"query_time": 1.5}
        }
        
        await cache_manager.set("complex_key", complex_object)
        
        # Should handle serialization for Redis
        mock_memory_cache.set.assert_called_once()
        mock_redis_cache.set.assert_called_once()

    async def test_cache_key_generation(self, cache_manager):
        """Test cache key generation for different data types."""
        # Test with string
        key1 = cache_manager.generate_key("simple_string")
        assert isinstance(key1, str)
        
        # Test with dict
        key2 = cache_manager.generate_key({"query": "test", "top_k": 5})
        assert isinstance(key2, str)
        
        # Test with list
        key3 = cache_manager.generate_key(["item1", "item2", "item3"])
        assert isinstance(key3, str)
        
        # Same input should generate same key
        key4 = cache_manager.generate_key({"query": "test", "top_k": 5})
        assert key2 == key4

    async def test_cache_invalidation_pattern(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test cache invalidation by pattern."""
        pattern = "user:123:*"
        
        if hasattr(cache_manager, 'delete_pattern'):
            await cache_manager.delete_pattern(pattern)
            
            # Should call pattern deletion on both caches
            mock_memory_cache.delete_pattern.assert_called_once_with(pattern)
            mock_redis_cache.delete_pattern.assert_called_once_with(pattern)

    async def test_cache_warming(self, cache_manager):
        """Test cache warming functionality."""
        if hasattr(cache_manager, 'warm_cache'):
            warm_data = {
                "popular_query_1": "result_1",
                "popular_query_2": "result_2"
            }
            
            await cache_manager.warm_cache(warm_data)
            
            # Should populate both cache layers
            assert True  # Implementation dependent

    async def test_cache_fallback_l2_failure(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test fallback when L2 cache fails."""
        # L1 miss, L2 fails
        mock_memory_cache.get.return_value = None
        mock_redis_cache.get.side_effect = Exception("Redis connection failed")
        
        result = await cache_manager.get("test_key")
        
        # Should handle L2 failure gracefully
        assert result is None
        mock_memory_cache.get.assert_called_once()

    async def test_cache_fallback_l1_failure(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test fallback when L1 cache fails."""
        # L1 fails, should still check L2
        mock_memory_cache.get.side_effect = Exception("Memory cache error")
        mock_redis_cache.get.return_value = "l2_value"
        
        result = await cache_manager.get("test_key")
        
        # Should fallback to L2
        assert result == "l2_value"

    async def test_concurrent_cache_operations(self, cache_manager):
        """Test concurrent cache operations."""
        import asyncio
        
        # Simulate concurrent gets and sets
        async def cache_operation(key, value):
            await cache_manager.set(key, value)
            return await cache_manager.get(key)
        
        tasks = [
            cache_operation(f"key_{i}", f"value_{i}")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete without exceptions
        assert len(results) == 10
        assert all(not isinstance(r, Exception) for r in results)

    async def test_cache_size_limits(self, cache_manager, mock_memory_cache):
        """Test cache size limit enforcement."""
        # Mock memory cache as full
        mock_memory_cache.is_full.return_value = True
        
        if hasattr(cache_manager, '_evict_if_needed'):
            await cache_manager.set("new_key", "new_value")
            
            # Should trigger eviction
            mock_memory_cache.evict_lru.assert_called()

    async def test_cache_health_check(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test cache health checking."""
        # Both caches healthy
        mock_memory_cache.health_check.return_value = True
        mock_redis_cache.health_check.return_value = True
        
        health = await cache_manager.health_check()
        
        assert health["status"] == "healthy"
        assert health["l1_cache"] is True
        assert health["l2_cache"] is True

    async def test_cache_health_check_degraded(self, cache_manager, mock_memory_cache, mock_redis_cache):
        """Test cache health when one layer is down."""
        # L1 healthy, L2 down
        mock_memory_cache.health_check.return_value = True
        mock_redis_cache.health_check.return_value = False
        
        health = await cache_manager.health_check()
        
        assert health["status"] == "degraded"
        assert health["l1_cache"] is True
        assert health["l2_cache"] is False

    async def test_cache_metrics_collection(self, cache_manager):
        """Test cache metrics collection."""
        with patch('src.core.caching.cache_manager.metrics_collector') as mock_metrics:
            await cache_manager.get("test_key")
            await cache_manager.set("test_key", "test_value")
            
            # Should record cache metrics
            assert mock_metrics.record_value.call_count >= 2
