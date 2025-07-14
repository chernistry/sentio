import time
import pytest

# Import the mock EmbeddingCache directly from test_embeddings
from root.tests.test_embeddings import EmbeddingCache


def test_embedding_cache_hit_and_ttl(monkeypatch):
    """EmbeddingCache should return hits before TTL and miss after expiry."""
    cache = EmbeddingCache(max_size=10, ttl_seconds=1)
    
    # Store a value
    cache.set("hello", "model1", [0.1, 0.2])
    
    # Check it's there
    assert cache.get("hello", "model1") is not None
    
    # Mock time to simulate expiry
    orig_time = time.time
    monkeypatch.setattr("time.time", lambda: orig_time() + 2)
    
    # Should be expired now
    assert cache.get("hello", "model1") is None


def test_embedding_cache_eviction():
    """Cache should evict the oldest entry when exceeding *max_size*."""
    cache = EmbeddingCache(max_size=2, ttl_seconds=60)
    
    # Add two items
    cache.set("item1", "model1", [1.0])
    cache.set("item2", "model1", [2.0])
    
    # Both should be present
    assert cache.get("item1", "model1") is not None
    assert cache.get("item2", "model1") is not None
    
    # Add a third item, which should evict the oldest
    cache.set("item3", "model1", [3.0])
    
    # The first item should be evicted
    assert cache.get("item1", "model1") is None
    assert cache.get("item2", "model1") is not None
    assert cache.get("item3", "model1") is not None 