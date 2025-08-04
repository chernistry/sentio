"""Cache strategies and policies for intelligent caching behavior.

This module provides various caching strategies to optimize
cache performance based on usage patterns and data characteristics.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class CacheStrategy(ABC):
    """Base class for cache strategies."""

    @abstractmethod
    def should_cache(self, key: str, value: Any, context: dict[str, Any]) -> bool:
        """Determine if value should be cached."""

    @abstractmethod
    def get_ttl(self, key: str, value: Any, context: dict[str, Any]) -> int | None:
        """Get TTL for cached value."""

    @abstractmethod
    def get_priority(self, key: str, value: Any, context: dict[str, Any]) -> float:
        """Get cache priority (0.0 to 1.0)."""


@dataclass
class CacheContext:
    """Context information for cache decisions."""
    operation_type: str  # 'embedding', 'query', 'retrieval'
    data_size: int
    processing_time: float
    access_frequency: float = 0.0
    user_context: dict[str, Any] | None = None


class TTLStrategy(CacheStrategy):
    """Time-based caching strategy with adaptive TTL.
    
    Adjusts TTL based on data characteristics and access patterns.
    """

    def __init__(
        self,
        base_ttl: int = 3600,
        min_ttl: int = 300,
        max_ttl: int = 86400,
        size_factor: float = 0.1,
        frequency_factor: float = 0.5,
    ):
        """Initialize TTL strategy.
        
        Args:
            base_ttl: Base TTL in seconds
            min_ttl: Minimum TTL in seconds
            max_ttl: Maximum TTL in seconds
            size_factor: Factor for size-based TTL adjustment
            frequency_factor: Factor for frequency-based TTL adjustment
        """
        self.base_ttl = base_ttl
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.size_factor = size_factor
        self.frequency_factor = frequency_factor

    def should_cache(self, key: str, value: Any, context: dict[str, Any]) -> bool:
        """Cache everything by default."""
        return True

    def get_ttl(self, key: str, value: Any, context: dict[str, Any]) -> int | None:
        """Calculate adaptive TTL."""
        ttl = self.base_ttl

        # Adjust based on data size (smaller data = longer TTL)
        if "data_size" in context:
            size_kb = context["data_size"] / 1024
            if size_kb < 1:  # Very small data
                ttl = int(ttl * 1.5)
            elif size_kb > 100:  # Large data
                ttl = int(ttl * 0.7)

        # Adjust based on processing time (expensive operations = longer TTL)
        if "processing_time" in context:
            if context["processing_time"] > 1.0:  # Expensive operation
                ttl = int(ttl * 1.3)
            elif context["processing_time"] < 0.1:  # Fast operation
                ttl = int(ttl * 0.8)

        # Adjust based on access frequency
        if "access_frequency" in context:
            frequency = context["access_frequency"]
            if frequency > 10:  # High frequency
                ttl = int(ttl * (1 + self.frequency_factor))
            elif frequency < 1:  # Low frequency
                ttl = int(ttl * (1 - self.frequency_factor))

        # Apply bounds
        return max(self.min_ttl, min(self.max_ttl, ttl))

    def get_priority(self, key: str, value: Any, context: dict[str, Any]) -> float:
        """Calculate cache priority."""
        priority = 0.5

        # Higher priority for expensive operations
        if "processing_time" in context:
            if context["processing_time"] > 2.0:
                priority += 0.3
            elif context["processing_time"] > 0.5:
                priority += 0.1

        # Higher priority for frequently accessed data
        if "access_frequency" in context:
            frequency = context["access_frequency"]
            if frequency > 5:
                priority += 0.2
            elif frequency > 1:
                priority += 0.1

        return min(1.0, priority)


class LRUStrategy(CacheStrategy):
    """Least Recently Used caching strategy.
    
    Prioritizes recently accessed items and evicts least recently used items.
    """

    def __init__(
        self,
        access_tracking: bool = True,
        popularity_threshold: float = 0.1,
    ):
        """Initialize LRU strategy.
        
        Args:
            access_tracking: Enable access pattern tracking
            popularity_threshold: Threshold for popular items
        """
        self.access_tracking = access_tracking
        self.popularity_threshold = popularity_threshold
        self._access_counts: dict[str, int] = {}
        self._last_access: dict[str, float] = {}

    def should_cache(self, key: str, value: Any, context: dict[str, Any]) -> bool:
        """Cache based on access patterns."""
        if not self.access_tracking:
            return True

        # Track access
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        self._last_access[key] = time.time()

        # Cache if accessed multiple times or expensive to compute
        access_count = self._access_counts[key]
        processing_time = context.get("processing_time", 0)

        return access_count > 1 or processing_time > 0.5

    def get_ttl(self, key: str, value: Any, context: dict[str, Any]) -> int | None:
        """TTL based on access frequency."""
        access_count = self._access_counts.get(key, 1)

        # More frequently accessed items get longer TTL
        base_ttl = 3600
        if access_count > 10:
            return base_ttl * 2
        if access_count > 5:
            return int(base_ttl * 1.5)
        return base_ttl

    def get_priority(self, key: str, value: Any, context: dict[str, Any]) -> float:
        """Priority based on recency and frequency."""
        access_count = self._access_counts.get(key, 1)
        last_access = self._last_access.get(key, time.time())

        # Recency score (0-1, higher = more recent)
        time_since_access = time.time() - last_access
        recency_score = max(0, 1 - (time_since_access / 3600))  # 1 hour decay

        # Frequency score (0-1, higher = more frequent)
        frequency_score = min(1.0, access_count / 10)

        # Combined priority
        return (recency_score * 0.6) + (frequency_score * 0.4)


class SizeBasedStrategy(CacheStrategy):
    """Size-based caching strategy.
    
    Makes caching decisions based on data size and available cache capacity.
    """

    def __init__(
        self,
        max_item_size: int = 1024 * 1024,  # 1MB
        min_item_size: int = 100,  # 100 bytes
        size_efficiency_threshold: float = 0.1,
    ):
        """Initialize size-based strategy.
        
        Args:
            max_item_size: Maximum size to cache
            min_item_size: Minimum size worth caching
            size_efficiency_threshold: Size/benefit threshold
        """
        self.max_item_size = max_item_size
        self.min_item_size = min_item_size
        self.size_efficiency_threshold = size_efficiency_threshold

    def should_cache(self, key: str, value: Any, context: dict[str, Any]) -> bool:
        """Cache based on size constraints."""
        data_size = context.get("data_size", 0)

        # Don't cache if too large or too small
        if data_size > self.max_item_size or data_size < self.min_item_size:
            return False

        # Consider size efficiency (benefit per byte)
        processing_time = context.get("processing_time", 0)
        if data_size > 0 and processing_time > 0:
            efficiency = processing_time / (data_size / 1024)  # benefit per KB
            return efficiency > self.size_efficiency_threshold

        return True

    def get_ttl(self, key: str, value: Any, context: dict[str, Any]) -> int | None:
        """TTL inversely related to size."""
        data_size = context.get("data_size", 1000)

        # Smaller items get longer TTL
        base_ttl = 3600
        size_factor = max(0.5, min(2.0, 1000 / (data_size / 1024)))

        return int(base_ttl * size_factor)

    def get_priority(self, key: str, value: Any, context: dict[str, Any]) -> float:
        """Priority inversely related to size."""
        data_size = context.get("data_size", 1000)
        processing_time = context.get("processing_time", 0)

        # Higher priority for smaller, expensive-to-compute items
        size_score = max(0.1, min(1.0, 10000 / data_size))
        time_score = min(1.0, processing_time / 5.0)

        return (size_score * 0.4) + (time_score * 0.6)


class AdaptiveStrategy(CacheStrategy):
    """Adaptive caching strategy that combines multiple strategies.
    
    Uses machine learning principles to adapt cache behavior
    based on observed patterns and performance.
    """

    def __init__(self):
        """Initialize adaptive strategy."""
        self.ttl_strategy = TTLStrategy()
        self.lru_strategy = LRUStrategy()
        self.size_strategy = SizeBasedStrategy()

        # Performance tracking
        self._hit_rates: dict[str, float] = {}
        self._performance_history: list[dict[str, Any]] = []

    def should_cache(self, key: str, value: Any, context: dict[str, Any]) -> bool:
        """Adaptive caching decision."""
        # All strategies must agree to cache
        strategies = [self.ttl_strategy, self.lru_strategy, self.size_strategy]

        decisions = [s.should_cache(key, value, context) for s in strategies]

        # Cache if majority agrees or if high processing time
        processing_time = context.get("processing_time", 0)
        if processing_time > 2.0:  # Very expensive operation
            return True

        return sum(decisions) >= len(decisions) / 2

    def get_ttl(self, key: str, value: Any, context: dict[str, Any]) -> int | None:
        """Adaptive TTL calculation."""
        # Get TTL from all strategies
        ttls = []
        for strategy in [self.ttl_strategy, self.lru_strategy, self.size_strategy]:
            ttl = strategy.get_ttl(key, value, context)
            if ttl is not None:
                ttls.append(ttl)

        if not ttls:
            return 3600

        # Use weighted average based on strategy performance
        return int(sum(ttls) / len(ttls))

    def get_priority(self, key: str, value: Any, context: dict[str, Any]) -> float:
        """Adaptive priority calculation."""
        # Get priorities from all strategies
        priorities = []
        for strategy in [self.ttl_strategy, self.lru_strategy, self.size_strategy]:
            priority = strategy.get_priority(key, value, context)
            priorities.append(priority)

        # Weighted average
        return sum(priorities) / len(priorities)

    def update_performance(self, key: str, hit: bool, context: dict[str, Any]):
        """Update performance metrics for adaptation."""
        # Track hit rates per operation type
        op_type = context.get("operation_type", "unknown")
        if op_type not in self._hit_rates:
            self._hit_rates[op_type] = 0.0

        # Update hit rate with exponential moving average
        alpha = 0.1
        self._hit_rates[op_type] = (
            alpha * (1.0 if hit else 0.0) +
            (1 - alpha) * self._hit_rates[op_type]
        )

        # Store performance history
        self._performance_history.append({
            "timestamp": time.time(),
            "key": key,
            "hit": hit,
            "operation_type": op_type,
            "hit_rate": self._hit_rates[op_type],
        })

        # Keep only recent history
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-500:]

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            "hit_rates": self._hit_rates.copy(),
            "total_operations": len(self._performance_history),
            "recent_performance": self._performance_history[-10:],
        }
