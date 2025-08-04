"""Resilience patterns for production reliability.

This module provides circuit breakers, retry mechanisms, and graceful degradation
patterns for handling external service failures and network issues.
"""

from .decorators import with_circuit_breaker, with_retry
from .fallbacks import FallbackManager
from .patterns import CircuitBreakerConfig, ResilientClient, RetryConfig

__all__ = [
    "CircuitBreakerConfig",
    "FallbackManager",
    "ResilientClient",
    "RetryConfig",
    "with_circuit_breaker",
    "with_retry",
]
