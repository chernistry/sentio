"""Decorators for adding resilience patterns to functions and methods.

Provides convenient decorators for circuit breakers and retry logic.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from .patterns import CircuitBreakerConfig, ResilientClient, RetryConfig

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def with_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
    timeout_seconds: float = 30.0,
) -> Callable[[F], F]:
    """Decorator to add circuit breaker protection to async functions.
    
    Args:
        name: Circuit breaker name for logging and monitoring
        config: Circuit breaker configuration
        timeout_seconds: Timeout for function calls
        
    Returns:
        Decorated function with circuit breaker protection
    """
    def decorator(func: F) -> F:
        client = ResilientClient(
            name=f"{name}_{func.__name__}",
            circuit_config=config,
            timeout_seconds=timeout_seconds,
        )

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await client.execute(func, *args, **kwargs)

        # Attach health status method to decorated function
        wrapper.get_health_status = client.get_health_status  # type: ignore
        return wrapper  # type: ignore

    return decorator


def with_retry(
    config: RetryConfig | None = None,
    circuit_breaker: CircuitBreakerConfig | None = None,
) -> Callable[[F], F]:
    """Decorator to add retry logic to async functions.
    
    Args:
        config: Retry configuration
        circuit_breaker: Optional circuit breaker configuration
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: F) -> F:
        client = ResilientClient(
            name=func.__name__,
            circuit_config=circuit_breaker,
            retry_config=config,
        )

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await client.execute(func, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def resilient_http_client(
    name: str,
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
) -> Callable[[F], F]:
    """Decorator for HTTP client methods with full resilience.
    
    Args:
        name: Client name for monitoring
        timeout_seconds: HTTP timeout
        max_retries: Maximum retry attempts
        
    Returns:
        Decorated HTTP client method
    """
    return with_circuit_breaker(
        name=name,
        config=CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=60,
            exceptions=(Exception,),
        ),
        timeout_seconds=timeout_seconds,
    )
