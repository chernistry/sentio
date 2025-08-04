"""Production-grade resilience patterns for external service interactions.

This module implements circuit breakers, retry mechanisms, and timeouts
following industry best practices for microservice reliability.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    exceptions: tuple = (Exception,)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    min_wait_seconds: float = 1.0
    max_wait_seconds: float = 10.0
    multiplier: float = 2.0
    exceptions: tuple = (
        httpx.RequestError,
        httpx.TimeoutException,
        ConnectionError,
        asyncio.TimeoutError,
    )


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""


class CircuitBreaker:
    """Circuit breaker implementation for external service calls.
    
    Prevents cascading failures by failing fast when a service is down.
    Automatically recovers when the service becomes available again.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            if isinstance(e, self.config.exceptions):
                await self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.timeout_seconds

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker reset to CLOSED state")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning("Circuit breaker moved to OPEN state during half-open test")
            elif (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.config.failure_threshold
            ):
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker tripped to OPEN state after {self.failure_count} failures"
                )

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == CircuitState.OPEN


class ResilientClient:
    """High-level client with built-in resilience patterns.
    
    Combines circuit breakers, retries, and timeouts for robust service calls.
    """

    def __init__(
        self,
        name: str,
        circuit_config: CircuitBreakerConfig | None = None,
        retry_config: RetryConfig | None = None,
        timeout_seconds: float = 30.0,
    ):
        self.name = name
        self.circuit_breaker = CircuitBreaker(
            circuit_config or CircuitBreakerConfig()
        )
        self.retry_config = retry_config or RetryConfig()
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(f"resilient_client.{name}")

    async def execute(
        self,
        func: Callable,
        *args,
        timeout: float | None = None,
        **kwargs,
    ) -> Any:
        """Execute function with full resilience patterns.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            timeout: Override default timeout
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit breaker is open
            RetryError: If all retries exhausted
            asyncio.TimeoutError: If operation times out
        """
        effective_timeout = timeout or self.timeout_seconds

        async def _execute_with_timeout():
            return await asyncio.wait_for(
                self.circuit_breaker.call(func, *args, **kwargs),
                timeout=effective_timeout
            )

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.retry_config.max_attempts),
                wait=wait_exponential(
                    multiplier=self.retry_config.multiplier,
                    min=self.retry_config.min_wait_seconds,
                    max=self.retry_config.max_wait_seconds,
                ),
                retry=retry_if_exception_type(self.retry_config.exceptions),
                reraise=True,
            ):
                with attempt:
                    start_time = time.time()
                    try:
                        result = await _execute_with_timeout()
                        duration = time.time() - start_time
                        self.logger.debug(
                            f"Call succeeded in {duration:.2f}s on attempt {attempt.retry_state.attempt_number}"
                        )
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        self.logger.warning(
                            f"Call failed in {duration:.2f}s on attempt {attempt.retry_state.attempt_number}: {e}"
                        )
                        raise

        except RetryError as e:
            self.logger.error(f"All retry attempts exhausted for {self.name}: {e}")
            raise
        except CircuitBreakerError as e:
            self.logger.error(f"Circuit breaker blocked call to {self.name}: {e}")
            raise

    @asynccontextmanager
    async def session(self):
        """Context manager for HTTP sessions with resilience."""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        ) as client:
            yield client

    def get_health_status(self) -> dict[str, Any]:
        """Get current health status of the client."""
        return {
            "name": self.name,
            "circuit_state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count,
            "success_count": self.circuit_breaker.success_count,
            "last_failure_time": self.circuit_breaker.last_failure_time,
            "is_healthy": not self.circuit_breaker.is_open,
        }


class HealthChecker:
    """Periodic health checking for services."""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.clients: dict[str, ResilientClient] = {}
        self._running = False
        self._task: asyncio.Task | None = None

    def register_client(self, client: ResilientClient):
        """Register a client for health monitoring."""
        self.clients[client.name] = client

    async def start(self):
        """Start periodic health checking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())
        logger.info("Health checker started")

    async def stop(self):
        """Stop health checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")

    async def _health_check_loop(self):
        """Periodic health check loop."""
        while self._running:
            try:
                for name, client in self.clients.items():
                    status = client.get_health_status()
                    if not status["is_healthy"]:
                        logger.warning(f"Service {name} is unhealthy: {status}")

                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.check_interval)

    def get_all_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status for all registered clients."""
        return {
            name: client.get_health_status()
            for name, client in self.clients.items()
        }


# Modern async circuit breaker implementation
class AsyncCircuitBreaker:
    """Modern async circuit breaker with state tracking and metrics.
    
    Features:
    - Async/await support
    - Detailed metrics tracking
    - Configurable thresholds
    - Automatic recovery testing
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        recovery_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self.total_calls = 0
        self.total_failures = 0

        self._lock = asyncio.Lock()

    async def call(self, coro_func):
        """Execute async function with circuit breaker protection."""
        async with self._lock:
            self.total_calls += 1

            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")

        try:
            result = await coro_func()
            await self._record_success()
            return result
        except Exception:
            await self._record_failure()
            raise

    async def _record_success(self):
        """Record successful call."""
        async with self._lock:
            self.success_count += 1
            self.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= 3:  # Recovery threshold
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker reset to CLOSED")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    async def _record_failure(self):
        """Record failed call."""
        async with self._lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker failed during recovery, back to OPEN")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "failure_rate": self.total_failures / max(self.total_calls, 1),
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
        }


class AsyncRetry:
    """Modern async retry mechanism with exponential backoff.
    
    Features:
    - Configurable retry attempts
    - Exponential backoff with jitter
    - Specific exception handling
    - Detailed logging
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_wait_time: float = 30.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff_multiplier = backoff_multiplier
        self.max_wait_time = max_wait_time
        self.jitter = jitter

    async def call(self, coro_func):
        """Execute async function with retry logic."""
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await coro_func()
                if attempt > 1:
                    logger.info(f"Retry succeeded on attempt {attempt}")
                return result

            except Exception as e:
                last_exception = e

                if attempt == self.max_attempts:
                    logger.error(f"All {self.max_attempts} retry attempts failed")
                    break

                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.backoff_multiplier ** (attempt - 1)),
                    self.max_wait_time
                )

                # Add jitter to prevent thundering herd
                if self.jitter:
                    import random
                    delay = delay * (0.5 + random.random() * 0.5)

                logger.warning(
                    f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s"
                )

                await asyncio.sleep(delay)

        raise last_exception


# Global health checker instance
health_checker = HealthChecker()
