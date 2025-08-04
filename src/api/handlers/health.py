"""Health check handlers for comprehensive system monitoring.

Provides detailed health status for all system components with
dependency checks and performance metrics.
"""

import asyncio
import logging
import time
from typing import Any

from src.core.embeddings import get_embedder
from src.core.resilience.patterns import health_checker
from src.core.vector_store import get_vector_store
from src.utils.settings import settings

logger = logging.getLogger(__name__)


class HealthHandler:
    """Comprehensive health checking for all system components.
    
    Provides both basic and detailed health endpoints with
    dependency verification and performance metrics.
    """

    def __init__(self):
        self._cached_health: dict[str, Any] | None = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 10.0  # Cache health checks for 10 seconds

    async def basic_health_check(self) -> dict[str, Any]:
        """Basic health check for load balancer/orchestrator.

        Returns:
            Simple health status
        """
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "3.0.0",
        }

    async def detailed_health_check(self) -> dict[str, Any]:
        """Comprehensive health check for all system components.

        Returns:
            Detailed health status with component breakdown
        """
        # Check cache first
        current_time = time.time()
        if (
            self._cached_health
            and current_time - self._cache_timestamp < self._cache_ttl
        ):
            return self._cached_health

        health_status = {
            "status": "healthy",
            "timestamp": current_time,
            "version": "3.0.0",
            "checks": {},
            "metrics": {},
        }

        # Run all health checks concurrently
        check_tasks = {
            "vector_store": self._check_vector_store(),
            "embeddings": self._check_embedding_service(),
            "llm": self._check_llm_service(),
            "external_dependencies": self._check_external_dependencies(),
            "circuit_breakers": self._check_circuit_breakers(),
        }

        # Execute all checks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*check_tasks.values(), return_exceptions=True),
                timeout=30.0
            )

            # Process results
            for i, (check_name, task) in enumerate(check_tasks.items()):
                result = results[i]
                if isinstance(result, Exception):
                    health_status["checks"][check_name] = {
                        "healthy": False,
                        "error": str(result),
                        "timestamp": current_time,
                    }
                else:
                    health_status["checks"][check_name] = result

        except TimeoutError:
            health_status["status"] = "timeout"
            health_status["error"] = "Health check timeout"

        # Determine overall status
        all_healthy = all(
            check.get("healthy", False)
            for check in health_status["checks"].values()
        )

        if not all_healthy:
            health_status["status"] = "unhealthy"

        # Add performance metrics
        health_status["metrics"] = await self._collect_metrics()

        # Cache result
        self._cached_health = health_status
        self._cache_timestamp = current_time

        return health_status

    async def _check_vector_store(self) -> dict[str, Any]:
        """Check vector store connectivity and performance."""
        start_time = time.time()

        try:
            vector_store = get_vector_store(
                name=settings.vector_store_name,
                collection_name=settings.collection_name,
                vector_size=768,  # Default dimension
            )

            # Test basic connectivity
            if hasattr(vector_store, "health_check"):
                is_healthy = vector_store.health_check()
            # Try a basic operation
            elif hasattr(vector_store, "_client"):
                collections = vector_store._client.get_collections()
                is_healthy = True
            else:
                is_healthy = True  # Assume healthy if no specific check

            response_time = time.time() - start_time

            return {
                "healthy": is_healthy,
                "response_time_ms": round(response_time * 1000, 2),
                "provider": settings.vector_store_name,
                "collection": settings.collection_name,
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": time.time(),
            }

    async def _check_embedding_service(self) -> dict[str, Any]:
        """Check embedding service availability and performance."""
        start_time = time.time()

        try:
            embedder = get_embedder(name=settings.embedder_name)

            # Test embedding generation
            test_embedding = await embedder.embed_async_single("health check test")

            response_time = time.time() - start_time

            return {
                "healthy": True,
                "response_time_ms": round(response_time * 1000, 2),
                "provider": settings.embedder_name,
                "model": settings.embedding_model,
                "dimension": len(test_embedding),
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": time.time(),
            }

    async def _check_llm_service(self) -> dict[str, Any]:
        """Check LLM service availability."""
        start_time = time.time()

        try:
            from src.core.llm.factory import create_generator

            generator = create_generator()

            # Test basic generation (we don't actually call it to avoid costs)
            # Just check if the generator can be created

            response_time = time.time() - start_time

            return {
                "healthy": True,
                "response_time_ms": round(response_time * 1000, 2),
                "provider": settings.llm_provider,
                "model": settings.chat_llm_model,
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": time.time(),
            }

    async def _check_external_dependencies(self) -> dict[str, Any]:
        """Check external service dependencies."""
        dependencies = {
            "jina_api": "https://api.jina.ai/v1/embeddings",
            "openai_api": "https://api.openai.com/v1/models",
        }

        results = {}

        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            for service, url in dependencies.items():
                start_time = time.time()
                try:
                    # Just check if the endpoint is reachable
                    response = await client.head(url)
                    response_time = time.time() - start_time

                    results[service] = {
                        "healthy": response.status_code < 500,
                        "status_code": response.status_code,
                        "response_time_ms": round(response_time * 1000, 2),
                    }

                except Exception as e:
                    results[service] = {
                        "healthy": False,
                        "error": str(e),
                        "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    }

        return {
            "healthy": all(dep.get("healthy", False) for dep in results.values()),
            "dependencies": results,
            "timestamp": time.time(),
        }

    async def _check_circuit_breakers(self) -> dict[str, Any]:
        """Check circuit breaker status."""
        try:
            circuit_status = health_checker.get_all_health_status()

            all_healthy = all(
                status.get("is_healthy", True)
                for status in circuit_status.values()
            )

            return {
                "healthy": all_healthy,
                "circuits": circuit_status,
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def _collect_metrics(self) -> dict[str, Any]:
        """Collect performance and usage metrics."""
        import os

        import psutil

        try:
            process = psutil.Process(os.getpid())

            return {
                "memory_usage_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()),
                "threads": process.num_threads(),
                "uptime_seconds": round(time.time() - process.create_time(), 2),
            }
        except Exception:
            return {"error": "Could not collect metrics"}

    async def readiness_check(self) -> dict[str, Any]:
        """Kubernetes readiness check - determines if pod can receive traffic.
        
        Returns:
            Readiness status
        """
        # Check critical components only
        try:
            vector_store_status = await self._check_vector_store()
            embedding_status = await self._check_embedding_service()

            ready = (
                vector_store_status.get("healthy", False) and
                embedding_status.get("healthy", False)
            )

            return {
                "ready": ready,
                "timestamp": time.time(),
                "critical_checks": {
                    "vector_store": vector_store_status.get("healthy", False),
                    "embeddings": embedding_status.get("healthy", False),
                }
            }

        except Exception as e:
            return {
                "ready": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def liveness_check(self) -> dict[str, Any]:
        """Kubernetes liveness check - determines if pod should be restarted.
        
        Returns:
            Liveness status
        """
        try:
            # Basic process health check
            import os
            return {
                "alive": True,
                "pid": os.getpid(),
                "timestamp": time.time(),
            }
        except Exception as e:
            return {
                "alive": False,
                "error": str(e),
                "timestamp": time.time(),
            }
