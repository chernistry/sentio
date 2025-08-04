"""Dependency injection system for the RAG application.

This module provides a centralized dependency injection system that eliminates
global state and improves testability and maintainability.
"""

import asyncio
import logging
from functools import lru_cache
from typing import Any

from fastapi import Request

from src.core.caching.redis_cache import RedisCache
from src.core.embeddings import get_embedder
from src.core.ingest import DocumentIngestor
from src.core.vector_store import get_async_vector_store, get_vector_store
from src.utils.auth import AuthManager, auth_manager
from src.utils.settings import settings

logger = logging.getLogger(__name__)


class DependencyContainer:
    """Centralized dependency container for managing application dependencies.
    
    This class follows the Dependency Injection pattern to provide
    thread-safe, properly initialized dependencies throughout the application.
    """

    def __init__(self):
        self._ingestor: DocumentIngestor | None = None
        self._vector_store: Any | None = None
        self._async_vector_store: Any | None = None
        self._auth_manager: AuthManager | None = None
        self._embedder: Any | None = None
        self._chat_handler: Any | None = None
        self._health_handler: Any | None = None
        self._redis_cache: RedisCache | None = None
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
        self._use_async_stores = True  # Enable async operations by default

    async def get_embedder(self) -> Any:
        """Get or initialize the embedder."""
        if self._embedder is None:
            async with self._initialization_lock:
                if self._embedder is None:
                    try:
                        self._embedder = get_embedder(settings.embedder_name)
                        logger.info(f"Initialized embedder: {settings.embedder_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize embedder: {e}")
                        raise
        return self._embedder

    async def get_vector_store(self) -> Any:
        """Get or initialize the vector store."""
        if self._vector_store is None:
            async with self._initialization_lock:
                if self._vector_store is None:
                    try:
                        embedder = await self.get_embedder()
                        self._vector_store = get_vector_store(
                            name=settings.vector_store_name,
                            collection_name=settings.collection_name,
                            vector_size=embedder.dimension,
                        )
                        logger.info(f"Initialized vector store: {settings.vector_store_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize vector store: {e}")
                        raise
        return self._vector_store

    async def get_ingestor(self) -> DocumentIngestor:
        """Get or initialize the document ingestor."""
        if self._ingestor is None:
            async with self._initialization_lock:
                if self._ingestor is None:
                    try:
                        self._ingestor = DocumentIngestor(
                            collection_name=settings.collection_name,
                            chunk_size=settings.chunk_size,
                            chunk_overlap=settings.chunk_overlap,
                            chunking_strategy=settings.chunking_strategy,
                            embedder_name=settings.embedder_name,
                            vector_store_name=settings.vector_store_name,
                        )
                        await self._ingestor.initialize()
                        logger.info("Initialized document ingestor")
                    except Exception as e:
                        logger.error(f"Failed to initialize ingestor: {e}")
                        raise
        return self._ingestor

    async def get_auth_manager(self) -> AuthManager:
        """Get or initialize the authentication manager."""
        if self._auth_manager is None:
            async with self._initialization_lock:
                if self._auth_manager is None:
                    try:
                        self._auth_manager = auth_manager
                        await self._auth_manager.initialize()
                        logger.info("Initialized authentication manager")
                    except Exception as e:
                        logger.error(f"Failed to initialize auth manager: {e}")
                        raise
        return self._auth_manager

    async def get_chat_handler(self) -> Any:
        """Get or initialize the chat handler."""
        if self._chat_handler is None:
            async with self._initialization_lock:
                if self._chat_handler is None:
                    try:
                        from src.api.handlers.chat import ChatHandler
                        self._chat_handler = ChatHandler()
                        logger.info("Initialized chat handler")
                    except Exception as e:
                        logger.error(f"Failed to initialize chat handler: {e}")
                        raise
        return self._chat_handler

    async def get_health_handler(self) -> Any:
        """Get or initialize the health handler."""
        if self._health_handler is None:
            async with self._initialization_lock:
                if self._health_handler is None:
                    try:
                        from src.api.handlers.health import HealthHandler
                        self._health_handler = HealthHandler()
                        logger.info("Initialized health handler")
                    except Exception as e:
                        logger.error(f"Failed to initialize health handler: {e}")
                        raise
        return self._health_handler

    async def get_async_vector_store(self) -> Any:
        """Get or initialize the async vector store for high performance."""
        if self._async_vector_store is None:
            async with self._initialization_lock:
                if self._async_vector_store is None:
                    try:
                        embedder = await self.get_embedder()
                        self._async_vector_store = await get_async_vector_store(
                            name=settings.vector_store_name,
                            collection_name=settings.collection_name,
                            vector_size=embedder.dimension,
                        )
                        logger.info(f"Initialized async vector store: {settings.vector_store_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize async vector store: {e}")
                        raise
        return self._async_vector_store

    async def get_redis_cache(self) -> RedisCache | None:
        """Get or initialize the Redis cache."""
        if self._redis_cache is None:
            async with self._initialization_lock:
                if self._redis_cache is None:
                    try:
                        self._redis_cache = RedisCache(
                            redis_url=settings.redis_url,
                            key_prefix="sentio:",
                            default_ttl=3600,
                        )
                        logger.info("Initialized Redis cache")
                    except Exception as e:
                        logger.error(f"Failed to initialize Redis cache: {e}")
                        # Don't raise - caching is optional
                        return None
        return self._redis_cache

    async def initialize_all(self) -> None:
        """Initialize all dependencies proactively."""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

            try:
                logger.info("Initializing all application dependencies...")

                # Initialize in dependency order
                await self.get_embedder()
                await self.get_vector_store()
                await self.get_async_vector_store()  # Initialize async vector store
                await self.get_redis_cache()         # Initialize caching (optional)
                await self.get_ingestor()
                await self.get_auth_manager()
                await self.get_chat_handler()
                await self.get_health_handler()

                self._initialized = True
                logger.info("All dependencies initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize dependencies: {e}")
                raise

    async def cleanup(self) -> None:
        """Clean up all dependencies."""
        logger.info("Cleaning up application dependencies...")

        if self._auth_manager:
            try:
                await self._auth_manager.close()
            except Exception as e:
                logger.error(f"Error closing auth manager: {e}")

        if self._ingestor:
            try:
                # Close any connections in the ingestor
                if hasattr(self._ingestor, "cleanup"):
                    await self._ingestor.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up ingestor: {e}")

        # Reset all dependencies
        self._ingestor = None
        self._vector_store = None
        self._auth_manager = None
        self._embedder = None
        self._chat_handler = None
        self._health_handler = None
        self._initialized = False

        logger.info("Dependency cleanup completed")


# Global dependency container instance
_container: DependencyContainer | None = None


def get_container() -> DependencyContainer:
    """Get the global dependency container."""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


# FastAPI dependency functions
async def get_ingestor() -> DocumentIngestor:
    """FastAPI dependency for document ingestor."""
    container = get_container()
    return await container.get_ingestor()


async def get_vector_store_dep() -> Any:
    """FastAPI dependency for vector store."""
    container = get_container()
    return await container.get_vector_store()


async def get_auth_manager_dep() -> AuthManager:
    """FastAPI dependency for auth manager."""
    container = get_container()
    return await container.get_auth_manager()


async def get_chat_handler() -> Any:
    """FastAPI dependency for chat handler."""
    container = get_container()
    return await container.get_chat_handler()


async def get_health_handler() -> Any:
    """FastAPI dependency for health handler."""
    container = get_container()
    return await container.get_health_handler()


# Request context dependencies
@lru_cache(maxsize=1)
def get_request_context() -> dict[str, Any]:
    """Get request context for tracking and observability."""
    return {
        "request_id": None,
        "user_id": None,
        "start_time": None,
        "metadata": {},
    }


async def get_request_metadata(request: Request) -> dict[str, Any]:
    """Extract request metadata for observability."""
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client_host": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "content_type": request.headers.get("content-type"),
        "request_id": getattr(request.state, "request_id", None),
    }


# Dependency injection decorators
def inject_dependencies(*deps):
    """Decorator to inject dependencies into functions.
    
    Args:
        deps: List of dependency functions
    
    Returns:
        Decorator function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Inject dependencies as keyword arguments
            for dep in deps:
                dep_name = dep.__name__.replace("get_", "")
                if dep_name not in kwargs:
                    kwargs[dep_name] = await dep()
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Health check for dependencies
async def check_dependency_health() -> dict[str, str]:
    """Check health of all dependencies."""
    container = get_container()
    health_status = {}

    try:
        # Check embedder
        embedder = await container.get_embedder()
        health_status["embedder"] = "healthy" if embedder else "unhealthy"
    except Exception as e:
        health_status["embedder"] = f"error: {e!s}"

    try:
        # Check vector store
        vector_store = await container.get_vector_store()
        health_status["vector_store"] = "healthy" if vector_store else "unhealthy"
    except Exception as e:
        health_status["vector_store"] = f"error: {e!s}"

    try:
        # Check ingestor
        ingestor = await container.get_ingestor()
        health_status["ingestor"] = "healthy" if ingestor else "unhealthy"
    except Exception as e:
        health_status["ingestor"] = f"error: {e!s}"

    try:
        # Check auth manager
        auth_mgr = await container.get_auth_manager()
        health_status["auth_manager"] = "healthy" if auth_mgr else "unhealthy"
    except Exception as e:
        health_status["auth_manager"] = f"error: {e!s}"

    return health_status


# Additional FastAPI dependencies for async components
async def get_async_vector_store_dep() -> Any:
    """FastAPI dependency for async vector store."""
    container = get_container()
    return await container.get_async_vector_store()


async def get_redis_cache_dep() -> RedisCache | None:
    """FastAPI dependency for Redis cache."""
    container = get_container()
    return await container.get_redis_cache()
