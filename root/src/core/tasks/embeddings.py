
#!/usr/bin/env python3
"""
Advanced embedding service with caching, error handling, and optimization.

This module provides a production-ready interface to multiple embedding
providers, including Sentence-Transformers and Jina AI. It features
automatic retries, caching, batch processing, and monitoring.
"""

# ==== IMPORTS & DEPENDENCIES ==== #

import asyncio
import hashlib
import json
import logging
logger = logging.getLogger(__name__)
import os
import time
from abc import ABC, abstractmethod
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Union

import httpx
# --- EVENT LOOP PATCHING ------------------------------------------------- #
# Allow synchronous helpers to call async code even inside a running loop.  #
# The patch prevents "Cannot enter into task ..." re-entrancy errors.       #
# Attempt to patch the current event loop to allow nested usage inside
# already-running loops (e.g. Jupyter, FastAPI). When `uvloop` is active the
# patch is incompatible, so we catch the error and continue gracefully.
import nest_asyncio

try:
    # A no-op if the loop was patched earlier.
    nest_asyncio.apply()
except ValueError as exc:  # Raised when the loop implementation is not supported.
    logger.debug("nest_asyncio patch skipped: %s", exc)
from pydantic import PrivateAttr
from llama_index.core.base.embeddings.base import (
    BaseEmbedding as LlamaIndexBaseEmbedding,
)
from root.src.utils.settings import settings

# ------------------------------------------------------------------
# Backwards-compatibility shim – delay import of the heavy adapter
# until *after* this module finished initialisation to avoid circular
# dependencies.
# ------------------------------------------------------------------

def EmbeddingModel(*args: Any, **kwargs: Any):
    """Lazy wrapper that defers adapter import to call time."""
    from root.src.core.embeddings.embeddings_adapter import get_embedding_model  # noqa: WPS433 – late import
    return get_embedding_model(*args, **kwargs)

# ==== CORE PROCESSING MODULE ==== #
# --► EXCEPTIONS & CONSTANTS ⚠️ POTENTIALLY ERROR-PRONE LOGIC

class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""


# Known embedding dimensions
KNOWN_MODEL_DIMENSIONS: Dict[str, int] = {
    "jina-embeddings-v2": 768,
    "jina-embeddings-v3": 1024,
}

DEFAULT_EMBEDDING_DIMENSION: int = 1024


# ==== CORE PROCESSING MODULE ==== #
# --► CACHE IMPLEMENTATION

class EmbeddingCache:
    """Simple in-memory cache for embeddings with TTL."""

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600
    ) -> None:
        """
        Initialize cache.

        Args:
            max_size (int): Maximum number of entries in cache.
            ttl_seconds (int): Time-to-live for entries, in seconds.
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.stats = {
            "hits": 0,
            "misses": 0,
            "size": 0,
        }

    def _get_key(self, text: str, model_name: str) -> str:
        """
        Generate cache key from text and model name.

        Args:
            text (str): Input text.
            model_name (str): Embedding model identifier.

        Returns:
            str: MD5 hash key.
        """
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(
        self,
        text: str,
        model_name: str
    ) -> Optional[List[float]]:
        """
        Retrieve embedding if present and not expired.

        Args:
            text (str): Input text.
            model_name (str): Embedding model identifier.

        Returns:
            Optional[List[float]]: Cached embedding or None.
        """
        key = self._get_key(text, model_name)
        entry = self.cache.get(key)

        if entry is None:
            self.stats["misses"] += 1
            return None

        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self.cache[key]
            self.stats["misses"] += 1
            return None

        self.stats["hits"] += 1
        return entry["embedding"]

    def set(
        self,
        text: str,
        model_name: str,
        embedding: List[float]
    ) -> None:
        """
        Store embedding in cache with timestamp.

        Args:
            text (str): Input text.
            model_name (str): Embedding model identifier.
            embedding (List[float]): Embedding vector.
        """
        if len(self.cache) >= self.max_size:
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k]["timestamp"]
            )
            del self.cache[oldest_key]

        key = self._get_key(text, model_name)
        self.cache[key] = {
            "embedding": embedding,
            "timestamp": time.time(),
        }
        self.stats["size"] = len(self.cache)

    def clear(self) -> None:
        """
        Clear all cached embeddings.
        """
        self.cache.clear()
        self.stats["size"] = 0

    @property
    def stats_summary(self) -> Dict[str, Union[int, float]]:
        """
        Summary of cache statistics.

        Returns:
            Dict[str, Union[int, float]]: Stats including hit rate.
        """
        hits = self.stats.get("hits", 0)
        misses = self.stats.get("misses", 0)
        total = hits + misses

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total > 0 else 0.0,
        }


# ==== CORE PROCESSING MODULE ==== #
# --► RETRY DECORATOR

def _retry(max_retries: int):
    """
    Decorator for async functions with exponential backoff.

    Args:
        max_retries (int): Number of retry attempts.
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return await fn(*args, **kwargs)
                except (
                    httpx.TimeoutException,
                    httpx.NetworkError
                ) as e:
                    if attempt == max_retries - 1:
                        raise EmbeddingError(
                            f"Request failed after {max_retries} attempts: {e}"
                        )
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/"
                        f"{max_retries}), retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code

                    if status == 429:
                        if attempt == max_retries - 1:
                            raise EmbeddingError(
                                f"Rate limited. Failed after {max_retries} attempts."
                            )
                        wait_time = 2 ** (attempt + 1)
                        logger.warning(
                            f"Rate limited, retrying in {wait_time}s."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise EmbeddingError(
                            f"HTTP error: {status} - {e.response.text}"
                        ) from e

        return wrapper

    return decorator


# ==== CORE PROCESSING MODULE ==== #
# --► BASE EMBEDDING MODEL

class BaseEmbeddingModel(ABC):
    """Abstract base class for all embedding models."""

    def __init__(
        self,
        model_name: str,
        cache_enabled: bool,
        cache_size: int,
        cache_ttl: int,
        **kwargs: Any
    ) -> None:
        """
        Initialize embedding model.

        Args:
            model_name (str): Identifier for embedding model.
            cache_enabled (bool): Enable in-memory caching.
            cache_size (int): Max entries in cache.
            cache_ttl (int): Cache entry TTL in seconds.
        """
        self.model_name = model_name
        self.cache_enabled = cache_enabled

        if cache_enabled:
            self.cache = EmbeddingCache(
                max_size=cache_size,
                ttl_seconds=cache_ttl
            )
            logger.info(
                "✓ Embedding cache enabled (size: %d, TTL: %ds)",
                cache_size,
                cache_ttl
            )
        else:
            self.cache = None

        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_time": 0.0,
        }

        skip_dimension_check = kwargs.get("_skip_dimension_check", False)

        if not skip_dimension_check:
            self._dimension = self._get_embedding_dimension()
        else:
            self._dimension = None  # type: ignore[assignment]

        self._model = self  # For LlamaIndex compatibility

    @abstractmethod
    def _get_embedding_dimension(self) -> int:
        """Probe to determine embedding dimension."""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._dimension  # type: ignore[return-value]

    def _update_stats(
        self,
        hit: bool = False,
        error: bool = False,
        duration: float = 0.0
    ) -> None:
        """
        Update performance statistics.

        Args:
            hit (bool): Cache hit indicator.
            error (bool): Error indicator.
            duration (float): Time taken for request.
        """
        self.stats["requests"] += 1
        if hit:
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1

        if error:
            self.stats["errors"] += 1

        self.stats["total_time"] += duration

    def _check_cache(
        self,
        text: str
    ) -> Optional[List[float]]:
        """
        Check cache for existing embedding.

        Args:
            text (str): Input text.

        Returns:
            Optional[List[float]]: Cached embedding or None.
        """
        if not self.cache_enabled or self.cache is None:
            return None

        return self.cache.get(text, self.model_name)

    def _store_cache(
        self,
        text: str,
        embedding: List[float]
    ) -> None:
        """
        Store embedding in cache.

        Args:
            text (str): Input text.
            embedding (List[float]): Embedding vector.
        """
        if self.cache_enabled and self.cache is not None:
            self.cache.set(text, self.model_name, embedding)

    @abstractmethod
    async def embed_async_single(
        self,
        text: str
    ) -> List[float]:
        """Get embedding for a single text asynchronously."""
        raise NotImplementedError

    @abstractmethod
    async def embed_async_many(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Get embeddings for multiple texts asynchronously."""
        raise NotImplementedError

    def embed(
        self,
        texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Get embedding(s) for text(s) synchronously.

        Args:
            texts (Union[str, List[str]]): Single text or list of texts.

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s).
        """
        # Synchronously embed text(s) using the underlying async methods. The
        # patched ``nest_asyncio`` allows calling ``run_until_complete`` even
        # when an event-loop is already running, removing the need for manual
        # ``loop._run_once`` hacks and preventing re-entrancy errors.

        # Select the appropriate coroutine depending on the input type.
        coroutine = (
            self.embed_async_single(texts)
            if isinstance(texts, str)
            else self.embed_async_many(texts)  # type: ignore[arg-type]
        )

        # Reuse the running loop if present; otherwise create a new one.
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coroutine)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dict[str, Any]: Stats including cache summary if enabled.
        """
        stats = self.stats.copy()

        if self.cache_enabled and self.cache is not None:
            stats["cache"] = self.cache.stats_summary

        return stats

    def clear_cache(self) -> None:
        """
        Clear the embedding cache.
        """
        if self.cache_enabled and self.cache is not None:
            self.cache.clear()

    async def close(self) -> None:
        """
        Clean up resources.

        Default no-op; override if needed.
        """
        pass


# ==== CORE PROCESSING MODULE ==== #
# --► JINA EMBEDDING MODEL

class JinaEmbedding(BaseEmbeddingModel):
    """Embedding model using the Jina AI API."""

    BASE_URL: str = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self,
        model_name: str = "jina-embeddings-v3",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        batch_size: int = 100,
        allow_empty_api_key: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize Jina AI embedding client.

        Args:
            model_name (str): Jina embedding model to use.
            api_key (Optional[str]): API key for Jina, or env var.
            max_retries (int): Max retry attempts.
            timeout (int): Request timeout in seconds.
            batch_size (int): Max texts per request batch.
            allow_empty_api_key (bool): Allow missing key in offline contexts.
        """
        self.api_key = api_key or os.environ.get("EMBEDDING_MODEL_API_KEY", os.environ.get("JINA_API_KEY"))

        if not self.api_key and not allow_empty_api_key:
            raise EmbeddingError("Jina API key not provided")

        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size

        # Extract caching arguments without duplication
        cache_enabled = kwargs.pop("cache_enabled", True)
        cache_size = kwargs.pop("cache_size", 10000)
        cache_ttl = kwargs.pop("cache_ttl", 3600)

        super().__init__(
            model_name=model_name,
            cache_enabled=cache_enabled,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            **kwargs,
        )

    def _get_embedding_dimension(self) -> int:
        """
        Determine embedding dimension from known constants or test.

        Returns:
            int: Embedding vector dimension.
        """
        if self.model_name in KNOWN_MODEL_DIMENSIONS:
            logger.info(
                "Using known dimension for %s: %d",
                self.model_name,
                KNOWN_MODEL_DIMENSIONS[self.model_name]
            )
            return KNOWN_MODEL_DIMENSIONS[self.model_name]

        logger.warning(
            "Unknown dimension for model %s; determining at runtime",
            self.model_name
        )
        loop = asyncio.get_event_loop()
        test_embedding = loop.run_until_complete(
            self.embed_async_single("test")
        )
        return len(test_embedding)

    @_retry(3)
    async def _execute_async_request(
        self,
        payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute API request with retries.

        Args:
            payload (Dict[str, Any]): JSON payload for request.

        Returns:
            List[Dict[str, Any]]: Raw API response data.
        """
        if not self.api_key:
            raise EmbeddingError("Jina API key not provided")

        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.BASE_URL,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()["data"]

    async def embed_async_single(
        self,
        text: str
    ) -> List[float]:
        """
        Get embedding for a single text asynchronously.

        Args:
            text (str): Input text.

        Returns:
            List[float]: Embedding vector.
        """
        start_time = time.time()

        cached = self._check_cache(text)
        if cached is not None:
            self._update_stats(
                hit=True,
                duration=time.time() - start_time
            )
            return cached

        payload = {
            "input": [text],
            "model": self.model_name,
        }

        try:
            result = await self._execute_async_request(payload)
        except Exception as exc:  # noqa: BLE001 – broad catch
            raise EmbeddingError("Jina embedding failed") from exc

        embedding = result[0]["embedding"]

        self._store_cache(text, embedding)
        self._update_stats(duration=time.time() - start_time)

        return embedding

    async def embed_async_many(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts asynchronously, with batching.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        if not texts:
            return []

        start_time = time.time()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            batch_embeddings = [
                self._check_cache(text) for text in batch
            ]

            texts_to_embed: List[str] = []
            indices_to_embed: List[int] = []

            for j, embedding in enumerate(batch_embeddings):
                if embedding is None:
                    texts_to_embed.append(batch[j])
                    indices_to_embed.append(j)

            if texts_to_embed:
                payload = {
                    "input": texts_to_embed,
                    "model": self.model_name,
                }

                try:
                    result = await self._execute_async_request(payload)
                except Exception as exc:  # noqa: BLE001 – broad catch
                    raise EmbeddingError("Jina embedding failed") from exc

                for idx, j in enumerate(indices_to_embed):
                    embedding = result[idx]["embedding"]
                    batch_embeddings[j] = embedding
                    self._store_cache(batch[j], embedding)

            all_embeddings.extend(
                [e for e in batch_embeddings if e is not None]
            )

        cache_hits = sum(
            1 for text in texts
            if self._check_cache(text) is not None
        )
        self._update_stats(duration=time.time() - start_time)
        self.stats["cache_hits"] = cache_hits
        self.stats["cache_misses"] = len(texts) - cache_hits

        return all_embeddings

    async def close(self) -> None:
        """
        Close HTTP client. No-op since client is scoped per request.
        """
        pass


# ==== CORE PROCESSING MODULE ==== #
# --► EMBEDDING MODEL FACTORY

class EmbeddingModel(LlamaIndexBaseEmbedding):
    """
    Factory for creating the appropriate embedding model based on
    configuration. Compatible with LlamaIndex's BaseEmbeddingModel.
    """

    _embed_model: Any = PrivateAttr(default=None)
    _dimension: int = PrivateAttr(default=0)

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_enabled: bool = True,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        timeout: int = 30,
        batch_size: int = 256,
        allow_empty_api_key: bool = False,
    ) -> None:
        """
        Initialize the embedding model.

        Args:
            provider: Embedding provider (e.g., 'jina', 'beam').
            model_name: Name of the embedding model.
            cache_enabled: Whether to enable embedding caching.
            cache_size: Maximum number of entries in the cache.
            cache_ttl: Time-to-live for cache entries in seconds.
            max_retries: Maximum number of retry attempts for API calls.
            timeout: Timeout for API calls in seconds.
            batch_size: Maximum number of texts to embed in a single API call.
            allow_empty_api_key: Allow empty API key for testing.
        """
        # Call the parent class initializer
        super().__init__()
        
        # Determine the provider from settings if not provided
        if provider is None:
            provider = settings.embedding_provider.lower()
            logger.info(f"Using embedding provider from settings: {provider}")

        # Ensure model is set before any embedding operations
        if model_name is None:
            model_name = settings.embedding_model

        # Common kwargs passed to all provider implementations
        common_kwargs = dict(
            model_name=model_name,
            max_retries=max_retries,
            timeout=timeout,
            batch_size=batch_size,
            cache_enabled=cache_enabled,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            allow_empty_api_key=allow_empty_api_key,
        )

        # Import lazily to avoid circular dependency issues and gracefully
        # handle the module/package name clash (``embeddings.py`` vs
        # ``embeddings/``). We attempt the regular import first; on failure we
        # load the adapter from its file path and register it in ``sys.modules``
        # so that subsequent imports succeed.
        import importlib
        import importlib.util as _importlib_util
        import sys
        from pathlib import Path

        try:
            adapter_mod = importlib.import_module(
                "root.src.core.embeddings.embeddings_adapter",
            )
        except ModuleNotFoundError:
            adapter_path = (
                Path(__file__).with_name("embeddings") / "embeddings_adapter.py"
            )
            spec = _importlib_util.spec_from_file_location(
                "root.src.core.embeddings.embeddings_adapter",
                adapter_path,
            )
            if spec is None or spec.loader is None:  # pragma: no cover – safety
                raise ImportError(
                    f"Cannot load embedding adapter from {adapter_path}",
                )

            adapter_mod = _importlib_util.module_from_spec(spec)
            sys.modules[spec.name] = adapter_mod  # type: ignore[arg-type]
            spec.loader.exec_module(adapter_mod)  # type: ignore[arg-type]

        get_embedding_model = adapter_mod.get_embedding_model

        _embed_model = get_embedding_model(provider=provider, **common_kwargs)

        object.__setattr__(self, "_embed_model", _embed_model)
        object.__setattr__(self, "_dimension", _embed_model.dimension)

        # Cache Pydantic sentinel attributes to satisfy BaseModel expectations
        object.__setattr__(self, "__pydantic_fields_set__", set())
        object.__setattr__(self, "__pydantic_extra__", None)
        object.__setattr__(self, "__pydantic_private__", {})
        object.__setattr__(self, "__pydantic_forward_refs__", {})

        logger.info(
            "Initialized embedding model: %s (dim=%s)",
            provider,
            _embed_model.dimension,
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding synchronously."""
        return self.get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding synchronously."""
        return self.get_text_embedding(text)

    def _get_text_embedding_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """Get batch embeddings synchronously."""
        return self.get_text_embedding_batch(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously."""
        return await self._embed_model.embed_async_single(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously."""
        return await self._embed_model.embed_async_single(text)

    async def _aget_text_embedding_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """Get batch embeddings asynchronously."""
        return await self._embed_model.embed_async_many(texts)

    # ------------------------------------------------------------------
    # Public async delegation methods expected by tests
    # ------------------------------------------------------------------

    async def embed_async_single(self, text: str) -> List[float]:
        """Asynchronously embed a single document.

        This mirrors the interface of the underlying embedding model, keeping
        the factory thin while still exposing the convenience API required by
        unit-tests and upstream code. All heavy-lifting is delegated to
        ``self._embed_model``.
        """
        return await self._embed_model.embed_async_single(text)

    async def embed_async_many(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed multiple documents.

        Args:
            texts: List of input documents.

        Returns:
            A list of embedding vectors in the same order as ``texts``.
        """
        return await self._embed_model.embed_async_many(texts)

    # ------------------------------------------------------------------
    # Thin wrappers for cache/debug statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return performance/cache statistics of the active embedder."""
        return self._embed_model.get_stats()

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._dimension

    async def aembed_query(self, query: str) -> List[float]:
        """Public method: async query embedding."""
        return await self._embed_model.embed_async_single(query)

    async def aembed_documents(
        self,
        documents: List[str]
    ) -> List[List[float]]:
        """Public method: async document embeddings."""
        return await self._embed_model.embed_async_many(documents)

    def get_text_embedding(
        self,
        text: str,
        **kwargs: Any
    ) -> List[float]:
        """Synchronously get embedding for text."""
        coroutine = self._embed_model.embed_async_single(text)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coroutine)

    def get_text_embedding_batch(
        self,
        texts: List[str],
        **kwargs: Any
    ) -> List[List[float]]:
        """Synchronously get embeddings for texts."""
        coroutine = self._embed_model.embed_async_many(texts)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coroutine)

    async def aget_text_embedding(
        self,
        text: str,
        **kwargs: Any
    ) -> List[float]:
        """Asynchronously get embedding for text."""
        return await self._embed_model.embed_async_single(text)

    async def aget_text_embedding_batch(
        self,
        texts: List[str],
        **kwargs: Any
    ) -> List[List[float]]:
        """Asynchronously get embeddings for texts."""
        return await self._embed_model.embed_async_many(texts)

    def embed(
        self,
        texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """Get embedding(s) for text(s) synchronously."""
        if isinstance(texts, str):
            return self.get_text_embedding(texts)

        return self.get_text_embedding_batch(texts)

    async def close(self) -> None:
        """Close any open resources."""
        await self._embed_model.close()

    async def warm_up(self) -> None:
        """
        Pre-load the model and perform a test embedding.
        """
        if hasattr(self._embed_model, "warm_up"):
            await self._embed_model.warm_up()
        else:
            # Do a test embedding to warm up the model
            await self.aget_text_embedding("This is a test embedding for warm-up.")