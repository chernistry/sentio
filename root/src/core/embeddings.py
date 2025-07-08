#!/usr/bin/env python3
"""
Advanced embedding service with caching, error handling, and optimization.

This module provides a production-ready interface to Jina AI's embedding models
with features like automatic retries, caching, batch processing, and monitoring.
"""

import asyncio
import hashlib
import logging
import os
import time
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Union

import httpx
import nest_asyncio

# Применяем патч для вложенных event loop
nest_asyncio.apply()

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


class EmbeddingCache:
    """Simple in-memory cache for embeddings with TTL."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.stats = {'hits': 0, 'misses': 0, 'size': 0}  # Add stats tracking
    
    def _get_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache if available and not expired."""
        key = self._get_key(text, model_name)
        entry = self.cache.get(key)
        
        if entry is None:
            self.stats['misses'] += 1  # Track cache miss
            return None
            
        if time.time() - entry['timestamp'] > self.ttl_seconds:
            del self.cache[key]
            self.stats['misses'] += 1  # Track expiration as miss
            return None
        
        self.stats['hits'] += 1  # Track cache hit    
        return entry['embedding']
    
    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """Store embedding in cache with timestamp."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        key = self._get_key(text, model_name)
        self.cache[key] = {
            'embedding': embedding,
            'timestamp': time.time()
        }
        self.stats['size'] = len(self.cache)  # Update size stat
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
        self.stats['size'] = 0  # Update size stat
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
        }


def _retry(max_retries: int):
    """Decorator for async functions with exponential backoff."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await fn(*args, **kwargs)
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    if attempt == max_retries - 1:
                        raise EmbeddingError(f"Request failed after {max_retries} attempts: {e}")
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429: # Rate limited
                        if attempt == max_retries - 1:
                           raise EmbeddingError(f"Rate limited. Failed after {max_retries} attempts.")
                        wait_time = 2 ** (attempt + 1) # Longer wait for rate limits
                        logger.warning(f"Rate limited, retrying in {wait_time}s.")
                        await asyncio.sleep(wait_time)
                    else:
                        raise EmbeddingError(f"HTTP error: {e.response.status_code} - {e.response.text}") from e
        return wrapper
    return decorator


class EmbeddingModel:
    """
    Advanced embedding model wrapper using direct HTTP calls to Jina AI.
    
    Features:
    - Automatic retries with exponential backoff
    - In-memory caching with TTL
    - Batch processing optimization
    - Comprehensive error handling
    - Performance monitoring
    - Async and sync interfaces
    """
    
    BASE_URL = "https://api.jina.ai/v1/embeddings"
    
    def __init__(
        self, 
        model_name: str = "jina-embeddings-v4",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        cache_enabled: bool = True,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        batch_size: int = 100,
        task: str = "retrieval.passage"
    ):
        """
        Initialize embedding model with configuration.
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = batch_size
        self.task = task
        
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise EmbeddingError("JINA_API_KEY environment variable not set. Get one from https://jina.ai/?sui=apikey")

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        self._async_client = httpx.AsyncClient(headers=self._headers, timeout=self.timeout)
        self._sync_client = httpx.Client(headers=self._headers, timeout=self.timeout)

        # Initialize cache settings BEFORE probing dimension to avoid AttributeError
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache = EmbeddingCache(max_size=cache_size, ttl_seconds=cache_ttl)
            logger.info(f"✓ Embedding cache enabled (size: {cache_size}, TTL: {cache_ttl}s)")
        else:
            self.cache = None

        # Stats tracking - initialize before getting dimension
        self.stats = {
            'requests': 0, 'cache_hits': 0, 'cache_misses': 0, 'errors': 0, 'total_time': 0.0
        }

        # Now that cache attributes are set, safely probe dimension
        self._dimension = self._get_embedding_dimension()

        # Expose self as LlamaIndex-compatible embedding model so that external
        # libraries like LlamaIndex can call `.get_text_embedding*` APIs.  This
        # also keeps backwards-compatibility with older code that expects an
        # attribute named `_model`.
        self._model = self
    
    def _get_embedding_dimension(self) -> int:
        """Probe the API to determine embedding dimension."""
        try:
            logger.debug(f"Probing for embedding dimension of model {self.model_name}")
            embedding = self.embed("probe")
            if isinstance(embedding, list) and isinstance(embedding[0], float):
                dim = len(embedding)
                logger.info(f"✓ Determined embedding dimension: {dim}")
                return dim
            raise EmbeddingError("Failed to determine dimension from response.")
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension automatically, falling back to 2048: {e}")
            return 2048

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def _update_stats(self, hit: bool = False, error: bool = False, duration: float = 0.0) -> None:
        """Update performance statistics."""
        self.stats['requests'] += 1
        if hit:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
        if error:
            self.stats['errors'] += 1
        self.stats['total_time'] += duration
    
    def _check_cache(self, text: str) -> Optional[List[float]]:
        """Check cache for existing embedding."""
        if not self.cache_enabled or not self.cache:
            return None
        return self.cache.get(text, self.model_name)
    
    def _store_cache(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        if self.cache_enabled and self.cache:
            self.cache.set(text, self.model_name, embedding)

    @_retry(3)
    async def _execute_async_request(self, payload: Dict) -> List[Dict[str, Any]]:
        response = await self._async_client.post(self.BASE_URL, json=payload)
        response.raise_for_status()
        return response.json().get("data", [])

    async def embed_async_single(self, text: str) -> List[float]:
        """Get embedding for a single text asynchronously."""
        start_time = time.time()
        
        cached = self._check_cache(text)
        if cached is not None:
            self._update_stats(hit=True, duration=time.time() - start_time)
            return cached
        
        try:
            payload = {"input": [text], "model": self.model_name, "task": self.task}
            data = await self._execute_async_request(payload)
            if not data or "embedding" not in data[0]:
                raise EmbeddingError("Invalid response format from Jina API")
            
            embedding = data[0]["embedding"]
            self._store_cache(text, embedding)
            
            duration = time.time() - start_time
            self._update_stats(hit=False, duration=duration)
            logger.debug(f"Generated embedding for text (length: {len(text)}, time: {duration:.2f}s)")
            return embedding
        except Exception as e:
            self._update_stats(error=True, duration=time.time() - start_time)
            raise EmbeddingError(f"Embedding failed: {e}") from e

    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts asynchronously with batching."""
        if not texts:
            return []
        
        all_embeddings: List[Optional[List[float]]] = [None] * len(texts)
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self._check_cache(text)
            if cached:
                all_embeddings[i] = cached
                self._update_stats(hit=True)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        if not uncached_texts:
            return all_embeddings # type: ignore

        # Process uncached texts in batches
        for i in range(0, len(uncached_texts), self.batch_size):
            batch_texts = uncached_texts[i:i + self.batch_size]
            batch_indices = uncached_indices[i:i + self.batch_size]
            
            try:
                start_time = time.time()
                payload = {"input": batch_texts, "model": self.model_name, "task": self.task}
                data = await self._execute_async_request(payload)
                
                if len(data) != len(batch_texts):
                    raise EmbeddingError("Mismatch between request and response batch size")

                new_embeddings = [item["embedding"] for item in data]
                duration = time.time() - start_time
                
                for j, (text, embedding) in enumerate(zip(batch_texts, new_embeddings)):
                    self._store_cache(text, embedding)
                    original_index = batch_indices[j]
                    all_embeddings[original_index] = embedding
                    self._update_stats(hit=False, duration=duration / len(batch_texts))
                
                logger.debug(f"Generated {len(new_embeddings)} embeddings in batch (time: {duration:.2f}s)")

            except Exception as e:
                logger.error(f"Batch embedding failed, falling back to individual requests: {e}")
                for j, text in enumerate(batch_texts):
                    original_index = batch_indices[j]
                    try:
                        embedding = await self.embed_async_single(text)
                        all_embeddings[original_index] = embedding
                    except EmbeddingError as individual_error:
                        logger.error(f"Failed to embed individual text '{text[:50]}...': {individual_error}")
                        all_embeddings[original_index] = [0.0] * self.dimension
                        self._update_stats(error=True)

        return all_embeddings # type: ignore

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Synchronous embedding interface."""
        if isinstance(texts, str):
            try:
                # Используем get_event_loop и run_until_complete вместо asyncio.run()
                # Это безопасно благодаря nest_asyncio.apply() в начале файла
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.embed_async_single(texts))
            except Exception as e:
                raise EmbeddingError(f"Failed to get embedding: {e}") from e
        else:
            try:
                # Используем get_event_loop и run_until_complete вместо asyncio.run()
                # Это безопасно благодаря nest_asyncio.apply() в начале файла
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.embed_async(texts))
            except Exception as e:
                raise EmbeddingError(f"Failed to get batch embeddings: {e}") from e

    # ------------------------------------------------------------------
    # LlamaIndex compatibility layer
    # ------------------------------------------------------------------

    def get_text_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """Return embedding for a single text in a sync context (LlamaIndex API)."""
        result = self.embed(text)
        # `embed` returns List[float] for single string input
        if isinstance(result, list) and (not result or isinstance(result[0], float)):
            return result  # type: ignore[return-value]
        raise ValueError("Unexpected embedding format for single text")

    def get_text_embedding_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:  # noqa: D401
        """Return embeddings for multiple texts (LlamaIndex API)."""
        result = self.embed(texts)
        if isinstance(result, list) and (not result or isinstance(result[0], list)):
            return result  # type: ignore[return-value]
        raise ValueError("Unexpected embedding format for batch of texts")

    async def aget_text_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """Async embedding for a single text (LlamaIndex API)."""
        return await self.embed_async_single(text)

    async def aget_text_embedding_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Async embeddings for multiple texts (LlamaIndex API)."""
        return await self.embed_async(texts)

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        total_requests = self.stats['requests']
        if total_requests == 0:
            return self.stats.copy()
        
        stats = self.stats.copy()
        stats.update({
            'cache_hit_rate': (self.stats['cache_hits'] / total_requests) if total_requests else 0,
            'error_rate': (self.stats['errors'] / total_requests) if total_requests else 0,
            'avg_time_per_request': (self.stats['total_time'] / total_requests) if total_requests else 0
        })
        
        if self.cache_enabled and self.cache:
            stats['cache_stats'] = self.cache.stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache_enabled and self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")
    
    async def close(self):
        """Close the async client."""
        await self._async_client.aclose()

    def __repr__(self) -> str:
        return f"EmbeddingModel(model={self.model_name}, cache={self.cache_enabled}, dim={self.dimension})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sync_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close() 