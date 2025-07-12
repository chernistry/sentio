#!/usr/bin/env python3
"""
Ollama-based embedding provider for Sentio RAG.

This module provides an implementation of embedding generation using local Ollama models.
It's designed to be compatible with the core EmbeddingModel interface.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from plugins.interface import SentioPlugin
from root.src.utils.settings import settings

# Default embedding dimension for Ollama models
DEFAULT_OLLAMA_DIMENSION = 1024  # Default for Qwen3 embedding models

logger = logging.getLogger(__name__)


class OllamaEmbedding:
    """Embedding model using local Ollama instance."""
    
    def __init__(
        self,
        model_name: str,
        ollama_url: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        batch_size: int = 256,
        max_parallel_requests: int = 8,
        **kwargs,
    ):
        """
        Initialize Ollama embedding provider.
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama API server
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            batch_size: Maximum number of texts to embed in one batch
            max_parallel_requests: Maximum number of concurrent requests
        """
        self.model_name = model_name
        self.base_url = (ollama_url or settings.ollama_url).rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_parallel_requests = max_parallel_requests
        
        # Track performance statistics
        self.stats = {"requests": 0, "errors": 0, "total_time": 0.0}
        
        # One shared AsyncClient with connection pooling & keep-alive
        limits = httpx.Limits(
            max_keepalive_connections=max_parallel_requests * 2,
            max_connections=max_parallel_requests * 4,
        )
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=limits,
        )

        # Semaphore to limit parallel requests
        self._semaphore = asyncio.Semaphore(max_parallel_requests)

        # Warm-up flag
        self._warmed_up = False
        
        # Embedding dimension (initialized lazily)
        self._dimension = None
        
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Use default dimension, actual dimension will be determined on first call
            return DEFAULT_OLLAMA_DIMENSION
        return self._dimension
        
    async def _get_actual_dimension(self) -> int:
        """Determine actual embedding dimension by making a test request."""
        try:
            embedding = await self.embed_async_single("Test embedding dimension")
            return len(embedding)
        except Exception as e:
            logger.warning(f"Failed to determine embedding dimension: {e}")
            return DEFAULT_OLLAMA_DIMENSION
            
    async def _execute_async_request(self, text: str) -> Dict[str, Any]:
        """Execute embedding request to Ollama API with retries."""
        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    response = await self._client.post(
                        "/api/embeddings",
                        json={"model": self.model_name, "prompt": text}
                    )
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to get embeddings after {self.max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Fallback return in case of exception
        return {}
            
    async def embed_async_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        start_time = time.time()
        self.stats["requests"] += 1
        
        try:
            response = await self._execute_async_request(text)
            embedding = response.get("embedding", [])
            
            # If this is first request, store the dimension
            if self._dimension is None and embedding:
                self._dimension = len(embedding)
                logger.info(f"Ollama embedding dimension determined: {self._dimension}")
                
            return embedding
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Embedding generation failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.stats["total_time"] += duration
            
    async def embed_async_many(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in parallel."""
        if not texts:
            return []
            
        # Process in smaller batches to control concurrency
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Create tasks for concurrent processing
            tasks = [self.embed_async_single(text) for text in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in batch embedding: {result}")
                    # Use empty vector of correct dimension as fallback
                    dim = self._dimension or DEFAULT_OLLAMA_DIMENSION
                    batch_results[idx] = [0.0] * dim
                    
            results.extend(batch_results)
            
        return results
            
    async def warm_up(self) -> None:
        """Pre-warm the model with a test embedding."""
        if self._warmed_up:
            return
            
        try:
            logger.info(f"Warming up Ollama embedding model: {self.model_name}")
            await self.embed_async_single("Warm-up text for embedding model initialization")
            self._warmed_up = True
            logger.info("✓ Ollama embedding model warmed up successfully")
        except Exception as e:
            logger.warning(f"Failed to warm up embedding model: {e}")
            
    async def close(self):
        """Close the HTTP client."""
        if hasattr(self, '_client'):
            await self._client.aclose()
            
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return {
            "model": self.model_name,
            "requests": self.stats["requests"],
            "errors": self.stats["errors"],
            "avg_time": self.stats["total_time"] / max(1, self.stats["requests"]),
            "dimension": self._dimension or DEFAULT_OLLAMA_DIMENSION,
        }
    
    def __repr__(self) -> str:
        return f"OllamaEmbedding(model={self.model_name}, dimension={self.dimension})"


class OllamaEmbeddingPlugin(SentioPlugin):
    """Plugin wrapper for Ollama embedding provider."""

    name = "ollama_embedding"
    plugin_type = "embedding"

    def __init__(self, model_name: str | None = None, **kwargs: Any) -> None:
        model_name = model_name or settings.ollama_model
        self.model = OllamaEmbedding(model_name=model_name, **kwargs)

    def register(self, pipeline: Any) -> None:
        pipeline.embed_model = self.model


def get_plugin() -> SentioPlugin:
    """Return plugin instance."""
    return OllamaEmbeddingPlugin()
