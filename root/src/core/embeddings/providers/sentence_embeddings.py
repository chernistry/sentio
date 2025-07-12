#!/usr/bin/env python3
"""
Local embedding provider using Sentence-Transformers.

This module is an optional component for running embedding models locally,
intended for development or offline usage.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np

from sentence_transformers import SentenceTransformer

from root.src.core.tasks.embeddings import BaseEmbeddingModel, EmbeddingError
from plugins.interface import SentioPlugin
from root.src.utils.settings import settings


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Embedding model using Sentence Transformers (HuggingFace)."""

    def __init__(self, model_name: str, **kwargs):
        self.model = SentenceTransformer(model_name)
        self.is_query_prefix_model = model_name.startswith(("BAAI/bge", "bge"))

        # Configure additional settings
        self.normalize = kwargs.get("normalize", True)
        self.batch_size = kwargs.get("batch_size", 32)
        
        # Some models (especially BGE) benefit from query/document prefixes
        # Based on https://huggingface.co/BAAI/bge-large-en-v1.5#model-card
        self.query_prefix = kwargs.get("query_prefix", "Represent this sentence for searching relevant passages: ")
        self.document_prefix = kwargs.get("document_prefix", "Represent this passage for retrieval: ")

        # Call parent init which will also determine dimension
        super().__init__(model_name=model_name, **kwargs)

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension by running a test embedding."""
        test_embedding = self.model.encode("test", normalize_embeddings=self.normalize)
        return len(test_embedding)

    def _prepare_text(self, text: str, is_query: bool = False) -> str:
        """Prepare text with appropriate prefix if model supports it."""
        if not self.is_query_prefix_model:
            return text
        
        prefix = self.query_prefix if is_query else self.document_prefix
        return f"{prefix}{text}"

    async def embed_async_single(self, text: str, is_query: bool = False) -> List[float]:
        """Get embedding for a single text asynchronously."""
        start_time = time.time()
        
        # Check cache
        cached = self._check_cache(text)
        if cached is not None:
            self._update_stats(hit=True, duration=time.time() - start_time)
            return cached
        
        try:
            # Process text based on query/document and model type
            prepared_text = self._prepare_text(text, is_query)
            
            # Run encoding in thread pool to prevent blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode(prepared_text, normalize_embeddings=self.normalize).tolist()
            )
            
            # Store in cache
            self._store_cache(text, embedding)
            
            self._update_stats(duration=time.time() - start_time)
            return embedding
        except Exception as e:
            self._update_stats(error=True, duration=time.time() - start_time)
            raise EmbeddingError(f"Failed to get embedding: {e}")

    async def embed_async_many(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Get embeddings for multiple texts asynchronously, with batching."""
        if not texts:
            return []
        
        if len(texts) == 1:
            return [await self.embed_async_single(texts[0], is_query)]
        
        start_time = time.time()
        result: List[List[float]] = []
        
        # Check cache for each text
        cache_hits = []
        to_embed = []
        indices = []
        
        for i, text in enumerate(texts):
            cached = self._check_cache(text)
            if cached is not None:
                cache_hits.append((i, cached))
            else:
                to_embed.append(self._prepare_text(text, is_query))
                indices.append(i)
        
        # Sort cache hits by index
        cache_hits.sort(key=lambda x: x[0])
        
        # Get embeddings for texts not in cache
        embeddings = []
        if to_embed:
            try:
                # Process embeddings in batches
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    None,
                    lambda: self.model.encode(
                        to_embed,
                        batch_size=self.batch_size,
                        normalize_embeddings=self.normalize
                    ).tolist()
                )
                
                # Store in cache
                for i, embedding in zip(indices, batch_embeddings):
                    self._store_cache(texts[i], embedding)
                    embeddings.append((i, embedding))
            except Exception as e:
                self._update_stats(error=True, duration=time.time() - start_time)
                raise EmbeddingError(f"Failed to get embeddings: {e}")
        
        # Combine cache hits and fresh embeddings, maintaining original order
        combined = cache_hits + embeddings
        combined.sort(key=lambda x: x[0])
        
        # Extract just the embeddings
        result = [emb for _, emb in combined]
        
        # Update stats
        self._update_stats(
            hit=False,
            duration=time.time() - start_time
        )
        self.stats["cache_hits"] += len(cache_hits)
        self.stats["cache_misses"] += len(to_embed)
        
        return result


class LocalEmbeddingPlugin(SentioPlugin):
    """Plugin wrapper for local embedding model."""

    name = "local_embedding"
    plugin_type = "embedding"

    def __init__(self, model_name: str | None = None, **kwargs: Any) -> None:
        model_name = model_name or settings.local_embed_model
        self.model = SentenceTransformerEmbedding(model_name, **kwargs)

    def register(self, pipeline: Any) -> None:
        pipeline.embed_model = self.model


def get_plugin() -> SentioPlugin:
    """Return plugin instance."""
    return LocalEmbeddingPlugin()
