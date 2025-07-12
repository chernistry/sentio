#!/usr/bin/env python3
"""
Web search retrieval module for Sentio RAG.

This module provides integration with web search engines (DuckDuckGo, Brave)
to augment RAG results with real-time information from the internet.
"""

import os
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional
from contextlib import suppress

import httpx
import diskcache as dc

from qdrant_client import QdrantClient
from plugins.interface import SentioPlugin

logger = logging.getLogger(__name__)

# Optional dependency imports with graceful degradation
try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    DDGS = None

class WebSearchRetriever:
    """
    Retrieve search snippets from web search engines.
    
    This class performs a web search and returns the page title + snippet as
    document text. Results can be cached locally to reduce API calls.
    """
    
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embed_model: Any = None,
        source: str = "duckduckgo",
        max_results: int = 10,
        fetch_full_pages: bool = False,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize web search retriever.
        
        Args:
            client: Optional Qdrant client for vector caching
            embed_model: Optional embedding model for vector caching
            source: Search engine to use ("duckduckgo" or "brave")
            max_results: Maximum number of results to return
            fetch_full_pages: Whether to fetch full page content
            cache_ttl_hours: Cache TTL in hours
        """
        self.source = source
        self.max_results = max_results
        self.fetch_full_pages = fetch_full_pages
        
        # Configure page fetcher if needed
        self._page_fetcher = None
        if fetch_full_pages:
            try:
                from .web_page_fetcher import AsyncWebPageFetcher
                self._page_fetcher = AsyncWebPageFetcher()
            except ImportError:
                logger.warning("Failed to load AsyncWebPageFetcher, full page fetching disabled")
                
        # Brave API setup
        self.api_key = os.getenv("BRAVE_API_KEY")
        self._use_brave = self.api_key is not None
        
        # DuckDuckGo availability check
        if not DUCKDUCKGO_AVAILABLE and not self._use_brave:
            logger.warning(
                "DuckDuckGo search is not available and no Brave API key is configured. "
                "Install duckduckgo-search or set BRAVE_API_KEY."
            )
        
        # Diskcache setup for caching results
        self._cache = dc.Cache(".cache/web_search")
        self._cache_ttl = timedelta(hours=cache_ttl_hours).total_seconds()
        
        # Vector cache in Qdrant (optional)
        self.client = client
        self.embed_model = embed_model
        self.cache_collection = "web_cache"
        if self.client is not None and self.embed_model is not None:
            self._ensure_cache_collection()
            
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return search snippets for a query.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            
        Returns:
            List of document dictionaries with text, URL, and metadata
        """
        limit = top_k or self.max_results
        
        # 1. Try cache first
        cached_docs = self._cache.get(query)
        if cached_docs:
            return cached_docs[:limit]
            
        # 2. Try vector cache in Qdrant
        if self.client and self.embed_model and self._has_cache_collection():
            docs = self._retrieve_from_qdrant(query, limit)
            if docs:
                self._cache.set(query, docs, expire=self._cache_ttl)
                return docs
                
        # 3. Perform web search
        docs = self._retrieve_brave(query, limit) if self._use_brave else self._retrieve_duckduckgo(query, limit)
        
        # 4. Cache results
        if docs:
            self._cache.set(query, docs, expire=self._cache_ttl)
            if self.client and self.embed_model:
                self._store_in_qdrant(docs)
                
        return docs
        
    async def retrieve_async(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Async version that can fetch full page content.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            
        Returns:
            List of document dictionaries with text, URL, and metadata
        """
        docs = self.retrieve(query, top_k=top_k)
        
        # If we don't have a page fetcher or no docs, just return results
        if not self._page_fetcher or not docs:
            return docs
            
        # Fetch full page content
        urls = [d.get("url") for d in docs if d.get("url")]
        if not urls:
            return docs
            
        try:
            full_pages = await self._page_fetcher.fetch_batch(urls)
            url_to_text = {item["url"]: item["text"] for item in full_pages}
            
            # Update documents with full page content
            for d in docs:
                if d.get("url") in url_to_text:
                    d["text"] = url_to_text[d["url"]]
        except Exception as e:
            logger.warning(f"Failed to fetch full page content: {e}")
            
        return docs
    
    def _retrieve_brave(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Retrieve search results from Brave Search API."""
        if not self.api_key:
            return self._retrieve_duckduckgo(query, limit)
            
        docs = []
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                    params={"q": query, "count": limit, "result_filter": "web"}
                )
                response.raise_for_status()
                data = response.json()
                
            results = data.get("web", {}).get("results", [])
            for rank, item in enumerate(results[:limit]):
                snippet = (
                    (item.get("title", "") + "\n" + item.get("description", "")).strip()
                )
                if not snippet:
                    continue
                    
                score = 1.0 / (rank + 1)  # Simple rank-based scoring
                docs.append({
                    "id": item.get("url", f"web:{rank}"),
                    "text": snippet,
                    "score": score,
                    "source": "web",
                    "url": item.get("url", ""),
                })
        except Exception as e:
            logger.warning(f"Brave search failed: {e}")
            return self._retrieve_duckduckgo(query, limit)
            
        return docs
        
    def _retrieve_duckduckgo(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Retrieve search results from DuckDuckGo."""
        if not DUCKDUCKGO_AVAILABLE:
            logger.warning("DuckDuckGo search not available, install duckduckgo-search package")
            return []
            
        docs = []
        try:
            with DDGS() as ddgs:
                for rank, result in enumerate(ddgs.text(query, max_results=limit)):
                    snippet = (result.get("title", "") + "\n" + result.get("body", "")).strip()
                    if not snippet:
                        continue
                        
                    score = 1.0 / (rank + 1)  # Simple rank-based scoring
                    docs.append({
                        "id": result.get("href", f"web:{rank}"),
                        "text": snippet,
                        "score": score,
                        "source": "web",
                        "url": result.get("href", ""),
                    })
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            
        return docs[:limit]
    
    def _ensure_cache_collection(self) -> None:
        """Create vector cache collection if it doesn't exist."""
        if not self.client:
            return
            
        try:
            if not self.client.collection_exists(collection_name=self.cache_collection):
                self.client.create_collection(
                    collection_name=self.cache_collection,
                    vectors_config={"size": self.embed_model.dimension, "distance": "Cosine"},
                )
        except Exception as e:
            logger.warning(f"Failed to create cache collection: {e}")
            
    def _has_cache_collection(self) -> bool:
        """Check if cache collection exists and has points."""
        if not self.client:
            return False
            
        try:
            info = self.client.get_collection(self.cache_collection)
            return (info.points_count or 0) > 0
        except Exception:
            return False
            
    def _retrieve_from_qdrant(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Retrieve cached results from Qdrant vector store."""
        if not self.client or not self.embed_model:
            return []
            
        try:
            # Get query embedding
            query_vector = self.embed_model.embed([query])[0]
            
            # Search Qdrant
            results = self.client.search(
                collection_name=self.cache_collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
            
            return [
                {
                    "id": str(result.id),
                    "text": result.payload.get("text", ""),
                    "score": result.score,
                    "source": "web",
                    "url": result.payload.get("url", ""),
                }
                for result in results
            ]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
            
    def _store_in_qdrant(self, docs: List[Dict[str, Any]]) -> None:
        """Store documents in Qdrant vector cache."""
        if not self.client or not self.embed_model or not docs:
            return
            
        try:
            points = []
            for doc in docs:
                # Skip documents without text
                text = doc.get("text")
                if not text:
                    continue
                    
                # Get embedding
                vector = self.embed_model.embed([text])[0]
                
                # Create point
                points.append({
                    "id": doc.get("id", f"web:{len(points)}"),
                    "vector": vector,
                    "payload": {
                        "text": text,
                        "url": doc.get("url", ""),
                        "source": doc.get("source", "web"),
                    }
                })
                
            if points:
                self.client.upsert(
                    collection_name=self.cache_collection,
                    points=points
                )
        except Exception as e:
            logger.warning(f"Failed to store documents in vector cache: {e}")


class WebSearchPlugin(SentioPlugin):
    """Plugin providing web search retrieval."""

    name = "web_search"
    plugin_type = "retriever"

    def __init__(self, **kwargs: Any) -> None:
        self.retriever = WebSearchRetriever(**kwargs)

    def register(self, pipeline: Any) -> None:
        pipeline.web_search = self.retriever


def get_plugin() -> SentioPlugin:
    """Return plugin instance."""
    return WebSearchPlugin()
