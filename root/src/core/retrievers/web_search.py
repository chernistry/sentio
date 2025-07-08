from __future__ import annotations

"""Web search retriever module.

Fetches web search results (DuckDuckGo) at query time and converts them into
RAG-compatible document dicts. No external state is preserved; results are
fetched ad-hoc and fed directly to the reranking stage.

Dependencies: `duckduckgo-search` which is lightweight and requires no API key.
"""

from typing import List, Dict, Any
import os
from contextlib import suppress
import httpx
from datetime import timedelta

import diskcache as dc

from qdrant_client import QdrantClient

from ..embeddings import EmbeddingModel

from duckduckgo_search import DDGS
from ..web.page_fetcher import AsyncWebPageFetcher

# Optional DDG client
with suppress(ImportError):
    from duckduckgo_search import DDGS  # type: ignore


class WebSearchRetriever:  # pylint: disable=too-few-public-methods
    """Retrieve search snippets from DuckDuckGo.

    This class performs a web search and returns the page *title + snippet* as
    the document text. Each document dict is structured identically to the
    `HybridRetriever` output so that it can be fused and reused downstream.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        embed_model: EmbeddingModel | None = None,
        source: str | None = "duckduckgo",
        max_results: int = 10,
        fetch_full_pages: bool = False,
        cache_ttl_hours: int = 24,
    ):
        self.source = source or "duckduckgo"
        self.max_results = max_results
        self.fetch_full_pages = fetch_full_pages
        self._page_fetcher: AsyncWebPageFetcher | None = AsyncWebPageFetcher() if fetch_full_pages else None
        self.api_key = os.getenv("BRAVE_API_KEY")
        self._use_brave = self.api_key is not None

        # Diskcache setup
        self._cache = dc.Cache(".cache/web_search")
        self._cache_ttl = timedelta(hours=cache_ttl_hours).total_seconds()

        # Vector cache in Qdrant
        self.client = client
        self.embed_model = embed_model
        self.cache_collection = "web_cache"
        if self.client is not None and self.embed_model is not None:
            self._ensure_cache_collection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:  # noqa: D401
        """Return search snippets for *query* limited to *top_k* results.

        If a `BRAVE_API_KEY` environment variable is set and the `brave-search`
        library is installed, Brave Search API is used. Otherwise, DuckDuckGo
        is used as a fallback (no key required).
        """
        limit = top_k or self.max_results

        # 1. Try in-memory diskcache first
        cached_docs = self._cache.get(query)
        if cached_docs:
            return cached_docs[:limit]

        # 2. Try vector search in Qdrant web_cache
        if self.client is not None and self.embed_model is not None and self._has_cache_vectors():
            docs = self._retrieve_from_qdrant(query, limit)
            if docs:
                # Store in diskcache for faster future hits
                self._cache.set(query, docs, expire=self._cache_ttl)
                return docs

        # 3. Fallback to live web search
        docs = self._retrieve_brave(query, limit) if self._use_brave else self._retrieve_duckduckgo(query, limit)

        # 4. Optionally fetch full pages asynchronously elsewhere (handled in async path)

        # 5. Persist new docs
        if self.client is not None and self.embed_model is not None and docs:
            self._upsert_docs(docs)

        # 6. Put into diskcache
        self._cache.set(query, docs, expire=self._cache_ttl)

        return docs

    async def retrieve_async(self, query: str, top_k: int | None = None):
        """Async version that optionally fetches full page content."""
        docs = self.retrieve(query, top_k=top_k)
        if self._page_fetcher is None:
            for d in docs:
                d.setdefault("source", "web")
            return docs
        urls = [d.get("url") for d in docs if d.get("url")]
        if not urls:
            return docs
        full_pages = await self._page_fetcher.fetch_batch(urls)
        url_to_text = {item["url"]: item["text"] for item in full_pages}
        for d in docs:
            if d.get("url") in url_to_text:
                d["text"] = url_to_text[d["url"]]
        return docs

    # ------------------------------------------------------------------
    # Brave implementation
    # ------------------------------------------------------------------
    def _retrieve_brave(self, query: str, limit: int) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
            params = {"q": query, "count": limit, "result_filter": "web"}
            with httpx.Client(timeout=10) as client:
                resp = client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
            web_results = data.get("web", {}).get("results", [])
            for rank, item in enumerate(web_results[:limit]):
                snippet = (
                    (item.get("title", "") + "\n" + item.get("description", "")).strip()
                )
                if not snippet:
                    continue
                score = 1.0 / (rank + 1)
                docs.append({
                    "id": item.get("url", f"web:{rank}"),
                    "text": snippet,
                    "score": score,
                    "source": "web",
                    "url": item.get("url", ""),
                })
        except Exception:  # pylint: disable=broad-except
            return self._retrieve_duckduckgo(query, limit)
        return docs

    # ------------------------------------------------------------------
    # DuckDuckGo implementation
    # ------------------------------------------------------------------
    def _retrieve_duckduckgo(self, query: str, limit: int) -> List[Dict[str, Any]]:
        if 'DDGS' not in globals():
            return []
        docs: List[Dict[str, Any]] = []
        with DDGS() as ddgs:  # type: ignore
            for rank, result in enumerate(ddgs.text(query, max_results=limit)):
                snippet = (result.get("title", "") + "\n" + result.get("body", "")).strip()
                if not snippet:
                    continue
                score = 1.0 / (rank + 1)
                docs.append({
                    "id": result.get("href", f"web:{rank}"),
                    "text": snippet,
                    "score": score,
                    "source": "web",
                    "url": result.get("href", ""),
                })

        # After building initial docs, optionally fetch full pages -- handled in async path only
        return docs[:limit]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _ensure_cache_collection(self) -> None:
        """Create vector cache collection in Qdrant if it doesn't exist."""
        try:
            if not self.client.collection_exists(collection_name=self.cache_collection):
                self.client.create_collection(
                    collection_name=self.cache_collection,
                    vectors_config={"size": self.embed_model.dimension, "distance": "Cosine"},
                )
        except Exception:  # pragma: no cover
            # Swallow errors – cache disabled
            self.client = None

    def _has_cache_vectors(self) -> bool:
        try:
            meta = self.client.get_collection(self.cache_collection)
            return (meta.points_count or 0) > 0
        except Exception:
            return False

    def _retrieve_from_qdrant(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Dense vector search in web_cache collection."""
        try:
            query_vec = self.embed_model.embed([query])[0]
            res = self.client.search(
                collection_name=self.cache_collection,
                query_vector=query_vec,
                limit=limit,
                with_payload=True,
            )
            return [
                {
                    "id": p.id,
                    "text": p.payload.get("text", "") if p.payload else "",
                    "score": p.score,
                    "source": "web",
                    "url": p.payload.get("url", "") if p.payload else "",
                }
                for p in res
            ]
        except Exception:
            return []

    def _upsert_docs(self, docs: List[Dict[str, Any]]) -> None:
        """Embed and upsert documents into Qdrant web_cache collection."""
        points = []
        for doc in docs:
            text = doc.get("text", "")
            if not text:
                continue
            vector = self.embed_model.embed([text])[0]
            points.append({"id": doc.get("id"), "vector": vector, "payload": doc})

        if not points:
            return
        try:
            self.client.upsert(collection_name=self.cache_collection, points=points)
        except Exception:  # pragma: no cover
            pass 