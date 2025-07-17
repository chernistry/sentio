from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Callable
import os
from collections import defaultdict
from functools import lru_cache

import numpy as np
# NOTE: switched from basic TF-IDF to BM25 (rank_bm25) for stronger lexical matching
# BM25 provides better term-frequency normalisation and generally outperforms raw
# TF-IDF across modern IR benchmarks while remaining lightweight.
# Lightweight BM25 implementation for fallback only
from rank_bm25 import BM25Okapi

# Optional high-performance BM25 via Pyserini
try:
    from ..retrievers.sparse import PyseriniBM25Retriever  # type: ignore
    _HAS_PYSERINI = True
except Exception:  # pragma: no cover – optional dependency
    _HAS_PYSERINI = False

from qdrant_client import QdrantClient
TEXT_VECTOR_NAME: str = os.getenv("QDRANT_VECTOR_NAME", "text-dense")


# ------------------------------------------------------------------
# Plugin interface
# ------------------------------------------------------------------


class HybridRetrieverPlugin:  # noqa: D101 – simple interface class
    """Interface for external retriever plugins compatible with HybridRetriever.

    A plugin must implement a ``retrieve`` method returning a list of ``(doc_id,
    score)`` tuples similar to the internal sparse retriever. The Hybrid
    retriever will fuse these scores using Reciprocal Rank Fusion (RRF).
    """

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:  # noqa: D401
        raise NotImplementedError

from root.src.core.tasks.embeddings import EmbeddingModel


class HybridRetriever:
    """Hybrid dense + sparse retriever with Reciprocal Rank Fusion.

    Dense retrieval is performed via Qdrant vector search, while sparse retrieval
    uses an *in-memory* TF-IDF index built on the same documents. Final results
    are merged with Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        client: QdrantClient,
        embed_model: EmbeddingModel,
        collection_name: str = "Sentio_docs",
        rrf_k: int = 60,
        tfidf_max_features: int = 50000,
        retriever_plugins: Optional[List["HybridRetrieverPlugin"]] = None,
    ) -> None:
        self.client = client
        self.embed_model = embed_model
        self.collection_name = collection_name
        self.rrf_k = rrf_k

        # External retriever plugins (e.g., SPLADE, ColBERT)
        self._plugins: List[HybridRetrieverPlugin] = retriever_plugins or []

        # ------------------------------------------------------------------
        # Built-in dense retriever (Qdrant) – always enabled
        # ------------------------------------------------------------------
        self.cache_collection_name = "web_cache"
        self._has_cache_collection = False
        # Detect if cache collection exists and has points
        try:
            if self.client.collection_exists(collection_name=self.cache_collection_name):
                meta = self.client.get_collection(self.cache_collection_name)
                self._has_cache_collection = (meta.points_count or 0) > 0
        except Exception:  # pragma: no cover
            # Any failure disables cache retrieval gracefully
            self._has_cache_collection = False

        # Load documents from Qdrant to build sparse index
        self._doc_ids: List[str] = []
        self._docs: List[str] = []
        self._load_documents()

        # ----------------- Sparse index (BM25) -----------------
        # 1) Try disk-backed Pyserini (preferred for large corpora)
        # 2) Fallback to in-memory rank_bm25
        self._pyserini: Optional[PyseriniBM25Retriever] = None
        self._bm25: Optional[BM25Okapi] = None

        if _HAS_PYSERINI and os.getenv("BM25_USE_PYSERINI", "0") == "1":
            try:
                self._pyserini = PyseriniBM25Retriever()
            except Exception:  # pragma: no cover – silently fall back
                self._pyserini = None

        # Build fallback in-memory BM25 only if Pyserini not available
        if self._pyserini is None and self._docs:
            tokenised_corpus = [doc.split() for doc in self._docs]
            if tokenised_corpus:  # safeguard against entirely blank docs
                self._bm25 = BM25Okapi(tokenised_corpus)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return *top_k* documents using hybrid retrieval.

        The returned list is sorted by the fused RRF score in descending order
        and each item has the following shape::

            {
                "id": <qdrant_point_id>,
                "text": <document text>,
                "score": <rrf_score>,
            }
        """

        # Dense search for original query (mandatory)
        dense_hits = self._dense_search(query, limit=top_k)

        # Sparse lexical search (BM25) on original query
        sparse_hits = self._sparse_search(query, limit=top_k)

        # Optional: attempt retrieval from cached web collection first
        dense_cache_hits: List[Tuple[str, dict, float]] = []
        if self._has_cache_collection:
            dense_cache_hits = self._dense_search(
                query,
                limit=top_k,
                collection_name=self.cache_collection_name,
            )

        # ------------- Plugin retrievers (optional) -------------
        plugin_scores: List[Tuple[str, float]] = []
        for plugin in self._plugins:
            try:
                plugin_results = plugin.retrieve(query, top_k)
                plugin_scores.extend(plugin_results)
            except Exception:  # pragma: no cover – keep pipeline robust
                continue

        # Combine dense results prioritizing cache hits
        dense_hits_combined = dense_cache_hits + dense_hits

        # RRF fusion
        scores: defaultdict[str, float] = defaultdict(float)
        for rank, (pid, _payload, _score) in enumerate(dense_hits_combined):
            scores[pid] += 1 / (self.rrf_k + rank)
        for rank, (pid, _score) in enumerate(sparse_hits):
            scores[pid] += 1 / (self.rrf_k + rank)

        # Include plugin-provided hits in fusion
        for rank, (pid, _score) in enumerate(plugin_scores):
            scores[pid] += 1 / (self.rrf_k + rank)

        # Sort by fused score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build output with texts
        id_to_text = {pid: self._id_to_text(pid) for pid, _ in sorted_items}
        return [
            {"id": pid, "text": id_to_text.get(pid, ""), "score": score}
            for pid, score in sorted_items
        ]

    async def retrieve_async(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:  # noqa: D401
        """Асинхронная обертка над retrieve() для безопасного вызова в event loop."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.retrieve(query, top_k))

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _dense_search(self, query: str, limit: int, collection_name: Optional[str] = None) -> List[Tuple[str, dict, float]]:
        """Dense vector similarity search in specified Qdrant collection."""
        target_collection = collection_name or self.collection_name
        # Embed the query
        query_vec = self.embed_model.embed([query])[0]
        results = self.client.search(
            collection_name=target_collection,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
            vector_name=TEXT_VECTOR_NAME,
        )
        return [(p.id, p.payload, p.score) for p in results]

    def _sparse_search(self, query: str, limit: int) -> List[Tuple[str, float]]:

        # 1) Preferred: Pyserini (disk-backed, scalable)
        if self._pyserini is not None:
            try:
                return self._pyserini.retrieve(query, top_k=limit)
            except Exception:  # pragma: no cover
                # Soft-fail – fall through to next option
                pass

        # 2) Fallback: in-memory rank_bm25
        if self._bm25:
            scores = self._bm25.get_scores(query.split())
            # Higher BM25 score indicates greater relevance
            top_idx = np.argsort(-np.array(scores))[:limit]
            return [(self._doc_ids[i], float(scores[i])) for i in top_idx]

        return []

    def _load_documents(self) -> None:
        """Load *all* documents from Qdrant (payload text) for sparse index."""
        next_offset: Optional[int] = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_offset,
            )
            if not points:
                break
            for p in points:
                text = ""
                if p.payload is not None:
                    # Common LlamaIndex payload key
                    text = (
                        p.payload.get("text")
                        or p.payload.get("document")
                        or p.payload.get("content")
                        or ""
                    )
                self._doc_ids.append(p.id)
                self._docs.append(text)
            if next_offset is None:
                break

    @lru_cache(maxsize=1024)
    def _id_to_text(self, pid: str) -> str:
        """Fetch document text for given point id (cached)."""
        res = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[pid],
            with_payload=True,
            with_vectors=False,
        )
        if res and res[0].payload:
            return res[0].payload.get("text", "")
        return "" 