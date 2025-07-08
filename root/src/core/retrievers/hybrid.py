from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient

from ..embeddings import EmbeddingModel


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
        collection_name: str = "Sentio_docs_v2",
        rrf_k: int = 60,
        tfidf_max_features: int = 50000,
    ) -> None:
        self.client = client
        self.embed_model = embed_model
        self.collection_name = collection_name
        self.rrf_k = rrf_k

        # Load documents from Qdrant to build sparse index
        self._doc_ids: List[str] = []
        self._docs: List[str] = []
        self._load_documents()

        # Fit TF-IDF Vectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            stop_words="english",
        )
        if self._docs:
            self._tfidf_matrix = self._vectorizer.fit_transform(self._docs)
        else:
            # Empty collection; create dummy matrix to avoid errors
            self._tfidf_matrix = None

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

        dense_hits = self._dense_search(query, limit=top_k)
        sparse_hits = self._sparse_search(query, limit=top_k)

        # Optional: attempt retrieval from cached web collection first
        dense_cache_hits: List[Tuple[str, dict, float]] = []
        if self._has_cache_collection:
            dense_cache_hits = self._dense_search(query, limit=top_k, collection_name=self.cache_collection_name)

        # Combine dense results prioritizing cache hits
        dense_hits_combined = dense_cache_hits + dense_hits

        # RRF fusion
        scores: defaultdict[str, float] = defaultdict(float)
        for rank, (pid, _payload, _score) in enumerate(dense_hits_combined):
            scores[pid] += 1 / (self.rrf_k + rank)
        for rank, (pid, _score) in enumerate(sparse_hits):
            scores[pid] += 1 / (self.rrf_k + rank)

        # Sort by fused score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build output with texts
        id_to_text = {pid: self._id_to_text(pid) for pid, _ in sorted_items}
        return [
            {"id": pid, "text": id_to_text.get(pid, ""), "score": score}
            for pid, score in sorted_items
        ]

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
        )
        return [(p.id, p.payload, p.score) for p in results]

    def _sparse_search(self, query: str, limit: int) -> List[Tuple[str, float]]:
        if self._tfidf_matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        # Cosine similarity for sparse vectors
        similarities = (self._tfidf_matrix @ query_vec.T).toarray().ravel()
        top_idx = np.argsort(-similarities)[:limit]
        return [(self._doc_ids[i], float(similarities[i])) for i in top_idx]

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