from __future__ import annotations

"""Hybrid dense + sparse retrieval with Reciprocal Rank Fusion (RRF)."""

import logging
import os
from collections import defaultdict

from src.core.models.document import Document

from .base import BaseRetriever, ScorerPlugin
from .dense import DenseRetriever
from .sparse import BM25Retriever, PyseriniBM25Retriever

logger = logging.getLogger(__name__)

# Try to import Pyserini for high-performance BM25
try:
    from pyserini.search import SimpleSearcher  # type: ignore
    _HAS_PYSERINI = True
except (ImportError, RuntimeError) as e:  # pragma: no cover – optional dependency
    logger.warning(f"Pyserini not available (Java may not be installed): {e}")
    _HAS_PYSERINI = False
    SimpleSearcher = None  # type: ignore


class HybridRetrieverPlugin:
    """Interface for external retriever plugins compatible with HybridRetriever.

    A plugin must implement a ``retrieve`` method returning a list of ``(doc_id,
    score)`` tuples. The Hybrid retriever will fuse these scores using 
    Reciprocal Rank Fusion (RRF).
    """

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Retrieve documents for the query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        raise NotImplementedError


class HybridRetriever(BaseRetriever):
    """Combine dense retrieval with lexical BM25 using RRF fusion.
    
    This retriever combines vector similarity search with lexical BM25 search,
    merging results using Reciprocal Rank Fusion (RRF). Additional scoring 
    plugins can be provided to further enhance retrieval quality.
    
    Args:
        dense_retriever: The DenseRetriever instance for vector search
        corpus_docs: Optional list of documents to build BM25 index from
        rrf_k: RRF constant (higher values reduce importance of rank position)
        scorer_plugins: Optional list of scoring plugins for additional ranking signals
        retriever_plugins: Optional list of additional retriever plugins
        use_pyserini: Whether to use Pyserini for BM25 if available
        sparse_retriever: Optional pre-configured sparse retriever to use
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BaseRetriever | None = None,
        corpus_docs: list[Document] | None = None,
        rrf_k: int = 60,
        scorer_plugins: list[ScorerPlugin] | None = None,
        retriever_plugins: list[HybridRetrieverPlugin] | None = None,
        use_pyserini: bool = False,
        fusion_method: str = "rrf",  # Support different fusion methods
        dense_weight: float = 0.5,  # Weight for dense retriever
        sparse_weight: float = 0.5,  # Weight for sparse retriever
    ) -> None:
        self._dense = dense_retriever
        self._rrf_k = rrf_k
        self._scorer_plugins = scorer_plugins or []
        self._retriever_plugins = retriever_plugins or []
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Cache collection support (for web results)
        self._has_cache_collection = False
        self._cache_collection_name = os.getenv("CACHE_COLLECTION_NAME", "web_cache")

        # Try to detect if cache collection exists
        try:
            client = getattr(self._dense, "_client", None)
            if client and hasattr(client, "collection_exists"):
                if client.collection_exists(collection_name=self._cache_collection_name):
                    self._has_cache_collection = True
                    logger.info("Cache collection detected: %s", self._cache_collection_name)
        except Exception as e:
            logger.warning("Failed to check cache collection: %s", e)

        # Initialize sparse retriever
        self._sparse_retriever = sparse_retriever

        # If no sparse retriever provided, create one based on configuration
        if self._sparse_retriever is None and corpus_docs:
            # Try to use Pyserini if requested and available
            if use_pyserini and _HAS_PYSERINI and os.getenv("BM25_INDEX_DIR"):
                try:
                    self._sparse_retriever = PyseriniBM25Retriever()
                    logger.info("Using Pyserini BM25 retriever")
                except Exception as e:
                    logger.warning("Failed to initialize Pyserini: %s", e)
                    self._sparse_retriever = None

            # Fall back to in-memory BM25 if Pyserini not available
            if self._sparse_retriever is None:
                self._sparse_retriever = BM25Retriever(documents=corpus_docs)
                logger.info("Using in-memory BM25 retriever")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Retrieve documents using hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Query string to search for
            top_k: Number of top results to return
            
        Returns:
            List of documents sorted by relevance
        """
        # Retrieve documents using dense search
        logger.debug("HybridRetriever: performing dense search for '%s'", query)
        dense_hits = self._dense.retrieve(query, top_k=top_k)

        # Optional: attempt retrieval from cached web collection first
        dense_cache_hits: list[Document] = []
        if self._has_cache_collection:
            try:
                # Use the same dense retriever but with cache collection
                client = getattr(self._dense, "_client", None)
                embedder = getattr(self._dense, "_embedder", None)
                vector_name = getattr(self._dense, "_vector_name", None)

                if client and embedder:
                    import inspect

                    from src.core.retrievers.dense import DenseRetriever

                    # Check if DenseRetriever accepts vector_name parameter
                    dense_retriever_sig = inspect.signature(DenseRetriever.__init__)

                    # Create cache retriever with appropriate parameters
                    if "vector_name" in dense_retriever_sig.parameters and vector_name:
                        # With vector_name parameter
                        cache_retriever = DenseRetriever(
                            client=client,
                            embedder=embedder,
                            collection_name=self._cache_collection_name,
                            vector_name=vector_name,
                        )
                    else:
                        # Without vector_name parameter
                        cache_retriever = DenseRetriever(
                            client=client,
                            embedder=embedder,
                            collection_name=self._cache_collection_name,
                        )

                    dense_cache_hits = cache_retriever.retrieve(query, top_k=top_k)
                    logger.debug("Retrieved %d hits from cache collection", len(dense_cache_hits))
            except Exception as e:
                logger.warning("Failed to retrieve from cache collection: %s", e)

        # Perform sparse search if retriever is available
        sparse_hits: list[tuple[str, float]] = []
        sparse_docs: list[Document] = []
        if self._sparse_retriever:
            logger.debug("HybridRetriever: performing sparse BM25 search")
            sparse_docs = self._sparse_retriever.retrieve(query, top_k=top_k)
            # Extract ID and score tuples for RRF fusion
            sparse_hits = [(doc.id, doc.metadata.get("bm25_score", 0.0)) for doc in sparse_docs]
            logger.debug("HybridRetriever: found %d sparse hits", len(sparse_hits))

        # Get results from retriever plugins
        plugin_hits: list[tuple[str, float]] = []
        for plugin in self._retriever_plugins:
            try:
                plugin_results = plugin.retrieve(query, top_k)
                plugin_hits.extend(plugin_results)
                logger.debug("Retrieved %d hits from plugin retriever", len(plugin_results))
            except Exception as e:
                logger.warning("Retriever plugin failed: %s", e)

        # Initialize fusion scores dictionary
        fused_scores = defaultdict(float)

        # Combine dense results prioritizing cache hits
        all_dense_hits = dense_cache_hits + dense_hits

        # Add dense ranking signals
        for rank, doc in enumerate(all_dense_hits):
            fused_scores[doc.id] += 1 / (self._rrf_k + rank)
            # Store dense score in metadata
            doc.metadata["dense_score"] = doc.metadata.get("score", 0.0)

        # Add sparse ranking signals
        for rank, (doc_id, score) in enumerate(sparse_hits):
            fused_scores[doc_id] += 1 / (self._rrf_k + rank)

        # Add plugin retriever signals
        for rank, (doc_id, score) in enumerate(plugin_hits):
            fused_scores[doc_id] += 1 / (self._rrf_k + rank)

        # Build complete document map from all sources
        id_to_doc: dict[str, Document] = {}

        # Add dense and cache documents to map
        for doc in all_dense_hits:
            id_to_doc[doc.id] = doc

        # Add sparse documents to map if not already present
        for doc in sparse_docs:
            if doc.id not in id_to_doc:
                id_to_doc[doc.id] = doc

        # Apply additional scoring plugins if available
        merged_docs = list(id_to_doc.values())
        for plugin_idx, scorer in enumerate(self._scorer_plugins):
            try:
                plugin_scores = scorer.score(query, merged_docs)
                # Update fused scores with normalized plugin scores
                for idx, (doc, score) in enumerate(zip(merged_docs, plugin_scores, strict=False)):
                    # Store plugin score in metadata
                    doc.metadata[f"plugin_{plugin_idx}_score"] = float(score)
                    # Add to fusion score
                    fused_scores[doc.id] += float(score)
            except Exception as e:
                logger.warning("Scorer plugin %d failed: %s", plugin_idx, e)

        # Sort by fused score
        ranked_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build final result list
        results = []
        for doc_id, score in ranked_items:
            if doc_id in id_to_doc:
                doc = id_to_doc[doc_id]
                # Store fused score under generic key expected by evaluators
                doc.metadata["hybrid_score"] = float(score)
                doc.metadata["score"] = float(score)
                results.append(doc)

        return results

    # ------------------------------------------------------------------
    # Plugin management
    # ------------------------------------------------------------------
    def add_scorer_plugin(self, scorer: ScorerPlugin) -> None:
        """Add a scoring plugin to the retriever.
        
        Args:
            scorer: A ScorerPlugin instance
        """
        if scorer not in self._scorer_plugins:
            self._scorer_plugins.append(scorer)
            logger.info("Added scorer plugin: %s", type(scorer).__name__)

    def add_retriever_plugin(self, retriever: HybridRetrieverPlugin) -> None:
        """Add a retriever plugin to the hybrid retriever.
        
        Args:
            retriever: A HybridRetrieverPlugin instance
        """
        if retriever not in self._retriever_plugins:
            self._retriever_plugins.append(retriever)
            logger.info("Added retriever plugin: %s", type(retriever).__name__)
