from __future__ import annotations

"""Factory for creating retrievers based on environment configuration."""

import logging
import os

from qdrant_client import QdrantClient

from src.core.embeddings.base import BaseEmbedder
from src.core.models.document import Document
from src.core.retrievers.base import BaseRetriever, ScorerPlugin
from src.core.retrievers.dense import DenseRetriever
from src.core.retrievers.hybrid import HybridRetriever
from src.core.retrievers.sparse import BM25Retriever, PyseriniBM25Retriever
from src.core.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def create_retriever_from_env(
    client: QdrantClient | QdrantStore,
    embedder: BaseEmbedder,
    collection_name: str | None = None,
    corpus_docs: list[Document] | None = None,
    scorer_plugins: list[ScorerPlugin] | None = None,
) -> BaseRetriever:
    """Create a retriever based on environment configuration."""
    # Default to the configuration that yielded the best average reranked score
    # during the latest quality experiments (see `retriever_quality_results.json`).
    #  * dense strategy
    #  * RRF_K=20 (only used by hybrid retriever but kept consistent)
    #  * RETRIEVAL_TOP_K=10
    # Environment variables can still override these values at runtime.
    strategy = os.getenv("RETRIEVAL_STRATEGY", "dense").lower()
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    rrf_k = int(os.getenv("RRF_K", "20"))
    bm25_variant = os.getenv("BM25_VARIANT", "okapi").lower()
    fusion_method = os.getenv("FUSION_METHOD", "rrf").lower()  # rrf | weighted_rrf | comb_sum
    try:
        dense_weight = float(os.getenv("DENSE_WEIGHT", "0.5"))
        sparse_weight = float(os.getenv("SPARSE_WEIGHT", "0.5"))
    except ValueError:
        dense_weight, sparse_weight = 0.5, 0.5

    if collection_name is None:
        collection_name = os.getenv("COLLECTION_NAME", "Sentio_docs")
    logger.info(f"Creating retriever with strategy: {strategy}, collection: {collection_name}")

    vector_name = os.getenv("TEXT_VECTOR_NAME", "text-dense")
    logger.debug(f"Using vector name: {vector_name}")

    # Dense retriever is the base for most strategies
    dense_retriever = DenseRetriever(
        client=client,
        embedder=embedder,
        collection_name=collection_name,
        vector_name=vector_name,
    )

    # ------------------------------------------------------------------
    # Default scoring plugins – enabled unless overridden by caller
    # ------------------------------------------------------------------
    if scorer_plugins is None:
        from src.core.retrievers.scorers import (
            KeywordMatchScorer,
            MMRScorer,
            SemanticSimilarityScorer,
        )

        scorer_plugins = [
            # High-weight semantic similarity provides a second-pass view that
            # leverages larger embedding context than the raw dense score.
            SemanticSimilarityScorer(embedder=embedder, weight=0.8),
            # Lightweight lexical matching helps catch corner cases where the
            # embedder misses exact terminology.
            KeywordMatchScorer(weight=0.2),
            # MMR diversification penalises near-duplicates to improve recall.
            MMRScorer(embedder=embedder, lambda_=0.5, weight=0.5),
        ]

    # Load corpus from Qdrant if not provided, required for sparse/hybrid
    if corpus_docs is None and strategy in ("hybrid", "bm25"):
        logger.info("Loading corpus from Qdrant for sparse index...")
        try:
            # Use the client directly to scroll through all documents
            corpus_docs = []
            next_offset = None

            # Get the actual QdrantClient instance
            qdrant_client = client._client if isinstance(client, QdrantStore) else client

            # Scroll through all documents in the collection
            while True:
                points, next_offset = qdrant_client.scroll(
                    collection_name=collection_name,
                    with_payload=True,
                    with_vectors=False,
                    limit=100,  # Process in batches
                    offset=next_offset,
                )

                if not points:
                    break

                # Convert points to Document objects
                for point in points:
                    if point.payload:
                        # Try common payload keys for text content
                        text = (
                            point.payload.get("text") or
                            point.payload.get("content") or
                            point.payload.get("document") or
                            point.payload.get("page_content") or
                            ""
                        )

                        # Create Document object
                        doc = Document(
                            id=str(point.id),
                            text=text,
                            metadata=point.payload.get("metadata", {})
                        )
                        corpus_docs.append(doc)

                # Exit loop if no more points
                if next_offset is None:
                    break

            logger.info(f"Loaded {len(corpus_docs)} documents from '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to load documents from Qdrant: {e}")
            corpus_docs = []

    if strategy == "dense":
        logger.info("Using dense retrieval strategy")
        return dense_retriever

    if strategy == "bm25":
        logger.info(f"Using BM25 retrieval strategy with variant: {bm25_variant}")
        if not corpus_docs:
            logger.warning("No corpus documents provided for BM25 retrieval.")
            return BM25Retriever(documents=[], variant=bm25_variant)
        return BM25Retriever(
            documents=corpus_docs,
            variant=bm25_variant,
            cache_dir=os.getenv("SPARSE_CACHE_DIR", ".sparse_cache"),
        )

    if strategy == "pyserini":
        logger.info("Using Pyserini BM25 retrieval strategy")
        try:
            return PyseriniBM25Retriever(
                index_dir=os.getenv("BM25_INDEX_DIR", "indexes/lucene-index"),
                k1=float(os.getenv("BM25_K1", "0.9")),
                b=float(os.getenv("BM25_B", "0.4")),
            )
        except RuntimeError as e:
            logger.error(f"Failed to initialize Pyserini, falling back to BM25: {e}")
            if not corpus_docs:
                logger.warning("No corpus documents for BM25 fallback.")
                return BM25Retriever(documents=[], variant=bm25_variant)
            return BM25Retriever(documents=corpus_docs, variant=bm25_variant)

    if strategy == "hybrid":
        logger.info("Using hybrid retrieval strategy")
        sparse_retriever = None
        # Prefer Pyserini if index exists
        index_dir = os.getenv("BM25_INDEX_DIR", "indexes/lucene-index")
        if os.path.isdir(index_dir):
            try:
                sparse_retriever = PyseriniBM25Retriever(index_dir=index_dir)
                logger.info(f"Using Pyserini for sparse retrieval from {index_dir}")
            except RuntimeError as e:
                logger.error(f"Failed to initialize Pyserini: {e}")

        # Fallback to in-memory BM25 if Pyserini is not used
        if sparse_retriever is None:
            if corpus_docs:
                sparse_retriever = BM25Retriever(documents=corpus_docs, variant=bm25_variant)
                logger.info(f"Using in-memory BM25 for sparse retrieval ({len(corpus_docs)} docs)")
            else:
                logger.warning("Hybrid search selected, but no sparse retriever available.")

        return HybridRetriever(
            dense_retriever=dense_retriever,
            corpus_docs=corpus_docs,  # Pass docs for consistency
            rrf_k=rrf_k,
            scorer_plugins=scorer_plugins,
            sparse_retriever=sparse_retriever,
            fusion_method=fusion_method,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    raise ValueError(f"Unknown retrieval strategy: {strategy}")


def create_retriever_for_graph() -> BaseRetriever:
    """Create a retriever for the LangGraph pipeline."""
    from src.core.embeddings import get_embedder
    from src.utils.settings import settings

    embedder = get_embedder(
        model_name=os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3"),
        api_key=os.getenv("EMBEDDING_MODEL_API_KEY", ""),
    )
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    return create_retriever_from_env(
        client=client,
        embedder=embedder,
        collection_name=os.getenv("COLLECTION_NAME", "Sentio_docs"),
    )
