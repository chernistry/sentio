from __future__ import annotations

"""Sparse retrievers implementation using BM25 and other sparse methods.

This module provides efficient sparse retrieval algorithms including BM25,
which generally outperforms classic TF-IDF due to better term frequency
normalization and document length compensation.
"""

import logging
import os
import pickle
import time

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus

from src.core.models.document import Document
from src.core.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)

# Try to import Pyserini for high-performance BM25
try:
    from pyserini.search import SimpleSearcher  # type: ignore
    _HAS_PYSERINI = True
except ImportError:  # pragma: no cover – optional dependency
    _HAS_PYSERINI = False


class BM25Retriever(BaseRetriever):
    """BM25 sparse retriever with persistence and optimization.
    
    Implements the BM25 algorithm for lexical search over a document collection.
    BM25 is a modern TF-IDF variant that accounts for document length and
    term frequency saturation, generally outperforming raw TF-IDF.
    
    Args:
        documents: Optional list of Document objects to index
        variant: BM25 variant to use ("okapi" or "plus")
        cache_dir: Directory to store/load persisted index
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
        variant: str = "okapi",
        cache_dir: str | None = None,
    ):
        """Initialize BM25 retriever with optional documents.
        
        Args:
            documents: Optional list of Document objects to index.
            variant: BM25 variant to use ("okapi" or "plus")
            cache_dir: Directory to store/load persisted index
        """
        self.bm25: BM25Okapi | BM25Plus | None = None
        self.doc_ids: list[str] = []
        self.doc_map: dict[str, Document] = {}
        self.tokenized_corpus: list[list[str]] = []
        self.variant = os.environ.get("BM25_VARIANT", variant).lower()
        self.cache_dir = cache_dir or os.environ.get("SPARSE_CACHE_DIR", ".sparse_cache")

        # Initialize with documents if provided
        if documents:
            self.index(documents)

    def index(self, documents: list[Document]) -> None:
        """Index documents using BM25.
        
        Args:
            documents: List of Document objects to index.
        """
        if not documents:
            logger.warning("Empty document list provided for BM25 indexing")
            return

        start_time = time.time()
        logger.info("Starting BM25 indexing for %d documents", len(documents))

        # Store document IDs and document map for retrieval
        self.doc_ids = [doc.id for doc in documents]
        self.doc_map = {doc.id: doc for doc in documents}

        # Tokenize corpus
        self.tokenized_corpus = [doc.text.lower().split() for doc in documents]

        # Create BM25 index
        if self.variant == "plus":
            self.bm25 = BM25Plus(self.tokenized_corpus)
            logger.info("Using BM25Plus variant")
        else:
            # Default to Okapi
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info("Using BM25Okapi variant")

        elapsed = time.time() - start_time
        logger.info("BM25 indexing completed in %.2f seconds", elapsed)

    def save(self, filepath: str | None = None) -> None:
        """Save BM25 index to disk.
        
        Args:
            filepath: Path to save the index. If None, uses default location.
        """
        if not self.bm25:
            logger.warning("Cannot save empty BM25 index")
            return

        if not filepath:
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            filepath = os.path.join(self.cache_dir, "bm25_index.pkl")

        try:
            with open(filepath, "wb") as f:
                pickle.dump({
                    "bm25": self.bm25,
                    "doc_ids": self.doc_ids,
                    "doc_map": self.doc_map,
                    "tokenized_corpus": self.tokenized_corpus,
                    "variant": self.variant,
                }, f)
            logger.info("BM25 index saved to %s", filepath)
        except Exception as e:
            logger.error("Failed to save BM25 index: %s", e)

    def load(self, filepath: str | None = None) -> bool:
        """Load BM25 index from disk.
        
        Args:
            filepath: Path to load the index from. If None, uses default location.
            
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not filepath:
            filepath = os.path.join(self.cache_dir, "bm25_index.pkl")

        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.doc_ids = data["doc_ids"]
                    self.doc_map = data.get("doc_map", {})
                    self.tokenized_corpus = data.get("tokenized_corpus", [])
                    self.variant = data.get("variant", "okapi")
                logger.info("BM25 index loaded from %s", filepath)
                return True
            logger.warning("BM25 index file not found: %s", filepath)
            return False
        except Exception as e:
            logger.error("Failed to load BM25 index: %s", e)
            return False

    def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Retrieve top-k documents for query using BM25.
        
        Args:
            query: Query string to search for.
            top_k: Number of top results to return.
            
        Returns:
            List of documents sorted by relevance.
        """
        if not self.bm25:
            logger.warning("BM25 index not initialized")
            return []

        # Tokenize query and get BM25 scores
        query_tokens = query.lower().split()

        try:
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k document indices and scores
            top_indices = np.argsort(-np.array(scores))[:top_k]
            results = []

            for idx in top_indices:
                if idx < len(self.doc_ids) and scores[idx] > 0:
                    doc_id = self.doc_ids[idx]
                    score = float(scores[idx])

                    # Get document from map or create a new one if not found
                    if doc_id in self.doc_map:
                        doc = self.doc_map[doc_id]
                    else:
                        # Reconstruct document from tokenized corpus
                        text = " ".join(self.tokenized_corpus[idx])
                        doc = Document(id=doc_id, text=text)

                    # Add score to metadata
                    doc.metadata["bm25_score"] = score
                    results.append(doc)

            return results
        except Exception as e:
            logger.error("BM25 retrieval error: %s", e)
            return []


class PyseriniBM25Retriever(BaseRetriever):
    """BM25 retriever backed by a Lucene index created with Pyserini.

    This retriever is significantly more scalable than the in-memory
    ``rank_bm25`` implementation because it stores postings on disk and can
    easily handle millions of documents.
    
    Args:
        index_dir: Path to the Lucene index directory
        k1: BM25 k1 parameter (term frequency saturation)
        b: BM25 b parameter (document length normalization)
    """

    def __init__(self, index_dir: str | None = None, k1: float = 0.9, b: float = 0.4):
        """Initialize PyseriniBM25Retriever with index path and parameters.
        
        Args:
            index_dir: Path to the Lucene index directory
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
            
        Raises:
            RuntimeError: If Pyserini is not installed or index directory not found
        """
        if not _HAS_PYSERINI:
            raise RuntimeError("Pyserini is not installed – pip install pyserini")

        self.index_dir: str = index_dir or os.getenv("BM25_INDEX_DIR", "indexes/lucene-index")
        if not os.path.isdir(self.index_dir):
            raise RuntimeError(f"Pyserini index directory not found: {self.index_dir}")

        # Create searcher and configure BM25 parameters
        self.searcher = SimpleSearcher(self.index_dir)
        self.searcher.set_bm25(k1, b)
        logger.info("PyseriniBM25Retriever initialized with index: %s", self.index_dir)

    def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Retrieve top-k documents for query using Pyserini BM25.
        
        Args:
            query: Query string to search for
            top_k: Number of top results to return
            
        Returns:
            List of documents sorted by relevance
        """
        try:
            hits = self.searcher.search(query, top_k)
            results = []

            for hit in hits:
                # Get document content from Pyserini
                raw_doc = self.searcher.doc(hit.docid)
                if raw_doc:
                    # Extract content from Pyserini document
                    text = raw_doc.get("contents", "")

                    # Create document with score in metadata
                    doc = Document(id=hit.docid, text=text)
                    doc.metadata["bm25_score"] = float(hit.score)

                    # Add source if available
                    if "source" in raw_doc:
                        doc.metadata["source"] = raw_doc.get("source")

                    results.append(doc)

            return results
        except Exception as e:
            logger.error("Pyserini retrieval error: %s", e)
            return []
