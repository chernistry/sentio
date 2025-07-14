"""
Sparse retrievers implementation using BM25 and other sparse methods.

This module provides efficient sparse retrieval algorithms including BM25,
which generally outperforms classic TF-IDF due to better term frequency
normalization and document length compensation.
"""

from __future__ import annotations

import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import pickle
import time

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus

# Attempt to import Pyserini only if installed to avoid hard dependency
try:
    # Pyserini 0.21+: SimpleSearcher moved under pyserini.search
    from pyserini.search import SimpleSearcher  # type: ignore
    _HAS_PYSERINI = True
except ImportError:  # pragma: no cover – optional dependency
    _HAS_PYSERINI = False


logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Simple document container with ID and text."""
    id: str
    text: str
    metadata: Optional[Dict] = None


class BM25Retriever:
    """BM25 sparse retriever with persistence and optimization.
    
    Implements the BM25 algorithm for lexical search over a document collection.
    BM25 is a modern TF-IDF variant that accounts for document length and
    term frequency saturation, generally outperforming raw TF-IDF.
    """
    
    def __init__(self, documents: Optional[List[Document]] = None):
        """Initialize BM25 retriever with optional documents.
        
        Args:
            documents: Optional list of Document objects to index.
        """
        self.bm25: Optional[Union[BM25Okapi, BM25Plus]] = None
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.variant = os.environ.get("BM25_VARIANT", "okapi").lower()
        self.cache_dir = os.environ.get("SPARSE_CACHE_DIR", ".sparse_cache")
        
        # Initialize with documents if provided
        if documents:
            self.index(documents)
    
    def index(self, documents: List[Document]) -> None:
        """Index documents using BM25.
        
        Args:
            documents: List of Document objects to index.
        """
        if not documents:
            logger.warning("Empty document list provided for BM25 indexing")
            return
            
        start_time = time.time()
        logger.info(f"Starting BM25 indexing for {len(documents)} documents")
        
        # Store document IDs and tokenize corpus
        self.doc_ids = [doc.id for doc in documents]
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
        logger.info(f"BM25 indexing completed in {elapsed:.2f} seconds")
    
    def save(self, filepath: str = None) -> None:
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
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'doc_ids': self.doc_ids,
                    'variant': self.variant,
                }, f)
            logger.info(f"BM25 index saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
    
    def load(self, filepath: str = None) -> bool:
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
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data['bm25']
                    self.doc_ids = data['doc_ids']
                    self.variant = data.get('variant', 'okapi')
                logger.info(f"BM25 index loaded from {filepath}")
                return True
            else:
                logger.warning(f"BM25 index file not found: {filepath}")
                return False
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k documents for query using BM25.
        
        Args:
            query: Query string to search for.
            top_k: Number of top results to return.
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance.
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
            results = [(self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
            
            return results
        except Exception as e:
            logger.error(f"BM25 retrieval error: {e}")
            return []
            

class SPLADERetriever:
    """SPLADE sparse retrieval using learned sparse representations.
    
    Note: This is a placeholder for future implementation. SPLADE requires
    installing additional dependencies and trained models.
    """
    
    def __init__(self):
        """Initialize SPLADE retriever."""
        logger.warning("SPLADERetriever is not yet implemented")
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Placeholder for SPLADE retrieval."""
        logger.warning("SPLADE retrieval not implemented")
        return [] 


class PyseriniBM25Retriever:
    """BM25 retriever backed by a Lucene index created with Pyserini.

    This retriever is significantly more scalable than the in-memory
    ``rank_bm25`` implementation because it stores postings on disk and can
    easily handle millions of documents. The index path must be provided via
    ``index_dir`` or the *BM25_INDEX_DIR* environment variable. If Pyserini is
    not installed or the index folder does not exist, initialise will raise
    ``RuntimeError`` so that callers can gracefully fall back to another sparse
    method (e.g., :class:`BM25Retriever`).
    """

    def __init__(self, index_dir: Optional[str] = None, k1: float = 0.9, b: float = 0.4):
        if not _HAS_PYSERINI:
            raise RuntimeError("Pyserini is not installed – pip install pyserini")

        self.index_dir: str = index_dir or os.getenv("BM25_INDEX_DIR", "indexes/lucene-index")
        if not os.path.isdir(self.index_dir):
            raise RuntimeError(f"Pyserini index directory not found: {self.index_dir}")

        # Create searcher and configure BM25 parameters
        self.searcher = SimpleSearcher(self.index_dir)
        self.searcher.set_bm25(k1, b)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-*k* doc ids and BM25 scores for *query*."""
        hits = self.searcher.search(query, top_k)
        return [(hit.docid, float(hit.score)) for hit in hits] 