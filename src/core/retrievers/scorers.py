from __future__ import annotations

"""Scoring plugins for document retrieval.

This module provides various scorer plugins that can be used with retrievers
to enhance document ranking based on different signals.
"""

import logging
import re

from src.core.models.document import Document
from src.core.retrievers.base import ScorerPlugin

__all__ = [
    "KeywordMatchScorer",
    "MMRScorer",
    "RecencyScorer",
    "SemanticSimilarityScorer",
]

logger = logging.getLogger(__name__)


class KeywordMatchScorer(ScorerPlugin):
    """Score documents based on keyword match counts.
    
    This scorer counts exact matches of keywords in the document text
    and produces a normalized score based on match density.
    
    Args:
        weight: Weight of the keyword score in the overall ranking
        case_sensitive: Whether to perform case-sensitive matching
    """

    def __init__(self, weight: float = 0.5, case_sensitive: bool = False):
        self.weight = weight
        self.case_sensitive = case_sensitive

    def score(self, query: str, docs: list[Document]) -> list[float]:
        """Score documents by counting keyword matches.
        
        Args:
            query: The query string to extract keywords from
            docs: List of documents to score
            
        Returns:
            List of scores corresponding to each document
        """
        # Extract keywords (simple approach - split on whitespace and punctuation)
        keywords = set(re.findall(r"\w+", query.lower()))
        if not keywords:
            return [0.0] * len(docs)

        # Prepare document texts based on case sensitivity
        doc_texts = []
        for doc in docs:
            if self.case_sensitive:
                doc_texts.append(doc.text)
            else:
                doc_texts.append(doc.text.lower())

        # Count keyword matches in each document
        scores = []
        for text in doc_texts:
            text_words = set(re.findall(r"\w+", text))
            matches = len(keywords.intersection(text_words))
            # Normalize by keyword count and apply weight
            score = (matches / len(keywords)) * self.weight
            scores.append(score)

        return scores


class RecencyScorer(ScorerPlugin):
    """Score documents based on recency information in metadata.
    
    This scorer assumes documents have timestamp information in metadata
    and assigns higher scores to more recent documents.
    
    Args:
        timestamp_field: Name of the metadata field containing timestamp
        weight: Weight of recency score in the overall ranking
        max_age_seconds: Maximum document age in seconds for normalization
    """

    def __init__(
        self,
        timestamp_field: str = "timestamp",
        weight: float = 0.3,
        max_age_seconds: int = 86400 * 30  # 30 days
    ):
        self.timestamp_field = timestamp_field
        self.weight = weight
        self.max_age_seconds = max_age_seconds

    def score(self, query: str, docs: list[Document]) -> list[float]:
        """Score documents based on recency.
        
        Args:
            query: The query string (not used in this scorer)
            docs: List of documents to score
            
        Returns:
            List of scores corresponding to each document
        """
        import time

        current_time = time.time()
        scores = []

        for doc in docs:
            # Default score is 0
            score = 0.0

            # Extract timestamp from metadata if available
            timestamp = doc.metadata.get(self.timestamp_field)
            if timestamp and isinstance(timestamp, (int, float)):
                age_seconds = current_time - timestamp
                if age_seconds >= 0:
                    # Newer documents get higher scores
                    # Normalize to 0-1 range based on max_age
                    normalized_age = min(age_seconds, self.max_age_seconds) / self.max_age_seconds
                    score = (1 - normalized_age) * self.weight

            scores.append(score)

        return scores


class SemanticSimilarityScorer(ScorerPlugin):
    """Score documents based on semantic similarity to query.
    
    This scorer uses a text embedding model to compute semantic similarity
    between the query and documents.
    
    Args:
        embedder: The embedding model to use
        weight: Weight of semantic score in the overall ranking
    """

    def __init__(self, embedder, weight: float = 0.7):
        """Initialize the semantic scorer.
        
        Args:
            embedder: Any embedding model with an embed_sync method
            weight: Weight of semantic score in the overall ranking
        """
        self.embedder = embedder
        self.weight = weight

    def score(self, query: str, docs: list[Document]) -> list[float]:
        """Score documents based on semantic similarity.
        
        Args:
            query: The query to compare documents against
            docs: List of documents to score
            
        Returns:
            List of scores corresponding to each document
        """
        try:
            import numpy as np

            # Embed query
            query_embedding = self.embedder.embed_sync(query)

            # Embed document texts
            doc_texts = [doc.text for doc in docs]
            if not doc_texts:
                return []

            doc_embeddings = self.embedder.embed_many_sync(doc_texts)

            # Calculate cosine similarities
            query_norm = np.linalg.norm(query_embedding)
            scores = []

            for doc_emb in doc_embeddings:
                doc_norm = np.linalg.norm(doc_emb)
                if query_norm > 0 and doc_norm > 0:
                    similarity = np.dot(query_embedding, doc_emb) / (query_norm * doc_norm)
                    scores.append(float(similarity * self.weight))
                else:
                    scores.append(0.0)

            return scores

        except Exception as e:
            logger.warning("Error in semantic scoring: %s", e)
            return [0.0] * len(docs)


class MMRScorer(ScorerPlugin):
    """Diversify results using Maximal Marginal Relevance (MMR).

    This scorer rewards documents that are both relevant to the query *and*
    novel with respect to the already-ranked documents.  It computes a
    combined score for each document:

        ``score = λ · relevance − (1 − λ) · redundancy``

    where *relevance* is the cosine similarity between the document and the
    query, and *redundancy* is the **maximum** cosine similarity between the
    document and any higher-ranked document.  The scorer returns a list of
    MMR scores that can be fused with existing signals.

    Args:
        embedder: Embedding backend with ``embed_sync``/``embed_batch_sync``.
        lambda_: Trade-off between relevance and diversity.  Should be in the
                 range ⟦0,1⟧.  A higher value favours relevance.
        weight:   Final weight applied when adding to the fusion score.
    """

    def __init__(self, embedder, lambda_: float = 0.7, weight: float = 0.5):
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda_ must be between 0 and 1 inclusive")
        self.embedder = embedder
        self.lambda_ = lambda_
        self.weight = weight

    def score(self, query: str, docs: list[Document]) -> list[float]:
        import numpy as np

        if not docs:
            return []

        try:
            # Embed query and documents
            query_emb = self.embedder.embed_sync(query)
            doc_embs = self.embedder.embed_many_sync([d.text for d in docs])

            # Pre-compute cosine similarities to the query
            def _cos(a: np.ndarray, b: np.ndarray) -> float:
                denom = (np.linalg.norm(a) * np.linalg.norm(b))
                return float(np.dot(a, b) / denom) if denom else 0.0

            rel_scores = [_cos(query_emb, d_emb) for d_emb in doc_embs]

            mmr_scores: list[float] = [0.0] * len(docs)
            selected: list[int] = []

            # Greedy MMR selection to estimate redundancy for each doc
            for _ in range(len(docs)):
                best_idx = None
                best_score = -1.0
                for idx, rel in enumerate(rel_scores):
                    if idx in selected:
                        continue
                    # Redundancy: max similarity to any already selected doc
                    redundancy = 0.0
                    for s_idx in selected:
                        redundancy = max(redundancy, _cos(doc_embs[idx], doc_embs[s_idx]))
                    score = self.lambda_ * rel - (1 - self.lambda_) * redundancy
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                mmr_scores[best_idx] = best_score * self.weight

            # For documents never selected, fall back to scaled relevance only
            for idx in range(len(docs)):
                if mmr_scores[idx] == 0.0:
                    mmr_scores[idx] = rel_scores[idx] * self.weight * self.lambda_

            # Ensure scores are non-negative after weighting
            mmr_scores = [max(0.0, s) for s in mmr_scores]
            return mmr_scores
        except Exception as exc:  # pragma: no cover – defensive
            logger.warning("MMR scorer failed: %s", exc)
            return [0.0] * len(docs)
