from __future__ import annotations

"""Jina reranker implementation using the Jina AI Reranker API.

This module provides a reranker that uses the Jina AI Reranker API for
document reranking without requiring local models.
"""

import logging
import time
from typing import Any

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)

from src.core.models.document import Document
from src.core.rerankers.base import Reranker
from src.core.resilience.patterns import AsyncCircuitBreaker, AsyncRetry
from src.utils.settings import settings

logger = logging.getLogger(__name__)


class JinaReranker(Reranker):
    """Reranker that uses the Jina AI Reranker API for document reranking.

    This reranker sends documents to the Jina AI Reranker API for scoring and
    reranking. It offers superior performance without requiring local GPU
    resources.
    
    Features comprehensive resilience patterns:
    - Circuit breaker for API failures
    - Exponential backoff retry with tenacity
    - Request timeout handling
    - Fallback mechanisms
    """

    def __init__(
        self,
        model_name: str | None = None,
        model: str | None = None,  # Alternative parameter name for compatibility
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
    ) -> None:
        """Initialize the JinaReranker instance.

        Args:
            model_name: Custom model name. If None, uses environment variable or default.
            model: Alternative parameter name for model_name (for compatibility).
            api_key: Jina API key. If None, uses environment variable.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            circuit_breaker_threshold: Number of failures before circuit breaker opens.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        # Support both model_name and model parameters
        self.model_name: str = model_name or model or settings.reranker_model
        logger.info("Using Jina reranker model: %s", self.model_name)

        raw_key: str = api_key or settings.embedding_model_api_key
        self.api_key: str = raw_key.strip()

        if not self.api_key:
            raise ValueError(
                "No embedding API key provided. Set EMBEDDING_MODEL_API_KEY or JINA_API_KEY "
                "environment variable or pass the api_key parameter."
            )

        self.headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Set request timeout to prevent hanging
        self.timeout: int = timeout or settings.reranker_timeout
        self.rerank_url: str = settings.reranker_url
        self.max_retries = max_retries

        # Initialize resilience patterns
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            timeout_seconds=60.0,
            recovery_timeout=30.0,
        )
        
        self.retry_handler = AsyncRetry(
            max_attempts=max_retries,
            base_delay=1.0,
            backoff_multiplier=2.0,
            max_wait_time=30.0,
            jitter=True,
        )

        logger.debug("Jina reranker initialized with resilience patterns")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            requests.RequestException,
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )
    def _make_api_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make API request with tenacity retry logic.
        
        Args:
            payload: Request payload for Jina API
            
        Returns:
            API response data
            
        Raises:
            requests.RequestException: If all retries fail
        """
        logger.debug(
            "JinaReranker: POST %s | model=%s | top_n=%d",
            self.rerank_url,
            payload.get("model"),
            payload.get("top_n", 0),
        )

        response = requests.post(
            self.rerank_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:
            logger.error(
                "Jina API error: %d - %s",
                response.status_code,
                response.text,
            )
            response.raise_for_status()

        return response.json()

    async def _rerank_with_resilience(
        self,
        query: str,
        doc_texts: list[str],
        top_k: int,
    ) -> dict[str, Any]:
        """Rerank documents with full resilience patterns.
        
        Args:
            query: Search query
            doc_texts: List of document texts
            top_k: Number of top results to return
            
        Returns:
            Reranking results from API
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": doc_texts,
            "top_n": min(len(doc_texts), top_k * 2),  # Request more results for robustness
        }

        async def _api_call():
            """Wrapper for API call to work with circuit breaker."""
            return self._make_api_request(payload)

        # Apply circuit breaker and retry patterns
        try:
            result = await self.circuit_breaker.call(_api_call)
            return result
        except Exception as e:
            logger.error(f"Reranking failed after all resilience attempts: {e}")
            # Circuit breaker will handle state management
            raise

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[Document]:
        """Rerank documents based on their relevance to the query using Jina API.

        Args:
            query: The search query.
            docs: List of documents to rerank.
            top_k: Number of top documents to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of documents sorted by relevance.
        """
        if not docs:
            logger.debug("No documents to rerank")
            return []

        if not query or query.strip() == "":
            logger.warning("Empty query received in reranker, using default ranking")
            return self._default_ranking(docs, top_k)

        logger.info("Reranking %d documents with Jina API (resilient)", len(docs))
        start_time = time.time()

        try:
            # Extract document texts with fallback to metadata.content
            doc_texts = []
            for i, doc in enumerate(docs):
                text = doc.text
                if not text and doc.metadata and 'content' in doc.metadata:
                    text = doc.metadata['content']
                    logger.info(f"JinaReranker - Doc {i}: Using fallback content from metadata")
                
                doc_texts.append(text)
                logger.debug(f"JinaReranker - Doc {i}: text_length={len(text)}, has_content={bool(text.strip())}")

            # Use asyncio to run the resilient reranking
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self._rerank_with_resilience(query, doc_texts, top_k)
            )

            logger.debug(
                "Jina API response received in %.2fs",
                time.time() - start_time,
            )

            if "results" not in result or not result["results"]:
                logger.warning("No results in Jina API response")
                return self._default_ranking(docs, top_k)

            # Process results
            scored_docs = []
            for item in result.get("results", []):
                index: int = item["index"]
                if index >= len(docs):
                    logger.warning("Jina returned invalid index %d, skipping", index)
                    continue

                score: float = item["relevance_score"]

                # Create a copy of the document to avoid modifying the original
                doc = docs[index]
                
                # Ensure document has text content with fallback
                if not doc.text and doc.metadata and 'content' in doc.metadata:
                    doc.text = doc.metadata['content']
                    logger.info(f"JinaReranker - Result {index}: Applied text fallback from metadata")
                
                doc.metadata["rerank_score"] = float(score)
                doc.metadata["score"] = float(score)
                scored_docs.append(doc)
                
                logger.debug(f"JinaReranker - Result {index}: final_text='{doc.text[:100]}...', score={score}")

            # Sort by rerank score
            ranked_docs = sorted(
                scored_docs,
                key=lambda d: d.metadata.get("rerank_score", 0.0),
                reverse=True,
            )[:top_k]

            logger.info(
                "Reranking completed in %.2fs, returned %d docs with content",
                time.time() - start_time,
                sum(1 for doc in ranked_docs if doc.text.strip())
            )
            return ranked_docs

        except Exception as error:
            logger.error("Error calling Jina Reranker API: %s", error)
            logger.warning("Falling back to original document order")
            return self._default_ranking(docs, top_k)

    def _default_ranking(self, docs: list[Document], top_k: int) -> list[Document]:
        """Provide a fallback ranking when the API call fails.
        
        Args:
            docs: List of documents to rank
            top_k: Number of top documents to return
            
        Returns:
            List of documents with default scoring
        """
        logger.info("Using default ranking for documents")

        # Limit to top_k documents
        result_docs = docs[:top_k]

        # Assign decreasing scores based on original order and ensure text content
        for idx, doc in enumerate(result_docs):
            # Apply content fallback if needed
            if not doc.text and doc.metadata and 'content' in doc.metadata:
                doc.text = doc.metadata['content']
                logger.info(f"JinaReranker - Default ranking Doc {idx}: Applied text fallback from metadata")
            
            doc.metadata["rerank_score"] = 1.0 - (idx * 0.1)
            logger.debug(f"JinaReranker - Default ranking Doc {idx}: text='{doc.text[:100]}...', score={doc.metadata['rerank_score']}")

        return result_docs

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of the reranker including circuit breaker state.
        
        Returns:
            Health status dictionary
        """
        circuit_stats = self.circuit_breaker.get_stats()
        
        return {
            "service": "jina_reranker",
            "model": self.model_name,
            "circuit_breaker": circuit_stats,
            "is_healthy": circuit_stats["state"] != "open",
            "api_url": self.rerank_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
