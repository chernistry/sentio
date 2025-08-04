from __future__ import annotations

"""Jina reranker implementation using the Jina AI Reranker API.

This module provides a reranker that uses the Jina AI Reranker API for
document reranking without requiring local models.
"""

import logging
import time
from typing import Any

import requests

from src.core.models.document import Document
from src.core.rerankers.base import Reranker
from src.utils.settings import settings

logger = logging.getLogger(__name__)


class JinaReranker(Reranker):
    """Reranker that uses the Jina AI Reranker API for document reranking.

    This reranker sends documents to the Jina AI Reranker API for scoring and
    reranking. It offers superior performance without requiring local GPU
    resources.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
    ) -> None:
        """Initialize the JinaReranker instance.

        Args:
            model_name: Custom model name. If None, uses environment variable or default.
            api_key: Jina API key. If None, uses environment variable.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.model_name: str = model_name or settings.reranker_model
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

        logger.debug("Jina reranker initialized successfully")

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

        logger.info("Reranking %d documents with Jina API", len(docs))
        start_time = time.time()

        try:
            # Extract document texts
            doc_texts = [doc.text for doc in docs]

            payload: dict[str, Any] = {
                "model": self.model_name,
                "query": query,
                "documents": doc_texts,
                "top_n": min(len(docs), top_k * 2),  # Request more results for robustness
            }

            logger.debug(
                "JinaReranker: POST %s | model=%s | top_n=%d",
                self.rerank_url,
                self.model_name,
                payload["top_n"],
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
                return self._default_ranking(docs, top_k)

            response.raise_for_status()
            result: dict[str, Any] = response.json()

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
                doc.metadata["rerank_score"] = float(score)
                doc.metadata["score"] = float(score)
                scored_docs.append(doc)

            # Sort by rerank score
            ranked_docs = sorted(
                scored_docs,
                key=lambda d: d.metadata.get("rerank_score", 0.0),
                reverse=True,
            )[:top_k]

            logger.info(
                "Reranking completed in %.2fs",
                time.time() - start_time,
            )
            return ranked_docs

        except requests.RequestException as error:
            logger.error("Error calling Jina Reranker API: %s", error)
            logger.warning("Falling back to original document order")
            return self._default_ranking(docs, top_k)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.error("Error processing Jina API response: %s", e)
            return self._default_ranking(docs, top_k)
        except Exception as e:
            logger.error("Unexpected error in reranker: %s", e)
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

        # Assign decreasing scores based on original order
        for idx, doc in enumerate(result_docs):
            doc.metadata["rerank_score"] = 1.0 - (idx * 0.1)

        return result_docs
