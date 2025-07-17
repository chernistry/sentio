"""Jina reranker using the Jina AI Reranker API.

This module provides a lightweight wrapper around the Jina AI Reranker API
to enable quick and accurate document reranking without the computational
overhead of running models locally. Ideal for scenarios where running large
reranker models locally is impractical due to performance constraints.
"""

from __future__ import annotations

import os
import logging
import requests
import time
import json
from typing import Any, Dict, List, Optional


# ==== CORE PROCESSING MODULE ==== #


# --► CONFIGURATION & CONSTANTS ------------------------------------------------

DEFAULT_JINA_RERANK_MODEL: str = "jina-reranker-v2-base-en"
JINA_RERANK_URL: str = "https://api.jina.ai/v1/rerank"

logger = logging.getLogger(__name__)


# --► RERANKER IMPLEMENTATION ---------------------------------------------------

class JinaReranker:
    """Reranker that uses the Jina AI Reranker API for document reranking.

    This reranker sends documents to the Jina AI Reranker API for scoring and
    reranking. It offers superior performance without requiring local GPU
    resources.

    Example:
        >>> reranker = JinaReranker()
        >>> ranked_docs = reranker.rerank(query, docs)
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,  # Added provider parameter
    ) -> None:
        """
        Initializes the JinaReranker instance.

        Args:
            model_name (str | None): Custom model name. If None, the
                RERANKER_MODEL environment variable or the default model
                is used.
            api_key (str | None): Jina API key. If None, the JINA_API_KEY
                environment variable is used.
            provider (str | None): Provider identifier, ignored by this implementation.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        # Preconditions:
        # - Either model_name or corresponding environment variable is set.
        # - Either api_key or corresponding environment variable is set.

        self.model_name: str = (
            model_name
            or os.getenv("RERANKER_MODEL", DEFAULT_JINA_RERANK_MODEL)
        )
        logger.info(f"Using Jina reranker model: {self.model_name}")

        raw_key: str = api_key or os.getenv("EMBEDDING_MODEL_API_KEY", os.getenv("JINA_API_KEY", ""))
        self.api_key: str = raw_key.strip()

        if not self.api_key:
            raise ValueError(
                "No embedding API key provided. Set EMBEDDING_MODEL_API_KEY environment variable "
                "or pass the api_key parameter."
            )

        self.headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # Установка таймаута запросов для предотвращения зависания
        self.timeout: int = int(os.getenv("RERANKER_TIMEOUT", "30"))
        
        logger.debug("Jina reranker initialized successfully")

    # --------------------------------------------------------------------------
    # --► DATA EXTRACTION & TRANSFORMATION
    # --------------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        *,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Returns the top_k documents ranked by relevance to the query string.

        Args:
            query (str): Search query string.
            docs (List[Dict[str, Any]]): List of document dicts, each with
                a 'text' field.
            top_k (int): Number of top documents to return.

        Returns:
            List[Dict[str, Any]]: Documents sorted by descending 'rerank_score'.

        Preconditions:
            - 'docs' is a list of dicts containing at least the 'text' key.

        Postconditions:
            - Returns an empty list if 'docs' is empty.
            - Each returned dict contains a 'rerank_score' float.
        """
        if not docs:
            logger.debug("No documents to rerank")
            return []

        if not query or query.strip() == "":
            logger.warning("Empty query received in reranker, using default ranking")
            return self._default_ranking(docs, top_k)

        logger.info(f"Reranking {len(docs)} documents with Jina API")
        start_time = time.time()

        try:
            doc_texts: List[str] = []
            for doc in docs:
                text = doc.get("text", "")
                if not text:
                    logger.warning("Document without 'text' field encountered")
                    text = json.dumps(doc)  # Используем JSON представление документа как fallback
                doc_texts.append(text)

            payload: Dict[str, Any] = {
                "model": self.model_name,
                "query": query,
                "documents": doc_texts,
                "top_n": min(len(docs), top_k * 2),  # Запрашиваем больше результатов для устойчивости
            }

            logger.debug(
                "[JinaReranker] POST %s | model=%s | top_n=%d | token_prefix=%s…",
                JINA_RERANK_URL,
                self.model_name,
                payload["top_n"],
                self.api_key[:5] + "..." if len(self.api_key) > 5 else "*****",
            )

            response = requests.post(
                JINA_RERANK_URL,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            
            if not response.ok:
                logger.error(
                    f"Jina API error: {response.status_code} - {response.text}"
                )
                return self._default_ranking(docs, top_k)
                
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            
            logger.debug(
                f"Jina API response received in "
                f"{time.time() - start_time:.2f}s"
            )

            if "results" not in result or not result["results"]:
                logger.warning("No results in Jina API response")
                return self._default_ranking(docs, top_k)

            scored_docs: List[Dict[str, Any]] = []
            for item in result.get("results", []):
                index: int = item["index"]
                if index >= len(docs):
                    logger.warning(f"Jina returned invalid index {index}, skipping")
                    continue
                    
                score: float = item["relevance_score"]

                doc_copy: Dict[str, Any] = docs[index].copy()
                doc_copy["rerank_score"] = float(score)

                scored_docs.append(doc_copy)

            ranked_docs: List[Dict[str, Any]] = sorted(
                scored_docs,
                key=lambda d: d["rerank_score"],
                reverse=True,
            )[:top_k]

            logger.info(
                f"Reranking completed in {time.time() - start_time:.2f}s"
            )
            return ranked_docs

        except requests.RequestException as error:
            logger.error(f"Error calling Jina Reranker API: {error}")
            logger.warning("Falling back to original document order")
            return self._default_ranking(docs, top_k)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.error(f"Error processing Jina API response: {e}")
            return self._default_ranking(docs, top_k)
        except Exception as e:
            logger.error(f"Unexpected error in reranker: {e}")
            return self._default_ranking(docs, top_k)

    def _default_ranking(self, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Provides a fallback ranking when the API call fails.
        
        Args:
            docs: List of documents to rank
            top_k: Number of top documents to return
            
        Returns:
            List of documents with default scoring
        """
        logger.info("Using default ranking for documents")
        fallback_docs: List[Dict[str, Any]] = []
        
        # Limit to top_k documents
        for idx, doc in enumerate(docs[:top_k]):
            doc_copy = doc.copy()
            # Assign decreasing scores based on original order
            doc_copy["rerank_score"] = 1.0 - (idx * 0.1)
            fallback_docs.append(doc_copy)

        return fallback_docs