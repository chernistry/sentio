#!/usr/bin/env python3
"""
Ollama-based reranker for Sentio RAG.

This module provides an implementation of reranking using local Ollama models.
It directly asks an LLM to score document relevance to a query.
"""

import logging
import os
import re
from typing import Dict, List, Any

import httpx
from plugins.interface import SentioPlugin

logger = logging.getLogger(__name__)

class OllamaReranker:
    """Reranker that uses Ollama LLM to score document relevance."""
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "phi3.5:3.8b",
        timeout: int = 15,
    ):
        """
        Initialize the Ollama reranker.
        
        Args:
            ollama_url: URL to the Ollama API server
            ollama_model: Name of the model to use for reranking
            timeout: Request timeout in seconds
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.timeout = timeout
        self.enable_llm_judge = True  # Can be disabled to bypass LLM reranking
        
    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        *,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The search query
            docs: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents
        """
        if not docs:
            return []
            
        # Return original docs if LLM judging is disabled
        if not self.enable_llm_judge:
            return docs[:top_k]
            
        # Only rerank top documents to save on LLM calls
        judge_docs = docs[:min(3, len(docs))]
        scores = self._llm_judge_relevance(query, judge_docs)
        
        # Update scores
        for d, s in zip(judge_docs, scores):
            d["llm_judge_score"] = s
            # Store original score if present
            if "score" in d:
                d["original_score"] = d["score"]
            # Set new score or blend if original exists
            d["score"] = s if "score" not in d else 0.7 * d["score"] + 0.3 * s
            
        # Combine reranked docs with remaining docs
        ranked = sorted(judge_docs, key=lambda d: d["score"], reverse=True)
        remaining = [d for d in docs[len(judge_docs):] if d not in judge_docs]
        
        # Return combined results
        return (ranked + remaining)[:top_k]
    
    def _llm_judge_relevance(self, query: str, docs: List[Dict[str, Any]]) -> List[float]:
        """Use Ollama LLM to estimate relevance on a 0-1 scale."""
        # Set max document size to avoid token limits
        MAX_DOC_CHARS = 1200  # ~400-500 tokens
        
        def _clip(text: str, limit: int = MAX_DOC_CHARS) -> str:
            """Trim text in the middle so head/tail are preserved equally."""
            if len(text) <= limit:
                return text
            half = limit // 2
            return text[:half] + "\n…\n" + text[-half:]
            
        # Default optimistic scores in case of errors
        default_scores = [0.9, 0.6, 0.3][:len(docs)]
        scores = default_scores.copy()
        
        for idx, doc in enumerate(docs):
            # Extract text from document
            snippet = _clip(doc.get("text", ""))
            
            # Build prompt for LLM
            prompt = (
                "Rate the relevance of the following *document* to the *query* on a scale "
                "from 0 to 10. Respond with **only** the number.\n\n"
                f"QUERY: {_clip(query, 200)}\n\nDOCUMENT: {snippet}\n\nRating (0-10):"
            )
            
            try:
                # Call Ollama API
                response = httpx.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_ctx": 4096}  # Explicit context length
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                output = response.json().get("response", "")
                
                # Extract numeric rating
                match = re.search(r"(\d+(?:\.\d+)?)", output)
                if match:
                    val = min(max(float(match.group(1)) / 10, 0.0), 1.0)
                    scores[idx] = val
            except Exception as e:
                logger.warning(f"LLM judge failed for document {idx}: {e}")
                # Keep default score on error
        
        return scores


class OllamaRerankerPlugin(SentioPlugin):
    """Plugin wrapper for Ollama reranker."""

    name = "ollama_reranker"
    plugin_type = "reranker"

    def __init__(self, **kwargs: Any) -> None:
        self.reranker = OllamaReranker(**kwargs)

    def register(self, pipeline: Any) -> None:
        pipeline.reranker = self.reranker


def get_plugin() -> SentioPlugin:
    """Return plugin instance."""
    return OllamaRerankerPlugin()
