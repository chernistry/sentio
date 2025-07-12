"""
Hypothetical Document Embeddings (HyDE) query expansion module.

Implements the HyDE technique (Gao et al., 2023) where a language model
generates a hypothetical document that would contain the answer to a query.
The embeddings of this document are then used to augment the original query,
improving retrieval performance for complex or ambiguous queries.
"""

import os
import time
import logging
import requests
from typing import Any, Dict, List, Optional, Tuple

from plugins.interface import SentioPlugin

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3.5:3.8b")
HYDE_ENABLED = os.environ.get("ENABLE_HYDE", "0") == "1"
HYDE_CACHE_SIZE = 100

# Simple in-memory cache for HyDE generations
_hyde_cache: Dict[str, Tuple[str, float]] = {}


def _get_from_cache(query: str) -> Optional[str]:
    """Get hypothetical document from cache if available and not expired."""
    if query not in _hyde_cache:
        return None
    
    doc, timestamp = _hyde_cache[query]
    # Cache entry expires after 1 hour
    if time.time() - timestamp > 3600:
        del _hyde_cache[query]
        return None
    
    return doc


def _add_to_cache(query: str, document: str) -> None:
    """Add hypothetical document to cache."""
    # Limit cache size with LRU strategy
    if len(_hyde_cache) >= HYDE_CACHE_SIZE:
        oldest_key = min(_hyde_cache.keys(), key=lambda k: _hyde_cache[k][1])
        del _hyde_cache[oldest_key]
    
    _hyde_cache[query] = (document, time.time())


async def expand_query_hyde_async(query: str) -> Optional[str]:
    """
    Generate a hypothetical document for the query using an async API call.

    Args:
        query: The user query to expand

    Returns:
        A hypothetical document that might contain the answer to the query,
        or None if generation failed
    """
    # Check if HyDE is enabled in configuration
    if not HYDE_ENABLED:
        return None
    
    # Check cache first
    cached_doc = _get_from_cache(query)
    if cached_doc:
        logger.debug(f"HyDE: cache hit for query: {query[:30]}...")
        return cached_doc
    
    try:
        import httpx
        
        # Use async httpx client for non-blocking operation
        async with httpx.AsyncClient(timeout=10.0) as client:
            prompt = _generate_hyde_prompt(query)
            
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"HyDE: Ollama API returned status code {response.status_code}")
                return None
            
            result = response.json()
            document = result.get("response", "")
            
            # Ensure we got a meaningful response
            if len(document.strip()) < 20:
                logger.warning("HyDE: Generated document too short, skipping")
                return None
                
            _add_to_cache(query, document)
            logger.debug(f"HyDE: Generated hypothetical document for query: {query[:30]}...")
            return document
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return None


def expand_query_hyde(query: str) -> Optional[str]:
    """
    Synchronous version of the HyDE expansion function.
    
    Args:
        query: The user query to expand

    Returns:
        A hypothetical document that might contain the answer to the query,
        or None if generation failed
    """
    # Check if HyDE is enabled in configuration
    if not HYDE_ENABLED:
        return None
    
    # Check cache first
    cached_doc = _get_from_cache(query)
    if cached_doc:
        logger.debug(f"HyDE: cache hit for query: {query[:30]}...")
        return cached_doc
        
    try:
        prompt = _generate_hyde_prompt(query)
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )
        
        if response.status_code != 200:
            logger.warning(f"HyDE: Ollama API returned status code {response.status_code}")
            return None
        
        result = response.json()
        document = result.get("response", "")
        
        # Ensure we got a meaningful response
        if len(document.strip()) < 20:
            logger.warning("HyDE: Generated document too short, skipping")
            return None
            
        _add_to_cache(query, document)
        logger.debug(f"HyDE: Generated hypothetical document for query: {query[:30]}...")
        return document
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return None


def _generate_hyde_prompt(query: str) -> str:
    """Generate prompt for HyDE document generation."""
    return f"""You are an expert research assistant tasked with creating hypothetical document passages. 
For the question below, write a detailed, factual passage that directly answers the question.
Write as if you are an authoritative document on this topic.

Question: {query}

Hypothetical Document:"""


class HyDEPlugin(SentioPlugin):
    """Plugin providing HyDE query expansion."""

    name = "hyde_expander"
    plugin_type = "expander"

    def register(self, pipeline: Any) -> None:
        pipeline.hyde_expand = expand_query_hyde_async


def get_plugin() -> SentioPlugin:
    """Return plugin instance."""
    return HyDEPlugin()
