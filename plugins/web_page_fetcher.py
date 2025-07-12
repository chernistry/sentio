#!/usr/bin/env python3
"""
Web page fetcher module for downloading and extracting content from web pages.

This module provides asynchronous utilities for fetching and processing web pages
to extract clean, readable content for use in RAG applications.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import trafilatura
from plugins.interface import SentioPlugin

logger = logging.getLogger(__name__)


class AsyncWebPageFetcher:
    """Asynchronous web page fetcher with content extraction."""
    
    def __init__(
        self,
        timeout: int = 10,
        max_concurrent: int = 5,
        user_agent: str = "Mozilla/5.0 Sentio/1.0 Web Fetcher",
        min_content_length: int = 50,
    ):
        """
        Initialize the web page fetcher.
        
        Args:
            timeout: HTTP request timeout in seconds
            max_concurrent: Maximum number of concurrent requests
            user_agent: User-Agent header for HTTP requests
            min_content_length: Minimum content length to be considered valid
        """
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.user_agent = user_agent
        self.min_content_length = min_content_length
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
    async def fetch(self, url: str) -> Dict[str, Any]:
        """
        Fetch a single web page and extract its content.
        
        Args:
            url: URL to fetch
            
        Returns:
            Dictionary with URL, title, and extracted text content
        """
        headers = {"User-Agent": self.user_agent}
        result = {"url": url, "title": "", "text": "", "error": None}
        
        try:
            async with self._semaphore:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url, headers=headers, follow_redirects=True)
                    response.raise_for_status()
                    html = response.text
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {str(e)}")
            result["error"] = str(e)
            return result
            
        # Use trafilatura to extract main content
        try:
            # Get cleaned text content
            text = trafilatura.extract(html, include_comments=False, include_tables=False)
            
            # Fall back to basic extraction if trafilatura fails
            if not text:
                text = trafilatura.extract(html, fallback=True, include_comments=False)
                
            # Get title if available
            if hasattr(response, "headers") and "content-type" in response.headers:
                if "html" in response.headers["content-type"].lower():
                    try:
                        # Try to extract title
                        import re
                        title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
                        if title_match:
                            title = title_match.group(1).strip()
                            result["title"] = title
                    except Exception:
                        pass
                        
            # Store the extracted text
            if text and len(text) >= self.min_content_length:
                result["text"] = text
            else:
                result["error"] = "Extracted content too short or empty"
                
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {str(e)}")
            result["error"] = f"Content extraction failed: {str(e)}"
            
        return result
        
    async def fetch_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch multiple web pages concurrently.
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            List of dictionaries with URL, title, and extracted text content
        """
        if not urls:
            return []
            
        # Create tasks for all URLs
        tasks = [self.fetch(url) for url in urls]
        
        # Run concurrently with asyncio.gather
        results = await asyncio.gather(*tasks)
        
        # Filter out failed results
        return [r for r in results if r.get("text")]
        
    @staticmethod
    def get_domain(url: str) -> str:
        """Extract the domain from a URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return "unknown"


class WebPageFetcherPlugin(SentioPlugin):
    """Plugin providing page fetching utilities."""

    name = "web_page_fetcher"
    plugin_type = "fetcher"

    def __init__(self, **kwargs: Any) -> None:
        self.fetcher = AsyncWebPageFetcher(**kwargs)

    def register(self, pipeline: Any) -> None:
        pipeline.page_fetcher = self.fetcher


def get_plugin() -> SentioPlugin:
    """Return plugin instance."""
    return WebPageFetcherPlugin()
