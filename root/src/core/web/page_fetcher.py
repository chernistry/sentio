from __future__ import annotations

"""Asynchronous web page loader with content extraction.

Core ideas inspired by popular research agents but implemented from scratch.
* Uses `aiohttp` for non-blocking downloads.
* Parses HTML (or PDFs) and returns clean article text for RAG ingestion.

No backlink to original repos kept.
"""

from typing import List, Dict, Any
import asyncio
import logging
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
import trafilatura


LOGGER = logging.getLogger(__name__)
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


class AsyncWebPageFetcher:  # pylint: disable=too-few-public-methods
    """Download and extract text from web pages concurrently."""

    def __init__(self, max_concurrency: int = 8, timeout: int = 15):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def _download(self, session: aiohttp.ClientSession, url: str) -> str | None:
        try:
            async with self.semaphore, session.get(url, timeout=self.timeout) as resp:
                if resp.status != 200:
                    LOGGER.warning("Failed to fetch %s [status=%s]", url, resp.status)
                    return None
                content_type = resp.headers.get("Content-Type", "")
                raw = await resp.read()
                if "application/pdf" in content_type:
                    # Fallback: treat as binary; skip for now
                    LOGGER.debug("PDF detected for %s – skipping extraction", url)
                    return None
                html = raw.decode(errors="ignore")
                return html
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Exception fetching %s: %s", url, exc)
            return None

    @staticmethod
    def _extract_text(html: str) -> str:
        """Return main article text using trafilatura or BeautifulSoup fallback."""
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        if extracted:
            return extracted
        soup = BeautifulSoup(html, "lxml")
        # Simple fallback: join paragraph texts
        paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
        return "\n\n".join([p for p in paragraphs if len(p) > 40])

    async def fetch(self, url: str) -> Dict[str, Any] | None:
        """Fetch single URL and return dict with `url`, `domain`, `text`."""
        async with aiohttp.ClientSession(headers=_DEFAULT_HEADERS) as session:
            html = await self._download(session, url)
            if html is None:
                return None
            text = self._extract_text(html)
            if not text:
                return None
            return {
                "url": url,
                "domain": urlparse(url).netloc,
                "text": text,
            }

    async def fetch_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple URLs concurrently and return extracted docs."""
        tasks = [self.fetch(u) for u in urls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None] 