import asyncio
import os
from types import SimpleNamespace
from typing import List

import pytest
import httpx

from root.src.core.pipeline import SentioRAGPipeline, PipelineError


class _StubAsyncClient:  # noqa: D401 – mimics minimal httpx.AsyncClient
    def __init__(self):
        self.calls: int = 0
        self.is_closed = False

    async def post(self, *_, **__):  # noqa: D401
        self.calls += 1
        if self.calls == 1:
            raise httpx.TimeoutException("boom")

        class _Resp:  # noqa: D401
            def __init__(self):
                self.status_code = 200

            def raise_for_status(self):  # noqa: D401
                pass

            def json(self):  # noqa: D401
                # Minimal payload expected by _generate_answer
                return {
                    "choices": [
                        {"message": {"content": "ok-answer"}}
                    ]
                }

        return _Resp()


class _SearchHit:  # noqa: D401 – stub for Qdrant search result
    def __init__(self, score: float):
        self.score = score
        self.payload = {"text": "stub text", "source": "s"}


class _StubQdrantClient:  # noqa: D401 – supplies .search()
    def __init__(self, hits: List[_SearchHit]):
        self._hits = hits

    def search(self, *_, **__):  # noqa: D401
        return self._hits


class _EmbedStub:  # noqa: D401 – returns deterministic vector
    async def embed_async_single(self, text):  # noqa: D401
        return [0.1, 0.2, 0.3]

    async def embed_async_many(self, texts):  # noqa: D401
        return [[0.1, 0.2, 0.3] for _ in texts]

    def clear_cache(self):  # noqa: D401
        pass

    def get_stats(self):  # noqa: D401
        return {}


@pytest.mark.asyncio
async def test_post_with_retries_success():  # noqa: D401
    """_post_with_retries should succeed on second attempt."""

    pip = SentioRAGPipeline()
    client = _StubAsyncClient()
    resp = await pip._post_with_retries(  # type: ignore[attr-defined]
        client,
        url="http://x",
        headers={},
        json={},
        max_retries=2,
    )
    assert client.calls == 2
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_dense_retrieval_flow(monkeypatch):  # noqa: D401
    """Cover _dense_retrieval logic including payload parsing."""

    pip = SentioRAGPipeline()
    pip.embed_model = _EmbedStub()
    pip.qdrant_client = _StubQdrantClient([_SearchHit(0.42)])
    pip.config.collection_name = "col"
    pip.index = object()  # non-None to pass internal guard

    docs = await pip._dense_retrieval("hello", top_k=1)
    assert docs and docs[0]["score"] == 0.42 and docs[0]["text"] == "stub text"


@pytest.mark.asyncio
async def test_generate_answer_openrouter(monkeypatch):  # noqa: D401
    """Exercise _generate_answer path with openrouter provider."""

    monkeypatch.setenv("CHAT_LLM_API_KEY", "tok")

    async def _fake_completion(payload):  # noqa: D401
        return {"choices": [{"message": {"content": "ok-answer"}}]}

    monkeypatch.setattr(
        "root.src.core.llm.chat_adapter.chat_completion",
        _fake_completion,
        raising=True,
    )

    pip = SentioRAGPipeline()

    answer = await pip._generate_answer("q", "ctx", {"temperature": 0.3, "max_tokens": 10})
    assert answer == "ok-answer" 