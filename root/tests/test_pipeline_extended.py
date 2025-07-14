import asyncio
from types import SimpleNamespace
from typing import List, Dict

import pytest

from root.src.core.pipeline import (
    SentioRAGPipeline,
    RetrievalStrategy,
    GenerationResult,
)


# ---------------------------------------------------------------------------
# Helper stubs used across multiple tests
# ---------------------------------------------------------------------------


class _StubChunker:  # noqa: D401 – provides *split* only
    def __init__(self, result_nodes: List):
        self._nodes = result_nodes

    def split(self, _docs):  # noqa: D401
        return self._nodes

    def reset_stats(self):  # noqa: D401
        pass


class _StubNode:  # noqa: D401 – mimics LlamaIndex Node behaviour
    def __init__(self, text: str):
        self._text = text
        self.metadata = {"source": "stub"}

    def get_content(self):  # noqa: D401
        return self._text


class _StubEmbedModel:  # noqa: D401 – minimal async embedder
    async def embed_async_many(self, texts: List[str]):  # noqa: D401
        # Map each text to vector [len(text)] * 3 (consistent, distinct)
        return [[float(len(t))] * 3 for t in texts]

    async def embed_async_single(self, text):  # noqa: D401
        return [0.0]

    def clear_cache(self):  # noqa: D401
        pass


class _StubQdrant:  # noqa: D401 – no-op client
    def upsert(self, *_, **__):  # noqa: D401
        pass

    async def close(self):  # noqa: D401
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", list(RetrievalStrategy))
async def test_retrieve_strategies(monkeypatch, strategy):  # noqa: D401
    """Exercise *retrieve* path for all strategies with stubbed helpers."""

    pl = SentioRAGPipeline()
    pl.initialized = True
    pl.config.retrieval_strategy = strategy

    # Stub retrieval helpers
    async def _ret(self, query: str, top_k: int):  # noqa: D401
        return [{"text": f"{strategy.value}-doc", "score": 0.5}]

    monkeypatch.setattr(SentioRAGPipeline, "_dense_retrieval", _ret)
    monkeypatch.setattr(SentioRAGPipeline, "_hybrid_retrieval", _ret)
    monkeypatch.setattr(SentioRAGPipeline, "_semantic_retrieval", _ret)

    result = await pl.retrieve("q", top_k=1)
    assert result.strategy == strategy.value
    assert result.documents[0]["text"].endswith("doc")


@pytest.mark.asyncio
async def test_query_stream(monkeypatch):  # noqa: D401
    """Validate *query_stream* yields the answer progressively."""

    pipeline = SentioRAGPipeline()
    pipeline.initialized = True

    from types import SimpleNamespace

    # Stub component behaviour to make *query_stream* deterministic
    async def _fake_retrieve(self, q, top_k=None):  # noqa: D401
        return SimpleNamespace(
            documents=[{"text": "ctx"}],
            total_time=0.0,
            strategy="dense",
        )

    async def _fake_rerank(self, q, docs, top_k=None):  # noqa: D401
        return docs

    async def _fake_generate(self, q, ctx, mode=None):  # noqa: D401
        return GenerationResult(
            answer="streaming",
            sources=[],
            query=q,
            mode="fast",
            total_time=0.0,
        )

    monkeypatch.setattr(SentioRAGPipeline, "retrieve", lambda *a, **k: _fake_retrieve(*a, **k))
    monkeypatch.setattr(SentioRAGPipeline, "rerank", _fake_rerank)
    monkeypatch.setattr(SentioRAGPipeline, "generate", _fake_generate)

    chunks = []
    async for token in pipeline.query_stream("what?"):
        chunks.append(token)
    assert "".join(chunks).strip() == "streaming"


@pytest.mark.asyncio
async def test_ingest_texts(monkeypatch):  # noqa: D401
    """Cover *ingest_texts* happy-path branch with full stubbed deps."""
    # Removed skip – uses patched PointStruct compat
    pipeline = SentioRAGPipeline()
    pipeline.initialized = True

    # Inject stubs
    texts = ["hello world", "foo bar"]
    nodes = [_StubNode(t) for t in texts]
    pipeline.chunker = _StubChunker(nodes)
    pipeline.embed_model = _StubEmbedModel()
    pipeline.qdrant_client = _StubQdrant()
    pipeline.config.collection_name = "test"

    # Provide fake index with insert_nodes method to avoid rebuild path
    class _FakeIndex:  # noqa: D401
        def insert_nodes(self, _):  # noqa: D401
            pass

    pipeline.index = _FakeIndex()

    count = await pipeline.ingest_texts(texts, sources=["s1", "s2"])
    assert count == 2 