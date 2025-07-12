import pytest

from root.src.core.pipeline import (
    SentioRAGPipeline,
    RetrievalResult,
    GenerationResult,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_pipeline_query_happy_path(monkeypatch):
    """query() should combine mocked stages into a final response and update stats."""
    pipeline = SentioRAGPipeline()
    pipeline.initialized = True  # Skip heavy init logic

    # ------------------------------------------------------------------
    # Mock internal async stages to avoid network/database access.
    # ------------------------------------------------------------------
    async def fake_retrieve(query: str, top_k: int | None = None):  # noqa: D401
        return RetrievalResult(
            documents=[{"text": "hello", "source": "unit", "score": 0.9}],
            query=query,
            strategy="dense",
            total_time=0.01,
            sources_found=1,
        )

    async def fake_rerank(query: str, documents: list[dict], top_k: int | None = None):  # noqa: D401
        return documents  # Identity

    async def fake_generate(query: str, context: list[dict], mode=None):  # noqa: D401
        return GenerationResult(
            answer="42",
            sources=context,
            query=query,
            mode="fast",
            total_time=0.02,
        )

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "rerank", fake_rerank)
    monkeypatch.setattr(pipeline, "generate", fake_generate)

    response = await pipeline.query("What is the answer?")

    assert response["answer"] == "42"
    assert response["sources"]
    assert response["metadata"]["retrieval_strategy"] == "dense"
    assert pipeline.stats["queries_processed"] == 1 