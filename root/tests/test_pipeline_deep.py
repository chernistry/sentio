import pytest


@pytest.mark.asyncio
async def test_pipeline_rerank_generate_health(monkeypatch):  # noqa: D401
    from root.src.core.pipeline import SentioRAGPipeline, GenerationResult

    pipeline = SentioRAGPipeline()
    pipeline.initialized = True

    # ---- Stub reranker -------------------------------------------------------
    class _Reranker:  # noqa: D401
        def rerank(self, query, docs, top_k=5):  # noqa: D401, ANN001
            for d in docs:
                d["rerank_score"] = 42.0
            return docs

    pipeline.reranker = _Reranker()

    docs = [{"text": "hello", "score": 0.9}]
    reranked = await pipeline.rerank("q", docs, top_k=1)
    assert reranked[0]["rerank_score"] == 42.0

    # ---- Patch LLM generation stack -----------------------------------------
    async def _fake_completion(payload):  # noqa: D401
        return {"choices": [{"message": {"content": "llm-answer"}}]}

    monkeypatch.setenv("CHAT_LLM_API_KEY", "test")
    monkeypatch.setattr(
        "root.src.core.llm.chat_adapter.chat_completion",
        _fake_completion,
        raising=True,
    )

    generated = await pipeline.generate("q", docs)
    assert generated.answer == "llm-answer"

    # ---- Embed model stub for health_check -----------------------------------
    class _StubEmbed:  # noqa: D401
        async def embed_async_single(self, text):  # noqa: D401
            return [0.0]

    pipeline.embed_model = _StubEmbed()
    health = await pipeline.health_check()
    assert health["status"] in {"healthy", "unhealthy"} 