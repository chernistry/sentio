import pytest
from root.src.core.pipeline import SentioRAGPipeline, GenerationResult, RetrievalResult


@pytest.mark.asyncio
async def test_pipeline_retrieve_and_query(monkeypatch):  # noqa: D401
    """Cover *retrieve* and *query* high-level flows with stubbed internals."""

    pipeline = SentioRAGPipeline()

    # --- Make initialize() cheap ------------------------------------------------
    async def _fake_initialize(self):  # noqa: D401
        self.initialized = True

    monkeypatch.setattr(SentioRAGPipeline, "initialize", _fake_initialize, raising=True)
    monkeypatch.setattr(SentioRAGPipeline, "_setup_vector_store", lambda *_: None, raising=False)
    monkeypatch.setattr(SentioRAGPipeline, "_setup_retrievers", lambda *_: None, raising=False)

    # --- Stub retrieval paths ----------------------------------------------------
    async def _fake_hybrid(self, query: str, top_k: int):  # noqa: D401
        return [{"text": "doc", "source": "s1", "score": 0.9}]

    async def _fake_dense(self, *_, **__):  # noqa: D401
        return []

    async def _fake_semantic(self, *_, **__):  # noqa: D401
        return []

    monkeypatch.setattr(SentioRAGPipeline, "_hybrid_retrieval", _fake_hybrid)
    monkeypatch.setattr(SentioRAGPipeline, "_dense_retrieval", _fake_dense)
    monkeypatch.setattr(SentioRAGPipeline, "_semantic_retrieval", _fake_semantic)

    await pipeline.initialize()

    # *retrieve* -----------------------------------------------------------------
    result: RetrievalResult = await pipeline.retrieve("q", top_k=1)
    assert result.documents[0]["text"] == "doc"
    assert result.sources_found == 1

    # --- Stub rerank & generate for *query* -------------------------------------
    async def _fake_rerank(self, query: str, documents, top_k=None):  # noqa: D401
        return documents

    async def _fake_generate(self, query: str, ctx, mode=None):  # noqa: D401
        return GenerationResult(
            answer="ans", sources=ctx, query=query, mode="fast", total_time=0.01
        )

    monkeypatch.setattr(SentioRAGPipeline, "rerank", _fake_rerank)
    monkeypatch.setattr(SentioRAGPipeline, "generate", _fake_generate)

    final = await pipeline.query("What is Sentio?")
    assert final["answer"] == "ans" and final["sources"][0]["text"] == "doc"

    # Stats reset ----------------------------------------------------------------
    pipeline.reset_stats()
    assert pipeline.get_stats()["queries_processed"] == 0 