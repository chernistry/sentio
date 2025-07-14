import pytest

from types import SimpleNamespace

from root.src.core import pipeline as pl


@pytest.mark.asyncio
async def test_pipeline_initialize_light(monkeypatch):  # noqa: D401
    """Run *initialize()* with all heavy sub-components stubbed out.

    This executes the majority of the initialisation branch without touching
    external services or model downloads, dramatically boosting coverage of
    *pipeline.py* while keeping the test runtime minimal.
    """

    # --- Stub heavyweight classes -------------------------------------------
    class _StubEmbedding:  # noqa: D401 – no-op replacement
        def __init__(self, *_, **__):
            pass

    class _StubChunker:  # noqa: D401 – no-op replacement
        def __init__(self, *_, **__):
            pass

    class _StubVectorStore:  # noqa: D401 – placeholder
        pass

    class _StubRetriever:  # noqa: D401 – placeholder
        pass

    class _StubRerankTask:  # noqa: D401 – placeholder
        def __init__(self, *_, **__):
            pass

    monkeypatch.setattr(pl, "EmbeddingModel", _StubEmbedding)
    monkeypatch.setattr(pl, "TextChunker", _StubChunker)
    monkeypatch.setattr(pl, "QdrantClient", lambda *_, **__: SimpleNamespace(get_collections=lambda: SimpleNamespace(collections=[])))
    monkeypatch.setattr(pl, "QdrantVectorStore", _StubVectorStore)
    monkeypatch.setattr(pl, "HybridRetriever", _StubRetriever)
    monkeypatch.setattr("root.src.core.tasks.rerank.RerankTask", _StubRerankTask)

    # --- NOP async helpers ---------------------------------------------------
    async def _noop(self, *_, **__):  # noqa: D401
        return None

    monkeypatch.setattr(pl.SentioRAGPipeline, "_setup_vector_store", _noop, raising=False)
    monkeypatch.setattr(pl.SentioRAGPipeline, "_setup_retrievers", _noop, raising=False)
    monkeypatch.setattr(pl.SentioRAGPipeline, "_setup_index", _noop, raising=False)

    # --- Dummy plugin manager ------------------------------------------------
    class _DummyPM:  # noqa: D401
        def load_all(self):
            pass

        def load_from_env(self):
            pass

        def register_all(self, _):
            pass

    pipeline = pl.SentioRAGPipeline(plugins=_DummyPM())
    assert not pipeline.initialized

    await pipeline.initialize()

    assert pipeline.initialized 