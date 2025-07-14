import asyncio
import importlib
import sys
import types
from typing import Any, Dict, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


class _StubQdrantClient:  # noqa: D101 – simple stub
    """Minimal Qdrant client stub for HybridRetriever tests."""

    # pylint: disable=unused-argument
    def collection_exists(self, collection_name: str) -> bool:  # noqa: D401
        return False

    def get_collection(self, collection_name: str) -> "Any":  # pragma: no cover
        class _Meta:  # noqa: D401, WPS431 – inner stub
            points_count: int = 0

        return _Meta()

    def search(self, *args: Any, **kwargs: Any) -> List[Any]:  # noqa: D401
        return []

    def scroll(self, *args: Any, **kwargs: Any) -> Tuple[List[Any], None]:  # noqa: D401
        return [], None

    def retrieve(self, *args: Any, **kwargs: Any) -> List[Any]:  # noqa: D401
        return []


class _StubEmbedModel:  # noqa: D101 – simple stub
    """Very small embedding stub returning fixed-size vectors."""

    # pylint: disable=unused-argument
    def embed(self, texts: List[str]) -> List[List[float]]:  # noqa: D401
        return [[0.0] * 4 for _ in texts]

    def get_stats(self) -> Dict[str, Any]:  # noqa: D401
        return {}

    def clear_cache(self) -> None:  # noqa: D401
        pass


# ---------------------------------------------------------------------------
# HybridRetriever RRF scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top_k", [3, 5])
def test_hybrid_retriever_rrf(monkeypatch: pytest.MonkeyPatch, top_k: int) -> None:
    """Ensure HybridRetriever fuses dense and sparse scores via RRF.

    The internal dense/sparse search methods are monkey-patched to avoid heavy
    external dependencies. The test checks that documents appearing in both
    sources receive higher scores and bubble to the top of the final ranking.
    """
    from root.src.core.retrievers.hybrid import HybridRetriever

    # Monkey-patch heavy I/O helpers before instantiation
    monkeypatch.setattr(HybridRetriever, "_load_documents", lambda self: None)

    retriever = HybridRetriever(
        client=_StubQdrantClient(),
        embed_model=_StubEmbedModel(),
    )

    # Patch internal retrieval helpers
    def _fake_dense(self, query: str, limit: int, collection_name: str | None = None):  # noqa: D401
        return [("d1", {}, 0.9), ("d2", {}, 0.8)]

    def _fake_sparse(self, query: str, limit: int):  # noqa: D401
        return [("d2", 1.0), ("d3", 0.5)]

    monkeypatch.setattr(HybridRetriever, "_dense_search", _fake_dense)
    monkeypatch.setattr(HybridRetriever, "_sparse_search", _fake_sparse)
    monkeypatch.setattr(HybridRetriever, "_id_to_text", lambda self, pid: f"text {pid}")

    results = retriever.retrieve("sentio", top_k=top_k)
    # Document *d2* appears in both dense and sparse hits, hence must top-rank
    assert results[0]["id"] == "d2"
    assert len(results) <= top_k


# ---------------------------------------------------------------------------
# Pipeline prompt & context helpers + stats flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_prompt_context_and_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate helper logic inside *SentioRAGPipeline* without external I/O."""
    from root.src.core.pipeline import (
        GenerationResult,
        RetrievalResult,
        SentioRAGPipeline,
    )

    pipeline = SentioRAGPipeline()

    # ---------------- Patch costly async components ----------------
    async def _noop_init(self):  # noqa: D401
        self.initialized = True

    async def _fake_retrieve(self, query: str, top_k: int | None = None):  # noqa: D401
        return RetrievalResult(
            documents=[{"text": "foo", "source": "bar", "score": 0.9}],
            query=query,
            strategy="dense",
            total_time=0.01,
            sources_found=1,
        )

    async def _fake_rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int | None = None):  # noqa: D401
        return docs[:1]

    async def _fake_generate(self, query: str, ctx: List[Dict[str, Any]], mode=None):  # noqa: D401
        return GenerationResult(
            answer="sentio answer",
            sources=ctx,
            query=query,
            mode="balanced",
            total_time=0.02,
        )

    monkeypatch.setattr(SentioRAGPipeline, "initialize", _noop_init)
    monkeypatch.setattr(SentioRAGPipeline, "retrieve", _fake_retrieve)
    monkeypatch.setattr(SentioRAGPipeline, "rerank", _fake_rerank)
    monkeypatch.setattr(SentioRAGPipeline, "generate", _fake_generate)

    # --------------- Execute query & validate ---------------
    result = await pipeline.query("What is Sentio?")
    assert result["answer"] == "sentio answer"
    assert result["sources"][0]["text"] == "foo"

    # ---------- Prompt / context helper coverage ----------
    ctx_string = pipeline._build_context_string(result["sources"])  # noqa: SLF001
    assert "Source 1" in ctx_string

    from root.src.core.pipeline import GenerationMode

    prompt = pipeline._build_prompt(
        "q",
        ctx_string,
        pipeline._generation_configs[GenerationMode.FAST],
    )
    assert "Question: q" in prompt and "Context:" in prompt

    # -------------------- Stats flow --------------------
    stats = pipeline.get_stats()
    assert stats["queries_processed"] == 1
    pipeline.reset_stats()
    assert pipeline.get_stats()["queries_processed"] == 0


# ---------------------------------------------------------------------------
# PluginManager load / unload / reload logic
# ---------------------------------------------------------------------------


def test_plugin_manager_load_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise PluginManager load, register, unload paths."""
    from root.src.core.plugin_manager import PluginManager

    # Dynamically create a dummy plugin module under the *plugins* namespace
    dummy_module = types.ModuleType("plugins.dummy")

    from plugins.interface import SentioPlugin

    class _DummyPlugin(SentioPlugin):  # noqa: D401 – simple stub
        name: str = "dummy"
        version: str = "1.0"
        plugin_type: str = "feature"

        def register(self, pipeline):  # noqa: D401
            pipeline._dummy = True  # type: ignore[attr-defined]

    def _get_plugin():  # noqa: D401
        return _DummyPlugin()

    dummy_module.get_plugin = _get_plugin  # type: ignore[attr-defined]

    # Ensure import machinery finds our dummy module
    sys.modules["plugins.dummy"] = dummy_module

    # Make importlib.import_module return the dummy module
    monkeypatch.setattr(importlib, "import_module", lambda name: dummy_module)

    pm = PluginManager()
    pm.load_plugin("dummy")
    assert "dummy" in pm._plugin_map

    class _Pipeline:  # noqa: D401 – minimal stub
        pass

    pipeline_obj = _Pipeline()
    pm.register_all(pipeline_obj)
    assert getattr(pipeline_obj, "_dummy", False)

    pm.unload_plugin("dummy", pipeline_obj)
    assert "dummy" not in pm._plugin_map


# ---------------------------------------------------------------------------
# BM25Retriever persistence & retrieval
# ---------------------------------------------------------------------------


def test_bm25_retriever_index_save_load(tmp_path):  # noqa: D401
    """Cover BM25Retriever end-to-end workflow."""
    from root.src.core.retrievers.sparse import BM25Retriever, Document

    docs = [
        Document(id="1", text="sentio provides rag pipeline"),
        Document(id="2", text="rag enables retrieval augmented generation"),
    ]

    retriever = BM25Retriever()
    retriever.index(docs)
    # Retrieval may return empty if BM25 scores are zero – ensure no exception
    _ = retriever.retrieve("rag")

    file_path = tmp_path / "bm25.pkl"
    retriever.save(str(file_path))

    loaded = BM25Retriever()
    assert loaded.load(str(file_path))
    _ = loaded.retrieve("rag")  # ensure call succeeds after loading


# ---------------------------------------------------------------------------
# EmbeddingCache stats & expiry
# ---------------------------------------------------------------------------


def test_embedding_cache_hit_miss(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: D401
    """Validate EmbeddingCache hit/miss logic and statistic tracking."""
    from root.tests.test_embeddings import EmbeddingCache
    
    cache = EmbeddingCache(max_size=2, ttl_seconds=1)
    text, model = "hello", "m"
    assert cache.get(text, model) is None  # initial miss
    
    cache.set(text, model, [0.1, 0.2])
    assert cache.get(text, model) is not None
    
    # Simulate expiry by advancing time
    import time as _time_mod
    _orig_time = _time_mod.time
    monkeypatch.setattr("time.time", lambda: _orig_time() + 100.0)
    assert cache.get(text, model) is None  # expired → miss 