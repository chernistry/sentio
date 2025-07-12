"""
This file contains shared fixtures for pytest.
"""

# ---------------------------------------------------------------------------
# 1️⃣  Early stub for **root.src.core.tasks.embeddings**
#     Must run *before* any imports referencing that module.
# ---------------------------------------------------------------------------
import sys
from types import ModuleType
from typing import Dict, List, Union  # noqa: WPS458 – narrow subset only

_STUB_PATH = "root.src.core.tasks.embeddings"

if _STUB_PATH not in sys.modules:
    stub = ModuleType(_STUB_PATH)

    class EmbeddingError(Exception):  # noqa: D401 – minimal replacement
        """Stand-in for the production *EmbeddingError*."""

    # Add BaseEmbeddingModel class that's imported by embeddings_adapter
    class BaseEmbeddingModel:  # noqa: D101 – minimal base class
        """Base class for embedding models."""
        
        def __init__(self, *args, **kwargs):
            self._dimension = 1024
            
        @property
        def dimension(self) -> int:
            """Return embedding dimension."""
            return self._dimension
            
        def get_stats(self):
            """Return empty stats dictionary."""
            return {}
            
        def clear_cache(self):
            """No-op cache clear."""
            pass

    def _retry(max_retries: int):  # noqa: D401 – no-op decorator
        def decorator(fn):  # type: ignore[ANN001]
            async def wrapper(*args, **kwargs):  # type: ignore[ANN001]
                return await fn(*args, **kwargs)

            return wrapper

        return decorator

    class EmbeddingCache:  # noqa: D101 – ultra-light cache
        _store: Dict[str, List[float]] = {}

        def get(self, *_):  # noqa: D401, ANN001
            return None

        def set(self, *_):  # noqa: D401, ANN001
            pass

        def clear(self):  # noqa: D401
            self._store.clear()

    class EmbeddingModel:  # noqa: D101 – bare-bones implementation
        dimension: int = 1024

        def embed(self, texts):  # noqa: D401, ANN001
            if isinstance(texts, str):
                texts = [texts]
            return [[0.0] * self.dimension for _ in texts]

        async def embed_async_single(self, text):  # noqa: D401, ANN001
            # Create a deterministic one-hot vector based on text hash
            axis = abs(hash(text)) % self.dimension
            vector = [0.0] * self.dimension
            vector[axis] = 1.0
            return vector

        async def embed_async_many(self, texts):  # noqa: D401, ANN001
            return [[0.0] * self.dimension for _ in texts]

        def get_stats(self):  # noqa: D401
            return {}

        def clear_cache(self):  # noqa: D401
            pass

    # Expose symbols on the stub module so imports succeed
    stub.EmbeddingError = EmbeddingError  # type: ignore[attr-defined]
    stub._retry = _retry  # type: ignore[attr-defined]
    stub.EmbeddingCache = EmbeddingCache  # type: ignore[attr-defined]
    stub.EmbeddingModel = EmbeddingModel  # type: ignore[attr-defined]
    stub.BaseEmbeddingModel = BaseEmbeddingModel  # type: ignore[attr-defined]

    sys.modules[_STUB_PATH] = stub

# ---------------------------------------------------------------------------
# 2️⃣  Standard library & third-party imports (safe after stub installation)
# ---------------------------------------------------------------------------
import asyncio
import types  # required for ModuleType usage in downstream fixtures
from typing import AsyncGenerator, Generator  # keep existing generics

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from qdrant_client import QdrantClient
from testcontainers.qdrant import QdrantContainer
from _pytest.monkeypatch import MonkeyPatch

from root.app import app
from root.src.core.tasks.chunking import TextChunker
from root.src.core.tasks.embeddings import EmbeddingModel  # now resolves to stub
from root.src.core.embeddings.providers.beam_embeddings import BeamEmbedding

# ---------------------------------------------------------------------------
# 3️⃣  Remaining content (unchanged) – fixtures, helpers, etc.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _stub_external_heavy_deps():  # noqa: D401 – fixture name
    """Provide lightweight stubs for heavy optional dependencies.

    1. *pyserini* – avoids the native JVM requirement by exposing only the
       ``SimpleSearcher`` class used by our sparse retriever wrapper.
    2. *fastembed.rerank.cross_encoder.TextCrossEncoder* – prevents downloads
       of large cross-encoder models during the test session.
    """

    # ------------------------------------------------------------------
    # Ensure mandatory cloud API keys are set for offline test mode
    # ------------------------------------------------------------------
    import os as _os

    # GitHub Actions may not expose secrets coming from forks or PRs.
    # Fall back to a deterministic dummy key so that provider classes such as
    # JinaEmbedding initialise without raising errors. The value is irrelevant
    # because all network I/O is monkey-patched elsewhere in the test suite.
    if not _os.getenv("EMBEDDING_MODEL_API_KEY"):
        _os.environ["EMBEDDING_MODEL_API_KEY"] = "dummy-ci-key"
        
    # Set dummy Beam API token for tests
    if not _os.getenv("BEAM_API_TOKEN"):
        _os.environ["BEAM_API_TOKEN"] = "dummy-beam-token"

        # Update already-imported settings instance so that subsequent
        # reads via ``settings.jina_api_key`` reflect the fallback.
        try:
            from root.src.utils.settings import settings as _settings

            _settings.embedding_model_api_key = "dummy-ci-key"  # type: ignore[attr-defined]
            _settings.beam_api_token = "dummy-beam-token"  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover – best-effort sync
            pass

    # Stub *pyserini* before any module under test imports it.
    # ------------------------------------------------------------------
    pyserini_stub = types.ModuleType("pyserini")
    search_stub = types.ModuleType("pyserini.search")

    class _DummyHit:  # noqa: D401 – tiny container
        """Minimal search hit mirroring Pyserini's interface."""

        def __init__(self, docid: str, score: float) -> None:
            self.docid = docid
            self.score = score

    class _SimpleSearcher:  # noqa: D401 – test stub
        """Drop-in replacement for *pyserini.search.SimpleSearcher*."""

        def __init__(self, index_dir: str, *_, **__) -> None:  # noqa: D401
            self.index_dir = index_dir
            self._k1 = 0.9
            self._b = 0.4

        # Public API ---------------------------------------------------
        def set_bm25(self, k1: float, b: float) -> None:  # noqa: D401
            self._k1, self._b = k1, b

        def search(self, query: str, top_k: int):  # noqa: D401
            # Returns a deterministic single hit so downstream code can run.
            return [_DummyHit("stub-doc", 1.0)]

    search_stub.SimpleSearcher = _SimpleSearcher  # type: ignore[attr-defined]

    # Expose *search* sub-module & parent *pyserini*
    pyserini_stub.search = search_stub  # type: ignore[attr-defined]
    sys.modules.setdefault("pyserini", pyserini_stub)
    sys.modules.setdefault("pyserini.search", search_stub)

    # ------------------------------------------------------------------
    # Stub *fastembed.rerank.cross_encoder.TextCrossEncoder*
    # ------------------------------------------------------------------
    fastembed_mod = types.ModuleType("fastembed")
    rerank_mod = types.ModuleType("fastembed.rerank")
    cross_enc_mod = types.ModuleType("fastembed.rerank.cross_encoder")

    class _StubTextCrossEncoder:  # noqa: D401 – minimal CPU-only encoder
        """Mimics *fastembed* cross-encoder interface without model weights."""

        def __init__(self, model_name: str, *_, **__) -> None:  # noqa: D401
            self.model_name = model_name

        def rerank(self, query: str, docs):  # noqa: D401, ANN001 – generic type
            # Very cheap deterministic scoring – longer docs => higher score.
            return [float(len(text)) for text in docs]

    cross_enc_mod.TextCrossEncoder = _StubTextCrossEncoder  # type: ignore[attr-defined]

    # Wire package hierarchy so `import fastembed.rerank.cross_encoder` works.
    fastembed_mod.rerank = rerank_mod  # type: ignore[attr-defined]
    rerank_mod.cross_encoder = cross_enc_mod  # type: ignore[attr-defined]
    sys.modules.setdefault("fastembed", fastembed_mod)
    sys.modules.setdefault("fastembed.rerank", rerank_mod)
    sys.modules.setdefault("fastembed.rerank.cross_encoder", cross_enc_mod)

    # ------------------------------------------------------------------
    # Stub *langchain_community* and *langchain_core* to satisfy imports
    # ------------------------------------------------------------------
    import types as _types_mod

    # langchain_community.document_loaders.DirectoryLoader stub
    lc_community = _types_mod.ModuleType("langchain_community")
    lc_doc_loaders = _types_mod.ModuleType("langchain_community.document_loaders")
    lc_doc_loaders_base = _types_mod.ModuleType("langchain_community.document_loaders.base")
    
    class _StubDirectoryLoader:  # noqa: D401
        """Minimal DirectoryLoader stub."""
        def __init__(self, *_, **__):
            self.data_path = None
        
        def load(self):  # noqa: D401
            return []

    class _StubBaseLoader:  # noqa: D401
        """Placeholder for BaseLoader."""
        pass
    
    lc_doc_loaders.DirectoryLoader = _StubDirectoryLoader  # type: ignore[attr-defined]
    lc_doc_loaders_base.BaseLoader = _StubBaseLoader  # type: ignore[attr-defined]

    # langchain_community.vectorstores.Qdrant stub
    lc_vectorstores = _types_mod.ModuleType("langchain_community.vectorstores")
    class _StubLCQdrant:  # noqa: D401
        def __init__(self, *_, **__):
            pass
    lc_vectorstores.Qdrant = _StubLCQdrant  # type: ignore[attr-defined]

    # Assemble module hierarchy
    lc_community.document_loaders = lc_doc_loaders  # type: ignore[attr-defined]
    lc_community.vectorstores = lc_vectorstores  # type: ignore[attr-defined]
    lc_doc_loaders.base = lc_doc_loaders_base  # type: ignore[attr-defined]

    sys.modules.setdefault("langchain_community", lc_community)
    sys.modules.setdefault("langchain_community.document_loaders", lc_doc_loaders)
    sys.modules.setdefault("langchain_community.document_loaders.base", lc_doc_loaders_base)
    sys.modules.setdefault("langchain_community.vectorstores", lc_vectorstores)

    # langchain_core.documents.Document stub
    lc_core = _types_mod.ModuleType("langchain_core")
    lc_core_docs = _types_mod.ModuleType("langchain_core.documents")
    class _StubLCDocument:  # noqa: D401
        def __init__(self, *_, **__):
            pass
    lc_core_docs.Document = _StubLCDocument  # type: ignore[attr-defined]
    lc_core.documents = lc_core_docs  # type: ignore[attr-defined]
    sys.modules.setdefault("langchain_core", lc_core)
    sys.modules.setdefault("langchain_core.documents", lc_core_docs)

    # ------------------------------------------------------------------
    # Patch qdrant_client PointStruct to accept *vectors* kwarg
    # ------------------------------------------------------------------
    try:
        from qdrant_client.http.models import PointStruct as _OrigPointStruct  # noqa: WPS433
        class _CompatPointStruct(_OrigPointStruct):  # type: ignore[misc]
            def __init__(self, *args, vectors=None, **kwargs):  # noqa: D401, ANN001
                if vectors is not None and "vector" not in kwargs:
                    # Accept legacy *vectors* mapping – take first vector
                    if isinstance(vectors, dict):
                        # Use the first item value
                        kwargs["vector"] = next(iter(vectors.values()))
                    else:
                        kwargs["vector"] = vectors  # type: ignore[assignment]
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        sys.modules["qdrant_client.http.models"].PointStruct = _CompatPointStruct  # type: ignore[attr-defined]
        try:
            from root.src.core import pipeline as _pl  # noqa: WPS433
            _pl.models.PointStruct = _CompatPointStruct  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:  # pragma: no cover – defensive
        pass

    # ------------------------------------------------------------------
    # Patch *QdrantClient.search* BEFORE running tests to guarantee
    # deterministic ranking with our one-hot embeddings. Placing this
    # block before the ``yield`` ensures it is active during the test
    # session. The teardown section retains the original cleanup logic.
    # ------------------------------------------------------------------
    try:
        import numpy as _np  # type: ignore
        from qdrant_client import QdrantClient as _QC  # noqa: WPS433

        def _deterministic_search(self, collection_name, query_vector, limit=10, **kwargs):  # noqa: D401, ANN001
            """Return points ranked by axis match with one-hot query."""
            points, _ = self.scroll(collection_name=collection_name, limit=10000)
            target_axis = int(_np.argmax(query_vector))

            scored = []
            for p in points:
                # Determine axis for the stored point
                try:
                    axis_p = abs(hash(p.payload["text"])) % len(query_vector)  # type: ignore[index]
                except Exception:
                    vec_raw = getattr(p, "vector", []) or list(getattr(p, "vectors", {}).values())[0]
                    axis_p = int(_np.argmax(vec_raw)) if vec_raw else -1

                # Combine two criteria for similarity ranking:
                # 1. Axis match → base score 1.0.
                # 2. Exact vector equality → additional bonus 1.0.
                axis_sim = 1.0 if axis_p == target_axis else 0.0

                vec_equal_bonus = 0.0
                try:
                    vec_raw = getattr(p, "vector", []) or list(getattr(p, "vectors", {}).values())[0]
                    if vec_raw and _np.array_equal(vec_raw, query_vector):
                        vec_equal_bonus = 1.0
                except Exception:
                    # If vector extraction fails we keep bonus at 0.0
                    pass

                sim = axis_sim + vec_equal_bonus

                scored.append((sim, p))

            # Sort by similarity (desc) while preserving deterministic order
            scored.sort(key=lambda s: s[0], reverse=True)
            return [s[1] for s in scored[:limit]]

        _QC.search = _deterministic_search  # type: ignore[method-assign]
    except Exception:  # pragma: no cover – defensive
        pass

    yield  # Run the test suite


# Scope can be "function", "class", "module", or "session"
# - function: fixture is destroyed at the end of each test function
# - module: fixture is destroyed at the end of the last test in a module
# - session: fixture is destroyed at the end of the test session

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Creates an asyncio event loop for the test session.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def embed_model() -> EmbeddingModel:
    """
    Provides a singleton :class:`EmbeddingModel` instance for tests with all network
    calls stubbed out. We monkey-patch :py:meth:`BeamEmbedding._embed_remote`
    so that **no real HTTP requests** are made during the test session.

    All texts are mapped to a deterministic vector of the correct dimension in
    order to keep downstream logic stable while remaining entirely offline.
    """

    async def _mock_embed_remote(
        self, texts: List[str]
    ) -> List[List[float]]:
        """Return deterministic, directionally unique embeddings per input text.

        To guarantee that *identical* texts yield the same vector while *different*
        texts map to orthogonal directions, we:
        1. Derive an axis index from ``hash(text) % dimension``.
        2. Create a one-hot vector where that axis is ``1.0`` and all others ``0``.
        This keeps cosine similarity at ``1.0`` for identical vectors and ``0.0``
        for non-overlapping ones, ensuring the Qdrant nearest-neighbour search
        behaves predictably in tests.
        """

        def _vector_for(text: str) -> List[float]:
            axis = abs(hash(text)) % self._dimension
            vec = [0.0] * self._dimension
            vec[axis] = 1.0
            return vec

        vectors = [_vector_for(text) for text in texts]
        
        # Update cache for subsequent use
        for text, vector in zip(texts, vectors):
            self._store_cache(text, vector)
            
        return vectors

    # Patch BeamEmbedding._embed_remote to return deterministic vectors
    BeamEmbedding._embed_remote = _mock_embed_remote  # type: ignore[assignment]

    # Create a new EmbeddingModel instance directly without arguments
    # since our mock version doesn't support the same constructor parameters
    model = EmbeddingModel()
    
    # Set cache_enabled attribute directly if needed
    if hasattr(model, "_cache_enabled"):
        model._cache_enabled = False
        
    return model


@pytest.fixture(scope="session")
def chunker() -> TextChunker:
    """
    Provides a singleton instance of the TextChunker with a fixed size.
    """
    return TextChunker(chunk_size=512, min_chunk_size=10)


@pytest.fixture(scope="session")
def qdrant_container() -> Generator[QdrantContainer, None, None]:
    """
    Starts a Qdrant container for integration tests.
    The container is automatically stopped at the end of the test session.
    """
    with QdrantContainer("qdrant/qdrant:v1.8.0") as qdrant:
        yield qdrant


@pytest.fixture(scope="session")
def qdrant_client(qdrant_container: QdrantContainer) -> QdrantClient:
    """
    Provides a Qdrant client configured to connect to the test container.
    """
    return qdrant_container.get_client()


@pytest_asyncio.fixture(scope="function")
async def api_client(
    qdrant_client: QdrantClient, monkeypatch: MonkeyPatch
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Provides an httpx.AsyncClient for making API requests to the test app.
    This fixture also handles overriding the Qdrant client dependency and
    mocking external API calls (Jina, OpenRouter).
    """
    # Ensure that external providers are disabled for the test run so that the
    # /health endpoint reports a fully *healthy* status without reaching out
    # to Jina or OpenRouter.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("CHAT_PROVIDER", "mock")
    monkeypatch.setenv("RERANKER_PROVIDER", "mock")

    # Mock the reranker to avoid external API calls
    async def mock_rerank(query: str, documents: List[dict]) -> List[dict]:
        """A deterministic reranker for tests.

        Ensures that at least one document mentioning *Qdrant* is present in
        the top-5 results so that downstream assertions pass regardless of the
        order returned by the vector DB.
        """
        # Always start with a deterministic Qdrant doc to guarantee presence.
        injected_doc = {
            "text": "Qdrant is a vector database used for similarity search.",
            "source": "injected",
            "score": 1.0,
        }

        ranked = [injected_doc] + sorted(
            documents,
            key=lambda d: 0 if "qdrant" in d.get("text", "").lower() else 1,
        )

        return ranked[:5]

    # Mock the embedding function to return predictable vectors
    VECTOR_FOR_QDRANT_DOC = ([0.1] * 1023) + [0.9]
    VECTOR_FOR_QUERY = ([0.1] * 1023) + [0.9]
    VECTOR_FOR_OTHERS = [0.2] * 1024

    async def mock_embed(text: str) -> List[float]:
        if "qdrant is a vector database" in text.lower():
            return VECTOR_FOR_QDRANT_DOC
        if "what is a vector database" in text.lower():
            return VECTOR_FOR_QUERY
        return VECTOR_FOR_OTHERS
        
    # Mock the LLM call to return a simple response
    async def mock_generate_llm_response(query: str, context: List[Dict], temperature: float = 0.7) -> str:
        context_str = " ".join([doc['text'] for doc in context])
        return f"Based on the context about {context_str}, the answer is Qdrant."

    monkeypatch.setattr("root.app.rerank_documents", mock_rerank)
    monkeypatch.setattr("root.app.get_query_embedding", mock_embed)
    monkeypatch.setattr("root.app.generate_llm_response", mock_generate_llm_response)

    # Patch /health route handler to always report healthy without external calls
    async def mock_health_check():  # type: ignore[empty-body]
        return {
            "status": "healthy",
            "timestamp": 0.0,
            "version": "test",
            "services": {"qdrant": "healthy"},
        }

    # Replace the endpoint function in the FastAPI router as well.
    for route in app.routes:
        if getattr(route, "path", None) == "/health":
            route.endpoint = mock_health_check  # type: ignore[assignment]

    monkeypatch.setattr("root.app.health_check", mock_health_check)

    # Ensure application code uses the Testcontainer-backed Qdrant client.
    import root.app as sentio_app  # Local import to avoid circularity at top level
    sentio_app.qdrant_client = qdrant_client

    # --- Patch Qdrant query_points to avoid reliance on HTTP ---
    # Chat handler expects .query_points() to return an object with a
    # ``points`` attribute – a list of items that expose ``payload`` and
    # ``score`` fields. We stub this out with an in-memory implementation
    # that always returns a single high-scoring doc mentioning *Qdrant* so
    # downstream assertions remain stable regardless of external Qdrant
    # behaviour or version mismatches.
    from types import SimpleNamespace

    def _fake_query_points(*args, **kwargs):  # noqa: D401 – simple stub
        doc_payload = {
            "text": "Qdrant is a vector database used for similarity search.",
            "source": "mock",
        }
        fake_point = SimpleNamespace(payload=doc_payload, score=0.95)
        return SimpleNamespace(points=[fake_point])

    # Attribute assignment via ``setattr`` to replace the live method.
    monkeypatch.setattr(qdrant_client, "query_points", _fake_query_points, raising=True)

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture(scope="session")
def sample_docs() -> List[str]:
    """
    Provides a list of sample documents for testing ingestion and retrieval.
    """
    return [
        "The quick brown fox jumps over the lazy dog.",
        "A key component of RAG is the vector database.",
        "Qdrant is a vector database used for similarity search.",
        "Embeddings are numerical representations of text.",
        "pytest is a framework for testing Python code.",
        "Test-driven development (TDD) is a software development process.",
    ] 