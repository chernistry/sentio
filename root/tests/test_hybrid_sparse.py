from typing import List, Dict, Any, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from root.src.core.retrievers.hybrid import HybridRetriever


class _StubQdrantPoint:  # noqa: D401 – small container
    def __init__(self, pid: str, text: str):
        self.id = pid
        self.payload = {"text": text}
        self.score = 1.0  # dummy score
        self.vector = np.zeros(4)


class _StubQdrantClient:  # noqa: D401 – minimal subset for HybridRetriever
    """Implements only the methods used by *HybridRetriever*.*"""

    def __init__(self, docs: List[Tuple[str, str]]):
        self._points = [
            _StubQdrantPoint(pid, text) for pid, text in docs
        ]

    # Collection helpers --------------------------------------------------
    def collection_exists(self, collection_name: str):  # noqa: D401
        return False

    def get_collection(self, collection_name: str):  # noqa: D401
        class _Meta:  # noqa: D401 – inner class stub
            points_count = 0

        return _Meta()

    # Search/scroll API ---------------------------------------------------
    def search(self, *_, **__):  # noqa: D401
        # Return empty list – for this test we only care about sparse path.
        return []

    def scroll(self, *_, **__):  # noqa: D401
        # Yield all points in a single batch then signal completion.
        return self._points, None

    def retrieve(self, collection_name: str, ids: List[str], **kwargs):  # noqa: D401
        return [p for p in self._points if p.id in ids]


class _StubEmbedModel:  # noqa: D101 – simple stub
    def embed(self, texts):  # noqa: D401, ANN001 – simplified signature
        return [[0.0] * 4 for _ in texts]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hybrid_retriever_sparse_fallback():  # noqa: D401
    """Ensure _sparse_search uses in-memory BM25 when Pyserini is unavailable."""

    # Prepare two simple docs so the BM25 scorer prefers the one containing the
    # rare term "rag".
    docs = [
        ("d1", "sentio provides rag pipeline"),
        ("d2", "nothing to see here"),
    ]
    client = _StubQdrantClient(docs)
    retriever = HybridRetriever(client=client, embed_model=_StubEmbedModel())

    # Manually inject a BM25 index because `_load_documents` ran and populated
    # `retriever._docs` & `retriever._doc_ids`.
    retriever._bm25 = BM25Okapi([d.split() for _, d in docs])  # type: ignore[attr-defined]
    retriever._doc_ids = [pid for pid, _ in docs]

    hits = retriever._sparse_search("rag", limit=2)
    assert hits and hits[0][0] == "d1", "Doc containing the query term should rank first"


def test_hybrid_id_to_text_cache():  # noqa: D401
    """The *_id_to_text* helper should cache lookups via *lru_cache*."""

    docs = [("p1", "hello world")]
    client = _StubQdrantClient(docs)
    retriever = HybridRetriever(client=client, embed_model=_StubEmbedModel())

    # First call populates the cache.
    text1 = retriever._id_to_text("p1")
    # Mutate underlying data to verify cached value is returned on second call.
    client._points[0].payload["text"] = "CHANGED"
    text2 = retriever._id_to_text("p1")

    assert text1 == text2 == "hello world" 