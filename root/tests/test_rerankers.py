import types
from typing import List, Dict

import pytest

# ---------------------------------------------------------------------------
# CrossEncoderReranker – primary, secondary-local & secondary-Jina paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secondary_type,secondary_model",
    [
        (None, None),  # primary only
        ("local", "dummy-sec"),  # primary + secondary local
        ("jina", "dummy-jina"),  # primary + secondary jina
    ],
)
def test_cross_encoder_reranker_paths(secondary_type, secondary_model):  # noqa: D401
    """Exercise the main rerank() code paths of *CrossEncoderReranker*.

    The heavy model classes are *stubbed globally* in the session-level
    fixture ``_stub_external_heavy_deps`` defined in *conftest.py* so that we
    can instantiate the reranker without downloading weights or hitting the
    network.
    """

    from root.src.core.rerankers.providers.cross_encoder import CrossEncoderReranker

    docs: List[Dict[str, str]] = [
        {"text": "a"},
        {"text": "bb"},
        {"text": "ccc"},
    ]

    # Inject a stub *TextCrossEncoder* so the constructor cannot attempt to
    # load real model weights (which would require large downloads).
    from root.src.core.rerankers.providers import cross_encoder as ce_mod

    class _StubTextCrossEncoder:  # noqa: D401 – minimal scorer
        def __init__(self, *args, **kwargs):
            pass

        def rerank(self, query: str, docs: List[str]):  # noqa: D401, ANN001
            # Simple deterministic score proportional to text length.
            return [float(len(t)) for t in docs]

    ce_mod.TextCrossEncoder = _StubTextCrossEncoder  # type: ignore[attr-defined]

    # When secondary == "jina" the class internally constructs *JinaReranker*;
    # we monkey-patch it to a lightweight scorer so the network is never used.
    if secondary_type == "jina":
        from root.src.core.rerankers.providers import jina_reranker as jina_mod

        class _StubJinaReranker:  # noqa: D401 – minimal API
            def __init__(self, *args, **kwargs):
                pass

            def rerank(self, query: str, docs: List[Dict[str, str]], top_k: int = 5):  # noqa: D401
                # Assign descending scores so we can assert ordering easily.
                scored: List[Dict[str, str]] = []
                for i, d in enumerate(docs):
                    d = d.copy()
                    d["rerank_score"] = float(len(d["text"]) + 100)  # huge bump
                    scored.append(d)
                return scored

        # The cross-encoder module imports JinaReranker, so we patch it there.
        ce_mod.JinaReranker = _StubJinaReranker  # type: ignore[attr-defined]

    reranker = CrossEncoderReranker(
        secondary_reranker_type=secondary_type,
        secondary_model=secondary_model,
    )

    ranked = reranker.rerank("query", docs, top_k=2, top_k_intermediate=3)

    # The longest string should end up on top irrespective of the path taken.
    assert ranked[0]["text"] == "ccc"
    assert len(ranked) == 2
    assert all("rerank_score" in d for d in ranked)


# ---------------------------------------------------------------------------
# JinaReranker – success & fallback branches
# ---------------------------------------------------------------------------


def test_jina_reranker_success(monkeypatch):  # noqa: D401
    """JinaReranker returns API scores when the request succeeds."""

    from root.src.core.rerankers.providers.jina_reranker import JinaReranker

    # Fake *requests.post* – returns mocked JSON payload.
    class _FakeResponse:  # noqa: D401 – tiny stub
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.8},
                ]
            }

    monkeypatch.setattr(
        "root.src.core.rerankers.providers.jina_reranker.requests.post", lambda *a, **k: _FakeResponse()
    )

    docs = [{"text": "doc0"}, {"text": "doc1"}]
    rr = JinaReranker(api_key="dummy-key", model_name="stub-model")
    ranked = rr.rerank("q", docs, top_k=2)

    # API ranked doc1 above doc0 according to the mocked payload.
    assert ranked[0]["text"] == "doc1"
    assert ranked[0]["rerank_score"] >= ranked[1]["rerank_score"]


def test_jina_reranker_fallback(monkeypatch):  # noqa: D401
    """JinaReranker falls back to heuristic ordering when the request fails."""

    from root.src.core.rerankers.providers.jina_reranker import JinaReranker
    import requests

    # Force *requests.post* to raise so we hit the fallback branch.
    monkeypatch.setattr(
        "root.src.core.rerankers.providers.jina_reranker.requests.post",  # noqa: S301
        lambda *a, **k: (_ for _ in ()).throw(requests.RequestException("boom")),
    )

    docs = [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}]
    rr = JinaReranker(api_key="dummy-key", model_name="stub-model")
    ranked = rr.rerank("q", docs, top_k=3)

    # Fallback assigns synthetic scores; ensure we still have them.
    assert len(ranked) == 3
    assert all("rerank_score" in d for d in ranked) 