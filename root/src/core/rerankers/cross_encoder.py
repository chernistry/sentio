from __future__ import annotations

from typing import List, Dict, Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Lightweight wrapper around a `sentence-transformers` cross-encoder.

    The default model is `cross-encoder/ms-marco-MiniLM-L-6-v2`, which is a
    compact yet accurate reranking model widely used for RAG pipelines.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Return reranked `docs` limited to *top_k* by relevance.

        Args:
            query: User query string.
            docs: List of candidate documents (dicts with at least a `text` key).
            top_k: Number of results to return.
        """
        if not docs:
            return []
        pairs = [(query, d.get("text", "")) for d in docs]
        scores = self.model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        ranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k] 