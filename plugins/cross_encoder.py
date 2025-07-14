"""Cross-encoder-based multi-stage reranker.

This module implements a *cascade* reranker following the recommendations
outlined in `plan_ai.md` (Iteration 2).  It performs three optional stages:

1. **Fast coarse ranking** – a lightweight MiniLM cross-encoder quickly
   scores *all* candidate passages.
2. **High-quality re-ranking** – the strongest open-weight model available
   (`BAAI/bge-reranker-base`) rescoring the top-*k* passages from stage 1.
   Alternatively, can use Jina AI's reranker API for superior performance.
3. **LLM judge** (conditional) – a compact local LLM (e.g. Phi-3-mini via
   Ollama) provides an additional relevance score for the final top-N.

Each stage can be toggled via environment variables so that the pipeline can
adapt to resource constraints at runtime.

```text
Primary model     : FAST_RERANK_MODEL  env→RERANKER_MODEL        (default: MiniLM)
Secondary model   : BGE_RERANK_MODEL   env→SECONDARY_RERANK_MODEL (default: BGE)
Secondary type    : SECONDARY_RERANKER_TYPE (default: local, can be 'local' or 'jina')
LLM judge enabled : env→ENABLE_LLM_JUDGE=1 (default: off)
```

The public API matches the previous implementation so downstream code does
**not** need to change.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Union

from plugins.interface import SentioPlugin

from sentence_transformers import CrossEncoder
from ..root.src.core.rerankers.jina_reranker import JinaReranker

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

# Fast model suitable for CPU ranking of hundreds of passages/second 
FAST_RERANK_MODEL = "ibm-granite/granite-embedding-107m-multilingual"

# Strong 500 M model – noticeably better quality, acceptable latency on Mac M1
BGE_RERANK_MODEL = "Alibaba-NLP/gte-multilingual-base"

# Default Jina reranker model - high quality multilingual model available via API
JINA_RERANK_MODEL = "jina-reranker-m0"

# Back-compat alias so old env vars still work
DEFAULT_RERANK_MODEL = FAST_RERANK_MODEL

# Reranker types
LOCAL_RERANKER = "local"
JINA_RERANKER = "jina"


# ---------------------------------------------------------------------------
# Reranker implementation
# ---------------------------------------------------------------------------


class CrossEncoderReranker:  # noqa: D101 – concise class
    """Cascade cross-encoder reranker."""

    def __init__(
        self,
        model_name: str | None = None,
        secondary_model: str | None = None,
        secondary_reranker_type: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialise reranker.

        Args:
            model_name: Primary (fast) cross-encoder.  If *None*, the value of
                the *RERANKER_MODEL* environment variable is used, falling back
                to :data:`FAST_RERANK_MODEL`.
            secondary_model: Optional secondary (high-quality) cross-encoder.
                If *None*, the *SECONDARY_RERANK_MODEL* env var is consulted
                then :data:`BGE_RERANK_MODEL` is used for local reranker, or
                :data:`JINA_RERANK_MODEL` for Jina reranker.
            secondary_reranker_type: Type of secondary reranker to use ('local' or 'jina').
                If *None*, the value of *SECONDARY_RERANKER_TYPE* env var is used, 
                defaulting to 'local'.
            device: Execution device (``cpu``/``cuda``).  Defaults to automatic
                selection performed by *sentence-transformers*.
        """

        # ---------------- Primary model ----------------
        if model_name is None:
            model_name = os.getenv("RERANKER_MODEL", DEFAULT_RERANK_MODEL)
        self.model_name: str = model_name
        self.model = CrossEncoder(model_name, device=device)

        # ---------------- Secondary reranker type ----------------
        if secondary_reranker_type is None:
            secondary_reranker_type = os.getenv("SECONDARY_RERANKER_TYPE", LOCAL_RERANKER)
        
        if secondary_reranker_type not in (LOCAL_RERANKER, JINA_RERANKER):
            raise ValueError(
                f"Invalid secondary_reranker_type: {secondary_reranker_type}. "
                f"Must be '{LOCAL_RERANKER}' or '{JINA_RERANKER}'."
            )
        
        self.secondary_reranker_type: str = secondary_reranker_type

        # ---------------- Secondary model ----------------
        self.secondary_model_name: Optional[str] = None
        self.secondary_model = None
        
        if secondary_model is None:
            if self.secondary_reranker_type == LOCAL_RERANKER:
                secondary_model = os.getenv("SECONDARY_RERANK_MODEL", BGE_RERANK_MODEL)
            else:  # JINA_RERANKER
                secondary_model = os.getenv("RERANKER_MODEL", JINA_RERANK_MODEL)
        
        if secondary_model and secondary_model.lower() != "none":
            self.secondary_model_name = secondary_model
            
            # Initialize the appropriate secondary reranker
            if self.secondary_reranker_type == LOCAL_RERANKER:
                # Loading the model is cheap – initialise eagerly to fail fast if
                # weights are missing.
                self.secondary_model = CrossEncoder(secondary_model, device=device)
            else:  # JINA_RERANKER
                # Initialize Jina reranker with the specified model
                self.secondary_model = JinaReranker(model_name=secondary_model)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        *,
        top_k: int = 5,
        top_k_intermediate: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return the *top_k* most relevant ``docs`` for *query*.

        The method performs:

        1. Score *all* docs with the primary model.
        2. If a secondary model is configured, rescore the ``top_k_intermediate``
           docs returned by stage 1.
        """

        if not docs:
            return []

        # --------------- Stage 1 – fast coarse ranking ---------------
        pairs = [(query, d.get("text", "")) for d in docs]
        scores = self.model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)

        ranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)

        # --------------- Stage 2 – high-quality rescoring ---------------
        if self.secondary_model is not None and len(ranked) > 1:
            top_docs = ranked[: min(top_k_intermediate, len(ranked))]
            
            if self.secondary_reranker_type == LOCAL_RERANKER:
                # Local reranker (CrossEncoder)
                sec_pairs = [(query, d.get("text", "")) for d in top_docs]
                sec_scores = self.secondary_model.predict(sec_pairs)
                
                for d, s in zip(top_docs, sec_scores):
                    d["rerank_score_primary"] = d["rerank_score"]  # preserve
                    d["rerank_score"] = float(s)
            
            else:  # JINA_RERANKER
                # Jina reranker API
                reranked_docs = self.secondary_model.rerank(
                    query=query, 
                    docs=top_docs, 
                    top_k=len(top_docs)
                )
                
                # Map back scores to original docs to preserve other fields
                scored_docs_map = {
                    id(doc): doc for doc in reranked_docs
                }
                
                for doc in top_docs:
                    doc["rerank_score_primary"] = doc["rerank_score"]  # preserve
                    # If the doc was scored by Jina, use that score
                    if id(doc) in scored_docs_map:
                        doc["rerank_score"] = scored_docs_map[id(doc)]["rerank_score"]

            # resort the rescored subset, keep tail unchanged
            rescored = sorted(top_docs, key=lambda d: d["rerank_score"], reverse=True)
            ranked = rescored + ranked[len(top_docs) :]

        return ranked[:top_k]


class CrossEncoderPlugin(SentioPlugin):
    """Plugin wrapper for cross-encoder reranker."""

    name = "cross_encoder"
    plugin_type = "reranker"

    def __init__(self, **kwargs: Any) -> None:
        self.reranker = CrossEncoderReranker(**kwargs)

    def register(self, pipeline: Any) -> None:
        pipeline.reranker = self.reranker


def get_plugin() -> SentioPlugin:
    """Return plugin instance."""
    return CrossEncoderPlugin()
