"""Cross-encoder-based multi-stage reranker.

This module implements a *cascade* reranker following the recommendations
outlined in `plan_ai.md` (Iteration 2).  It performs three optional stages:

1. **Fast coarse ranking** – a lightweight MiniLM cross-encoder quickly
   scores *all* candidate passages.
2. **High-quality re-ranking** – the strongest open-weight model available
   (`BAAI/bge-reranker-base`) rescoring the top-*k* passages from stage 1.
   Alternatively, can use Jina AI's reranker API for superior performance.
3. **LLM judge** (conditional) – a lightweight local LLM can optionally
   provide an additional relevance score for the final top-N.

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

import re
from typing import List, Dict, Any, Optional, Union

from fastembed.rerank.cross_encoder import TextCrossEncoder
from .jina_reranker import JinaReranker
from root.src.utils.settings import settings

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

# Removed hard-coded model constants; models are now configured via environment
# variables and loaded through `settings` in `root.src.utils.settings`.

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
        device: str | None = None, # device is no longer used but kept for compatibility
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
            device: Execution device (``cpu``/``cuda``). This parameter is ignored
                as `fastembed` handles device management automatically.
        """

        # ---------------- Primary model ----------------
        if model_name is None:
            model_name = settings.reranker_model
        self.model_name: str = model_name
        self.model = TextCrossEncoder(model_name)

        # ---------------- Secondary reranker type ----------------
        if secondary_reranker_type is None:
            secondary_reranker_type = settings.secondary_reranker_type

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
                secondary_model = settings.secondary_rerank_model
            else:  # JINA_RERANKER
                secondary_model = settings.reranker_model

        if secondary_model and secondary_model.lower() != "none":
            self.secondary_model_name = secondary_model

            # Initialize the appropriate secondary reranker
            if self.secondary_reranker_type == LOCAL_RERANKER:
                # Loading the model is cheap – initialise eagerly to fail fast if
                # weights are missing.
                self.secondary_model = TextCrossEncoder(secondary_model)
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
        doc_texts = [d.get("text", "") for d in docs]
        scores = self.model.rerank(query, doc_texts)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)

        ranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)

        # --------------- Stage 2 – high-quality rescoring ---------------
        if self.secondary_model is not None and len(ranked) > 1:
            top_docs = ranked[: min(top_k_intermediate, len(ranked))]

            if self.secondary_reranker_type == LOCAL_RERANKER:
                # Local reranker (TextCrossEncoder)
                sec_doc_texts = [d.get("text", "") for d in top_docs]
                sec_scores = self.secondary_model.rerank(query, sec_doc_texts)

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