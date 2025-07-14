"""RAGAS-based evaluation for Retrieval-Augmented Generation.

This module provides quality assessment for RAG systems using the RAGAS
framework (Retrieval Augmented Generation Assessment). It offers metrics to
evaluate faithfulness, answer relevancy, and context relevancy without requiring
ground-truth answers. A fallback LLM judge is used when RAGAS is unavailable.
"""




# ==== MODULE IMPORTS & SETUP ==== #
# --► IMPORT STANDARD & THIRD-PARTY LIBRARIES

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ⚠️ OPTIONAL DEPENDENCIES
HAS_RAGAS: bool = False

try:
    from ragas.metrics import (
        answer_relevancy,
        context_relevancy,
        faithfulness,
    )  # type: ignore

    HAS_RAGAS = True
    logging.getLogger(__name__).info(
        "✓ RAGAS metrics initialised successfully"
    )
except ImportError:
    logging.getLogger(__name__).warning(
        "RAGAS not available – install with `pip install ragas`"
    )




# ==== CONFIGURATION THRESHOLDS ==== #

FAITHFULNESS_THRESHOLD: float = float(
    os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", "0.5")
)
ANSWER_RELEVANCY_THRESHOLD: float = float(
    os.getenv("RAGAS_ANSWER_RELEVANCY_THRESHOLD", "0.6")
)
CONTEXT_RELEVANCY_THRESHOLD: float = float(
    os.getenv("RAGAS_CONTEXT_RELEVANCY_THRESHOLD", "0.7")
)

# ---------------------------------------------------------------------------
# Centralised threshold lookup & model defaults
# ---------------------------------------------------------------------------

THRESHOLDS: Dict[str, float] = {
    "faithfulness": FAITHFULNESS_THRESHOLD,
    "answer_relevancy": ANSWER_RELEVANCY_THRESHOLD,
    "context_relevancy": CONTEXT_RELEVANCY_THRESHOLD,
}

DEFAULT_OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3.5:3.8b")




# ==== CORE EVALUATOR CLASS ==== #
# --► RAGEvaluator: Evaluate RAG outputs & retry logic


class RAGEvaluator:
    """
    Evaluate RAG outputs and optionally retry on low-quality answers.

    Core processing for faithfulness, relevancy, and context metrics.
    """

    def __init__(
        self,
        use_llm_judge: bool = False,
        llm_provider: Optional[str] = None,
    ) -> None:
        """
        Initialise the evaluator.

        Args:
            use_llm_judge (bool): Enable LLM judge fallback when RAGAS is unavailable.
            llm_provider (Optional[str]): Provider id for the LLM judge (default: 'ollama').

        Preconditions:
            If use_llm_judge is True, environment variable ENABLE_LLM_JUDGE must be '1'.

        Postconditions:
            Sets self.metrics based on available evaluation methods.
        """
        self.use_llm_judge: bool = bool(use_llm_judge) and os.getenv(
            "ENABLE_LLM_JUDGE", "0"
        ) == "1"

        self.llm_provider: str = llm_provider or os.getenv(
            "LLM_PROVIDER", "ollama"
        )
        self.metrics: Dict[str, Any] = {}

        if HAS_RAGAS:
            self.metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_relevancy": context_relevancy,
            }
        elif self.use_llm_judge:
            logging.getLogger(__name__).info(
                "Using LLM judge as RAGAS alternative"
            )
        else:
            logging.getLogger(__name__).warning(
                "No evaluation method available (RAGAS + LLM judge disabled)"
            )




    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for an answer given contexts.

        Args:
            query (str): The user query string.
            answer (str): Generated answer text.
            contexts (List[str]): Context passages used for retrieval.
            metrics (Optional[List[str]]): List of metric names to compute.

        Returns:
            Dict[str, float]: Mapping of metric names to computed scores.
        """
        selected_metrics = metrics or [
            "faithfulness",
            "answer_relevancy",
            "context_relevancy",
        ]
        results: Dict[str, float] = {}

        if HAS_RAGAS:
            try:
                data = {
                    "question": [query],
                    "answer": [answer],
                    "contexts": [contexts],
                }

                for name in selected_metrics:
                    if name in self.metrics:
                        score = self.metrics[name].score(data)

                        if hasattr(score, "mean"):
                            results[name] = float(score.mean())
                        else:
                            results[name] = float(score)

                        logging.getLogger(__name__).debug(
                            "RAGAS %s score: %.3f", name, results[name]
                        )

            except Exception as exc:
                logging.getLogger(__name__).error(
                    "RAGAS evaluation failed: %s", exc
                )

                if self.use_llm_judge:
                    return self._llm_judge_evaluation(
                        query, answer, contexts, selected_metrics
                    )

        elif self.use_llm_judge:
            return self._llm_judge_evaluation(
                query, answer, contexts, selected_metrics
            )

        return results




    def is_answer_reliable(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Determine whether an answer passes quality thresholds.

        Args:
            query (str): The user query string.
            answer (str): Generated answer text.
            contexts (List[str]): Context passages used for retrieval.

        Returns:
            Tuple[bool, Dict[str, float]]: (
                reliability flag,
                mapping of metric names to scores
            )
        """
        metrics = self.evaluate(query, answer, contexts)

        # Identify metrics falling below their configured thresholds
        low_metrics: Dict[str, float] = {
            name: score
            for name, score in metrics.items()
            if score < THRESHOLDS.get(name, 0.0)
        }

        for name, score in low_metrics.items():
            logger.warning(
                "Low %s: %.3f (threshold %.3f)",
                name,
                score,
                THRESHOLDS[name],
            )

        return not low_metrics, metrics




    def answer_with_retry(
        self,
        query: str,
        answer_fn: Callable[[str], Tuple[str, List[str]]],
        max_attempts: int = 2,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Generate answer and retry with a rewritten query when thresholds are not met.

        Args:
            query (str): The initial query string.
            answer_fn (Callable[[str], Tuple[str, List[str]]]): Function that
                generates answer and context list.
            max_attempts (int): Maximum retry attempts.

        Returns:
            Tuple[str, Dict[str, float]]: (final answer, last attempt metrics)
        """
        current_query = query

        for attempt in range(max_attempts):
            answer, contexts = answer_fn(current_query)

            reliable, metrics = self.is_answer_reliable(
                current_query, answer, contexts
            )

            if reliable or attempt == max_attempts - 1:
                return answer, metrics

            try:
                from ..query_expansion.rewriter import rewrite_query

                rewritten = rewrite_query(current_query)
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Query rewrite failed: %s", exc
                )
                rewritten = None

            if not rewritten or rewritten == current_query:
                return answer, metrics

            current_query = rewritten

        return answer, metrics




    def _llm_judge_evaluation(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        metrics: List[str],
    ) -> Dict[str, float]:
        """
        Fallback evaluation using a lightweight local LLM judge.

        Args:
            query (str): Query string.
            answer (str): Generated answer text.
            contexts (List[str]): Context passages.
            metrics (List[str]): Metric names to compute.

        Returns:
            Dict[str, float]: Mapping of metrics to LLM judge scores (0.0-1.0).

        Assumptions:
            LLM provider must be 'ollama'; uses OLLAMA_URL and OLLAMA_MODEL env vars.
        """
        if self.llm_provider != "ollama":
            logging.getLogger(__name__).warning(
                "LLM provider %s not supported", self.llm_provider
            )
            return {}

        import requests

        ollama_url = DEFAULT_OLLAMA_URL

        model = DEFAULT_OLLAMA_MODEL

        max_ctx = min(3, len(contexts))
        context_text = "\n\n".join(contexts[:max_ctx])

        results: Dict[str, float] = {}

        for name in metrics:
            prompt = self._build_judge_prompt(
                name, query, answer, context_text
            )

            try:
                resp = requests.post(
                    f"{ollama_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=10,
                )
                resp.raise_for_status()

                score = self._extract_score(
                    resp.json().get("response", "")
                )
                results[name] = score
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "LLM judge error: %s", exc
                )
                results[name] = 0.5  # neutral

        return results




    @staticmethod
    def _build_judge_prompt(
        metric: str,
        query: str,
        answer: str,
        context: str,
    ) -> str:
        """
        Construct prompt for LLM judge based on metric type.

        Args:
            metric (str): Metric name ('faithfulness', 'answer_relevancy', etc.).
            query (str): Query text.
            answer (str): Answer text (if applicable).
            context (str): Concatenated context text.

        Returns:
            str: Formatted prompt string for LLM judge.
        """
        if metric == "faithfulness":
            return (
                "Rate how factual the answer is w.r.t. the context on a 0-10 "
                f"scale.\nQUESTION: {query}\nCONTEXT: {context}\n"
                f"ANSWER: {answer}\nScore:"
            )

        if metric == "answer_relevancy":
            return (
                "Rate how relevant the answer is to the question on a 0-10 "
                f"scale.\nQUESTION: {query}\nANSWER: {answer}\nScore:"
            )

        if metric == "context_relevancy":
            return (
                "Rate how relevant the context is to the question on a 0-10 "
                f"scale.\nQUESTION: {query}\nCONTEXT: {context}\nScore:"
            )

        return (
            "Provide a quality score (0-10).\nQUESTION: {query}\nANSWER: "
            f"{answer}\nScore:"
        )




    @staticmethod
    def _extract_score(text: str) -> float:
        """
        Extract first numeric token and normalize to 0-1.

        Args:
            text (str): Response text containing a numeric score (0-10 scale).

        Returns:
            float: Normalized score between 0.0 and 1.0.
        """
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return 0.5

        val = float(match.group(1))
        return min(max(val / 10.0, 0.0), 1.0)




# ==== PLUGIN DEFINITION ==== #
# --► RAGASPlugin for Sentio integration


class RAGASPlugin(SentioPlugin):
    """
    Plugin providing RAGAS evaluation capabilities.
    """

    name: str = "ragas_eval"
    plugin_type: str = "evaluator"

    def __init__(self, use_llm_judge: bool = False) -> None:
        """
        Initialise plugin instance.

        Args:
            use_llm_judge (bool): Enable LLM judge fallback.
        """
        self.evaluator: RAGEvaluator = RAGEvaluator(
            use_llm_judge=use_llm_judge
        )




    def register(self, pipeline: Any) -> None:
        """
        Register evaluator with a processing pipeline.

        Args:
            pipeline (Any): Pipeline object to attach evaluator to.
        """
        pipeline.evaluator = self.evaluator




# --► PLUGIN FACTORY


def get_plugin() -> SentioPlugin:
    """
    Return a new instance of RAGASPlugin.

    Returns:
        SentioPlugin: A new plugin instance.
    """
    return RAGASPlugin()
