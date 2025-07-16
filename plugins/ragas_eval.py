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
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from root.src.core.llm.chat_adapter import chat_completion
from root.src.utils.settings import settings

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

FAITHFULNESS_THRESHOLD: float = settings.ragas_faithfulness_threshold
ANSWER_RELEVANCY_THRESHOLD: float = settings.ragas_answer_relevancy_threshold
CONTEXT_RELEVANCY_THRESHOLD: float = settings.ragas_context_relevancy_threshold

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
        self.use_llm_judge: bool = bool(use_llm_judge) or settings.enable_llm_judge

        self.llm_provider: str = llm_provider or settings.ragas_provider
        self.metrics: Dict[str, Any] = {}
        # Store evaluation history
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # Load RAGAS prompt template
        self.ragas_prompt_path = settings.ragas_prompt
        self.ragas_prompt = self._load_ragas_prompt()
        
        # Get RAGAS model from settings
        self.ragas_model = settings.ragas_model

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
    
    def _load_ragas_prompt(self) -> str:
        """Load RAGAS prompt template from file."""
        try:
            with open(self.ragas_prompt_path, "r") as f:
                return f.read().strip()
        except (FileNotFoundError, IOError):
            logger.warning(f"RAGAS prompt file not found at {self.ragas_prompt_path}, using default prompt")
            return """
            You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
            
            Evaluate the following answer based on the provided context and query.
            
            Return a JSON object with the following metrics, each scored from 0.0 to 1.0:
            - faithfulness: How factually consistent the answer is with the provided context
            - answer_relevancy: How relevant the answer is to the query
            - context_relevancy: How relevant the provided context is to the query
            
            Each score should be between 0.0 (worst) and 1.0 (best).
            
            QUERY: {query}
            
            CONTEXT: {context}
            
            ANSWER: {answer}
            
            EVALUATION (return only valid JSON):
            """




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
                elif settings.ragas_provider == "openrouter":
                    return self._openrouter_evaluation(
                        query, answer, contexts, selected_metrics
                    )

        elif self.use_llm_judge:
            return self._llm_judge_evaluation(
                query, answer, contexts, selected_metrics
            )
        elif settings.ragas_provider == "openrouter":
            return self._openrouter_evaluation(
                query, answer, contexts, selected_metrics
            )

        # Store evaluation results in history
        evaluation_entry = {
            "query": query,
            "answer": answer,
            "metrics": results,
            "timestamp": import_time_and_get_current_time()
        }
        self.evaluation_history.append(evaluation_entry)

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


    async def _openrouter_evaluation(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        metrics: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate using OpenRouter LLM via chat_adapter.

        Args:
            query (str): Query string.
            answer (str): Generated answer text.
            contexts (List[str]): Context passages.
            metrics (List[str]): Metric names to compute.

        Returns:
            Dict[str, float]: Mapping of metrics to scores (0.0-1.0).
        """
        try:
            context_text = "\n\n".join(contexts)
            
            # Format the prompt with query, answer, and context
            prompt = self.ragas_prompt.format(
                query=query,
                answer=answer,
                context=context_text
            )
            
            # Prepare the payload for the chat_completion function
            payload = {
                "model": self.ragas_model,
                "messages": [
                    {"role": "system", "content": "You are an expert RAG system evaluator."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "stream": False,
            }
            
            # Call the chat_completion function
            response = await chat_completion(payload)
            
            if isinstance(response, dict) and "choices" in response:
                response_text = response["choices"][0]["message"]["content"]
                
                # Try to parse JSON from the response
                try:
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        metrics_data = json.loads(json_str)
                        
                        # Extract metrics from the parsed JSON
                        results = {}
                        for metric_name in metrics:
                            if metric_name in metrics_data:
                                results[metric_name] = float(metrics_data[metric_name])
                        
                        # Store evaluation results in history
                        evaluation_entry = {
                            "query": query,
                            "answer": answer,
                            "metrics": results,
                            "timestamp": import_time_and_get_current_time(),
                            "method": "openrouter"
                        }
                        self.evaluation_history.append(evaluation_entry)
                        
                        return results
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse JSON from OpenRouter response: {e}")
            
            # Fallback to default scores if parsing fails
            logger.warning("OpenRouter evaluation failed, using default scores")
            results = {metric: 0.5 for metric in metrics}
            
            # Store evaluation results in history
            evaluation_entry = {
                "query": query,
                "answer": answer,
                "metrics": results,
                "timestamp": import_time_and_get_current_time(),
                "method": "openrouter_fallback"
            }
            self.evaluation_history.append(evaluation_entry)
            
            return results
        except Exception as e:
            logger.error(f"OpenRouter evaluation error: {e}")
            return {metric: 0.5 for metric in metrics}


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

        # Store evaluation results in history
        evaluation_entry = {
            "query": query,
            "answer": answer,
            "metrics": results,
            "timestamp": import_time_and_get_current_time(),
            "method": "llm_judge"
        }
        self.evaluation_history.append(evaluation_entry)

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


    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of all evaluations performed.

        Returns:
            List[Dict[str, Any]]: List of evaluation entries with metrics
        """
        return self.evaluation_history


    def get_average_metrics(self) -> Dict[str, float]:
        """
        Calculate average scores across all evaluations.

        Returns:
            Dict[str, float]: Average scores for each metric
        """
        if not self.evaluation_history:
            return {}
        
        # Collect all metrics
        all_metrics = {}
        for entry in self.evaluation_history:
            for metric_name, score in entry.get("metrics", {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(score)
        
        # Calculate averages
        return {
            metric_name: sum(scores) / len(scores)
            for metric_name, scores in all_metrics.items()
        }


def import_time_and_get_current_time():
    """Import time module and get current time to avoid circular imports"""
    import time
    return time.time()




# ==== PLUGIN DEFINITION ==== #
# --► RAGASPlugin for Sentio integration


class RAGASPlugin:
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
        
        # Only monkey patch if automatic evaluation is enabled
        if settings.enable_automatic_evaluation:
            # Monkey patch the query method to automatically evaluate answers
            original_query = pipeline.query
            
            async def query_with_evaluation(question: str, top_k: Optional[int] = None) -> Dict:
                # Call the original query method
                result = await original_query(question, top_k)
                
                # Extract the necessary data for evaluation
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                contexts = [source.get("text", "") for source in sources if "text" in source]
                
                # Run evaluation
                metrics = await self.evaluator._openrouter_evaluation(question, answer, contexts, 
                                                               ["faithfulness", "answer_relevancy", "context_relevancy"])
                
                # Add evaluation results to the response
                result["evaluation"] = {
                    "metrics": metrics,
                    "thresholds": THRESHOLDS,
                    "passed_thresholds": all(
                        score >= THRESHOLDS.get(name, 0.0) 
                        for name, score in metrics.items()
                    )
                }
                
                return result
                
            # Replace the original query method
            pipeline.query = query_with_evaluation
        
        # Add methods to get evaluation data
        pipeline.get_evaluation_history = self.evaluator.get_evaluation_history
        pipeline.get_average_metrics = self.evaluator.get_average_metrics




# --► PLUGIN FACTORY


def get_plugin() -> RAGASPlugin:
    """
    Return a new instance of RAGASPlugin.

    Returns:
        RAGASPlugin: A new plugin instance.
    """
    use_llm_judge = settings.enable_llm_judge
    return RAGASPlugin(use_llm_judge=use_llm_judge)
