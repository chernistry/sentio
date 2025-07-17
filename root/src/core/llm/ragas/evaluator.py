"""RAGAS evaluator implementation for RAG quality assessment.

This module contains the core RAGEvaluator class that handles evaluation
of RAG outputs using either the RAGAS library or fallback LLM-based evaluation.
"""

from __future__ import annotations

import logging
import os
import re
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import aiofiles  # Добавляем импорт aiofiles

from root.src.core.llm.chat_adapter import chat_completion
from root.src.utils.settings import settings
from root.src.core.llm.llm_reply_extractor import extract_json_dict_sync

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

# ---------------------------------------------------------------------------
# Centralised threshold lookup & model defaults
# ---------------------------------------------------------------------------

THRESHOLDS: Dict[str, float] = {
    "faithfulness": settings.ragas_faithfulness_threshold,
    "answer_relevancy": settings.ragas_answer_relevancy_threshold,
    "context_relevancy": settings.ragas_context_relevancy_threshold,
}

DEFAULT_OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3.5:3.8b")


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
        self.ragas_prompt: Optional[str] = None
        
        # Get RAGAS model from settings
        self.ragas_model = settings.ragas_model

        try:
            loop = asyncio.get_running_loop()
            # If the loop is already running, schedule the async init
            # otherwise, run it to completion.
            if loop.is_running():
                loop.create_task(self._init_async())
            else:
                loop.run_until_complete(self._init_async())
        except RuntimeError:
            # No running loop, so create a new one
            asyncio.run(self._init_async())

    async def _init_async(self) -> None:
        """Asynchronously initialize the evaluator."""
        self.ragas_prompt = await self._load_ragas_prompt()

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

    async def _load_ragas_prompt(self) -> str:
        """Load RAGAS prompt template from file asynchronously."""
        try:
            # Используем asyncio.to_thread для неблокирующего чтения файла
            async with aiofiles.open(self.ragas_prompt_path, "r") as f:
                return await f.read()
        except (FileNotFoundError, IOError, ImportError):
            # Если aiofiles не установлен или возникла другая ошибка
            logger.warning(f"RAGAS prompt file not found at {self.ragas_prompt_path}, using default prompt")
            return """
            You are an expert evaluator of Retrieval-Augmented Generation (RAG) answers.

            ## Objective
            Return three quality scores between **0.0** and **1.0** (inclusive):

            1. "faithfulness" – factual consistency of ANSWER with CONTEXT.  
            2. "answer_relevancy" – how well ANSWER addresses QUERY.  
            3. "context_relevancy" – usefulness of CONTEXT for answering QUERY.

            If CONTEXT is empty or clearly unrelated, set "context_relevancy" ≤ 0.20 and cap "faithfulness" at the same value.  
            If ANSWER is empty, output **0.0** for all metrics.

            ## Input
            QUERY: {query}

            CONTEXT: {context}

            ANSWER: {answer}

            ## Output
            Respond **only** with valid UTF-8 JSON, no extra text:

            ```json
            {
              "faithfulness": <float>,
              "answer_relevancy": <float>,
              "context_relevancy": <float>
            }
            ```
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
            "timestamp": time.time()
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
                from root.src.core.query_expansion.rewriter import rewrite_query

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
            # Check if we have contexts and answer
            if not contexts:
                logger.warning("Empty contexts provided for evaluation")
            if not answer.strip():
                logger.warning("Empty answer provided for evaluation")
                
            context_text = "\n\n".join(contexts)
            
            # Format the prompt with query, answer, and context
            prompt = self.ragas_prompt.format(
                query=query,
                answer=answer,
                context=context_text
            )
            
            # Prepare the payload for the chat_completion function
            # NOTE: Some models (including many served via OpenRouter) do **not** yet
            # support the official OpenAI `response_format` parameter and will raise
            # a 400 validation error when it is present.  This previously caused the
            # call to always fail and triggered the fallback branch that returns a
            # neutral 0.5 score for every metric.  We therefore omit the parameter
            # completely and rely on prompt-based JSON forcing instead.

            # NOTE: capturing raw response and potential parsing errors for debugging
            raw_response = None
            raw_text = None
            parse_error = None
            payload = {
                "model": self.ragas_model,
                "messages": [
                    {"role": "system", "content": "You are an expert RAG system evaluator. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "stream": False,
            }
            
            # Log the request payload and model being used
            logger.info(f"RAGAS evaluation using model: {self.ragas_model}")
            logger.info(f"RAGAS evaluation API key prefix: {settings.chat_llm_api_key[:8] if settings.chat_llm_api_key else 'None'}")
            logger.info(f"RAGAS evaluation base URL: {settings.chat_llm_base_url}")
            logger.debug(f"RAGAS evaluation payload: {payload}")
            
            # Call the chat_completion function
            logger.info("Sending request to OpenRouter API for RAGAS evaluation")
            response = await chat_completion(payload)
            raw_response = response
            
            # Log the raw response for debugging
            if isinstance(response, dict):
                logger.info(f"RAGAS evaluation response type: dict with keys {list(response.keys())}")
            else:
                logger.info(f"RAGAS evaluation response type: {type(response)}")
            
            logger.debug(f"RAGAS evaluation raw response: {response}")
            
            if isinstance(response, dict) and "choices" in response:
                response_text = response["choices"][0]["message"]["content"]
                raw_text = response_text
                logger.info(f"RAGAS evaluation response text: {response_text[:200]}...")
                
                # Try to parse JSON from the response
                try:
                    # First try using the robust JSON extractor
                    metrics_data = extract_json_dict_sync(response_text)
                    
                    if metrics_data:
                        logger.info(f"Successfully extracted JSON using llm_reply_extractor: {metrics_data}")
                    else:
                        # Extract JSON from the response text which may contain markdown code blocks
                        json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response_text)
                        if json_match:
                            json_str = json_match.group(1)
                            logger.info(f"Extracted JSON from code block: {json_str}")
                            metrics_data = json.loads(json_str)
                        else:
                            # If no code block, try to find JSON directly
                            json_start = response_text.find("{")
                            json_end = response_text.rfind("}") + 1
                            if json_start >= 0 and json_end > json_start:
                                json_str = response_text[json_start:json_end]
                                logger.info(f"Extracted JSON string: {json_str}")
                                metrics_data = json.loads(json_str)
                            else:
                                # Try one more approach - look for numbers in the text
                                logger.info("No JSON found, trying to extract scores directly from text")
                                metrics_data = {}
                                
                                # Look for metrics in text format
                                for metric_name in metrics:
                                    pattern = rf"{metric_name}[:\s=]+(\d+(?:\.\d+)?)"
                                    match = re.search(pattern, response_text.lower(), re.IGNORECASE)
                                    if match:
                                        metrics_data[metric_name] = float(match.group(1))
                                        logger.info(f"Found {metric_name} score: {metrics_data[metric_name]}")
                                
                                if not metrics_data:
                                    raise ValueError("No JSON or metrics found in response")
                    
                    logger.info(f"Parsed metrics data: {metrics_data}")
                    
                    # ------------------------------------------------------------------
                    # Normalise keys (strip non-alpha chars & lower-case) so that minor
                    # spelling variations like "answer relevancy" or "answer_relevancy"
                    # still map correctly.  Afterwards, coerce values to the 0-1 range.
                    # ------------------------------------------------------------------

                    norm_map = {
                        re.sub(r"[^a-z]", "", k.lower()): v for k, v in metrics_data.items()
                    }

                    results = {}
                    for metric_name in metrics:
                        norm_key = re.sub(r"[^a-z]", "", metric_name.lower())
                        if norm_key in norm_map:
                            value = float(norm_map[norm_key])
                            # Convert 0-10 scale → 0-1 if needed
                            if value > 1.0:
                                value = value / 10.0 if value <= 10.0 else 1.0
                            results[metric_name] = value
                        else:
                            logger.warning(
                                "Metric %s not found in response (keys=%s)",
                                metric_name,
                                list(metrics_data.keys()),
                            )
                            results[metric_name] = 0.5  # Fallback neutral value
                    
                    logger.info(f"Final RAGAS metrics: {results}")
                    
                    # Store evaluation results in history
                    evaluation_entry = {
                        "query": query,
                        "answer": answer,
                        "metrics": results,
                        "timestamp": time.time(),
                        "method": "openrouter",
                        "raw_response": raw_response,
                        "raw_text": raw_text,
                        "parse_error": parse_error,
                    }
                    self.evaluation_history.append(evaluation_entry)
                    
                    return results
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse JSON from OpenRouter response: {e}")
                    logger.error(f"Response text that failed to parse: {response_text}")
            else:
                logger.error(f"Unexpected response format from OpenRouter: {response}")
            
            # Fallback to default scores if parsing fails
            logger.warning("OpenRouter evaluation failed, using default scores")
            results = {metric: 0.5 for metric in metrics}
            
            # Store evaluation results in history
            evaluation_entry = {
                "query": query,
                "answer": answer,
                "metrics": results,
                "timestamp": time.time(),
                "method": "openrouter_fallback",
                "raw_response": raw_response,
                "raw_text": raw_text,
                "parse_error": parse_error,
            }
            self.evaluation_history.append(evaluation_entry)
            
            return results
        except Exception as e:
            logger.error(f"OpenRouter evaluation error: {e}", exc_info=True)
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
            "timestamp": time.time(),
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
        
    async def test_openrouter_evaluation(self, query: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Test method to directly evaluate using OpenRouter with detailed debugging.
        
        Args:
            query: User query
            answer: Generated answer
            contexts: List of context passages
            
        Returns:
            Dict with full response details for debugging
        """
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
        
        # Log everything
        logger.info(f"TEST EVALUATION - Model: {self.ragas_model}")
        logger.info(f"TEST EVALUATION - API Key: {settings.chat_llm_api_key[:8] if settings.chat_llm_api_key else 'None'}")
        logger.info(f"TEST EVALUATION - Base URL: {settings.chat_llm_base_url}")
        logger.info(f"TEST EVALUATION - Prompt: {prompt[:200]}...")
        
        # Direct call to OpenRouter API
        import httpx
        
        headers = {
            "Authorization": f"Bearer {settings.chat_llm_api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            async with httpx.AsyncClient(base_url=settings.chat_llm_base_url, timeout=60.0) as client:
                logger.info("TEST EVALUATION - Sending direct request to OpenRouter API")
                response = await client.post("/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"TEST EVALUATION - Response status: {response.status_code}")
                logger.info(f"TEST EVALUATION - Response headers: {response.headers}")
                logger.info(f"TEST EVALUATION - Response body: {result}")
                
                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    logger.info(f"TEST EVALUATION - Content: {content[:200]}...")
                    
                    # Try to extract metrics
                    metrics_data = extract_json_dict_sync(content)
                    if metrics_data:
                        logger.info(f"TEST EVALUATION - Extracted metrics: {metrics_data}")
                    else:
                        logger.warning("TEST EVALUATION - Failed to extract metrics from response")
                
                return {
                    "status": "success",
                    "raw_response": result,
                    "extracted_metrics": metrics_data if 'metrics_data' in locals() else None
                }
                
        except Exception as e:
            logger.error(f"TEST EVALUATION - Error: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            } 