"""RAGAS plugin for integration with the Sentio RAG pipeline.

This module provides a plugin that integrates RAGAS evaluation capabilities
with the Sentio RAG pipeline.
"""

from typing import Any, Dict, List, Optional

from root.src.utils.settings import settings
from .evaluator import RAGEvaluator


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
                metrics = await self.evaluator._openrouter_evaluation(
                    question, 
                    answer, 
                    contexts, 
                    ["faithfulness", "answer_relevancy", "context_relevancy"]
                )
                
                # Add evaluation results to the response
                result["evaluation"] = {
                    "metrics": metrics,
                    "thresholds": {
                        "faithfulness": settings.ragas_faithfulness_threshold,
                        "answer_relevancy": settings.ragas_answer_relevancy_threshold,
                        "context_relevancy": settings.ragas_context_relevancy_threshold,
                    },
                    "passed_thresholds": all(
                        score >= settings.ragas_faithfulness_threshold 
                        if name == "faithfulness" else
                        score >= settings.ragas_answer_relevancy_threshold 
                        if name == "answer_relevancy" else
                        score >= settings.ragas_context_relevancy_threshold
                        for name, score in metrics.items()
                    )
                }
                
                return result
                
            # Replace the original query method
            pipeline.query = query_with_evaluation
        
        # Add methods to get evaluation data
        pipeline.get_evaluation_history = self.evaluator.get_evaluation_history
        pipeline.get_average_metrics = self.evaluator.get_average_metrics


def get_plugin() -> RAGASPlugin:
    """
    Return a new instance of RAGASPlugin.

    Returns:
        RAGASPlugin: A new plugin instance.
    """
    use_llm_judge = settings.enable_llm_judge
    return RAGASPlugin(use_llm_judge=use_llm_judge) 