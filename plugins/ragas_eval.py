"""RAGAS Evaluation Plugin for Sentio RAG.

This plugin integrates RAGAS evaluation capabilities with the Sentio RAG pipeline.
"""

from typing import Any

from plugins.interface import SentioPlugin
from root.src.core.llm.ragas.evaluator import RAGEvaluator
from root.src.utils.settings import settings
import logging

logger = logging.getLogger(__name__)

class RAGASPlugin(SentioPlugin):
    """
    Plugin providing RAGAS evaluation capabilities.
    """

    name: str = "ragas_eval"
    plugin_type: str = "evaluator"
    version: str = "0.2.0"
    description: str = "RAGAS evaluation for RAG pipeline"

    def __init__(self, use_llm_judge: bool = False) -> None:
        """
        Initialize plugin instance.

        Args:
            use_llm_judge (bool): Enable LLM judge fallback.
        """
        self.evaluator: RAGEvaluator = RAGEvaluator(
            use_llm_judge=use_llm_judge
        )
        logger.info("RAGAS Plugin initialized with evaluator")

    def register(self, pipeline: Any) -> None:
        """
        Register evaluator with a processing pipeline.

        Args:
            pipeline (Any): Pipeline object to attach evaluator to.
        """
        logger.info("Registering RAGAS plugin with pipeline")
        
        # Add evaluator to pipeline
        pipeline.evaluator = self.evaluator
        
        # Add direct methods to pipeline for compatibility
        pipeline.get_evaluation_history = self.get_evaluation_history
        pipeline.get_average_metrics = self.get_average_metrics
        
        # Only monkey patch if automatic evaluation is enabled
        if settings.enable_automatic_evaluation:
            # Store original query method
            original_query = pipeline.query
            
            async def query_with_evaluation(question: str, top_k: int = None) -> dict:
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
            logger.info("Monkey patched pipeline.query with evaluation")
        
        logger.info(f"RAGAS plugin registration complete. Pipeline has evaluator: {hasattr(pipeline, 'evaluator')}")
        logger.info(f"Pipeline methods: get_evaluation_history={hasattr(pipeline, 'get_evaluation_history')}, get_average_metrics={hasattr(pipeline, 'get_average_metrics')}")

    def get_evaluation_history(self):
        """
        Get the history of all evaluations performed.
        
        Returns:
            List of evaluation entries with metrics
        """
        return self.evaluator.get_evaluation_history()
        
    def get_average_metrics(self):
        """
        Calculate average scores across all evaluations.
        
        Returns:
            Average scores for each metric
        """
        return self.evaluator.get_average_metrics()


def get_plugin():
    """
    Return a new instance of RAGASPlugin.
    
    Returns:
        RAGASPlugin: A new plugin instance.
    """
    use_llm_judge = settings.enable_llm_judge
    return RAGASPlugin(use_llm_judge=use_llm_judge)
