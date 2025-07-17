"""RAGAS Evaluation Plugin for Sentio RAG.

This plugin integrates RAGAS evaluation capabilities with the Sentio RAG pipeline.
"""
import logging
from typing import Any

from plugins.interface import SentioPlugin
from root.src.core.llm.ragas.evaluator import RAGEvaluator
from root.src.utils.settings import settings

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
        logger.info("Initializing RAGAS Plugin...")
        try:
            self.evaluator: RAGEvaluator = RAGEvaluator(
                use_llm_judge=use_llm_judge
            )
            logger.info("✅ RAGAS Plugin initialized with evaluator.")
        except Exception as e:
            logger.error(f"❌ Error initializing RAGAS Plugin: {e}", exc_info=True)
            raise

    def register(self, pipeline: Any) -> None:
        """
        Register evaluator with a processing pipeline.

        Args:
            pipeline (Any): Pipeline object to attach evaluator to.
        """
        logger.info("Registering RAGAS plugin with pipeline...")
        try:
            # Add evaluator to pipeline
            pipeline.evaluator = self.evaluator

            # Add direct methods to pipeline for compatibility
            pipeline.get_evaluation_history = self.get_evaluation_history
            pipeline.get_average_metrics = self.get_average_metrics

            # Only monkey patch if automatic evaluation is enabled AND pipeline has query method
            if settings.enable_automatic_evaluation and hasattr(pipeline, 'query'):
                logger.info("Monkey patching pipeline.query with evaluation...")
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
                logger.info("✅ Monkey patched pipeline.query with evaluation.")
            elif settings.enable_automatic_evaluation:
                logger.warning("Pipeline does not have 'query' method, skipping monkey patching.")

            logger.info("✅ RAGAS plugin registration complete.")
        except Exception as e:
            logger.error(f"❌ Error registering RAGAS plugin: {e}", exc_info=True)
            raise

    def get_evaluation_history(self) -> list:
        """Get the full RAGAS evaluation history."""
        return self.evaluator.get_evaluation_history()

    def get_average_metrics(self) -> dict:
        """Get average scores for all RAGAS metrics."""
        return self.evaluator.get_average_metrics()


_plugin_instance = None


def get_plugin(use_llm_judge: bool = False) -> RAGASPlugin:
    """
    Get a singleton instance of the RAGAS plugin.

    Args:
        use_llm_judge (bool): Enable LLM judge fallback.

    Returns:
        RAGASPlugin: Singleton instance of the plugin.
    """
    global _plugin_instance
    logger.info("🔍 Creating RAGAS plugin instance")
    if _plugin_instance is None:
        try:
            _plugin_instance = RAGASPlugin(use_llm_judge=use_llm_judge)
            logger.info("✅ RAGAS plugin instance created successfully.")
        except Exception as e:
            logger.error(f"❌ Error creating RAGAS plugin: {e}", exc_info=True)
            raise
    return _plugin_instance
