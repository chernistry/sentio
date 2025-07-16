"""RAGAS Evaluation Plugin for Sentio RAG.

This plugin integrates RAGAS evaluation capabilities with the Sentio RAG pipeline.
"""

# Добавляем вывод в консоль при импорте модуля
import sys
print("🔍 RAGAS_EVAL MODULE IMPORTED - PRINT TO STDOUT", file=sys.stdout)
print("🔍 RAGAS_EVAL MODULE IMPORTED - PRINT TO STDERR", file=sys.stderr)

from typing import Any
import traceback
import os

from plugins.interface import SentioPlugin
from root.src.core.llm.ragas.evaluator import RAGEvaluator
from root.src.utils.settings import settings
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Настройка обработчика для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# Добавляем логирование при импорте модуля
print("🔍 Loading RAGAS evaluation plugin module - PRINT")
logger.critical("🔍 Loading RAGAS evaluation plugin module - CRITICAL")
logger.error("🔍 Loading RAGAS evaluation plugin module - ERROR")
logger.warning("🔍 Loading RAGAS evaluation plugin module - WARNING")
logger.info("🔍 Loading RAGAS evaluation plugin module - INFO")
logger.debug("🔍 Loading RAGAS evaluation plugin module - DEBUG")

# Проверяем переменные окружения
logger.info(f"SENTIO_PLUGINS env: {os.getenv('SENTIO_PLUGINS', 'not set')}")
logger.info(f"Python path: {sys.path}")

try:
    # Проверяем доступность импортов
    import plugins.interface
    logger.info("✅ Successfully imported plugins.interface")
except ImportError as e:
    logger.error(f"❌ Failed to import plugins.interface: {e}")
    logger.error(traceback.format_exc())

try:
    # Проверяем доступность импортов
    from root.src.core.llm.ragas.evaluator import RAGEvaluator
    logger.info("✅ Successfully imported RAGEvaluator")
except ImportError as e:
    logger.error(f"❌ Failed to import RAGEvaluator: {e}")
    logger.error(traceback.format_exc())


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
        logger.info("🔧 Initializing RAGAS Plugin")
        try:
            self.evaluator: RAGEvaluator = RAGEvaluator(
                use_llm_judge=use_llm_judge
            )
            logger.info("✅ RAGAS Plugin initialized with evaluator")
            
            # Проверяем наличие методов в evaluator
            logger.info(f"Evaluator has get_evaluation_history: {hasattr(self.evaluator, 'get_evaluation_history')}")
            logger.info(f"Evaluator has get_average_metrics: {hasattr(self.evaluator, 'get_average_metrics')}")
        except Exception as e:
            logger.error(f"❌ Error initializing RAGAS Plugin: {e}")
            logger.error(traceback.format_exc())
            raise

    def register(self, pipeline: Any) -> None:
        """
        Register evaluator with a processing pipeline.

        Args:
            pipeline (Any): Pipeline object to attach evaluator to.
        """
        logger.info("🔄 Registering RAGAS plugin with pipeline")
        try:
            # Add evaluator to pipeline
            pipeline.evaluator = self.evaluator
            logger.info(f"✅ Added evaluator to pipeline: {hasattr(pipeline, 'evaluator')}")
            
            # Add direct methods to pipeline for compatibility
            pipeline.get_evaluation_history = self.get_evaluation_history
            logger.info(f"✅ Added get_evaluation_history to pipeline: {hasattr(pipeline, 'get_evaluation_history')}")
            
            pipeline.get_average_metrics = self.get_average_metrics
            logger.info(f"✅ Added get_average_metrics to pipeline: {hasattr(pipeline, 'get_average_metrics')}")
            
            # Only monkey patch if automatic evaluation is enabled AND pipeline has query method
            if settings.enable_automatic_evaluation and hasattr(pipeline, 'query'):
                logger.info("🔄 Monkey patching pipeline.query with evaluation")
                # Store original query method
                original_query = pipeline.query
                
                async def query_with_evaluation(question: str, top_k: int = None) -> dict:
                    logger.info(f"📝 Evaluating query: {question}")
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
                logger.info("✅ Monkey patched pipeline.query with evaluation")
            elif settings.enable_automatic_evaluation:
                logger.info("⚠️ Pipeline does not have 'query' method, skipping monkey patching")
            
            logger.info(f"✅ RAGAS plugin registration complete. Pipeline has evaluator: {hasattr(pipeline, 'evaluator')}")
            logger.info(f"Pipeline methods: get_evaluation_history={hasattr(pipeline, 'get_evaluation_history')}, get_average_metrics={hasattr(pipeline, 'get_average_metrics')}")
        except Exception as e:
            logger.error(f"❌ Error registering RAGAS plugin: {e}")
            logger.error(traceback.format_exc())
            raise

    def get_evaluation_history(self):
        """
        Get the history of all evaluations performed.
        
        Returns:
            List of evaluation entries with metrics
        """
        logger.info("📊 Calling get_evaluation_history")
        try:
            history = self.evaluator.get_evaluation_history()
            logger.info(f"📊 Retrieved {len(history)} evaluation history entries")
            return history
        except Exception as e:
            logger.error(f"❌ Error in get_evaluation_history: {e}")
            logger.error(traceback.format_exc())
            return []
        
    def get_average_metrics(self):
        """
        Calculate average scores across all evaluations.
        
        Returns:
            Average scores for each metric
        """
        logger.info("📊 Calling get_average_metrics")
        try:
            metrics = self.evaluator.get_average_metrics()
            logger.info(f"📊 Retrieved average metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"❌ Error in get_average_metrics: {e}")
            logger.error(traceback.format_exc())
            return {}


def get_plugin():
    """
    Return a new instance of RAGASPlugin.
    
    Returns:
        RAGASPlugin: A new plugin instance.
    """
    logger.info("🔍 Creating RAGAS plugin instance")
    try:
        use_llm_judge = settings.enable_llm_judge
        plugin = RAGASPlugin(use_llm_judge=use_llm_judge)
        logger.info("✅ Created RAGAS plugin instance")
        return plugin
    except Exception as e:
        logger.error(f"❌ Error creating RAGAS plugin: {e}")
        logger.error(traceback.format_exc())
        raise
