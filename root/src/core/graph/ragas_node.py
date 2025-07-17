#!/usr/bin/env python3
"""
RAGAS Evaluation Node for LangGraph.

This module provides a LangGraph node that integrates with the RAGAS evaluation 
capabilities, evaluating the quality of generated answers.
"""

import logging
from typing import Dict, Any, List, Optional

from ...utils.settings import settings

logger = logging.getLogger(__name__)


async def ragas_evaluation_node(state, pipeline) -> Dict[str, Any]:
    """
    Evaluate the RAG answer quality using RAGAS metrics.
    
    This node runs after the answer generation and evaluates the quality of the
    generated answer against the retrieved context using RAGAS metrics.
    
    Args:
        state: The current graph state containing the query, answer, and sources
        pipeline: The main pipeline instance with the evaluator
        
    Returns:
        Updated state with evaluation results added
    """
    # Skip evaluation if no evaluator is available
    if not hasattr(pipeline, "evaluator") or not settings.enable_automatic_evaluation:
        logger.info("Skipping RAGAS evaluation - evaluator not available or disabled")
        return state
    
    logger.info(f"Running RAGAS evaluation for query: {state.query}")
    
    try:
        # Extract necessary data for evaluation
        question = state.query
        answer = state.answer
        
        # Extract text from the source documents
        contexts = []
        if hasattr(state, "reranked_documents") and state.reranked_documents:
            contexts = [doc.get("text", "") for doc in state.reranked_documents if "text" in doc]
        elif hasattr(state, "sources") and state.sources:
            contexts = [doc.get("text", "") for doc in state.sources if "text" in doc]
        
        if not contexts:
            logger.warning("No contexts available for evaluation")
            return state
            
        # Run evaluation using the pipeline's evaluator
        metrics = await pipeline.evaluator._openrouter_evaluation(
            question, 
            answer, 
            contexts, 
            ["faithfulness", "answer_relevancy", "context_relevancy"]
        )
        
        # Add evaluation results to the state metadata
        state.metadata["evaluation"] = {
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
        
        logger.info(f"RAGAS evaluation complete: {metrics}")
        
    except Exception as e:
        logger.error(f"Error in RAGAS evaluation: {e}", exc_info=True)
        # Add error information to metadata but don't fail the pipeline
        state.metadata["evaluation_error"] = str(e)
        
    return state 