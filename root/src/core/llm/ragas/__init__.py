"""RAGAS-based evaluation for Retrieval-Augmented Generation.

This module provides quality assessment for RAG systems using the RAGAS
framework (Retrieval Augmented Generation Assessment). It offers metrics to
evaluate faithfulness, answer relevancy, and context relevancy without requiring
ground-truth answers. A fallback LLM judge is used when RAGAS is unavailable.
"""

from .evaluator import RAGEvaluator
from .plugin import RAGASPlugin, get_plugin

__all__ = ["RAGEvaluator", "RAGASPlugin", "get_plugin"] 