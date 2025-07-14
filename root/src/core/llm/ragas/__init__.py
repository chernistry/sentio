"""RAGAS evaluation module for Sentio RAG.

This module provides evaluation capabilities for RAG systems using RAGAS metrics.
"""

import logging

from .evaluator import RAGEvaluator
from .plugin import RAGASPlugin, get_plugin

__all__ = ["RAGEvaluator", "RAGASPlugin", "get_plugin"]

logger = logging.getLogger(__name__)
logger.info("✅ RAGAS module initialized") 