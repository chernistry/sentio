"""LLM integration for Sentio.

This package provides LLM integration for the Sentio RAG pipeline.

Note: Avoid importing heavy submodules at package import time to prevent
import cycles. Import submodules explicitly where needed, e.g.:

    from src.core.llm.factory import create_generator
    from src.core.llm.generator import LLMGenerator
    from src.core.llm.chat_adapter import ChatAdapter
"""

__all__ = []
