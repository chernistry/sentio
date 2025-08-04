"""LLM integration for Sentio.

This package provides LLM integration for the Sentio RAG pipeline.
"""

from src.core.llm.chat_adapter import ChatAdapter, chat_completion
from src.core.llm.factory import create_generator
from src.core.llm.generator import LLMGenerator
from src.core.llm.prompt_builder import PromptBuilder
from src.core.llm.reply_extractor import extract_json_dict, extract_json_dict_sync

__all__ = [
    "ChatAdapter",
    "LLMGenerator",
    "PromptBuilder",
    "chat_completion",
    "create_generator",
    "extract_json_dict",
    "extract_json_dict_sync",
]
