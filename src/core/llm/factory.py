"""Factory for creating LLM generators."""

import logging
import os

from src.core.llm.chat_adapter import ChatAdapter
from src.core.llm.generator import LLMGenerator
from src.core.llm.prompt_builder import PromptBuilder
from src.utils.settings import settings

logger = logging.getLogger(__name__)


def create_generator(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    mode: str | None = None,
    max_tokens: int | None = None,
) -> LLMGenerator:
    """Create an LLM generator with the specified configuration.
    
    Args:
        provider: LLM provider (openai, azure, etc.)
        model: Model name
        api_key: API key
        base_url: API base URL
        mode: Generation mode (fast, balanced, quality, creative)
        max_tokens: Maximum tokens to generate
        
    Returns:
        An LLMGenerator instance
    """
    # Use environment variables as defaults
    provider = provider or settings.llm_provider
    model = model or settings.chat_llm_model
    api_key = api_key or settings.chat_llm_api_key
    base_url = base_url or settings.chat_llm_base_url
    mode = mode or os.getenv("GENERATION_MODE", "balanced")
    max_tokens = max_tokens or int(os.getenv("GENERATION_MAX_TOKENS", "1024"))

    # Create chat adapter
    chat_adapter = ChatAdapter(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    # Create prompt builder
    prompt_builder = PromptBuilder()

    # Create generator
    generator = LLMGenerator(
        chat_adapter=chat_adapter,
        prompt_builder=prompt_builder,
        mode=mode,
        max_tokens=max_tokens,
    )

    logger.info(
        "Created LLM generator with provider=%s, model=%s, mode=%s",
        provider,
        model,
        mode,
    )

    return generator
