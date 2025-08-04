"""Application settings and configuration.

This module centralizes configuration management for the Sentio application.
It loads configuration from environment variables and provides default values.
"""

import logging
import os
from pathlib import Path
from typing import Any

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logging.getLogger(__name__).info(f"Loaded environment variables from {env_path}")
except ImportError:
    logging.getLogger(__name__).warning("python-dotenv not installed, skipping .env loading")

logger = logging.getLogger(__name__)


class Settings:
    """Application settings loaded from environment variables with sensible defaults.
    
    This class follows the Singleton pattern to ensure consistent configuration
    across the application.
    """

    def __init__(self) -> None:
        """Initialize settings with defaults and environment variables."""
        # Vector Store
        self.vector_store_name = os.getenv("VECTOR_STORE", "qdrant")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

        # Collection
        self.collection_name = os.getenv("COLLECTION_NAME", "Sentio_docs")

        # Embeddings
        self.embedder_name = os.getenv("EMBEDDER_NAME", "jina")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3")
        self.embedding_model_api_key = os.getenv(
            "EMBEDDING_MODEL_API_KEY",
            os.getenv("JINA_API_KEY", "")  # Backwards compatibility
        )

        # Reranker
        self.reranker_model = os.getenv("RERANKER_MODEL", "jina-reranker-m0")
        self.reranker_url = os.getenv("RERANKER_URL", "https://api.jina.ai/v1/rerank")
        self.reranker_timeout = int(os.getenv("RERANKER_TIMEOUT", "30"))

        # LLM
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        # Chat LLM (from .env)
        self.chat_llm_base_url = os.getenv("CHAT_LLM_BASE_URL", "https://api.openai.com/v1")
        self.chat_llm_model = os.getenv("CHAT_LLM_MODEL", "gpt-3.5-turbo")
        self.chat_llm_api_key = os.getenv("CHAT_LLM_API_KEY", self.openai_api_key)

        # OpenRouter specific settings
        self.openrouter_referer = os.getenv("OPENROUTER_REFERER", "https://sentio.ai/")
        self.openrouter_title = os.getenv("OPENROUTER_TITLE", "Sentio")

        # Chunking
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
        self.chunking_strategy = os.getenv("CHUNKING_STRATEGY", "recursive")

        # Retrieval
        self.top_k_retrieval = int(os.getenv("TOP_K_RETRIEVAL", "10"))
        self.top_k_rerank = int(os.getenv("TOP_K_RERANK", "5"))
        self.min_relevance_score = float(os.getenv("MIN_RELEVANCE_SCORE", "0.05"))

        # API
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.log_level = os.getenv("LOG_LEVEL", "info").upper()
        self.enable_cors = os.getenv("ENABLE_CORS", "true").lower() == "true"

        # Data paths
        self.data_dir = Path(os.getenv("DATA_DIR", "data"))

    def chunking_kwargs(self) -> dict[str, Any]:
        """Return a dictionary of chunking parameters for the TextChunker."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "strategy": self.chunking_strategy,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to a dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# Create a singleton instance
settings = Settings()
