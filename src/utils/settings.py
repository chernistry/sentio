"""Application settings and configuration.

This module centralizes configuration management for the Sentio application.
It loads configuration from environment variables and provides default values.
"""

import logging
import os
from pathlib import Path
from typing import Any
from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic import Field

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


class Settings(PydanticBaseSettings):
    """Application settings loaded from environment variables with sensible defaults.
    
    This class follows the Singleton pattern to ensure consistent configuration
    across the application.
    """

    # Environment and App Configuration
    environment: str = Field(default="development", alias="ENVIRONMENT")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    # Vector Store
    vector_store_name: str = Field(default="qdrant", alias="VECTOR_STORE")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_api_key_header: str = Field(default="api-key", alias="QDRANT_API_KEY_HEADER")

    # Collection
    collection_name: str = Field(default="Sentio_docs", alias="COLLECTION_NAME")

    # Embeddings
    embedder_name: str = Field(default="jina", alias="EMBEDDER_NAME")
    embedding_model: str = Field(default="jina-embeddings-v3", alias="EMBEDDING_MODEL")
    embedding_model_api_key: str = Field(default="", alias="EMBEDDING_MODEL_API_KEY")
    jina_api_key: str = Field(default="", alias="JINA_API_KEY")  # Backwards compatibility

    # Reranker
    reranker_model: str = Field(default="jina-reranker-m0", alias="RERANKER_MODEL")
    reranker_url: str = Field(default="https://api.jina.ai/v1/rerank", alias="RERANKER_URL")
    reranker_timeout: int = Field(default=30, alias="RERANKER_TIMEOUT")
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")

    # LLM
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", alias="OPENAI_MODEL")

    # Chat LLM
    chat_llm_base_url: str = Field(default="https://api.openai.com/v1", alias="CHAT_LLM_BASE_URL")
    chat_llm_model: str = Field(default="gpt-3.5-turbo", alias="CHAT_LLM_MODEL")
    chat_llm_api_key: str = Field(default="", alias="CHAT_LLM_API_KEY")

    # OpenRouter specific settings
    openrouter_referer: str = Field(default="https://sentio.ai/", alias="OPENROUTER_REFERER")
    openrouter_title: str = Field(default="Sentio", alias="OPENROUTER_TITLE")

    # LangChain/LangSmith Configuration
    langchain_tracing: bool = Field(default=False, alias="LANGCHAIN_TRACING")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="sense-rag", alias="LANGSMITH_PROJECT")

    # LangGraph Server Configuration
    langgraph_server_port: int = Field(default=2024, alias="LANGGRAPH_SERVER_PORT")
    langgraph_server_host: str = Field(default="127.0.0.1", alias="LANGGRAPH_SERVER_HOST")
    use_langgraph: bool = Field(default=True, alias="USE_LANGGRAPH")

    # Evaluation
    enable_automatic_evaluation: bool = Field(default=True, alias="ENABLE_AUTOMATIC_EVALUATION")

    # Chunking
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=50, alias="MIN_CHUNK_SIZE")
    max_chunk_size: int = Field(default=1024, alias="MAX_CHUNK_SIZE")
    preserve_code_blocks: bool = Field(default=True, alias="PRESERVE_CODE_BLOCKS")
    preserve_tables: bool = Field(default=True, alias="PRESERVE_TABLES")
    chunking_strategy: str = Field(default="recursive", alias="CHUNKING_STRATEGY")

    # Data paths
    sample_docs_folder: str = Field(default="data/raw", alias="SAMPLE_DOCS_FOLDER")
    data_dir: str = Field(default="data", alias="DATA_DIR")

    # Retrieval Strategy Configuration
    retrieval_strategy: str = Field(default="dense", alias="RETRIEVAL_STRATEGY")
    retrieval_top_k: int = Field(default=20, alias="RETRIEVAL_TOP_K")
    reranking_top_k: int = Field(default=5, alias="RERANKING_TOP_K")
    selection_top_k: int = Field(default=3, alias="SELECTION_TOP_K")
    rrf_k: int = Field(default=20, alias="RRF_K")  # Reciprocal Rank Fusion constant
    
    # BM25/Sparse Configuration
    bm25_variant: str = Field(default="okapi", alias="BM25_VARIANT")
    bm25_index_dir: str = Field(default="indexes/lucene-index", alias="BM25_INDEX_DIR")
    sparse_cache_dir: str = Field(default=".sparse_cache", alias="SPARSE_CACHE_DIR")

    # Backend Configuration
    sentio_backend_url: str = Field(default="http://localhost:8000", alias="SENTIO_BACKEND_URL")

    # Legacy fields for backwards compatibility
    top_k_retrieval: int = Field(default=10, alias="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=5, alias="TOP_K_RERANK")
    min_relevance_score: float = Field(default=0.05, alias="MIN_RELEVANCE_SCORE")

    # API
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    enable_cors: bool = Field(default=True, alias="ENABLE_CORS")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")

    # Authentication
    disable_auth: bool = Field(default=False, alias="DISABLE_AUTH")

    def __init__(self, **kwargs):
        """Initialize settings with proper field handling."""
        super().__init__(**kwargs)
        
        # Handle backwards compatibility for embedding API key
        if not self.embedding_model_api_key and self.jina_api_key:
            self.embedding_model_api_key = self.jina_api_key
            
        # Handle chat LLM API key fallback
        if not self.chat_llm_api_key and self.openai_api_key:
            self.chat_llm_api_key = self.openai_api_key
            
        # Convert string paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.sample_docs_folder = Path(self.sample_docs_folder)
        self.bm25_index_dir = Path(self.bm25_index_dir)
        self.sparse_cache_dir = Path(self.sparse_cache_dir)
        
        # Handle boolean string conversions from environment
        if isinstance(self.langchain_tracing, str):
            self.langchain_tracing = self.langchain_tracing.lower() == "true"
        if isinstance(self.use_langgraph, str):
            self.use_langgraph = self.use_langgraph.lower() == "true"
        if isinstance(self.enable_automatic_evaluation, str):
            self.enable_automatic_evaluation = self.enable_automatic_evaluation.lower() == "true"
        if isinstance(self.preserve_code_blocks, str):
            self.preserve_code_blocks = self.preserve_code_blocks.lower() == "true"
        if isinstance(self.preserve_tables, str):
            self.preserve_tables = self.preserve_tables.lower() == "true"
        if isinstance(self.use_reranker, str):
            self.use_reranker = self.use_reranker.lower() == "true"
        if isinstance(self.enable_cors, str):
            self.enable_cors = self.enable_cors.lower() == "true"
        if isinstance(self.disable_auth, str):
            self.disable_auth = self.disable_auth.lower() == "true"

    @property
    def auth_enabled(self) -> bool:
        return not self.disable_auth

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

    class Config:
        env_file = ".env"
        case_sensitive = False


# Create a singleton instance
settings = Settings()
