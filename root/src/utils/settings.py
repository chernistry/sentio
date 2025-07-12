from __future__ import annotations

"""Centralised runtime configuration for Sentio.

This thin wrapper around ``pydantic.BaseSettings`` exposes environment
variables as a typed, cached singleton.  Other modules should import
``settings`` and use attribute access instead of reaching for ``os.environ``
directly.  This improves testability and documents expected configuration
keys in one place.

Example
-------
>>> from root.src.utils.settings import settings
>>> print(settings.embedding_provider)
'jina'

The class deliberately **does not** hard-fail on unknown variables — that keeps
backward-compat and allows plugins to read custom env vars on their own.
"""

import functools
import sys
from typing import Optional, ClassVar

# Detect major Pydantic version at import-time to avoid incompatible config declarations.
try:
    import pydantic  # type: ignore

    _IS_PYD_V2 = pydantic.version.VERSION.startswith("2")
except Exception:  # pragma: no cover – very defensive
    _IS_PYD_V2 = False  # Assume v1 if detection fails

# ---------------------------------------------------------------------------
# Compatibility shim for *pydantic* v1 → v2 migration.
# ---------------------------------------------------------------------------
try:
    # Pydantic < 2.0
    from pydantic import BaseSettings, Field  # type: ignore
except ImportError:  # pragma: no cover – runtime fallback for v2+
    from pydantic_settings import BaseSettings  # type: ignore
    from pydantic import Field  # Field still exists in v2

# Import SettingsConfigDict at module level (optional dependency, only used when
# running on pydantic>=2). Importing here avoids leaking the symbol into the
# class namespace which would otherwise be interpreted as a model field.
try:
    from pydantic_settings import SettingsConfigDict  # type: ignore
except ImportError:  # pragma: no cover – optional for pydantic v1
    SettingsConfigDict = None  # type: ignore


# ---------------------------------------------------------------------------
# Model definition – conditional *Config* vs *model_config* for Pydantic v1/v2
# ---------------------------------------------------------------------------


class SentioSettings(BaseSettings):
    """Environment-driven settings for the whole application."""

    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Embeddings
    embedding_provider: str = Field("jina", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("jina-embeddings-v3", env="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(16, env="EMBEDDING_BATCH_SIZE")

    # Embedding provider API key (e.g., Jina AI)
    embedding_model_api_key: Optional[str] = Field(None, env="EMBEDDING_MODEL_API_KEY")

    # Ollama specific (optional)
    ollama_url: str = Field("http://localhost:11434", env="OLLAMA_URL")
    ollama_model: str = Field("qwen3:embedding", env="OLLAMA_MODEL")

    # Local Sentence-Transformers model (optional)
    local_embed_model: str = Field("BAAI/bge-base-en", env="LOCAL_EMBED_MODEL")

    # OpenRouter proxy configuration
    openrouter_api_base: str = Field("https://openrouter.ai/api/v1", env="OPENROUTER_API_BASE")
    openrouter_keys_csv: str = Field("/app/keys.csv", env="OPENROUTER_KEYS_CSV")
    openrouter_referer: str = Field("https://openwebui.com/", env="OPENROUTER_REFERER")
    openrouter_title: str = Field("SentioGateway", env="OPENROUTER_TITLE")

    # Generic OpenAI-compatible chat LLM configuration
    chat_provider: str = Field("openrouter", env="CHAT_PROVIDER")
    chat_llm_base_url: str = Field("https://openrouter.ai/api/v1", env="CHAT_LLM_BASE_URL")
    chat_llm_model: str = Field("deepseek/deepseek-chat-v3-0324:free", env="CHAT_LLM_MODEL")
    chat_llm_api_key: str | None = Field(None, env="CHAT_LLM_API_KEY")

    # Beam Cloud configuration
    beam_api_token: str | None = Field(None, env="BEAM_API_TOKEN")
    beam_volume: str = Field("comfy-weights", env="BEAM_VOLUME")
    beam_model_id: str = Field("mistral-7b", env="BEAM_MODEL_ID")
    beam_gpu: str = Field("A10G", env="BEAM_GPU")
    beam_memory: str = Field("32Gi", env="BEAM_MEMORY")
    beam_cpu: int = Field(4, env="BEAM_CPU")

    # Runtime mode – "cloud" (default) or "local". Influences provider logic.
    beam_mode: str = Field("cloud", env="BEAM_MODE")

    # Unified embedding URL that will be set by setup-env.sh
    beam_embedding_base_url: str | None = Field(
        None,
        env="BEAM_EMBEDDING_BASE_URL",
    )

    # Optional – remote embedding endpoint deployed on Beam (cloud)
    BEAM_EMBEDDING_BASE_CLOUD_URL: str | None = Field(
        None,
        env="BEAM_EMBEDDING_BASE_CLOUD_URL",
    )

    # Optional – local development embedding endpoint (e.g. http://localhost:8000/embed)
    beam_embedding_base_local_url: str | None = Field(
        None,
        env="BEAM_EMBEDDING_BASE_LOCAL_URL",
    )

    # Reranker -----------------------------------------------------------------
    reranker_provider: str = Field("cross_encoder", env="RERANKER_PROVIDER")
    reranker_model: str = Field(
        "ibm-granite/granite-embedding-107m-multilingual",
        env="RERANKER_MODEL",
    )
    secondary_reranker_type: str = Field(
        "local",
        env="SECONDARY_RERANKER_TYPE",
    )
    secondary_rerank_model: str = Field(
        "Alibaba-NLP/gte-multilingual-base",
        env="SECONDARY_RERANK_MODEL",
    )
    enable_llm_judge: bool = Field(
        False,
        env="ENABLE_LLM_JUDGE",
    )

    if not _IS_PYD_V2:
        # Pydantic < 2.0 – classic *Config* subclass.
        class Config:  # noqa: WPS431 (inner class by design)
            case_sensitive = False
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

    else:
        # Pydantic ≥ 2.0 – new ``SettingsConfigDict`` mechanism.
        if SettingsConfigDict is None:
            raise RuntimeError(
                "pydantic>=2 detected but the optional 'pydantic-settings' package "
                "is missing. Install it or pin pydantic<2."
            )

        # Mark the config object as a :class:`~typing.ClassVar` so Pydantic
        # does not treat it as a regular model field.
        model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
            extra="ignore",
            case_sensitive=False,
            env_file=".env",
            env_file_encoding="utf-8",
        )

    # Convenience helpers -------------------------------------------------

    def to_common_kwargs(self) -> dict[str, str | int | bool | None]:
        """Return dict with generic embedding-constructor kwargs."""
        return {
            "model_name": self.embedding_model,
            "batch_size": self.embedding_batch_size,
            # Additional fields can be mapped as needed.
        }


@functools.lru_cache(maxsize=1)
def get_settings() -> SentioSettings:  # pragma: no cover – deterministic
    """Return cached settings instance (singleton)."""
    return SentioSettings()  # type: ignore[return-value]


# Eagerly create default instance for import-style usage
settings: SentioSettings = get_settings()
