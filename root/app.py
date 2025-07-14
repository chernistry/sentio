#!/usr/bin/env python3


"""
Sentio RAG System - Main FastAPI Application

A production-ready, enterprise-grade RAG system with advanced features:
- Hybrid retrieval (dense + sparse)
- Intelligent reranking
- Streaming responses
- OpenAI-compatible API
- Comprehensive monitoring and logging
- Error handling and resilience
"""


# ==== IMPORTS & DEPENDENCIES ================================================== #

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import (
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Union
)

import httpx
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator
from qdrant_client import models
from qdrant_client import QdrantClient
import json
import re
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import VectorParams, Distance
from root.src.core.llm.chat_adapter import chat_completion
from root.src.core.embeddings.embeddings_adapter import get_embedding_model
from root.src.core.tasks.embeddings import BaseEmbeddingModel, EmbeddingError

# Import Azure integration modules with graceful degradation
try:
    from root.src.azure.queue import AzureQueueClient
    from root.src.azure.monitoring import (
        configure_azure_app_insights,
        track_event,
        track_metric
    )
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    AzureQueueClient = None  # type: ignore
    configure_azure_app_insights = lambda: None  # type: ignore
    track_event = lambda *args, **kwargs: None  # type: ignore
    track_metric = lambda *args, **kwargs: None  # type: ignore
    print(
        "[Sentio] Azure integration modules not found, continuing without Azure support"
    )


# ==== LOGGING CONFIGURATION =================================================== #

# Define a function to convert string log level to logging level
def get_log_level(level_name: str) -> int:
    """Convert string log level to logging module level value."""
    level_name = level_name.upper()
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_name, logging.INFO)

# Configure basic logging
logging.basicConfig(
    level=get_log_level(os.environ.get("LOG_LEVEL", "info")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/Sentio.log') if os.path.exists('logs') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==== ENVIRONMENT CONFIGURATION =============================================== #

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not available, using system environment only")


# ==== APPLICATION CONFIGURATION ============================================== #

class Config:
    """
    Centralized configuration management with validation.
    """

    # API Configuration
    EMBEDDING_MODEL_API_KEY: str = os.environ.get("EMBEDDING_MODEL_API_KEY", os.environ.get("JINA_API_KEY", "")).strip()
    QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY", "")

    # Collection and Model Settings
    COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "Sentio_docs")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "jina-embeddings-v3")
    RERANKER_MODEL: str = os.environ.get("RERANKER_MODEL", "jjina-reranker-m0")

    @property
    def EMBEDDING_PROVIDER(self) -> str:
        """
        Returns the embedding provider from environment variables.
        """
        return os.environ.get("EMBEDDING_PROVIDER", "jina").lower()

    @property
    def RERANKER_PROVIDER(self) -> str:
        """
        Returns the reranker provider from environment variables.
        """
        return os.environ.get("RERANKER_PROVIDER", "jina").lower()

    @property
    def CHAT_PROVIDER(self) -> str:
        """
        Returns the chat provider from environment variables.
        """
        return os.environ.get("CHAT_PROVIDER", "openrouter").lower()

    # Retrieval Parameters
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = int(os.environ.get("TOP_K_RERANK", "5"))
    MIN_RELEVANCE_SCORE: float = float(
        os.environ.get("MIN_RELEVANCE_SCORE", os.environ.get("SENTIO_MIN_SCORE", "0.05"))
    )
    REQUEST_TIMEOUT: int = 60
    MAX_RETRIES: int = 3

    # Performance Settings
    ENABLE_METRICS: bool = os.environ.get("ENABLE_METRICS", "true").lower() == "true"
    ENABLE_CORS: bool = os.environ.get("ENABLE_CORS", "true").lower() == "true"
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "info")

    # Azure-specific settings
    USE_AZURE: bool = (
        os.environ.get("USE_AZURE", "true").lower() == "true"
        and 'AZURE_AVAILABLE' in globals()
        and AZURE_AVAILABLE
    )

    @classmethod
    def validate(cls) -> None:
        """
        Validate critical configuration.

        Raises:
            ValueError: If any configuration is invalid or missing.
        """
        embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "jina").lower()
        reranker_provider = os.environ.get("RERANKER_PROVIDER", "jina").lower()
        chat_provider = os.environ.get("CHAT_PROVIDER", "openrouter").lower()
        jina_api_key = os.environ.get("EMBEDDING_MODEL_API_KEY", os.environ.get("JINA_API_KEY", ""))

        if embedding_provider == "jina" and not jina_api_key:
            raise ValueError(
                "EMBEDDING_MODEL_API_KEY is required when using 'jina' embedding provider. "
                "Get your key from https://jina.ai/?sui=apikey"
            )

        for url_name, url in [("QDRANT_URL", cls.QDRANT_URL)]:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"{url_name} must be a valid HTTP/HTTPS URL")

        # ------------------------------------------------------------------
        # Validate embedding provider against the list supported by
        # *embeddings_adapter*.  This enables providers such as *beam*,
        # *sentence-transformers* and *ollama* without hard-coding a single
        # option here.  Failing fast prevents mis-configuration at runtime.
        # ------------------------------------------------------------------
        allowed_embedding_providers = [
            "jina",
            "beam",
            "sentence",
            "sentence_transformers",
            "ollama",
        ]

        if embedding_provider not in allowed_embedding_providers:
            raise ValueError(
                "Unknown EMBEDDING_PROVIDER '{0}'. Allowed values: {1}".format(
                    embedding_provider, ", ".join(allowed_embedding_providers)
                )
            )

        if chat_provider not in ["openrouter"]:
            raise ValueError(
                f"CHAT_PROVIDER must be 'openrouter'. Got: {chat_provider}"
            )

        if reranker_provider not in ["jina"]:
            raise ValueError(
                f"RERANKER_PROVIDER must be 'jina'. Got: {reranker_provider}"
            )


config = Config()


# ==== GLOBAL CLIENTS ========================================================= #

qdrant_client: Optional[QdrantClient] = None
azure_queue_client: Optional[AzureQueueClient] = None
jina_headers: Dict[str, str] = {}

# Lazily-initialised embedding model instance (global for reuse)
embedding_model: Optional[BaseEmbeddingModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan with proper initialization and cleanup.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Allows FastAPI to run the application.
    """
    global qdrant_client, azure_queue_client, jina_headers

    logger.info("🚀 Starting Sentio RAG System...")

    try:
        config.validate()
        logger.info("✓ Configuration validated")

        embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "jina").lower()
        reranker_provider = os.environ.get("RERANKER_PROVIDER", "jina").lower()
        chat_provider = os.environ.get("CHAT_PROVIDER", "openrouter").lower()

        logger.info(f"⚙️ EMBEDDING_PROVIDER: {embedding_provider}")
        logger.info(f"⚙️ EMBEDDING_MODEL: {os.environ.get('EMBEDDING_MODEL', 'jina-embeddings-v3')}")
        logger.info(f"⚙️ RERANKER_PROVIDER: {reranker_provider}")
        logger.info(f"⚙️ RERANKER_MODEL: {os.environ.get('RERANKER_MODEL', 'unknown')}")
        logger.info(f"⚙️ CHAT_PROVIDER: {chat_provider}")
        logger.info(f"⚙️ OPENROUTER_MODEL: {os.environ.get('OPENROUTER_MODEL', 'unknown')}")

        if config.USE_AZURE:
            try:
                configure_azure_app_insights()
                logger.info("✓ Azure Application Insights configured")
            except Exception as e:
                logger.warning(f"Failed to configure Azure Application Insights: {e}")

        if config.EMBEDDING_PROVIDER == "jina" or config.RERANKER_PROVIDER == "jina":
            api_key = config.EMBEDDING_MODEL_API_KEY.strip()
            if not api_key:
                logger.warning("EMBEDDING_MODEL_API_KEY is not set, but is required for Jina API")

            jina_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            logger.info("✓ Jina API headers configured")
            logger.debug(f"Jina API key prefix: {api_key[:8] if api_key else 'None'}")

        for attempt in range(config.MAX_RETRIES):
            try:
                if config.QDRANT_API_KEY:
                    api_key_header = os.getenv("QDRANT_API_KEY_HEADER", "Authorization")
                    logger.info(
                        f"Connecting to Qdrant with API key using header: {api_key_header}"
                    )

                    if api_key_header == "api-key":
                        qdrant_client = QdrantClient(
                            url=config.QDRANT_URL,
                            api_key=config.QDRANT_API_KEY
                        )
                    else:
                        qdrant_client = QdrantClient(
                            url=config.QDRANT_URL,
                            api_key=config.QDRANT_API_KEY
                        )
                else:
                    logger.info(f"Connecting to Qdrant without API key")
                    qdrant_client = QdrantClient(url=config.QDRANT_URL)

                collections = qdrant_client.get_collections()
                logger.info(f"✓ Connected to Qdrant at {config.QDRANT_URL}")
                logger.info(f"  Found {len(collections.collections)} collections")
                break
            except Exception as e:
                if attempt == config.MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Failed to connect to Qdrant after {config.MAX_RETRIES} attempts: {e}"
                    )
                logger.warning(
                    f"Qdrant connection attempt {attempt + 1} failed, retrying: {str(e)}"
                )
                await asyncio.sleep(2 ** attempt)

        if config.USE_AZURE:
            try:
                azure_queue_client = AzureQueueClient()
                logger.info(f"✓ Connected to Azure Storage Queue")
            except Exception as e:
                logger.error(f"Failed to initialize Azure Queue Client: {e}")

        logger.info(f"✓ Using {config.EMBEDDING_PROVIDER.upper()} as embedding provider")
        logger.info(f"✓ Using {config.CHAT_PROVIDER.upper()} as chat provider")
        logger.info("🎉 Sentio RAG System started successfully!")

        yield

    except Exception as e:
        logger.error(f"Failed to start Sentio RAG System: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down Sentio RAG System...")
        if qdrant_client:
            qdrant_client.close()

        # Gracefully close embedding model if initialised
        if embedding_model is not None:
            try:
                await embedding_model.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to close embedding model: %s", exc)
        logger.info("✓ Shutdown complete")


# ==== FASTAPI APPLICATION INITIALIZATION ===================================== #

app = FastAPI(
    title="Sentio RAG API",
    description="Enterprise-grade Retrieval-Augmented Generation system with hybrid search and intelligent reranking",
    version="2.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
    lifespan=lifespan
)


if config.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if config.ENABLE_METRICS:
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/health", "/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="inprogress",
        inprogress_labels=True,
    )
    instrumentator.instrument(app).expose(app)


# ==== DATA MODELS ============================================================ #

class ChatRequest(BaseModel):
    """
    Chat request with comprehensive validation.

    Attributes:
        question (str): User's question.
        history (Optional[List[Dict[str, str]]]): Chat history.
        top_k (Optional[int]): Number of results to return.
        temperature (Optional[float]): Generation temperature.
    """
    question: str = Field(
        ..., min_length=1, max_length=2000, description="User's question"
    )
    history: Optional[List[Dict[str, str]]] = Field(
        default=[], description="Chat history"
    )
    top_k: Optional[int] = Field(
        default=3, ge=1, le=20, description="Number of results to return"
    )
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="Generation temperature"
    )

    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class Source(BaseModel):
    """
    Source document with metadata.

    Attributes:
        text (str): Source text content.
        source (str): Document source identifier.
        score (float): Relevance score.
        metadata (Optional[Dict]): Additional metadata.
    """
    text: str = Field(..., description="Source text content")
    source: str = Field(..., description="Document source identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")


class ChatResponse(BaseModel):
    """
    Chat response with sources and metadata.

    Attributes:
        answer (str): Generated answer.
        sources (List[Source]): Source documents used.
        metadata (Optional[Dict]): Response metadata.
    """
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source documents used")
    metadata: Optional[Dict] = Field(default=None, description="Response metadata")


class EmbedRequest(BaseModel):
    """
    Request for document embedding/ingestion.

    Attributes:
        id (Optional[Union[int, str]]): Document identifier.
        content (str): Raw text content to embed.
        metadata (Optional[Dict]): Arbitrary metadata for the document.
    """
    id: Optional[Union[int, str]] = Field(
        default=None, description="Document identifier (auto-generated if omitted)"
    )
    content: str = Field(
        ..., min_length=1, max_length=50000, description="Raw text content to embed"
    )
    metadata: Optional[Dict] = Field(default=None, description="Arbitrary metadata for the document")


class SearchRequest(BaseModel):
    """
    Request for vector search.

    Attributes:
        query (str): Query vector.
        collection_name (str): Name of the collection to search.
        limit (int): Number of results to return.
    """
    query: str = Field(
        ..., min_length=1, max_length=2000, description="Query vector"
    )
    collection_name: str = Field(
        default=config.COLLECTION_NAME, description="Name of the collection to search"
    )
    limit: int = Field(
        default=10, ge=1, le=100, description="Number of results to return"
    )


class SearchResponse(BaseModel):
    """
    Response for vector search.

    Attributes:
        results (List[Dict]): Search results.
    """
    results: List[Dict] = Field(..., description="Search results")


class HealthResponse(BaseModel):
    """
    Health check response.

    Attributes:
        status (str): Health status.
        timestamp (float): Timestamp of the health check.
        version (str): Application version.
        services (Dict[str, str]): Status of dependent services.
    """
    status: str
    timestamp: float
    version: str
    services: Dict[str, str]


# ==== UTILITY FUNCTIONS ======================================================= #

async def get_query_embedding(query: str) -> List[float]:
    """Return vector embedding for *query* using the configured provider.

    The first call lazily instantiates the provider via
    ``embeddings_adapter.get_embedding_model`` and caches the instance for
    subsequent requests to avoid redundant initialisation overhead.
    """
    global embedding_model  # pylint: disable=global-statement

    provider = config.EMBEDDING_PROVIDER
    logger.info("🔍 Using embedding provider: %s", provider)

    # Lazy instantiation --------------------------------------------------
    if embedding_model is None:
        try:
            embedding_model = get_embedding_model(provider=provider)
            logger.info("✓ Instantiated embedding model '%s'", provider)
        except EmbeddingError as exc:
            logger.error("Failed to load embedding model '%s': %s", provider, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Delegate embedding generation --------------------------------------
    try:
        return await embedding_model.embed_async_single(query)
    except Exception as exc:  # noqa: BLE001
        logger.error("Embedding generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def get_jina_embedding(query: str) -> List[float]:
    """
    Get embedding using Jina's Embeddings API with retries.

    Args:
        query (str): The query string to embed.

    Returns:
        List[float]: The embedding vector for the query.

    Raises:
        HTTPException: If the embedding service fails or times out.
    """
    for attempt in range(config.MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers=jina_headers,
                    json={
                        "input": [query],
                        "model": config.EMBEDDING_MODEL,
                        "task": "retrieval.query"
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]

        except httpx.TimeoutException:
            logger.warning(f"Jina API timeout on attempt {attempt + 1}")
            if attempt == config.MAX_RETRIES - 1:
                raise HTTPException(status_code=408, detail="Embedding service timeout")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                await asyncio.sleep(wait_time)
                continue
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Jina API error: {e}"
            )
        except Exception as e:
            if attempt == config.MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Embedding service error: {e}"
                )
            await asyncio.sleep(2 ** attempt)


async def rerank_documents(
    query: str,
    documents: List[Dict]
) -> List[Dict]:
    """
    Rerank documents using the configured reranker provider (Jina cloud API).

    Args:
        query (str): The query string.
        documents (List[Dict]): List of document dictionaries to rerank.

    Returns:
        List[Dict]: Reranked list of document dictionaries.
    """
    if not documents:
        return []

    provider = os.environ.get("RERANKER_PROVIDER", "jina").lower()
    logger.info(f"🔄 Using reranker provider: {provider}")
    logger.info(f"Reranking {len(documents)} documents with {provider}")
    total_chars = sum(
        len(doc.get("payload", {}).get("text", doc.get("text", ""))) for doc in documents
    )
    logger.info(f"Total document text size: {total_chars} characters")

    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(
            f"Memory usage before reranking: {memory_info.rss / 1024 / 1024:.2f} MB"
        )
    except ImportError:
        logger.debug("psutil not available, skipping memory usage logging")

    texts_to_rerank = [
        doc.get("payload", {}).get("text", doc.get("text", "")) for doc in documents
    ]

    if texts_to_rerank:
        sample_text = (
            texts_to_rerank[0][:200] + "..." if len(texts_to_rerank[0]) > 200 else texts_to_rerank[0]
        )
        logger.info(f"Sample document: {sample_text}")

    if provider == "jina":
        try:
            async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
                jina_model = os.getenv("RERANKER_MODEL", config.RERANKER_MODEL)
                logger.debug(
                    "[app] Jina rerank using model=%s token_prefix=%s… docs=%d",
                    jina_model,
                    config.EMBEDDING_MODEL_API_KEY[:8] if config.EMBEDDING_MODEL_API_KEY else "None",
                    len(texts_to_rerank)
                )

                logger.debug(f"Jina headers: {jina_headers}")

                payload = {
                    "model": jina_model,
                    "query": query,
                    "documents": texts_to_rerank,
                    "top_n": config.TOP_K_RERANK,
                }
                logger.debug(
                    f"Jina payload: model={payload['model']}, top_n={payload['top_n']}, "
                    f"query={payload['query'][:30]}..."
                )

                response = await client.post(
                    "https://api.jina.ai/v1/rerank",
                    headers=jina_headers,
                    json=payload,
                )

                logger.debug(f"Jina response status: {response.status_code}")
                if response.status_code != 200:
                    logger.error(
                        f"Jina API error: {response.status_code}, response: {response.text}"
                    )

                response.raise_for_status()
                reranked_results = response.json()["results"]

                final_docs: List[Dict] = []
                for res in reranked_results:
                    original_doc = documents[res["index"]]
                    final_docs.append(
                        {
                            "text": texts_to_rerank[res["index"]],
                            "source": original_doc.get("payload", {}).get("source", "unknown"),
                            "score": res["relevance_score"],
                            "payload": original_doc.get("payload", {}),
                            "metadata": original_doc.get("payload", {}),
                        }
                    )
                return final_docs
        except Exception as e:
            logger.error(f"Jina reranking failed: {e}")
            return documents[: config.TOP_K_RERANK]

    logger.warning(
        "Unknown RERANKER_PROVIDER '%s', returning original documents", provider
    )
    return documents[: config.TOP_K_RERANK]


async def generate_llm_response(
    query: str,
    context: List[Dict],
    temperature: float = 0.7
) -> str:
    """
    Generate response via the generic ChatAdapter (OpenAI-compatible).

    Args:
        query: User question.
        context: Retrieved context documents.
        temperature: LLM sampling temperature.

    Returns:
        Assistant answer as plain text.
    """

    context_str = "\n\n---\n\n".join([doc["text"] for doc in context])
    prompt = (
        "You are Sentio, an expert AI assistant with access to a curated knowledge "
        "base. Your responses should be:\n"
        "- Accurate and based solely on the provided context\n"
        "- Comprehensive yet concise\n"
        "- Well-structured and easy to understand\n"
        "- Professional in tone\n\n"
        "Context Information:\n" f"{context_str}\n\n"
        f"User Question: {query}\n\n"
        "Instructions:\n"
        "1. If the context contains relevant information, provide a detailed answer\n"
        "2. If the context is insufficient, clearly state what information is missing\n"
        "3. Always cite relevant sources when possible\n"
        "4. Be honest about limitations\n\n"
        "Answer:"
    )

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }

    try:
        result = await chat_completion(payload)
        if isinstance(result, dict):
            return result["choices"][0]["message"]["content"]
        # Streaming fall-back → concatenate chunks.
        chunks: List[str] = []
        async for part in result:  # type: ignore[assignment]
            chunks.append(part)
        return "".join(chunks).strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("ChatAdapter generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Language model error: {exc}") from exc


# ==== API ROUTES ============================================================== #

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Comprehensive health check with service status.

    Returns:
        HealthResponse: Health status and dependent service states.
    """
    services: Dict[str, str] = {}

    try:
        if qdrant_client:
            qdrant_client.get_collections()
            services["qdrant"] = "healthy"
        else:
            services["qdrant"] = "unavailable"
    except Exception:
        services["qdrant"] = "unhealthy"

    if config.CHAT_PROVIDER == "openrouter":
        try:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_api_key:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(
                        f"{os.getenv('OPENROUTER_URL', 'https://openrouter.ai/api/v1')}/models",
                        headers={"Authorization": f"Bearer {openrouter_api_key}"},
                    )
                    response.raise_for_status()
                    services["openrouter_llm"] = "healthy"
            else:
                services["openrouter_llm"] = "unavailable (API key not set)"
        except Exception as e:
            logger.error(f"OpenRouter LLM health check failed: {e}")
            services["openrouter_llm"] = "unhealthy"

    if config.EMBEDDING_PROVIDER == "jina":
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    response = await client.get(
                        "https://api.jina.ai/v1/models", headers=jina_headers
                    )
                    response.raise_for_status()
                    services["jina"] = "healthy"
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in [404, 403]:
                        logger.warning(
                            "Jina /models endpoint not available, probing /embeddings as fallback."
                        )
                        probe_response = await client.post(
                            "https://api.jina.ai/v1/embeddings",
                            headers=jina_headers,
                            json={"input": ["probe"], "model": config.EMBEDDING_MODEL}
                        )
                        probe_response.raise_for_status()
                        services["jina"] = "healthy (fallback probe)"
                    else:
                        raise
        except Exception as e:
            logger.error(f"Jina health check failed: {e}")
            services["jina"] = "unhealthy"

    reranker_provider = os.environ.get("RERANKER_PROVIDER", "jina").lower()
    if reranker_provider == "jina":
        if "jina" in services and services["jina"].startswith("healthy"):
            services["jina_reranker"] = "healthy"
        else:
            services["jina_reranker"] = "status depends on jina API"

    return HealthResponse(
        status="healthy" if all(
            status in ["healthy", "unavailable", "healthy (fallback probe)"]
            for status in services.values()
        ) else "degraded",
        timestamp=time.time(),
        version="2.0.0",
        services=services
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_handler(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request with Azure integration.

    Performs the following steps:
        1. Generate query embedding
        2. Retrieve relevant documents from vector store
        3. Rerank documents using semantic search
        4. Generate an answer using a language model
        5. Track request using Azure App Insights (if enabled)

    Args:
        request (ChatRequest): The chat request payload.

    Returns:
        ChatResponse: The chat response with answer and sources.

    Raises:
        HTTPException: If any processing step fails.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Processing chat request: {request_id}")

        if config.USE_AZURE:
            track_event("chat_request", {"request_id": request_id})

        query = request.question
        embedding = await get_query_embedding(query)

        try:
            collections = qdrant_client.get_collections()
            collection_exists = config.COLLECTION_NAME in [
                c.name for c in collections.collections
            ]

            if not collection_exists:
                logger.info(
                    "Collection %s does not exist. Creating with dim=%d",
                    config.COLLECTION_NAME,
                    len(embedding)
                )
                qdrant_client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=len(embedding),
                        distance=models.Distance.COSINE
                    ),
                )
            else:
                meta = qdrant_client.get_collection(config.COLLECTION_NAME)
                # Check collection metadata structure
                if (
                    hasattr(meta, "config")
                    and hasattr(meta.config, "params")
                    and hasattr(meta.config.params, "vectors")
                ):
                    # Qdrant <1.4 returns an object, >=1.4 may be a dict.
                    vectors_meta = meta.config.params.vectors  # type: ignore[attr-defined]
                    if isinstance(vectors_meta, dict):
                        current_dim = vectors_meta.get("size", len(embedding))  # type: ignore[assignment]
                    else:
                        current_dim = getattr(vectors_meta, "size", len(embedding))  # type: ignore[assignment]
                elif isinstance(meta, dict) and 'config' in meta and 'params' in meta['config'] and 'vectors' in meta['config']['params']:
                    # New API format - dictionary
                    current_dim = meta['config']['params']['vectors']['size']
                else:
                    # If we can't determine the dimension, use the current one
                    logger.warning("Cannot determine vector dimension from collection metadata, using current embedding dimension")
                    current_dim = len(embedding)
                
                if current_dim != len(embedding):
                    logger.warning(
                        "Vector dimension mismatch for collection %s (expected %d, got %d). Recreating collection.",
                        config.COLLECTION_NAME,
                        len(embedding),
                        current_dim
                    )
                    
                    # Deprecated method `recreate_collection` is replaced by explicit delete and create
                    try:
                        optimizers_config = qdrant_client.get_collection(config.COLLECTION_NAME).optimizers_config
                        qdrant_client.delete_collection(collection_name=config.COLLECTION_NAME)
                        qdrant_client.create_collection(
                            collection_name=config.COLLECTION_NAME,
                            vectors_config=models.VectorParams(
                                size=len(embedding),
                                distance=models.Distance.COSINE
                            ),
                            optimizers_config=optimizers_config
                        )
                        logger.info("Recreated collection with new embeddings.")
                    except Exception as e:
                        logger.error(f"Error during collection recreation: {e}")
                        # Depending on the desired behavior, you might want to re-raise or handle differently
                        raise HTTPException(status_code=500, detail="Could not recreate collection.")


        except Exception as e:
            logger.error(f"Error during collection check/creation: {e}")
            raise HTTPException(status_code=500, detail="Failed to ensure collection exists.")

        # ------------------------------------------------------------------
        # Qdrant collections that utilise *named vectors* (multi-vector
        # collections) require the caller to explicitly specify the vector
        # name in the ``using`` parameter.  When the collection was created
        # by a different ingestion pipeline (e.g. the `build_index` CLI), the
        # default name is often ``text-dense``.  If we omit the name Qdrant
        # responds with a 400 error and includes the required name in the
        # payload.  We therefore perform an optimistic query first and, on
        # failure, parse the response and retry with the advertised name.
        # ------------------------------------------------------------------
        try:
            query_response = qdrant_client.query_points(
                collection_name=config.COLLECTION_NAME,
                query=embedding,
                limit=request.top_k or config.TOP_K_RETRIEVAL,
            )
        except UnexpectedResponse as exc:
            msg = str(exc)
            # Example message:
            # "Wrong input: Collection requires specified vector name in the request, available names: text-dense"
            if (
                "requires specified vector name" in msg
                and "available names" in msg
            ):
                import re as _re  #  µ-import to keep local scope clean
                match = _re.search(r"available names:\s*([\w\-]+)", msg)
                if match:
                    vector_name = match.group(1)
                    logger.warning(
                        "Collection requires vector name '%s'. Retrying query using that vector.",
                        vector_name,
                    )
                    query_response = qdrant_client.query_points(
                        collection_name=config.COLLECTION_NAME,
                        query=embedding,
                        limit=request.top_k or config.TOP_K_RETRIEVAL,
                        using=vector_name,
                    )
                else:
                    # Pattern not found – re-raise the original exception.
                    raise
            else:
                # Unrelated error – bubble up.
                raise

        retrieved_docs = query_response.points

        documents = []
        for idx, doc in enumerate(retrieved_docs):
            if idx == 0:
                logger.debug(
                    "Sample retrieved payload: %s (score=%.4f)",
                    doc.payload,
                    doc.score
                )
            payload = doc.payload or {}

            text_segment = payload.get("text", "")
            if not text_segment and payload.get("_node_content"):
                try:
                    node_data = json.loads(payload["_node_content"])
                    text_segment = node_data.get("text", "")
                except Exception:
                    text_segment = ""

            source = (
                payload.get("source")
                or payload.get("file_name")
                or payload.get("file_path")
                or "unknown"
            )

            payload["text"] = text_segment
            payload.setdefault("source", source)

            documents.append({
                "text": text_segment,
                "payload": payload,
                "score": doc.score
            })

        reranked_docs = await rerank_documents(query, documents)
        filtered_docs = [
            d for d in reranked_docs if d["score"] >= config.MIN_RELEVANCE_SCORE
        ]
        if not filtered_docs:
            filtered_docs = reranked_docs
        top_docs = filtered_docs[: config.TOP_K_RERANK]
        logger.debug("Top_docs after relevance filtering: %d", len(top_docs))

        answer = await generate_llm_response(
            query, top_docs, request.temperature or 0.7
        )

        sources = []
        for doc in top_docs:
            payload_data = doc.get("payload") or doc.get("metadata") or {}
            src_val = payload_data.get("source", doc.get("source", "unknown"))
            sources.append(Source(
                text=doc["text"],
                source=src_val,
                score=doc["score"],
                metadata=payload_data
            ))

        response = ChatResponse(
            answer=answer,
            sources=sources,
            metadata={"request_id": request_id}
        )

        if config.USE_AZURE and azure_queue_client:
            try:
                message_payload = {
                    "event_type": "chat_completion",
                    "timestamp": time.time(),
                    "request_id": request_id,
                    "query": query,
                    "response_id": request_id,
                }
                azure_queue_client.send_message(message_payload)
                logger.debug(f"Sent event to Azure Queue: {request_id}")
            except Exception as e:
                logger.warning(f"Failed to send event to Azure Queue: {e}")

        if config.USE_AZURE:
            processing_time = time.time() - start_time
            track_metric(
                "chat_processing_time",
                processing_time,
                {"success": "true", "documents_retrieved": len(retrieved_docs)}
            )

        return response

    except Exception as e:
        if config.USE_AZURE:
            track_event(
                "chat_request_error",
                {"request_id": request_id, "error": str(e)}
            )
        logger.error(f"Error processing chat request {request_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", tags=["Ingestion"])
async def embed_document(request: EmbedRequest) -> Dict[str, Union[str, int, bool]]:
    """
    Embed a single document and enqueue for worker processing.

    For cloud (Azure) deployments, forward the payload to Azure Storage Queue so that
    the async worker can pick it up and perform heavy vectorization & indexing.
    For local Docker-Compose usage (no Azure Queue), synchronously index the text
    directly into Qdrant to keep the developer feedback loop tight.

    Args:
        request (EmbedRequest): The document embedding request.

    Returns:
        Dict[str, Union[str, int, bool]]: Embedding operation result.

    Raises:
        HTTPException: If embedding or indexing fails.
    """
    try:
        doc_id: Union[int, str]
        if request.id is None:
            doc_id = int(uuid.uuid4().int & (1 << 63) - 1)
        elif isinstance(request.id, int):
            doc_id = request.id
        elif isinstance(request.id, str):
            try:
                doc_id = str(uuid.UUID(request.id))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid document ID format: '{request.id}'. Must be an integer or a valid UUID string."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid document ID type."
            )

        payload = {
            "id": doc_id,
            "text": request.content,
            "source": request.content[:40],
            "metadata": request.metadata,
        }

        queued = False
        if config.USE_AZURE and azure_queue_client is not None:
            try:
                azure_queue_client.send_message(payload)
                queued = True
            except Exception as e:
                logger.warning(f"Azure queue unavailable, falling back to direct index: {e}")

        if not queued:
            embedding = await get_query_embedding(request.content)
            collections = qdrant_client.get_collections()
            collection_exists = config.COLLECTION_NAME in [
                c.name for c in collections.collections
            ]

            if not collection_exists:
                qdrant_client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=len(embedding),
                        distance=models.Distance.COSINE
                    ),
                )

            qdrant_client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )

        return {
            "status": "success",
            "queued": queued,
            "id": doc_id,
            "document_id": doc_id,
            "chunks_upserted": 1,
        }
    except Exception as e:
        logger.error(f"/embed failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(request: SearchRequest) -> SearchResponse:
    """
    Search for documents in the collection.

    Args:
        request (SearchRequest): The search request payload.

    Returns:
        SearchResponse: The search results.

    Raises:
        HTTPException: If search or embedding fails.
    """
    try:
        query_vector = await get_query_embedding(request.query)

        if not qdrant_client.collection_exists(collection_name=request.collection_name):
            raise HTTPException(status_code=404, detail="Collection not found")

        res = qdrant_client.search(
            collection_name=request.collection_name,
            query_vector=query_vector,
            limit=request.limit,
        )

        return SearchResponse(results=[r.dict() for r in res])
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", tags=["System"])
async def system_info() -> Dict[str, Union[str, Dict[str, Union[str, int]]]]:
    """
    Get system information and configuration.

    Returns:
        Dict[str, Union[str, Dict[str, Union[str, int]]]]: System info and config.
    """
    return {
        "name": "Sentio RAG System",
        "version": "2.0.0",
        "configuration": {
            "collection_name": config.COLLECTION_NAME,
            "embedding_provider": config.EMBEDDING_PROVIDER,
            "embedding_model": config.EMBEDDING_MODEL,
            "reranker_model": config.RERANKER_MODEL,
            "openrouter_model": os.getenv("OPENROUTER_MODEL"),
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_rerank": config.TOP_K_RERANK
        }
    }


# ==== ERROR HANDLERS ========================================================== #

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> Response:
    """
    Custom HTTP exception handler with logging.

    Args:
        request (Request): The incoming request.
        exc (HTTPException): The HTTP exception.

    Returns:
        Response: JSON error response.
    """
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return Response(
        content=f'{{"error": "{exc.detail}", "status_code": {exc.status_code}}}',
        status_code=exc.status_code,
        media_type="application/json"
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
) -> Response:
    """
    General exception handler for unexpected errors.

    Args:
        request (Request): The incoming request.
        exc (Exception): The exception.

    Returns:
        Response: JSON error response.
    """
    logger.error(f"Unhandled exception: {exc} - {request.url}", exc_info=True)
    return Response(
        content='{"error": "Internal server error", "status_code": 500}',
        status_code=500,
        media_type="application/json"
    )


# ==== DEVELOPMENT SERVER ENTRYPOINT ========================================== #

if __name__ == "__main__":
    import asyncio

    logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8910,
        reload=os.getenv("ENVIRONMENT") != "production",
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    ) 
