"""Sentio RAG System - FastAPI Application (LangGraph Version)

A production-ready RAG system built with LangGraph architecture:
- High-performance vector retrieval with resilience patterns
- Intelligent reranking and hybrid search
- Comprehensive security and monitoring
- Rate limiting and input validation
- OpenAI-compatible API with fallback mechanisms
"""

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, field_validator
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

from src.core.dependencies import (
    check_dependency_health,
    get_auth_manager_dep,
    get_chat_handler,
    get_container,
    get_health_handler,
    get_ingestor,
    get_vector_store_dep,
)
from src.core.ingest import DocumentIngestor
from src.observability import (
    instrument_fastapi,
    instrument_http_clients,
    performance_monitor,
    resource_monitor,
    setup_tracing,
    track_request_metrics,
)
from src.utils.auth import (
    AuthManager,
    AuthScope,
    TokenData,
)
from src.utils.exceptions import (
    ErrorHandler,
    SentioException,
)
from src.utils.security import (
    InputValidator,
    SecurityHeaders,
    setup_log_sanitization,
)
from src.utils.settings import settings

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set up security features
setup_log_sanitization()

# Initialize observability
setup_tracing(
    service_name="sentio-rag-api",
    service_version="3.0.0",
    jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
    console_export=settings.log_level.upper() == "DEBUG",
)

# Simple in-memory rate limiter
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        async with self.lock:
            now = datetime.now()
            # Clean old requests
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < timedelta(seconds=window_seconds)
            ]
            
            if len(self.requests[key]) >= max_requests:
                return False
            
            self.requests[key].append(now)
            return True

rate_limiter = RateLimiter()

# Initialize authentication manager
auth_manager = AuthManager()

# Required authentication
security = HTTPBearer()


# Allowed CORS origins including Streamlit UI
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://sentio-ui:8501,http://localhost:8501",
).split(",")


# Data Models
class ChatRequest(BaseModel):
    """Chat request with comprehensive validation.

    Attributes:
        question (str): User's question.
        history (Optional[List[Dict[str, str]]]): Chat history.
        top_k (Optional[int]): Number of results to return.
        temperature (Optional[float]): Generation temperature.
    """
    question: str = Field(
        ..., min_length=1, max_length=2000, description="User's question"
    )
    history: list[dict[str, str]] | None = Field(
        default_factory=list, description="Chat history"
    )
    top_k: int | None = Field(
        default=3, ge=1, le=20, description="Number of results to return"
    )
    temperature: float | None = Field(
        default=0.7, ge=0.0, le=2.0, description="Generation temperature"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        return InputValidator.validate_query(v)


class Source(BaseModel):
    """Source document with metadata.

    Attributes:
        text (str): Source text content.
        source (str): Document source identifier.
        score (float): Relevance score.
        metadata (Optional[Dict]): Additional metadata.
    """
    text: str = Field(..., description="Source text content")
    source: str = Field(..., description="Document source identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: dict | None = Field(default=None, description="Additional metadata")


class ChatResponse(BaseModel):
    """Chat response with sources and metadata.

    Attributes:
        answer (str): Generated answer.
        sources (List[Source]): Source documents used.
        metadata (Optional[Dict]): Response metadata.
    """
    answer: str = Field(..., description="Generated answer")
    sources: list[Source] = Field(..., description="Source documents used")
    metadata: dict | None = Field(default=None, description="Response metadata")


class EmbedRequest(BaseModel):
    """Request for document embedding/ingestion.

    Attributes:
        id (Optional[Union[int, str]]): Document identifier.
        content (str): Raw text content to embed.
        metadata (Optional[Dict]): Arbitrary metadata for the document.
    """
    id: int | str | None = Field(
        default=None, description="Document identifier (auto-generated if omitted)"
    )
    content: str = Field(
        ..., min_length=1, max_length=50000, description="Raw text content to embed"
    )
    metadata: dict | None = Field(default=None, description="Arbitrary metadata for the document")


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status (str): Health status.
        timestamp (float): Timestamp of the health check.
        version (str): Application version.
        services (Dict[str, str]): Status of dependent services.
    """
    status: str
    timestamp: float
    version: str
    services: dict[str, str]


async def startup_tasks():
    """Initialize all application dependencies and services on startup."""
    try:
        logger.info("Starting application startup tasks...")
        # Initialize all dependencies
        logger.info("About to call container.initialize_all()")
        await container.initialize_all()
        logger.info("All application dependencies initialized")

        # Start monitoring
        logger.info("Starting resource monitoring...")
        # Temporarily disable resource monitoring to avoid blocking
        # await resource_monitor.start_monitoring(interval_seconds=30.0)
        logger.info("Resource monitoring started (disabled for debugging)")
    except Exception as e:
        logger.error(f"Failed to start services: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


async def shutdown_tasks():
    """Clean up all services and dependencies on shutdown."""
    try:
        # Stop monitoring
        await resource_monitor.stop_monitoring()
        logger.info("Resource monitoring stopped")

        # Clean up all dependencies
        await container.cleanup()
        logger.info("All dependencies cleaned up")
    except Exception as e:
        logger.error(f"Error stopping services: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    await startup_tasks()
    yield
    await shutdown_tasks()


# Initialize FastAPI app with security
app = FastAPI(
    title="Sentio RAG API",
    description="Production-ready LangGraph-based Retrieval-Augmented Generation system",
    version="3.0.0",
    docs_url="/docs" if settings.log_level.upper() == "DEBUG" else None,
    redoc_url="/redoc" if settings.log_level.upper() == "DEBUG" else None,
    lifespan=lifespan,
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware."""
    client_ip = request.client.host if request.client else "unknown"
    
    # Rate limits: 100 requests per minute for most endpoints, 10 per minute for embed
    if request.url.path == "/embed":
        max_requests, window = 10, 60
    else:
        max_requests, window = 100, 60
    
    if not await rate_limiter.is_allowed(client_ip, max_requests, window):
        return Response(
            content=json.dumps({"detail": "Rate limit exceeded"}),
            status_code=429,
            media_type="application/json"
        )
    
    return await call_next(request)

# Add comprehensive error handlers
@app.exception_handler(SentioException)
async def sentio_exception_handler(request: Request, exc: SentioException):
    """Handle custom Sentio exceptions."""
    return await ErrorHandler.handle_exception(exc, request)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTP exceptions."""
    return await ErrorHandler.handle_exception(exc, request)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    return await ErrorHandler.handle_exception(exc, request)

# Set up observability instrumentation
instrument_fastapi(app)
instrument_http_clients()


# Security middleware
# @app.middleware("http")
# async def security_middleware(request: Request, call_next):
#     """Comprehensive security middleware."""
#     # Generate request ID for tracking
#     request_id = str(uuid.uuid4())
#     request.state.request_id = request_id

#     # CSRF validation for state-changing methods
#     if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
#         csrf_token = request.headers.get("X-CSRF-Token")
#         origin = request.headers.get("origin")
#         csrf_exempt = {"/embed", "/chat"}
#         if (
#             not csrf_token
#             and request.url.path not in ["/auth/login", "/auth/token"]
#             and not (request.url.path in csrf_exempt and origin in allowed_origins)
#         ):
#             return Response(
#                 content=json.dumps({"detail": "CSRF token required"}),
#                 status_code=403,
#                 media_type="application/json"
#             )

#     # Process request
#     start_time = time.time()
#     response = await call_next(request)
#     process_time = time.time() - start_time

#     # Add security headers
#     for name, value in SecurityHeaders.get_security_headers().items():
#         response.headers[name] = value

#     # Add custom headers
#     response.headers["X-Request-ID"] = request_id
#     response.headers["X-Process-Time"] = str(process_time)

#     # Log request for audit
#     if request.url.path not in ["/health", "/metrics"]:
#         await auth_manager.log_security_event(
#             event_type="request",
#             action=f"{request.method} {request.url.path}",
#             result=str(response.status_code),
#             ip_address=request.client.host,
#             user_agent=request.headers.get("user-agent"),
#             details={
#                 "request_id": request_id,
#                 "process_time": process_time,
#                 "status_code": response.status_code,
#             }
#         )

#     return response

# Enable CORS with strict security
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST"],
#     allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
#     max_age=3600,
#     expose_headers=["X-Total-Count", "X-Request-ID"],
# )

# If auth is disabled, we can also loosen CORS for local dev
if not settings.auth_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )


# Dependency functions for auth
async def get_metrics_auth():
    """Get auth dependency for metrics endpoints."""
    if not settings.auth_enabled:
        return None
    auth_manager = await get_auth_manager_dep()
    return auth_manager.require_scopes([AuthScope.METRICS])

async def get_delete_auth():
    """Get auth dependency for delete endpoints."""
    if not settings.auth_enabled:
        return None
    auth_manager = await get_auth_manager_dep()
    return auth_manager.require_scopes([AuthScope.DELETE, AuthScope.ADMIN])


# Dependency container for managing application state
container = get_container()


@app.get("/health", response_model=HealthResponse)
async def health_check(
    health_handler: Any = Depends(get_health_handler)
) -> HealthResponse:
    """Basic health check endpoint for load balancers.
    """
    result = await health_handler.basic_health_check()

    # Check dependency health
    dependency_health = await check_dependency_health()

    return HealthResponse(
        status=result["status"],
        timestamp=result["timestamp"],
        version=result["version"],
        services=dependency_health
    )


@app.get("/health/detailed")
async def detailed_health_check(
    health_handler: Any = Depends(get_health_handler)
) -> dict[str, Any]:
    """Comprehensive health check with component details.
    """
    base_health = await health_handler.detailed_health_check()
    dependency_health = await check_dependency_health()

    # Merge health information
    base_health["dependencies"] = dependency_health

    return base_health


@app.get("/health/ready")
async def readiness_check(
    health_handler: Any = Depends(get_health_handler)
) -> dict[str, Any]:
    """Kubernetes readiness check.
    """
    base_readiness = await health_handler.readiness_check()
    dependency_health = await check_dependency_health()

    # Check if all dependencies are healthy
    all_healthy = all(
        status == "healthy" for status in dependency_health.values()
    )

    if not all_healthy:
        base_readiness["status"] = "not_ready"
        base_readiness["dependencies"] = dependency_health

    return base_readiness


@app.get("/health/live")
async def liveness_check(
    health_handler: Any = Depends(get_health_handler)
) -> dict[str, Any]:
    """Kubernetes liveness check.
    """
    return await health_handler.liveness_check()


@app.post("/embed")
async def embed_document(
    request: EmbedRequest,
    request_obj: Request,
    ingestor: DocumentIngestor = Depends(get_ingestor),
    # Auth optional via env toggle
    # auth_manager: AuthManager = Depends(get_auth_manager_dep),
    # token_data: TokenData = Depends(auth_manager.require_scopes([AuthScope.EMBED]))
) -> dict[str, Any]:
    """Embed a document and store it in the vector database.
    """
    request_id = getattr(request_obj.state, "request_id", "unknown")
    logger.info("/embed called request_id=%s", request_id)
    try:

        doc_id = request.id or str(uuid.uuid4())

        # Validate and sanitize input
        content = InputValidator.validate_document_content(request.content)
        metadata = InputValidator.validate_metadata(
            request.metadata or {
                "source": "api_upload",
                "timestamp": time.time()
            }
        )

        # Create document with validated data
        from src.core.models.document import Document
        doc = Document(
            id=str(doc_id),
            text=content,
            metadata=metadata
        )

        # Use public ingestion API to process a single document
        result = await ingestor.ingest_document(doc)

        return {
            "status": "success",
            "id": doc_id,
            "chunks_created": result.get("chunks_created", 0),
            "embeddings_generated": result.get("embeddings_generated", 0),
        }

    except Exception as e:
        logger.error(f"Error embedding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    request_obj: Request,
    chat_handler: Any = Depends(get_chat_handler),
    # Auth optional via env toggle
    # auth_manager: AuthManager = Depends(get_auth_manager_dep),
    # token_data: TokenData = Depends(auth_manager.require_scopes([AuthScope.CHAT]))
) -> ChatResponse:
    """Process a chat request using the full RAG pipeline with observability.
    """
    with track_request_metrics("/chat"):
        try:
            # Process the request
            result = await chat_handler.process_chat_request(
                question=request.question,
                history=request.history,
                top_k=request.top_k,
                temperature=request.temperature,
            )

            # Record custom metrics
            performance_monitor.record_value("chat.requests", 1.0, {"status": "success", "user": "debug"})
            performance_monitor.record_value("chat.query_length", len(request.question))
            performance_monitor.record_value("chat.sources_returned", len(result["sources"]))

            # Skip auth-related logging for debugging
            # await auth_manager.log_security_event(...)

            return ChatResponse(
                answer=result["answer"],
                sources=[
                    Source(
                        text=source.get("content", source.get("text", "")),
                        source=source["source"],
                        score=source["score"],
                        metadata=source.get("metadata", {})
                    )
                    for source in result["sources"]
                ],
                metadata=result["metadata"]
            )

        except Exception as e:
            performance_monitor.record_value("chat.requests", 1.0, {"status": "error", "user": "debug"})
            logger.error(f"Error processing chat request: {e}")

            # Skip auth-related logging for debugging
            # await auth_manager.log_security_event(...)

            raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/clear")
async def clear_collection(
    vector_store: Any = Depends(get_vector_store_dep),
    token_data: TokenData | None = Depends(get_delete_auth),
) -> dict[str, str]:
    """Clear the vector store collection.
    """
    # If auth disabled, proceed without token validation
    try:

        # Method depends on the specific vector store implementation
        if hasattr(vector_store, "delete_collection"):
            # QdrantStore has delete_collection
            collection_name = settings.collection_name
            vector_store.delete_collection(collection_name)

            # Recreate collection
            if hasattr(vector_store, "create_collection"):
                from src.core.embeddings import get_embedder
                vector_embedder = get_embedder(settings.embedder_name)
                vector_size = vector_embedder.dimension

                vector_store.create_collection(
                    collection_name=collection_name,
                    vector_size=vector_size
                )

        return {"status": "success", "message": "Collection cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def system_info() -> dict[str, Any]:
    """Get system information with performance metrics."""
    try:
        resource_summary = resource_monitor.get_resource_summary()
    except Exception as e:
        logger.error(f"resource_monitor.get_resource_summary() failed: {e}")
        resource_summary = {}
    try:
        resource_health = resource_monitor.check_resource_health()
    except Exception as e:
        logger.error(f"resource_monitor.check_resource_health() failed: {e}")
        resource_health = {}

    return {
        "name": "Sentio LangGraph RAG System",
        "version": "3.0.0",
        "configuration": {
            "collection_name": settings.collection_name,
            "embedding_provider": settings.embedder_name,
            "vector_store": settings.vector_store_name,
            "chunking_strategy": settings.chunking_strategy,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "performance": {
            "resource_summary": resource_summary,
            "resource_health": resource_health,
        }
    }


@app.get("/metrics")
async def get_metrics(
    token_data: TokenData | None = Depends(get_metrics_auth),
):
    """Prometheus-compatible metrics endpoint.
    """
    from src.observability.metrics import metrics_collector

    metrics_data = metrics_collector.get_metrics_export()

    if metrics_data.startswith("{"):
        # JSON format (fallback metrics)
        return Response(
            content=metrics_data,
            media_type="application/json"
        )
    # Prometheus format
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.get("/metrics/performance")
async def get_performance_metrics(
    token_data: TokenData | None = Depends(get_metrics_auth),
) -> dict[str, Any]:
    """Get detailed performance metrics in JSON format.
    """
    return {
        "resource_summary": resource_monitor.get_resource_summary(),
        "resource_trends": resource_monitor.get_resource_trends(60),
        "resource_health": resource_monitor.check_resource_health(),
        "performance_history": performance_monitor.get_all_metrics_summary(),
    }
