"""Chat handler for RAG query processing.

This module implements the core chat functionality using the LangGraph
pipeline with proper error handling and monitoring.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from src.core.caching import get_cache_manager
from src.core.graph.factory import GraphConfig, build_basic_graph
from src.core.graph.state import RAGState, create_initial_state
from src.core.llm.factory import create_generator
from src.core.rerankers import get_reranker
from src.core.resilience.fallbacks import fallback_manager, llm_fallback
from src.core.retrievers.factory import create_retriever_for_graph
from src.utils.settings import settings

logger = logging.getLogger(__name__)


class ChatHandler:
    """Handles chat requests with full RAG pipeline integration.
    
    Manages the complete flow from query to response using LangGraph,
    with fallback mechanisms and proper error handling.
    """

    def __init__(self):
        self._graph = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._cache_manager = get_cache_manager()

    async def _ensure_initialized(self):
        """Ensure the graph is initialized (lazy loading)."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                logger.info("Initializing RAG graph components...")

                # Create retriever
                retriever = create_retriever_for_graph()

                # Create reranker if enabled
                reranker = None
                if settings.reranker_model:
                    try:
                        reranker = get_reranker(kind="jina")
                        logger.info("Reranker initialized successfully")
                    except Exception as e:
                        logger.warning(f"Failed to initialize reranker: {e}")

                # Create LLM generator
                llm_generator = create_generator(
                    provider=settings.llm_provider,
                    model=settings.chat_llm_model,
                    api_key=settings.chat_llm_api_key,
                    base_url=settings.chat_llm_base_url,
                )

                # Build graph with configuration
                graph_config = GraphConfig(
                    retriever=retriever,
                    reranker=reranker,
                    llm=llm_generator,
                    retrieval_top_k=settings.top_k_retrieval,
                    reranking_top_k=settings.top_k_rerank,
                    selection_top_k=settings.selection_top_k,
                    max_tokens=2000,
                )

                self._graph = build_basic_graph(graph_config)
                self._initialized = True
                logger.info("RAG graph initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize RAG graph: {e}")
                raise

    async def process_chat_request(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int = 5,
        temperature: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a chat request through the RAG pipeline.

        Args:
            question: User's question
            history: Previous conversation history
            top_k: Number of documents to retrieve
            temperature: LLM generation temperature
            metadata: Additional request metadata

        Returns:
            Dictionary containing response and metadata

        Raises:
            Exception: If processing fails after all fallbacks
        """
        request_start = time.time()
        query_id = str(uuid.uuid4())

        logger.info(f"Processing chat request {query_id}: {question[:100]}...")

        try:
            # Check cache first - DISABLED FOR DEBUGGING
            # cache_params = {"top_k": top_k, "temperature": temperature}
            # cached_result = await self._cache_manager.get_query_cache(question, cache_params)

            # if cached_result:
            #     logger.info(f"Cache hit for query {query_id}")
            #     cached_result["metadata"]["from_cache"] = True
            #     cached_result["metadata"]["query_id"] = query_id
            #     return cached_result

            # Ensure graph is initialized
            await self._ensure_initialized()

            # Create cache key for potential fallback
            cache_key = fallback_manager.generate_cache_key(
                question, {"top_k": top_k, "temperature": temperature}
            )

            # Prepare RAG state and pass user controls via metadata
            rag_state = create_initial_state(question)
            rag_state["metadata"].update({
                "query_id": query_id,
                "temperature": temperature,
                "request_timestamp": request_start,
                "user_top_k": top_k,
                **(metadata or {}),
            })

            # Execute RAG pipeline
            try:
                result = await self._graph.ainvoke(
                    rag_state,
                    config={"configurable": {"thread_id": query_id}}
                )

                # DEBUG: Log the result object structure
                logger.info(f"DEBUG - Result type: {type(result)}")
                logger.info(f"DEBUG - Result keys: {list(result.keys()) if hasattr(result, 'keys') else 'No keys method'}")
                if hasattr(result, '__dict__'):
                    logger.info(f"DEBUG - Result attributes: {list(result.__dict__.keys())}")
                
                # Log document counts at each stage
                retrieved_docs = result.get('retrieved_documents', [])
                reranked_docs = result.get('reranked_documents', [])
                selected_docs = result.get('selected_documents', [])
                
                logger.info(f"DEBUG - Retrieved documents: {len(retrieved_docs)}")
                logger.info(f"DEBUG - Reranked documents: {len(reranked_docs)}")
                logger.info(f"DEBUG - Selected documents: {len(selected_docs)}")

                # Extract response and sources
                response_text = result.get("response", "")
                documents = result.get("selected_documents", [])

                logger.info(f"Retrieved {len(documents)} documents from RAG pipeline")
                logger.info(f"Response text: {response_text[:200]}...")

                # Prepare sources
                sources = []
                for i, doc in enumerate(documents):
                    logger.info(f"Document {i}: text='{doc.text[:100]}...', metadata={doc.metadata}")
                    sources.append({
                        "text": doc.text,
                        "content": doc.text,  # Add content field for API compatibility
                        "source": doc.metadata.get("source", "unknown"),
                        "score": float(doc.metadata.get("score", 0.0)),
                        "metadata": doc.metadata,
                        "debug_text_length": len(doc.text),  # Debug info
                    })

                processing_time = time.time() - request_start

                # Cache successful response
                response_data = {
                    "answer": response_text,
                    "sources": sources,
                    "metadata": {
                        "query_id": query_id,
                        "processing_time": processing_time,
                        "model_used": settings.chat_llm_model,
                        "documents_retrieved": len(documents),
                        "success": True,
                    }
                }

                # Cache in both fallback manager and main cache - DISABLED FOR DEBUGGING
                # fallback_manager.cache_response(cache_key, response_data, ttl_seconds=1800)
                # await self._cache_manager.set_query_cache(
                #     question, cache_params, response_data, ttl=1800
                # )

                logger.info(
                    f"Chat request {query_id} completed successfully in {processing_time:.2f}s"
                )

                return response_data

            except Exception as e:
                logger.warning(f"RAG pipeline failed for {query_id}: {e}")

                # Try to get cached response
                cached_response = fallback_manager.get_cached_response(cache_key)
                if cached_response:
                    logger.info(f"Using cached response for {query_id}")
                    cached_response["metadata"]["from_cache"] = True
                    return cached_response

                # Use LLM fallback
                fallback_response = await llm_fallback.generate_fallback_response(
                    query=question,
                    context_docs=None,
                    response_type="search" if "retriev" in str(e).lower() else "default"
                )

                processing_time = time.time() - request_start

                return {
                    "answer": fallback_response,
                    "sources": [],
                    "metadata": {
                        "query_id": query_id,
                        "processing_time": processing_time,
                        "fallback_used": True,
                        "error": str(e),
                        "success": False,
                    }
                }

        except Exception as e:
            processing_time = time.time() - request_start
            logger.error(f"Chat request {query_id} failed completely: {e}")

            return {
                "answer": "I apologize, but I'm currently experiencing technical difficulties. Please try again in a few moments.",
                "sources": [],
                "metadata": {
                    "query_id": query_id,
                    "processing_time": processing_time,
                    "error": str(e),
                    "success": False,
                }
            }

    async def health_check(self) -> dict[str, Any]:
        """Check health of chat handler components.

        Returns:
            Health status dictionary
        """
        health_status = {
            "chat_handler": "healthy",
            "graph_initialized": self._initialized,
            "components": {}
        }

        if self._initialized and self._graph:
            try:
                # Test basic graph functionality
                test_state = create_initial_state("test query")
                test_state["metadata"]["health_check"] = True

                # Run a minimal test (just retrieval step)
                await asyncio.wait_for(
                    self._graph.get_graph().get_node("retriever")(test_state),
                    timeout=5.0
                )

                health_status["components"]["retriever"] = "healthy"

            except TimeoutError:
                health_status["components"]["retriever"] = "timeout"
                health_status["chat_handler"] = "degraded"
            except Exception as e:
                health_status["components"]["retriever"] = f"error: {e!s}"
                health_status["chat_handler"] = "unhealthy"

        return health_status
