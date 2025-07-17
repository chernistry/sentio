
#!/usr/bin/env python3
"""
Sentio RAG Pipeline - Enterprise-grade RAG orchestration system.

This module provides a comprehensive, production-ready RAG pipeline
with advanced retrieval, reranking, and generation capabilities.
"""




# ==== MODULE IMPORTS & DEPENDENCIES ==== #


import asyncio
import logging
import os
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union, Any
import uuid
import random
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential  # robust async retries
from root.src.core.llm.chat_adapter import chat_completion

import httpx
from httpx import HTTPError  # granular exception handling
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document as _LlamaDoc
from llama_index.core.storage import StorageContext
try:
    from llama_index.vector_stores.qdrant import QdrantVectorStore
except ImportError:
    from llama_index_vector_stores_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .tasks.chunking import ChunkingStrategy, TextChunker
from .tasks.embeddings import EmbeddingModel, EmbeddingError
from .retrievers import HybridRetriever
# The concrete reranker implementation is resolved at runtime via the
# adapter-based orchestration layer (see ``core.rerank``).
from .plugin_manager import PluginManager
from .llm.prompt_builder import PromptBuilder
from root.src.utils.settings import settings

logger = logging.getLogger(__name__)
TEXT_VECTOR_NAME: str = os.getenv("QDRANT_VECTOR_NAME", "text-dense")




# ==== EXCEPTIONS & ENUMS ==== #



class PipelineError(Exception):
    """Exception raised for errors occurring within the pipeline.

    This is the base exception for all pipeline-related errors.
    """
    pass


class RetrievalStrategy(Enum):
    """Enumeration of available retrieval strategies.

    Attributes:
        DENSE: Dense vector search only.
        HYBRID: Dense + sparse fusion.
        SEMANTIC: Semantic-aware retrieval.
    """
    DENSE = "dense"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


class GenerationMode(Enum):
    """Enumeration of generation modes for different use cases.

    Attributes:
        BALANCED: Balance speed and quality.
        FAST: Optimize for speed.
        QUALITY: Optimize for quality.
        CREATIVE: Higher temperature, more creative.
    """
    BALANCED = "balanced"
    FAST = "fast"
    QUALITY = "quality"
    CREATIVE = "creative"




# ==== DATA CLASSES ==== #



@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline.

    Args:
        collection_name: Name of the Qdrant collection to use.
        data_dir: Optional path to the data directory for document ingestion.
        retrieval_strategy: The retrieval strategy to use (dense, hybrid, semantic).
        top_k_retrieval: Number of documents to retrieve initially.
        top_k_final: Number of top documents to use for generation.
        min_relevance_score: Minimum relevance score threshold for filtering results.
        generation_mode: The mode for answer generation (fast, balanced, quality, creative).
        temperature: Default temperature for generation.
        max_tokens: Optional maximum number of tokens for generation.
        chunk_size: Size of each text chunk during chunking.
        chunk_overlap: Overlap between chunks.
        chunking_strategy: The strategy for chunking text (sentence, paragraph, etc.).
        cache_enabled: Whether to enable embedding/model caching.
        max_retries: Maximum number of retries for transient errors.
        timeout: Timeout in seconds for external requests.
        enable_reranking: Whether to enable reranking of retrieved documents.
        enable_source_filtering: Whether to enable filtering by source relevance.

    Attributes:
        See Args above.
    """
    collection_name: str = "Sentio_docs"
    data_dir: Optional[Path] = None
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k_retrieval: int = 10
    top_k_final: int = 3
    min_relevance_score: float = 0.1
    generation_mode: GenerationMode = GenerationMode.BALANCED
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE
    cache_enabled: bool = True
    max_retries: int = 3
    timeout: int = 60
    enable_reranking: bool = True
    enable_source_filtering: bool = True


@dataclass(slots=True, kw_only=True)
class RetrievalResult:
    """Container for retrieval results with metadata."""

    documents: List[Dict]
    query: str
    strategy: str
    total_time: float
    sources_found: int
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True, kw_only=True)
class GenerationResult:
    """Container for generation results with metadata."""

    answer: str
    sources: List[Dict]
    query: str
    mode: str
    total_time: float
    token_count: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


class SentioRAGPipeline:
    """
    Enterprise-grade RAG pipeline with advanced features.
    
    This pipeline provides a complete RAG solution with:
    - Multiple retrieval strategies
    - Intelligent document processing
    - Advanced reranking and filtering
    - Streaming generation capabilities
    - Comprehensive monitoring and analytics
    - Fault tolerance and graceful degradation
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, *, plugins: PluginManager | None = None):
        """
        Initialize the RAG pipeline with configuration.
        
        Args:
            config: Pipeline configuration object.
        """
        self.config = config or PipelineConfig()
        self.plugin_manager = plugins or PluginManager()
        self.initialized = False

        self.embed_model: Optional[EmbeddingModel] = None
        self.chunker: Optional[TextChunker] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.vector_store: Optional[QdrantVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[HybridRetriever] = None
        from typing import Any  # local import to avoid heavy static deps

        self.reranker: Optional[Any] = None  # resolved lazily

        # Centralised prompt builder (re-usable across components)
        self.prompt_builder: PromptBuilder = PromptBuilder()

        self.stats = {
            'queries_processed': 0,
            'total_retrieval_time': 0.0,
            'total_generation_time': 0.0,
            'cache_hits': 0,
            'errors': 0,
            'avg_sources_per_query': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in RetrievalStrategy}
        }

        self._generation_configs = {
            GenerationMode.FAST: {'temperature': 0.3, 'max_tokens': 150},
            GenerationMode.BALANCED: {'temperature': 0.7, 'max_tokens': 300},
            GenerationMode.QUALITY: {'temperature': 0.5, 'max_tokens': 500},
            GenerationMode.CREATIVE: {'temperature': 0.9, 'max_tokens': 400}
        }
        # retry back‑off (seconds)
        self._base_delay: float = 0.5
        # Shared HTTP client (lazy‑initialised)
        self.http_client: Optional[httpx.AsyncClient] = None
    async def _post_with_retries(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        headers: Dict[str, str],
        json: Dict,
        max_retries: int,
    ) -> httpx.Response:
        """
        POST with exponential back‑off retry for transient HTTP errors.

        Args:
            client: Shared HTTPX AsyncClient.
            url: Target URL.
            headers: HTTP headers.
            json: JSON payload.
            max_retries: Maximum number of retry attempts.

        Returns:
            HTTPX Response object.

        Raises:
            HTTPError: Propagated if all retries fail.
        """
        retryer = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=self._base_delay, min=self._base_delay, max=8.0),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                response = await client.post(
                    url, headers=headers, json=json, timeout=self.config.timeout
                )
                response.raise_for_status()
                return response



    async def initialize(self) -> None:
        """Initialize the pipeline components."""
        logger.info("🚀 Initializing Sentio RAG Pipeline...")

        try:
            # Load plugins before initializing other components
            self.plugin_manager.load_from_env()

            self.embed_model = EmbeddingModel(
                provider=settings.embedding_provider,
                model_name=settings.embedding_model,
                cache_enabled=self.config.cache_enabled,
                batch_size=settings.embedding_batch_size,
                max_retries=self.config.max_retries,
                allow_empty_api_key=True,
            )
            logger.info('✓ Embedding model ready')

            logger.info('Initializing text chunker...')
            self.chunker = await TextChunker.create(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                strategy=self.config.chunking_strategy,
            )
            logger.info('✓ Text chunker ready')

            logger.info('Initializing vector database...')
            await self._setup_vector_store()
            logger.info('✓ Vector database ready')

            logger.info('Initializing retrieval components...')
            await self._setup_retrievers()
            logger.info('✓ Retrieval components ready')

            if self.config.enable_reranking:
                reranker_provider = os.environ.get('RERANKER_PROVIDER', 'local').lower()
                logger.info(
                    'Initializing reranker (provider: %s)...', reranker_provider
                )
                # Dispatch to the new adapter-based orchestration layer.
                from root.src.core.tasks.rerank import RerankTask

                self.reranker = RerankTask(provider=reranker_provider)

                logger.info("✓ Reranker ready")

            # Register plugins which may override components
            self.plugin_manager.register_all(self)

            self.initialized = True
            logger.info('🎉 Sentio RAG Pipeline initialization complete!')
        except Exception as e:
            logger.error(f'Pipeline initialization failed: {e}')
            raise PipelineError(f'Failed to initialize pipeline: {e}')



    async def _setup_vector_store(self) -> None:
        """Set up Qdrant client and vector store."""
        qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        qdrant_api_key_header = os.getenv('QDRANT_API_KEY_HEADER', 'api-key')
        try:
            client_kwargs: Dict[str, Any] = {
                'url': qdrant_url,
            }
            if qdrant_api_key:
                client_kwargs['api_key'] = qdrant_api_key
                if qdrant_api_key_header.lower() != 'api-key':
                    client_kwargs['api_key_header'] = qdrant_api_key_header
            client_kwargs['prefer_grpc'] = False
            self.qdrant_client = QdrantClient(**client_kwargs)
            collections = self.qdrant_client.get_collections()
            logger.debug(f'Connected to Qdrant with {len(collections.collections)} collections')
            TEXT_VECTOR_NAME: str = os.getenv("QDRANT_VECTOR_NAME", "text-dense")
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.config.collection_name,
                vector_name=TEXT_VECTOR_NAME,
            )
            await self._setup_index()
        except Exception as e:
            raise PipelineError(f'Vector store setup failed: {e}')



    async def _setup_index(self) -> None:
        """Set up or load the vector index."""
        try:
            collection_exists = self.qdrant_client.collection_exists(
                collection_name=self.config.collection_name
            )
            if collection_exists:
                collection_info = self.qdrant_client.get_collection(self.config.collection_name)
                point_count = collection_info.points_count or 0
                if point_count > 0:
                    logger.info(f'Loading existing index with {point_count} documents')
                    storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                    self.index = VectorStoreIndex.from_vector_store(
                        self.vector_store,
                        embed_model=self.embed_model,
                        storage_context=storage_context,
                    )
                    return
            if self.config.data_dir and self.config.data_dir.exists():
                logger.info(f'Building new index from {self.config.data_dir}')
                await self._build_index_from_directory()
            else:
                logger.warning('No existing index and no data directory provided')
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                self.index = VectorStoreIndex(
                    [],
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                )
        except Exception as e:
            raise PipelineError(f'Index setup failed: {e}')



    async def _build_index_from_directory(self) -> None:
        """Build index from documents in the data directory."""
        try:
            docs = SimpleDirectoryReader(str(self.config.data_dir)).load_data()
            if not docs:
                raise ValueError(f'No documents found in {self.config.data_dir}')
            logger.info(f'Loaded {len(docs)} documents')
            nodes = self.chunker.split(docs)
            logger.info(f'Created {len(nodes)} text chunks')
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True
            )
            logger.info('✓ Index built successfully')
        except Exception as e:
            raise PipelineError(f'Index building failed: {e}')



    async def _setup_retrievers(self) -> None:
        """Set up retrieval components."""
        try:
            if self.config.retrieval_strategy in [RetrievalStrategy.HYBRID]:
                self.retriever = HybridRetriever(
                    client=self.qdrant_client,
                    embed_model=self.embed_model,
                    collection_name=self.config.collection_name
                )
                logger.debug('Hybrid retriever initialized')
        except Exception as e:
            raise PipelineError(f'Retriever setup failed: {e}')



    async def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query.
            top_k: Number of documents to retrieve.

        Returns:
            RetrievalResult with documents and metadata.
        """
        if not self.initialized:
            raise PipelineError('Pipeline not initialized. Call initialize() first.')
        
        # Проверяем, запущен ли уже event loop
        try:
            # Пробуем получить текущий event loop
            loop = asyncio.get_running_loop()
            # Если мы здесь, значит loop уже запущен, используем синхронную версию
            logger.warning("Event loop already running, using synchronous retrieval")
            return self.retrieve_sync(query, top_k)
        except RuntimeError:
            # Если loop не запущен, продолжаем с асинхронной версией
            pass
            
        start_time = time.time()
        top_k = top_k or self.config.top_k_retrieval
        strategy = self.config.retrieval_strategy
        try:
            documents = []
            if strategy == RetrievalStrategy.DENSE:
                documents = await self._dense_retrieval(query, top_k)
            elif strategy == RetrievalStrategy.HYBRID:
                documents = await self._hybrid_retrieval(query, top_k)
            elif strategy == RetrievalStrategy.SEMANTIC:
                documents = await self._semantic_retrieval(query, top_k)
            if self.config.enable_source_filtering:
                documents = [
                    doc for doc in documents
                    if doc.get('score', 0) >= self.config.min_relevance_score
                ]
            retrieval_time = time.time() - start_time
            self.stats['total_retrieval_time'] += retrieval_time
            self.stats['strategy_usage'][strategy.value] += 1
            logger.debug(
                f'Retrieved {len(documents)} documents in {retrieval_time:.2f}s using {strategy.value}'
            )
            return RetrievalResult(
                documents=documents,
                query=query,
                strategy=strategy.value,
                total_time=retrieval_time,
                sources_found=len(documents)
            )
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f'Retrieval failed: {e}')
            raise PipelineError(f'Retrieval failed: {e}')
            
    def retrieve_sync(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Синхронная версия метода retrieve для использования в случаях,
        когда event loop уже запущен.

        Args:
            query: Search query.
            top_k: Number of documents to retrieve.

        Returns:
            RetrievalResult with documents and metadata.
        """
        if not self.initialized:
            raise PipelineError('Pipeline not initialized. Call initialize() first.')
            
        start_time = time.time()
        top_k = top_k or self.config.top_k_retrieval
        strategy = self.config.retrieval_strategy
        try:
            documents = []
            # Используем синхронные методы для извлечения документов
            if not self.index:
                logger.warning("No index available for retrieval")
            elif strategy == RetrievalStrategy.DENSE:
                # Синхронная версия dense retrieval
                try:
                    if self.embed_model:
                        query_embedding = self.embed_model.embed_sync(query)
                        if self.qdrant_client:
                            search_results = self.qdrant_client.search(
                                collection_name=self.config.collection_name,
                                query_vector=query_embedding,
                                limit=top_k,
                                with_payload=True,
                                vector_name=TEXT_VECTOR_NAME,
                            )
                            
                            text_keys = ('text', 'document', 'content', 'chunk', 'chunk_text')
                            for res in search_results:
                                payload = res.payload or {}
                                text_segment = next((payload.get(k) for k in text_keys if payload.get(k)), '')
                                if not text_segment and payload.get('_node_content'):
                                    try:
                                        node_data = json.loads(payload['_node_content'])
                                        text_segment = node_data.get('text', '')
                                    except Exception:
                                        text_segment = ''
                                source = (
                                    payload.get('source')
                                    or payload.get('file_name')
                                    or payload.get('file_path')
                                    or 'unknown'
                                )
                                payload.setdefault('source', source)
                                doc = {
                                    'text': text_segment,
                                    'source': source,
                                    'score': float(res.score),
                                    'metadata': payload,
                                }
                                documents.append(doc)
                except Exception as e:
                    logger.error(f"Error in sync dense retrieval: {e}")
            elif strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.SEMANTIC):
                # Синхронная версия hybrid/semantic retrieval
                if self.retriever:
                    results = self.retriever.retrieve(query, top_k=top_k)
                    documents = [
                        {
                            'text': result.get('text', ''),
                            'source': result.get('source', 'unknown'),
                            'score': float(result.get('score', 0)),
                            'metadata': result
                        }
                        for result in results
                    ]
                    
            if self.config.enable_source_filtering:
                documents = [
                    doc for doc in documents
                    if doc.get('score', 0) >= self.config.min_relevance_score
                ]
                
            retrieval_time = time.time() - start_time
            self.stats['total_retrieval_time'] += retrieval_time
            self.stats['strategy_usage'][strategy.value] += 1
            logger.debug(
                f'Retrieved {len(documents)} documents in {retrieval_time:.2f}s using {strategy.value} (sync)'
            )
            return RetrievalResult(
                documents=documents,
                query=query,
                strategy=strategy.value,
                total_time=retrieval_time,
                sources_found=len(documents)
            )
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f'Sync retrieval failed: {e}')
            # Возвращаем пустой результат вместо исключения для более стабильной работы
            return RetrievalResult(
                documents=[],
                query=query,
                strategy=strategy.value,
                total_time=time.time() - start_time,
                sources_found=0
            )



    async def _dense_retrieval(self, query: str, top_k: int) -> List[Dict]:
        """Perform dense vector retrieval."""
        if not self.index:
            return []
        query_embedding = await self.embed_model.embed_async_single(query)
        search_results = self.qdrant_client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            vector_name=TEXT_VECTOR_NAME,
        )
        text_keys = ('text', 'document', 'content', 'chunk', 'chunk_text')
        docs: List[Dict] = []
        for res in search_results:
            payload = res.payload or {}
            text_segment = next((payload.get(k) for k in text_keys if payload.get(k)), '')
            if not text_segment and payload.get('_node_content'):
                try:
                    node_data = json.loads(payload['_node_content'])
                    text_segment = node_data.get('text', '')
                except Exception:
                    text_segment = ''
            source = (
                payload.get('source')
                or payload.get('file_name')
                or payload.get('file_path')
                or 'unknown'
            )
            payload.setdefault('source', source)
            doc = {
                'text': text_segment,
                'source': source,
                'score': float(res.score),
                'metadata': payload,
            }
            docs.append(doc)
        if os.getenv('SENTIO_DEBUG') == '1':
            for idx, d in enumerate(docs[: min(5, len(docs))]):
                logger.debug(
                    '[DEBUG] Retrieved #%d score=%.4f source=%s text_snippet=%s payload_keys=%s',
                    idx + 1,
                    d['score'],
                    d['source'],
                    (d['text'] or '').replace('\n', ' ')[:120] +
                    ('…' if len(d['text']) > 120 else ''),
                    list(d['metadata'].keys()),
                )
        return docs



    async def _hybrid_retrieval(self, query: str, top_k: int) -> List[Dict]:
        """Perform hybrid retrieval using dense + sparse."""
        if not self.retriever:
            return await self._dense_retrieval(query, top_k)
        results = self.retriever.retrieve(query, top_k=top_k)
        return [
            {
                'text': result.get('text', ''),
                'source': result.get('source', 'unknown'),
                'score': float(result.get('score', 0)),
                'metadata': result
            }
            for result in results
        ]



    async def _semantic_retrieval(self, query: str, top_k: int) -> List[Dict]:
        """Perform semantic-aware retrieval."""
        return await self._hybrid_retrieval(query, top_k)



    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Rerank documents using the configured reranker.

        Args:
            query: Original query.
            documents: Documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            Reranked documents.
        """
        if not self.config.enable_reranking or not self.reranker or not documents:
            return documents[:top_k] if top_k else documents
        try:
            top_k = top_k or self.config.top_k_final
            reranked = self.reranker.rerank(query, documents, top_k=top_k)
            logger.debug(f'Reranked {len(documents)} documents to top {len(reranked)}')
            return reranked
        except Exception as e:
            logger.warning(f'Reranking failed, using original order: {e}')
            return documents[:top_k] if top_k else documents



    async def generate(
        self,
        query: str,
        context: List[Dict],
        mode: Optional[GenerationMode] = None
    ) -> GenerationResult:
        """Generate an answer based on query and context.

        Args:
            query: User query.
            context: Retrieved context documents.
            mode: Generation mode to use.

        Returns:
            GenerationResult with answer and metadata.
        """
        start_time = time.time()
        mode = mode or self.config.generation_mode
        try:
            gen_config = self._generation_configs[mode]
            context_str = self._build_context_string(context)
            answer = await self._generate_answer(query, context_str, gen_config)
            generation_time = time.time() - start_time
            self.stats['total_generation_time'] += generation_time
            logger.debug(f'Generated answer in {generation_time:.2f}s using {mode.value} mode')
            return GenerationResult(
                answer=answer,
                sources=context,
                query=query,
                mode=mode.value,
                total_time=generation_time,
                token_count=len(answer.split())
            )
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f'Generation failed: {e}')
            raise PipelineError(f'Generation failed: {e}')



    def _build_context_string(self, context: List[Dict]) -> str:
        """Build formatted context string from documents.

        Args:
            context: List of context documents.

        Returns:
            String formatted for prompt context.
        """
        if not context:
            return ''
        context_parts = []
        for i, doc in enumerate(context, 1):
            text = doc.get('text', '')
            source = doc.get('source', 'unknown')
            score = doc.get('score', 0.0)
            context_parts.append(
                f'[Source {i}: {source} (relevance: {score:.2f})]\n{text}'
            )
        return '\n\n---\n\n'.join(context_parts)



    async def _generate_answer(self, query: str, context: str, config: Dict) -> str:
        """Generate answer using the configured LLM.

        Args:
            query: The user query.
            context: The formatted context string.
            config: Generation configuration dict.

        Returns:
            Generated answer string.
        """
        prompt = self._build_prompt(query, context, config)

        payload = {
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens', 300),
            'stream': False,
        }

        try:
            # Import on-demand to avoid early binding which breaks unit-test monkey-patching.
            from root.src.core.llm import chat_adapter as _chat

            result = await _chat.chat_completion(payload)
            if isinstance(result, dict):
                return result['choices'][0]['message']['content']

            # Streaming generator fallback – concatenate chunks.
            parts: list[str] = []
            async for chunk in result:  # type: ignore[assignment]
                parts.append(chunk)
            return ''.join(parts).strip()
        except HTTPError as e:
            raise PipelineError(f'LLM generation failed after retries: {e}') from e
        except Exception as e:  # pragma: no cover
            raise PipelineError(f'Unexpected LLM error: {e}') from e
    async def aclose(self) -> None:
        """Close any open network resources."""
        if self.http_client and not self.http_client.is_closed:
            await self.http_client.aclose()
        if self.qdrant_client:
            await asyncio.to_thread(self.qdrant_client.close)



    def _build_prompt(self, query: str, context: str, config: Dict) -> str:
        """Return final prompt string via :class:`PromptBuilder`."""
        mode_key = next(
            (k for k, v in self._generation_configs.items() if v == config),
            'balanced',
        )
        # Convert Enum to its lowercase string value expected by PromptBuilder.
        if isinstance(mode_key, Enum):  # pragma: no cover – defensive
            mode_key = mode_key.value
        return self.prompt_builder.build_generation_prompt(query, context, str(mode_key))



    async def query(self, question: str, top_k: Optional[int] = None) -> Dict:
        """Execute the complete RAG pipeline for a question.

        Args:
            question: User question.
            top_k: Number of sources to use for generation.

        Returns:
            Complete response with answer, sources, and metadata.
        """
        if not self.initialized:
            await self.initialize()
        start_time = time.time()
        top_k = top_k or self.config.top_k_final
        try:
            retrieval_result = await self.retrieve(question, top_k=self.config.top_k_retrieval)
            reranked_docs = await self.rerank(question, retrieval_result.documents, top_k=top_k)
            generation_result = await self.generate(question, reranked_docs)
            total_time = time.time() - start_time
            self.stats['queries_processed'] += 1
            self.stats['avg_sources_per_query'] = (
                (self.stats['avg_sources_per_query'] * (self.stats['queries_processed'] - 1) +
                 len(reranked_docs)) / self.stats['queries_processed']
            )
            return {
                'answer': generation_result.answer,
                'sources': reranked_docs,
                'metadata': {
                    'query_time': total_time,
                    'retrieval_time': retrieval_result.total_time,
                    'generation_time': generation_result.total_time,
                    'sources_found': len(retrieval_result.documents),
                    'sources_used': len(reranked_docs),
                    'retrieval_strategy': retrieval_result.strategy,
                    'generation_mode': generation_result.mode,
                    'timestamp': time.time()
                }
            }
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f'Query processing failed: {e}')
            raise PipelineError(f'Query processing failed: {e}')



    async def query_stream(
        self, question: str, top_k: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Execute RAG pipeline with streaming response.

        Args:
            question: User question.
            top_k: Number of sources to use.

        Yields:
            Streaming response tokens.
        """
        result = await self.query(question, top_k)
        answer = result['answer']
        words = answer.split()
        for word in words:
            yield f'{word} '
            await asyncio.sleep(0.01)



    def get_stats(self) -> Dict:
        """Get comprehensive pipeline statistics.

        Returns:
            Dictionary of pipeline statistics and metrics.
        """
        stats = self.stats.copy()
        if self.stats['queries_processed'] > 0:
            stats['avg_retrieval_time'] = (
                self.stats['total_retrieval_time'] / self.stats['queries_processed']
            )
            stats['avg_generation_time'] = (
                self.stats['total_generation_time'] / self.stats['queries_processed']
            )
            stats['error_rate'] = (
                self.stats['errors'] / self.stats['queries_processed']
            )
        if self.embed_model:
            stats['embedding_stats'] = self.embed_model.get_stats()
        if self.chunker:
            stats['chunking_stats'] = self.chunker.get_stats()
        return stats



    def reset_stats(self) -> None:
        """Reset all pipeline statistics."""
        self.stats = {
            'queries_processed': 0,
            'total_retrieval_time': 0.0,
            'total_generation_time': 0.0,
            'cache_hits': 0,
            'errors': 0,
            'avg_sources_per_query': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in RetrievalStrategy}
        }
        if self.embed_model:
            self.embed_model.clear_cache()
        if self.chunker:
            self.chunker.reset_stats()
        logger.info('Pipeline statistics reset')



    async def health_check(self) -> Dict:
        """Perform comprehensive health check of all components.

        Returns:
            Dictionary indicating health status and component checks.
        """
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': time.time()
        }
        try:
            if self.embed_model:
                test_embedding = await self.embed_model.embed_async_single('test')
                health['components']['embeddings'] = 'healthy' if test_embedding else 'unhealthy'
            if self.qdrant_client:
                collections = self.qdrant_client.get_collections()
                health['components']['qdrant'] = 'healthy'
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        return health



    def __repr__(self) -> str:
        """Return string representation of the pipeline instance."""
        status = 'initialized' if self.initialized else 'uninitialized'
        return f"SentioRAGPipeline(strategy={self.config.retrieval_strategy.value}, status={status})"



    # ----► INGESTION API ---- #



    async def ingest_texts(
        self, texts: List[str], sources: Optional[List[str]] = None
    ) -> int:
        """Ingest raw text documents into the vector store.

        Args:
            texts: List of raw document strings.
            sources: Optional list of source identifiers (filenames, URLs, etc.).

        Returns:
            Total number of chunks indexed.
        """
        if not self.initialized:
            await self.initialize()
        if not texts:
            return 0
        sources = sources or [f'doc_{i}' for i in range(len(texts))]
        documents = [
            _LlamaDoc(text=texts[i], metadata={'source': sources[i]}) for i in range(len(texts))
        ]
        nodes = self.chunker.split(documents)
        if not nodes:
            return 0
        node_texts = [n.get_content() for n in nodes]
        embeddings = await self.embed_model.embed_async_many(node_texts)
        points: List[models.PointStruct] = []
        for idx, node in enumerate(nodes):
            vec = embeddings[idx]
            payload = {
                'text': node.get_content(),
                'source': node.metadata.get('source', 'unknown'),
            }
            point = models.PointStruct(id=str(uuid.uuid4()), vectors={TEXT_VECTOR_NAME: vec}, payload=payload)
            points.append(point)
        self.qdrant_client.upsert(collection_name=self.config.collection_name, points=points)
        if self.index is not None:
            try:
                self.index.insert_nodes(nodes)
            except Exception:
                logger.warning('Failed to insert nodes into local index; will rebuild on next init')
        logger.info('Ingested %d chunks from %d documents', len(nodes), len(texts))
        return len(nodes)