#!/usr/bin/env python3
"""
Sentio RAG Pipeline - Enterprise-grade RAG orchestration system.

This module provides a comprehensive, production-ready RAG pipeline with:
- Advanced retrieval strategies (hybrid, semantic, dense)
- Intelligent reranking and fusion
- Streaming and async generation
- Performance monitoring and optimization
- Fault tolerance and graceful degradation
- Comprehensive error handling and logging
"""

import asyncio
import logging
import os
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union
import uuid

import httpx
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document as _LlamaDoc
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .chunking import ChunkingStrategy, TextChunker
from .embeddings import EmbeddingModel, EmbeddingError
from .retrievers import HybridRetriever
from .rerankers import CrossEncoderReranker

# Optional web search integration
try:
    from .retrievers.web_search import WebSearchRetriever
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    WebSearchRetriever = None

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    pass


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    DENSE = "dense"                 # Dense vector search only
    HYBRID = "hybrid"               # Dense + sparse fusion
    SEMANTIC = "semantic"           # Semantic-aware retrieval
    WEB_AUGMENTED = "web_augmented" # Hybrid + web search


class GenerationMode(Enum):
    """Generation modes for different use cases."""
    BALANCED = "balanced"           # Balance speed and quality
    FAST = "fast"                  # Optimize for speed
    QUALITY = "quality"            # Optimize for quality
    CREATIVE = "creative"          # Higher temperature, more creative


@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline."""
    
    # Collection settings
    collection_name: str = "Sentio_docs_v2"
    data_dir: Optional[Path] = None
    
    # Retrieval settings
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k_retrieval: int = 10
    top_k_final: int = 3
    min_relevance_score: float = 0.1
    
    # Generation settings
    generation_mode: GenerationMode = GenerationMode.BALANCED
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE
    
    # Performance settings
    cache_enabled: bool = True
    web_search_enabled: bool = False
    max_retries: int = 3
    timeout: int = 60
    
    # Quality settings
    enable_reranking: bool = True
    enable_source_filtering: bool = True
    enable_answer_validation: bool = True


class RetrievalResult:
    """Container for retrieval results with metadata."""
    
    def __init__(
        self,
        documents: List[Dict],
        query: str,
        strategy: str,
        total_time: float,
        sources_found: int
    ):
        self.documents = documents
        self.query = query
        self.strategy = strategy
        self.total_time = total_time
        self.sources_found = sources_found
        self.timestamp = time.time()


class GenerationResult:
    """Container for generation results with metadata."""
    
    def __init__(
        self,
        answer: str,
        sources: List[Dict],
        query: str,
        mode: str,
        total_time: float,
        token_count: Optional[int] = None
    ):
        self.answer = answer
        self.sources = sources
        self.query = query
        self.mode = mode
        self.total_time = total_time
        self.token_count = token_count
        self.timestamp = time.time()


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
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the RAG pipeline with configuration.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config or PipelineConfig()
        self.initialized = False
        
        # Components (initialized during startup)
        self.embed_model: Optional[EmbeddingModel] = None
        self.chunker: Optional[TextChunker] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.vector_store: Optional[QdrantVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[HybridRetriever] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.web_retriever: Optional[WebSearchRetriever] = None
        
        # Performance tracking
        self.stats = {
            'queries_processed': 0,
            'total_retrieval_time': 0.0,
            'total_generation_time': 0.0,
            'cache_hits': 0,
            'errors': 0,
            'avg_sources_per_query': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in RetrievalStrategy}
        }
        
        # Generation configuration
        self._generation_configs = {
            GenerationMode.FAST: {'temperature': 0.3, 'max_tokens': 150},
            GenerationMode.BALANCED: {'temperature': 0.7, 'max_tokens': 300},
            GenerationMode.QUALITY: {'temperature': 0.5, 'max_tokens': 500},
            GenerationMode.CREATIVE: {'temperature': 0.9, 'max_tokens': 400}
        }
        
        logger.info("Sentio RAG Pipeline initialized")

        # Elevate log level for deep debugging if env flag is set
        if os.getenv("SENTIO_DEBUG") == "1":
            logger.setLevel(logging.DEBUG)
    
    async def initialize(self) -> None:
        """Initialize all pipeline components asynchronously."""
        if self.initialized:
            logger.warning("Pipeline already initialized")
            return
        
        logger.info("🚀 Initializing Sentio RAG Pipeline...")
        
        try:
            # Initialize embedding model
            logger.info("Initializing embedding model...")
            self.embed_model = EmbeddingModel(
                cache_enabled=self.config.cache_enabled,
                max_retries=self.config.max_retries
            )
            logger.info("✓ Embedding model ready")
            
            # Initialize text chunker
            logger.info("Initializing text chunker...")
            self.chunker = TextChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                strategy=self.config.chunking_strategy
            )
            logger.info("✓ Text chunker ready")
            
            # Initialize Qdrant client and vector store
            logger.info("Initializing vector database...")
            await self._setup_vector_store()
            logger.info("✓ Vector database ready")
            
            # Initialize retrieval components
            logger.info("Initializing retrieval components...")
            await self._setup_retrievers()
            logger.info("✓ Retrieval components ready")
            
            # Initialize reranker if enabled
            if self.config.enable_reranking:
                logger.info("Initializing reranker...")
                self.reranker = CrossEncoderReranker()
                logger.info("✓ Reranker ready")
            
            # Initialize web search if enabled
            if self.config.web_search_enabled and WEB_SEARCH_AVAILABLE:
                logger.info("Initializing web search...")
                self.web_retriever = WebSearchRetriever(
                    client=self.qdrant_client,
                    embed_model=self.embed_model
                )
                logger.info("✓ Web search ready")
            elif self.config.web_search_enabled:
                logger.warning("Web search requested but not available")
            
            self.initialized = True
            logger.info("🎉 Sentio RAG Pipeline initialization complete!")
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise PipelineError(f"Failed to initialize pipeline: {e}")
    
    async def _setup_vector_store(self) -> None:
        """Set up Qdrant client and vector store."""
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        try:
            self.qdrant_client = QdrantClient(url=qdrant_url)
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            logger.debug(f"Connected to Qdrant with {len(collections.collections)} collections")
            
            # Set up vector store
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.config.collection_name
            )
            
            # Load or build index
            await self._setup_index()
            
        except Exception as e:
            raise PipelineError(f"Vector store setup failed: {e}")
    
    async def _setup_index(self) -> None:
        """Set up or load the vector index."""
        try:
            # Check if collection exists and has data
            collection_exists = self.qdrant_client.collection_exists(
                collection_name=self.config.collection_name
            )
            
            if collection_exists:
                collection_info = self.qdrant_client.get_collection(self.config.collection_name)
                point_count = collection_info.points_count or 0
                
                if point_count > 0:
                    # Load existing index
                    logger.info(f"Loading existing index with {point_count} documents")
                    storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                    self.index = VectorStoreIndex.from_vector_store(
                        self.vector_store,
                        embed_model=self.embed_model._model,
                        storage_context=storage_context
                    )
                    return
            
            # Build new index from documents if data directory exists
            if self.config.data_dir and self.config.data_dir.exists():
                logger.info(f"Building new index from {self.config.data_dir}")
                await self._build_index_from_directory()
            else:
                logger.warning("No existing index and no data directory provided")
                # Create empty index
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                self.index = VectorStoreIndex(
                    [],
                    storage_context=storage_context,
                    embed_model=self.embed_model._model
                )
                
        except Exception as e:
            raise PipelineError(f"Index setup failed: {e}")
    
    async def _build_index_from_directory(self) -> None:
        """Build index from documents in the data directory."""
        try:
            # Load documents
            docs = SimpleDirectoryReader(str(self.config.data_dir)).load_data()
            if not docs:
                raise ValueError(f"No documents found in {self.config.data_dir}")
            
            logger.info(f"Loaded {len(docs)} documents")
            
            # Chunk documents
            nodes = self.chunker.split(docs)
            logger.info(f"Created {len(nodes)} text chunks")
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Build index
            self.index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model._model,
                show_progress=True
            )
            
            logger.info("✓ Index built successfully")
            
        except Exception as e:
            raise PipelineError(f"Index building failed: {e}")
    
    async def _setup_retrievers(self) -> None:
        """Set up retrieval components."""
        try:
            if self.config.retrieval_strategy in [RetrievalStrategy.HYBRID, RetrievalStrategy.WEB_AUGMENTED]:
                self.retriever = HybridRetriever(
                    client=self.qdrant_client,
                    embed_model=self.embed_model,
                    collection_name=self.config.collection_name
                )
                logger.debug("Hybrid retriever initialized")
            
        except Exception as e:
            raise PipelineError(f"Retriever setup failed: {e}")
    
    async def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            RetrievalResult with documents and metadata
        """
        if not self.initialized:
            raise PipelineError("Pipeline not initialized. Call initialize() first.")
        
        start_time = time.time()
        top_k = top_k or self.config.top_k_retrieval
        strategy = self.config.retrieval_strategy
        
        try:
            documents = []
            
            if strategy == RetrievalStrategy.DENSE:
                # Dense vector search only
                documents = await self._dense_retrieval(query, top_k)
                
            elif strategy == RetrievalStrategy.HYBRID:
                # Hybrid retrieval (dense + sparse)
                documents = await self._hybrid_retrieval(query, top_k)
                
            elif strategy == RetrievalStrategy.SEMANTIC:
                # Semantic-aware retrieval
                documents = await self._semantic_retrieval(query, top_k)
                
            elif strategy == RetrievalStrategy.WEB_AUGMENTED:
                # Hybrid + web search
                documents = await self._web_augmented_retrieval(query, top_k)
            
            # Filter by relevance score
            if self.config.enable_source_filtering:
                documents = [
                    doc for doc in documents 
                    if doc.get('score', 0) >= self.config.min_relevance_score
                ]
            
            # Update statistics
            retrieval_time = time.time() - start_time
            self.stats['total_retrieval_time'] += retrieval_time
            self.stats['strategy_usage'][strategy.value] += 1
            
            logger.debug(f"Retrieved {len(documents)} documents in {retrieval_time:.2f}s using {strategy.value}")
            
            return RetrievalResult(
                documents=documents,
                query=query,
                strategy=strategy.value,
                total_time=retrieval_time,
                sources_found=len(documents)
            )
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Retrieval failed: {e}")
            raise PipelineError(f"Retrieval failed: {e}")
    
    async def _dense_retrieval(self, query: str, top_k: int) -> List[Dict]:
        """Perform dense vector retrieval."""
        if not self.index:
            return []
        
        # Get query embedding
        query_embedding = await self.embed_model.embed_async_single(query)
        
        # Search Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        # Build list of candidate keys to extract text from payload
        text_keys = ("text", "document", "content", "chunk", "chunk_text")

        docs: List[Dict] = []
        for res in search_results:
            payload = res.payload or {}
            text_segment = next((payload.get(k) for k in text_keys if payload.get(k)), "")
            # Fallback: handle LlamaIndex payloads stored under `_node_content` as JSON string
            if not text_segment and payload.get("_node_content"):
                try:
                    node_data = json.loads(payload["_node_content"])
                    text_segment = node_data.get("text", "")
                except Exception:
                    text_segment = ""

            # Derive a reasonable source identifier
            source = (
                payload.get("source")
                or payload.get("file_name")
                or payload.get("file_path")
                or "unknown"
            )

            # Ensure metadata contains source for downstream processing
            payload.setdefault("source", source)

            doc = {
                "text": text_segment,
                "source": source,
                "score": float(res.score),
                "metadata": payload,
            }

            docs.append(doc)

        # Extra debugging – show top retrieved chunks and their scores
        if os.getenv("SENTIO_DEBUG") == "1":
            for idx, d in enumerate(docs[: min(5, len(docs))]):
                logger.debug(
                    "[DEBUG] Retrieved #%d score=%.4f source=%s text_snippet=%s payload_keys=%s",
                    idx + 1,
                    d["score"],
                    d["source"],
                    (d["text"] or "").replace("\n", " ")[:120] + ("…" if len(d["text"]) > 120 else ""),
                    list(d["metadata"].keys()),
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
        # For now, use hybrid retrieval as semantic baseline
        # This could be enhanced with semantic analysis
        return await self._hybrid_retrieval(query, top_k)
    
    async def _web_augmented_retrieval(self, query: str, top_k: int) -> List[Dict]:
        """Perform web-augmented retrieval."""
        documents = await self._hybrid_retrieval(query, top_k)
        
        # Add web search results if available
        if self.web_retriever:
            try:
                web_results = await self.web_retriever.retrieve_async(query, top_k=min(3, top_k))
                web_docs = [
                    {
                        'text': result.get('text', ''),
                        'source': result.get('url', 'web'),
                        'score': 0.8,  # Fixed score for web results
                        'metadata': {'type': 'web', **result}
                    }
                    for result in web_results
                ]
                documents.extend(web_docs)
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        return documents
    
    async def rerank(self, query: str, documents: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """
        Rerank documents using the configured reranker.
        
        Args:
            query: Original query
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents
        """
        if not self.config.enable_reranking or not self.reranker or not documents:
            return documents[:top_k] if top_k else documents
        
        try:
            top_k = top_k or self.config.top_k_final
            reranked = self.reranker.rerank(query, documents, top_k=top_k)
            logger.debug(f"Reranked {len(documents)} documents to top {len(reranked)}")
            return reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return documents[:top_k] if top_k else documents
    
    async def generate(
        self, 
        query: str, 
        context: List[Dict], 
        mode: Optional[GenerationMode] = None
    ) -> GenerationResult:
        """
        Generate an answer based on query and context.
        
        Args:
            query: User query
            context: Retrieved context documents
            mode: Generation mode to use
            
        Returns:
            GenerationResult with answer and metadata
        """
        start_time = time.time()
        mode = mode or self.config.generation_mode
        
        try:
            # Get generation config for mode
            gen_config = self._generation_configs[mode]
            
            # Build context string
            context_str = self._build_context_string(context)
            
            # Generate answer
            answer = await self._generate_answer(query, context_str, gen_config)
            
            # Validate answer if enabled
            if self.config.enable_answer_validation:
                answer = self._validate_answer(answer, query, context)
            
            generation_time = time.time() - start_time
            self.stats['total_generation_time'] += generation_time
            
            logger.debug(f"Generated answer in {generation_time:.2f}s using {mode.value} mode")
            
            return GenerationResult(
                answer=answer,
                sources=context,
                query=query,
                mode=mode.value,
                total_time=generation_time,
                token_count=len(answer.split())  # Rough token estimate
            )
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Generation failed: {e}")
            raise PipelineError(f"Generation failed: {e}")
    
    def _build_context_string(self, context: List[Dict]) -> str:
        """Build formatted context string from documents."""
        if not context:
            return ""
        
        context_parts = []
        for i, doc in enumerate(context, 1):
            text = doc.get('text', '')
            source = doc.get('source', 'unknown')
            score = doc.get('score', 0.0)
            
            context_parts.append(f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    async def _generate_answer(self, query: str, context: str, config: Dict) -> str:
        """Generate answer using the configured LLM."""
        # Get Ollama configuration
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "phi3.5:3.8b")
        
        # Build prompt
        prompt = self._build_prompt(query, context, config)
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": config.get("temperature", 0.7),
                            "num_predict": config.get("max_tokens", 300)
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
                
        except Exception as e:
            raise PipelineError(f"LLM generation failed: {e}")
    
    def _build_prompt(self, query: str, context: str, config: Dict) -> str:
        """Build the generation prompt and optionally log it for debugging."""
        mode_instructions = {
            "fast": "Provide a concise, direct answer.",
            "balanced": "Provide a comprehensive but focused answer.",
            "quality": "Provide a detailed, well-structured answer with proper explanations.",
            "creative": "Provide an engaging, creative answer while staying factual."
        }
        
        mode = next((k for k, v in self._generation_configs.items() if v == config), "balanced")
        instruction = mode_instructions.get(mode.value if hasattr(mode, 'value') else str(mode), 
                                           mode_instructions["balanced"])
        
        prompt = f"""You are Sentio, an expert AI assistant with access to a comprehensive knowledge base.

Your task: {instruction}

Guidelines:
- Base your answer strictly on the provided context
- If the context is insufficient, clearly state what information is missing
- Cite sources when relevant
- Be honest about limitations
- Maintain a professional yet accessible tone

Context:
{context}

Question: {query}

Answer:"""

        # Debug: log prompt length and optional preview
        if os.getenv("SENTIO_DEBUG") == "1":
            logger.debug("[DEBUG] Prompt length: %d tokens (approx chars=%d)", len(prompt.split()), len(prompt))
            logger.debug("[DEBUG] Prompt preview:\n%s", prompt[:1000] + ("…" if len(prompt) > 1000 else ""))

        return prompt
    
    def _validate_answer(self, answer: str, query: str, context: List[Dict]) -> str:
        """Validate and potentially improve the generated answer."""
        # Simple validation - could be enhanced with more sophisticated checks
        if not answer or len(answer.strip()) < 10:
            return "I apologize, but I couldn't generate a sufficient answer based on the available information."
        
        # Check for basic coherence
        sentences = answer.split('.')
        if len(sentences) < 2 and len(answer) > 100:
            # Likely a run-on sentence, could be improved
            pass
        
        return answer.strip()
    
    async def query(self, question: str, top_k: Optional[int] = None) -> Dict:
        """
        Execute the complete RAG pipeline for a question.
        
        Args:
            question: User question
            top_k: Number of sources to use for generation
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        top_k = top_k or self.config.top_k_final
        
        try:
            # Step 1: Retrieve relevant documents
            retrieval_result = await self.retrieve(question, top_k=self.config.top_k_retrieval)
            
            # Step 2: Rerank documents
            reranked_docs = await self.rerank(question, retrieval_result.documents, top_k=top_k)
            
            # Step 3: Generate answer
            generation_result = await self.generate(question, reranked_docs)
            
            # Update statistics
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
            logger.error(f"Query processing failed: {e}")
            raise PipelineError(f"Query processing failed: {e}")
    
    async def query_stream(self, question: str, top_k: Optional[int] = None) -> AsyncGenerator[str, None]:
        """
        Execute RAG pipeline with streaming response.
        
        Args:
            question: User question
            top_k: Number of sources to use
            
        Yields:
            Streaming response tokens
        """
        # For now, implement as non-streaming with chunked output
        # Could be enhanced with true streaming generation
        result = await self.query(question, top_k)
        answer = result['answer']
        
        # Simulate streaming by yielding words
        words = answer.split()
        for word in words:
            yield f"{word} "
            await asyncio.sleep(0.01)  # Small delay for streaming effect
    
    def get_stats(self) -> Dict:
        """Get comprehensive pipeline statistics."""
        stats = self.stats.copy()
        
        # Add computed metrics
        if self.stats['queries_processed'] > 0:
            stats['avg_retrieval_time'] = self.stats['total_retrieval_time'] / self.stats['queries_processed']
            stats['avg_generation_time'] = self.stats['total_generation_time'] / self.stats['queries_processed']
            stats['error_rate'] = self.stats['errors'] / self.stats['queries_processed']
        
        # Add component stats
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
        
        # Reset component stats
        if self.embed_model:
            self.embed_model.clear_cache()
        
        if self.chunker:
            self.chunker.reset_stats()
        
        logger.info("Pipeline statistics reset")
    
    async def health_check(self) -> Dict:
        """Perform comprehensive health check of all components."""
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': time.time()
        }
        
        try:
            # Check embedding model
            if self.embed_model:
                test_embedding = await self.embed_model.embed_async_single("test")
                health['components']['embeddings'] = 'healthy' if test_embedding else 'unhealthy'
            
            # Check Qdrant
            if self.qdrant_client:
                collections = self.qdrant_client.get_collections()
                health['components']['qdrant'] = 'healthy'
            
            # Check Ollama
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{ollama_url}/api/tags")
                health['components']['ollama'] = 'healthy' if response.status_code == 200 else 'unhealthy'
        
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health
    
    def __repr__(self) -> str:
        status = "initialized" if self.initialized else "uninitialized"
        return f"SentioRAGPipeline(strategy={self.config.retrieval_strategy.value}, status={status})"

    # -------------------- Ingestion API --------------------

    async def ingest_texts(self, texts: List[str], sources: Optional[List[str]] = None) -> int:
        """Ingest raw text documents into the vector store.

        Args:
            texts: List of raw document strings.
            sources: Optional list of source identifiers (filenames, URLs, etc.)

        Returns:
            Total number of chunks indexed.
        """
        if not self.initialized:
            await self.initialize()

        if not texts:
            return 0

        sources = sources or [f"doc_{i}" for i in range(len(texts))]

        # Build LlamaIndex Document objects to leverage existing chunker
        documents = [
            _LlamaDoc(text=texts[i], metadata={"source": sources[i]}) for i in range(len(texts))
        ]

        # Chunk into nodes
        nodes = self.chunker.split(documents)

        if not nodes:
            return 0

        # Embed in batches
        node_texts = [n.get_content() for n in nodes]
        embeddings = await self.embed_model.embed_async_many(node_texts)

        # Prepare Qdrant points
        points: List[models.PointStruct] = []
        for idx, node in enumerate(nodes):
            vec = embeddings[idx]
            payload = {
                "text": node.get_content(),
                "source": node.metadata.get("source", "unknown"),
            }
            point = models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
            points.append(point)

        # Upsert into collection
        self.qdrant_client.upsert(collection_name=self.config.collection_name, points=points)

        # Also update the VectorStoreIndex if present
        if self.index is not None:
            # Insert nodes into index for local retrieval path
            try:
                self.index.insert_nodes(nodes)
            except Exception:
                # Fallback: rebuild index lazily on next startup
                logger.warning("Failed to insert nodes into local index; will rebuild on next init")

        logger.info("Ingested %d chunks from %d documents", len(nodes), len(texts))

        return len(nodes) 