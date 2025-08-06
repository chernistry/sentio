"""High-performance async Qdrant vector store with connection pooling.

This implementation provides:
- Async operations for better concurrency
- Connection pooling for reduced latency
- Batch operations for improved throughput
- Circuit breaker pattern for resilience
- Connection pool management
- Performance monitoring
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest

from src.core.resilience.patterns import AsyncCircuitBreaker, AsyncRetry
from src.utils.settings import settings

__all__ = ["AsyncQdrantStore", "QdrantConnectionPool"]

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for Qdrant connections."""

    url: str | None = None
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    pool_size: int = 2
    max_pool_size: int = 4
    pool_recycle_seconds: int = 3600
    health_check_interval: float = 30.0


class QdrantConnectionPool:
    """Connection pool for Qdrant clients with automatic management.
    
    Features:
    - Connection pooling with automatic expansion/contraction
    - Health checking and automatic reconnection
    - Load balancing across connections
    - Connection lifecycle management
    - Performance metrics
    """

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._pool: list[AsyncQdrantClient] = []
        self._pool_lock = asyncio.Lock()
        self._health_check_task: asyncio.Task | None = None
        self._is_closed = False
        self._connection_count = 0
        self._last_health_check = 0.0

        # Performance metrics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._avg_response_time = 0.0

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        async with self._pool_lock:
            if self._pool:
                return  # Already initialized

            # Create initial connections
            for _ in range(self.config.pool_size):
                client = await self._create_client()
                if client:
                    self._pool.append(client)

            # Start health check task
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

            logger.info(f"Initialized Qdrant connection pool with {len(self._pool)} connections")

    async def _create_client(self) -> AsyncQdrantClient | None:
        """Create a new Qdrant client."""
        try:
            url = self.config.url or settings.qdrant_url
            api_key = self.config.api_key or settings.qdrant_api_key

            if not url or not api_key:
                raise ValueError("Qdrant URL and API key must be provided")

            # Ensure HTTPS for cloud instances
            if ".cloud.qdrant.io" in url and not url.startswith("https://"):
                url = f"https://{url.removeprefix('http://')}"

            client = AsyncQdrantClient(
                url=url,
                api_key=api_key,
                timeout=self.config.timeout,
            )

            # Test connection
            await client.get_collections()
            self._connection_count += 1

            logger.debug(f"Created Qdrant client #{self._connection_count}")
            return client

        except Exception as e:
            logger.error(f"Failed to create Qdrant client: {e}")
            return None

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[AsyncQdrantClient, None]:
        """Get a client from the pool."""
        if self._is_closed:
            raise RuntimeError("Connection pool is closed")

        start_time = time.time()
        client = None

        try:
            # Get client from pool
            async with self._pool_lock:
                if not self._pool:
                    # Try to create a new client if pool is empty
                    client = await self._create_client()
                    if not client:
                        raise RuntimeError("No healthy connections available")
                else:
                    client = self._pool.pop(0)

            self._total_requests += 1
            yield client
            self._successful_requests += 1

        except Exception as e:
            self._failed_requests += 1
            logger.error(f"Error using Qdrant client: {e}")
            raise

        finally:
            # Return client to pool
            if client and not self._is_closed:
                async with self._pool_lock:
                    if len(self._pool) < self.config.max_pool_size:
                        self._pool.append(client)
                    else:
                        # Pool is full, close the extra client
                        try:
                            await client.close()
                        except Exception:
                            pass

            # Update performance metrics
            response_time = time.time() - start_time
            self._avg_response_time = (
                (self._avg_response_time * (self._total_requests - 1) + response_time)
                / self._total_requests
            )

    async def _health_check_loop(self) -> None:
        """Background task to check connection health."""
        while not self._is_closed:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if self._is_closed:
                    break

                await self._health_check()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _health_check(self) -> None:
        """Check health of all connections in the pool."""
        current_time = time.time()
        self._last_health_check = current_time

        async with self._pool_lock:
            healthy_clients = []

            for client in self._pool:
                try:
                    # Quick health check
                    await asyncio.wait_for(
                        client.get_collections(),
                        timeout=5.0
                    )
                    healthy_clients.append(client)

                except Exception as e:
                    logger.warning(f"Unhealthy Qdrant client detected: {e}")
                    try:
                        await client.close()
                    except Exception:
                        pass

            # Replace pool with healthy clients
            self._pool = healthy_clients

            # Add new clients if needed
            target_size = max(1, self.config.pool_size)
            while len(self._pool) < target_size:
                client = await self._create_client()
                if client:
                    self._pool.append(client)
                else:
                    break

            logger.debug(f"Health check complete: {len(self._pool)} healthy connections")

    async def close(self) -> None:
        """Close all connections in the pool."""
        self._is_closed = True

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        async with self._pool_lock:
            for client in self._pool:
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing Qdrant client: {e}")

            self._pool.clear()

        logger.info("Qdrant connection pool closed")

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "pool_size": len(self._pool),
            "max_pool_size": self.config.max_pool_size,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                self._successful_requests / self._total_requests
                if self._total_requests > 0 else 0
            ),
            "avg_response_time": self._avg_response_time,
            "last_health_check": self._last_health_check,
            "is_healthy": len(self._pool) > 0,
        }


@dataclass(slots=True)
class AsyncQdrantStore(VectorStore):
    """High-performance async Qdrant vector store.
    
    Features:
    - Async operations for better concurrency
    - Connection pooling for reduced latency
    - Batch operations for improved throughput
    - Circuit breaker pattern for resilience
    - Performance monitoring
    """

    collection_name: str
    vector_size: int = 1024
    distance: str = "Cosine"
    embedding: Embeddings | None = None
    content_payload_key: str = "content"
    metadata_payload_key: str = "metadata"
    batch_size: int = 100
    max_concurrent_requests: int = 4

    # Connection configuration
    connection_config: ConnectionConfig = field(default_factory=ConnectionConfig)

    # Internal state
    _pool: QdrantConnectionPool | None = field(init=False, repr=False, default=None)
    _circuit_breaker: AsyncCircuitBreaker | None = field(init=False, repr=False, default=None)
    _retry_policy: AsyncRetry | None = field(init=False, repr=False, default=None)
    _semaphore: asyncio.Semaphore | None = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    async def initialize(self) -> None:
        """Initialize the async vector store."""
        if self._initialized:
            return

        # Initialize connection pool
        self._pool = QdrantConnectionPool(self.connection_config)
        await self._pool.initialize()

        # Initialize circuit breaker
        self._circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=5,
            timeout_seconds=30.0,
            recovery_timeout=60.0,
        )

        # Initialize retry policy
        self._retry_policy = AsyncRetry(
            max_attempts=3,
            backoff_multiplier=2.0,
            max_wait_time=10.0,
        )

        # Initialize concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # Ensure collection exists
        await self._bootstrap_collection()

        self._initialized = True
        logger.info(f"AsyncQdrantStore initialized for collection '{self.collection_name}'")

    async def close(self) -> None:
        """Close the vector store and cleanup resources."""
        if self._pool:
            await self._pool.close()
        self._initialized = False

    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        if not self._initialized or not self._pool:
            return False

        try:
            async with self._pool.get_client() as client:
                await client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store with async embeddings."""
        if not self._initialized:
            await self.initialize()

        if self.embedding is None:
            raise ValueError("Embeddings must be provided for adding texts.")

        texts_list = list(texts)

        # Generate embeddings asynchronously if possible
        if hasattr(self.embedding, "aembed_documents"):
            embeddings = await self.embedding.aembed_documents(texts_list)
        else:
            # Fallback to sync embedding in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self.embedding.embed_documents,
                texts_list
            )

        return await self.add_embeddings(
            texts=texts_list,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            **kwargs
        )

    async def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add embeddings to the vector store in batches."""
        if not self._initialized:
            await self.initialize()

        texts_list = list(texts)
        if len(texts_list) != len(embeddings):
            raise ValueError("Number of texts and embeddings must be the same.")

        metadatas = metadatas or [{}] * len(texts_list)
        if len(metadatas) != len(texts_list):
            raise ValueError("Number of texts and metadatas must be the same.")

        # Process in batches for better performance
        all_ids = []
        for i in range(0, len(texts_list), self.batch_size):
            batch_texts = texts_list[i:i + self.batch_size]
            batch_embeddings = embeddings[i:i + self.batch_size]
            batch_metadatas = metadatas[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size] if ids else None

            batch_result_ids = await self._add_batch(
                batch_texts,
                batch_embeddings,
                batch_metadatas,
                batch_ids
            )
            all_ids.extend(batch_result_ids)

        return all_ids

    async def _add_batch(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add a batch of embeddings."""
        async with self._semaphore:
            points = []
            ids_out = []

            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas, strict=False)):
                point_id = str(ids[i]) if ids and i < len(ids) else str(int(time.time() * 1000000) + i)
                point = rest.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        self.content_payload_key: text,
                        self.metadata_payload_key: metadata,
                    },
                )
                points.append(point)
                ids_out.append(point_id)

            async def _upsert():
                async with self._pool.get_client() as client:
                    await client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )

            await self._circuit_breaker.call(_upsert)

            return ids_out

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return documents most similar to query with relevance scores."""
        if not self._initialized:
            await self.initialize()

        if self.embedding is None:
            raise ValueError("Embeddings must be provided for similarity search.")

        # Generate query embedding asynchronously if possible
        if hasattr(self.embedding, "aembed_query"):
            embedding = await self.embedding.aembed_query(query)
        else:
            # Fallback to sync embedding in thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.embedding.embed_query,
                query
            )

        async with self._semaphore:
            async def _search():
                async with self._pool.get_client() as client:
                    return await client.search(
                        collection_name=self.collection_name,
                        query_vector=embedding,
                        limit=k,
                        query_filter=self._convert_filter(filter) if filter else None,
                        search_params=kwargs.get("search_params"),
                    )

            results = await self._circuit_breaker.call(_search)

        documents = []
        for i, result in enumerate(results):
            payload = result.payload
            text = payload.get(self.content_payload_key, "")
            metadata = payload.get(self.metadata_payload_key, {})
            
            # If text is empty but content exists in metadata, use it
            if not text and "content" in metadata:
                text = metadata["content"]
                logger.info(f"Vector store - Result {i}: Using content from metadata: '{text[:100]}...'")
            
            # Create document with correct text
            doc = Document(page_content=text, metadata=metadata)
            documents.append((doc, result.score))
            
            logger.info(f"Vector store - Result {i}: Final document text='{doc.text[:100]}...'")

        return documents

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents most similar to query."""
        docs_and_scores = await self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    async def delete(self, ids: list[str], **kwargs: Any) -> bool | None:
        """Delete vectors by ID."""
        if not self._initialized:
            await self.initialize()

        async with self._semaphore:
            async def _delete():
                async with self._pool.get_client() as client:
                    await client.delete(
                        collection_name=self.collection_name,
                        points_selector=rest.PointIdsList(points=ids),
                    )

            await self._circuit_breaker.call(_delete)

        return True

    async def _bootstrap_collection(self) -> None:
        """Create the collection if it does not exist."""
        async with self._pool.get_client() as client:
            collections = await client.get_collections()
            collection_names = {c.name for c in collections.collections}

            if self.collection_name in collection_names:
                return  # Already exists

            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' "
                f"(vector size={self.vector_size}, distance={self.distance})"
            )

            await client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.vector_size,
                    distance=rest.Distance[self.distance.upper()],
                ),
            )

    def _convert_filter(self, filter_dict: dict[str, Any]) -> Any:
        """Convert LangChain filter dict to Qdrant filter."""
        filter_clauses = []
        for key, value in filter_dict.items():
            metadata_key = f"{self.metadata_payload_key}.{key}"
            filter_clauses.append(rest.FieldCondition(
                key=metadata_key,
                match=rest.MatchValue(value=value),
            ))

        if len(filter_clauses) == 1:
            return filter_clauses[0]
        return rest.Filter(must=filter_clauses)

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "collection_name": self.collection_name,
            "vector_size": self.vector_size,
            "batch_size": self.batch_size,
            "max_concurrent_requests": self.max_concurrent_requests,
            "initialized": self._initialized,
        }

        if self._pool:
            stats["connection_pool"] = self._pool.get_stats()

        if self._circuit_breaker:
            stats["circuit_breaker"] = self._circuit_breaker.get_stats()

        return stats

    @classmethod
    async def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        collection_name: str = "documents",
        vector_size: int | None = None,
        **kwargs: Any,
    ) -> "AsyncQdrantStore":
        """Create AsyncQdrantStore instance from texts."""
        # Determine vector size if not provided
        if vector_size is None and texts:
            if hasattr(embedding, "aembed_query"):
                sample_embedding = await embedding.aembed_query(texts[0])
            else:
                sample_embedding = embedding.embed_query(texts[0])
            vector_size = len(sample_embedding)

        vector_size = vector_size or 1024

        instance = cls(
            collection_name=collection_name,
            vector_size=vector_size,
            embedding=embedding,
            **kwargs,
        )

        await instance.initialize()

        if texts:
            await instance.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return instance
