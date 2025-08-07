from __future__ import annotations

"""Qdrant vector-store wrapper.

This thin abstraction hides the concrete *qdrant-client* API and provides
some ergonomic helpers (collection bootstrap, health-check).

The implementation is intentionally minimal – it is *not* a full-blown
repository pattern – but rather a convenience layer that centralises
configuration and future-proofs against API shape changes.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from src.core.models.document import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from src.utils.settings import settings

__all__ = ["QdrantStore"]

logger = logging.getLogger(__name__)


def _get_default_timeout() -> float:
    return 10.0


@dataclass(slots=True)
class QdrantStore(VectorStore):
    """Wrapper around :pymod:`qdrant_client` with helper routines.

    The class supports Qdrant Cloud instances.
    *URL* and *API key* are resolved from environment variables unless
    provided explicitly.

    Note:
    ----
    The sync client is used for now.  Async operations can be introduced
    later by swapping to :class:`qdrant_client.AsyncQdrantClient`.
    """

    collection_name: str
    vector_size: int = 1024
    distance: str = "Cosine"
    url: str | None = None
    api_key: str | None = None
    timeout: float = field(default_factory=_get_default_timeout)
    embedding: Embeddings | None = None
    content_payload_key: str = "content"
    metadata_payload_key: str = "metadata"

    _client: QdrantClient = field(init=False, repr=False)

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Initialise underlying Qdrant client and ensure collection exists."""
        # Resolve configuration lazily to make the class test-friendly.
        resolved_url = self.url or settings.qdrant_url
        if resolved_url is None:
            raise ValueError("QDRANT_URL env var or 'url' argument must be provided.")

        resolved_key = self.api_key or settings.qdrant_api_key
        if resolved_key is None:
            raise ValueError("QDRANT_API_KEY env var or 'api_key' argument must be provided.")

        # Ensure HTTPS for cloud instances
        if ".cloud.qdrant.io" in resolved_url:
            if not resolved_url.startswith("https://"):
                resolved_url = f"https://{resolved_url.removeprefix('http://')}"
                logger.info("Converted Qdrant cloud URL to HTTPS: %s", resolved_url)

        # For debugging
        logger.debug("Connecting to Qdrant at %s", resolved_url)

        # Instantiate client (sync for now).
        self._client = QdrantClient(url=resolved_url, api_key=resolved_key, timeout=self.timeout)

        logger.debug("Qdrant client initialised (url=%s, collection=%s)", resolved_url, self.collection_name)

        # Ensure collection exists.
        self._bootstrap_collection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, query: str, search_type: str, **kwargs: Any) -> list[Document]:
        """Search for documents using the specified search type.
        
        Args:
            query: Text query to search for
            search_type: Type of search (e.g., "similarity")
            **kwargs: Additional search parameters
            
        Returns:
            List of documents
        """
        if search_type == "similarity":
            k = kwargs.get("k", 4)
            filter_dict = kwargs.get("filter", None)
            return self.similarity_search(query, k=k, filter=filter_dict, **kwargs)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")

    async def search_async(
        self,
        query: str,
        search_type: str = "similarity",
        query_vector: list[float] = None,
        top_k: int = 4,
        filter_dict: dict[str, Any] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for similar documents.
        
        Args:
            query: Text query (if query_vector not provided)
            search_type: Type of search (similarity, etc.)
            query_vector: Pre-computed query vector
            top_k: Number of results to return
            filter_dict: Filter by metadata
            **kwargs: Additional arguments
            
        Returns:
            List of similar documents
        """
        if query_vector is None:
            if self.embedding is None:
                raise ValueError("Embeddings must be provided for text search.")
            query_vector = self.embedding.embed_query(query)
            
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=self._convert_filter(filter_dict) if filter_dict else None,
        )

        documents = []
        for result in results:
            payload = result.payload
            text = payload.get(self.content_payload_key, "")
            metadata = payload.get(self.metadata_payload_key, {})
            
            # If text is empty but content exists in metadata, use it
            if not text and "content" in metadata:
                text = metadata["content"]
                
            documents.append(Document(text=text, metadata=metadata))

        return documents

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            embeddings: Pre-computed embeddings (optional)
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        if embeddings is not None and len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
            
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.id for doc in documents]
        
        if embeddings is None:
            return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        else:
            return self.add_embeddings(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                **kwargs
            )

    async def delete_documents(self, document_ids: list[str]) -> bool:
        """Delete documents by ID.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        result = self.delete(document_ids)
        return result

    async def get_collection_info(self) -> dict[str, Any]:
        """Get collection information.
        
        Returns:
            Collection information
        """
        collection_info = self._client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": collection_info.config.params.vectors.distance.name,
            "points_count": collection_info.points_count,
        }

    async def health_check(self) -> bool:
        """Check if the vector store is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            self._client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant health-check failed: %s", exc)
            return False

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine"
    ) -> None:
        """Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of vectors
            distance: Distance metric
        """
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance[distance.upper()],
            ),
        )

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
        """
        self._client.delete_collection(collection_name)

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists
        """
        try:
            self._client.get_collection(collection_name)
            return True
        except Exception:
            return False

    async def get_document_count(self) -> int:
        """Get the number of documents in the collection.
        
        Returns:
            Number of documents
        """
        collection_info = self._client.get_collection(self.collection_name)
        return collection_info.points_count

    def health_check(self) -> bool:
        """Return *True* when Qdrant instance is reachable."""
        try:
            self._client.get_collections()
            return True
        except Exception as exc:  # pragma: no cover – network exceptions vary
            logger.warning("Qdrant health-check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # LangChain VectorStore API Implementation
    # ------------------------------------------------------------------
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store.

        Args:
            texts: Iterable of strings to add to the vector store.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs to associate with the texts.
            **kwargs: Optional keyword arguments.

        Returns:
            List of IDs of the added texts.
        """
        if self.embedding is None:
            raise ValueError("Embeddings must be provided for adding texts.")

        embeddings = self.embedding.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            **kwargs
        )

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add embeddings to the vector store.

        Args:
            texts: Iterable of strings to add to the vector store.
            embeddings: List of embeddings to add.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs to associate with the texts.
            **kwargs: Optional keyword arguments.

        Returns:
            List of IDs of the added embeddings.
        """
        texts_list = list(texts)
        if len(texts_list) != len(embeddings):
            raise ValueError("Number of texts and embeddings must be the same.")

        metadatas = metadatas or [{}] * len(texts_list)
        # Pad metadatas to match texts length if needed
        if len(metadatas) < len(texts_list):
            metadatas.extend([{}] * (len(texts_list) - len(metadatas)))
        elif len(metadatas) > len(texts_list):
            metadatas = metadatas[:len(texts_list)]

        points = []
        ids_out = []

        for i, (text, embedding, metadata) in enumerate(zip(texts_list, embeddings, metadatas, strict=False)):
            point_id = str(ids[i]) if ids and i < len(ids) else str(i)
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

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return ids_out

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return documents most similar to query with relevance scores.

        Args:
            query: Text to search for.
            k: Number of results to return.
            filter: Filter by metadata.
            **kwargs: Additional arguments.

        Returns:
            List of tuples of (document, relevance score).
        """
        if self.embedding is None:
            raise ValueError("Embeddings must be provided for similarity search.")

        embedding = self.embedding.embed_query(query)
        search_params = kwargs.get("search_params")

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=k,
            query_filter=self._convert_filter(filter) if filter else None,
            search_params=search_params,
        )

        documents = []
        for result in results:
            payload = result.payload
            text = payload.get(self.content_payload_key, "")
            metadata = payload.get(self.metadata_payload_key, {})
            
            # If text is empty but content exists in metadata, use it
            if not text and "content" in metadata:
                text = metadata["content"]
                logger.info(f"Vector store - Result: Using content from metadata: '{text[:100]}...'")
            
            documents.append((Document(text=text, metadata=metadata), result.score))

        return documents

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents most similar to query.

        Args:
            query: Text to search for.
            k: Number of results to return.
            filter: Filter by metadata.
            **kwargs: Additional arguments.

        Returns:
            List of documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def delete(self, ids: list[str], **kwargs: Any) -> bool | None:
        """Delete by vector ID.

        Args:
            ids: List of IDs to delete.
            **kwargs: Additional arguments.

        Returns:
            Boolean indicating whether the operation was successful.
        """
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=rest.PointIdsList(
                points=ids,
            ),
        )
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _bootstrap_collection(self) -> None:
        """Create the collection if it does not yet exist."""
        collections = {c.name for c in self._client.get_collections().collections}
        if self.collection_name in collections:
            return  # ✅ Already exists

        logger.info("Creating Qdrant collection '%s' (vector size=%s, distance=%s)", self.collection_name, self.vector_size, self.distance)

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(  # type: ignore[arg-type]
                size=self.vector_size,
                distance=rest.Distance[self.distance.upper()],
            ),
        )

    def _convert_filter(self, filter_dict: dict[str, Any]) -> Any:
        """Convert LangChain filter dict to Qdrant filter."""
        # Simple implementation for basic filters
        filter_clauses = []
        for key, value in filter_dict.items():
            metadata_key = f"{self.metadata_payload_key}.{key}"
            filter_clauses.append(rest.FieldCondition(
                key=metadata_key,
                match=rest.MatchValue(value=value),
            ))

        if len(filter_clauses) == 1:
            return filter_clauses[0]
        return rest.Filter(
            must=filter_clauses
        )

    # ------------------------------------------------------------------
    # Convenience passthroughs
    # ------------------------------------------------------------------
    def __getattr__(self, item: str) -> Any:
        """Delegate unknown attributes to the underlying client."""
        return getattr(self._client, item)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        collection_name: str = "documents",
        vector_size: int | None = None,
        **kwargs: Any,
    ) -> QdrantStore:
        """Return VectorStore instance from texts.

        Args:
            texts: List of texts.
            embedding: Embeddings to use.
            metadatas: Optional list of metadatas.
            ids: Optional list of IDs.
            collection_name: Name of the collection to store the vectors in.
            vector_size: Size of the embedding vectors.
            **kwargs: Additional arguments.

        Returns:
            A QdrantStore instance.
        """
        # Determine vector size if not provided
        if vector_size is None:
            if texts:
                # Embed the first text to get the vector size
                vector_size = len(embedding.embed_query(texts[0]))
            else:
                vector_size = 1024  # Default size

        instance = cls(
            collection_name=collection_name,
            vector_size=vector_size,
            embedding=embedding,
            **kwargs,
        )

        if texts:
            instance.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return instance
