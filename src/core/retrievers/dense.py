from __future__ import annotations

"""Dense similarity search via Qdrant."""

import logging

from qdrant_client import QdrantClient

from src.core.embeddings.base import BaseEmbedder
from src.core.models.document import Document
from src.core.vector_store.qdrant_store import QdrantStore

from .base import BaseRetriever

logger = logging.getLogger(__name__)

# Default vector name for Qdrant points – overridable via env var
_DEFAULT_VECTOR_NAME = "text-dense"


class DenseRetriever(BaseRetriever):
    """Dense vector retriever powered by Qdrant."""

    def __init__(
        self,
        client: QdrantClient | QdrantStore,
        embedder: BaseEmbedder,
        collection_name: str,
        vector_name: str | None = None,
    ) -> None:
        self._client: QdrantClient = (
            client._client if isinstance(client, QdrantStore) else client  # type: ignore[attr-defined]
        )
        self._embedder = embedder
        self._collection = collection_name
        self._vector_name = vector_name or _DEFAULT_VECTOR_NAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        logger.debug("DenseRetriever: searching for '%s' (top_k=%s)", query, top_k)
        query_vec = self._embedder.embed_sync(query)

        # Prepare search params, handling vector_name parameter compatibility
        search_params = {
            "collection_name": self._collection,
            "query_vector": query_vec,
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,
        }

        # Add vector_name parameter safely
        try:
            # Check if the client's search method accepts vector_name parameter
            import inspect
            search_sig = inspect.signature(self._client.search)
            if "vector_name" in search_sig.parameters:
                # If vector_name is supported, include it
                search_params["vector_name"] = self._vector_name

            # Perform the search with appropriate parameters
            points = self._client.search(**search_params)
        except TypeError as e:
            if "vector_name" in str(e):
                # If vector_name causes TypeError, try without it
                logger.warning("Vector name parameter not supported by Qdrant client: %s", e)
                search_params.pop("vector_name", None)
                points = self._client.search(**search_params)
            else:
                # Re-raise if it's a different error
                raise

        docs: list[Document] = []
        for p in points:
            text = (p.payload or {}).get("text") if isinstance(p.payload, dict) else None
            docs.append(
                Document(
                    id=str(p.id),
                    text=text or "",
                    metadata={"score": p.score, **(p.payload or {})},
                )
            )
        return docs
