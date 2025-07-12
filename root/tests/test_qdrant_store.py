"""
Integration tests for Qdrant vector store operations.
"""
import uuid
from typing import List

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, PointStruct, UpdateStatus

from root.src.core.tasks.embeddings import EmbeddingModel

pytestmark = pytest.mark.integration


@pytest.fixture(scope="function")
def collection_name() -> str:
    """
    Generates a unique collection name for each test function to ensure isolation.
    """
    return f"test-collection-{uuid.uuid4().hex}"


@pytest.mark.asyncio
async def test_qdrant_collection_create_upsert_and_search(
    qdrant_client: QdrantClient,
    embed_model: EmbeddingModel,
    collection_name: str,
    sample_docs: List[str],
):
    """
    Tests the end-to-end flow of:
    1. Creating a collection in Qdrant.
    2. Embedding documents and upserting them.
    3. Searching for a document and verifying the result.
    """
    # Skip this test as the mock EmbeddingModel doesn't create deterministic vectors
    # Test enabled – deterministic vectors implemented in EmbeddingModel stub
    
    # 1. Create collection
    qdrant_client.delete_collection(collection_name=collection_name)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embed_model.dimension, distance=models.Distance.COSINE
        ),
    )

    # 2. Embed and Upsert
    doc_to_search = sample_docs[2]  # "Qdrant is a vector database..."
    doc_to_search_vector = await embed_model.embed_async_single(doc_to_search)

    upsert_result = qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=await embed_model.embed_async_single(doc),
                payload={"text": doc},
            )
            for doc in sample_docs
        ],
        wait=True,
    )
    assert upsert_result.status == UpdateStatus.COMPLETED

    # 3. Search
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=doc_to_search_vector,
        limit=1,
    )

    # The top result should be the document we searched for.
    assert len(search_results) == 1
    top_hit = search_results[0]
    assert top_hit.payload["text"] == doc_to_search


def test_qdrant_dimension_mismatch(
    qdrant_client: QdrantClient,
    embed_model: EmbeddingModel,
    collection_name: str,
):
    """
    Tests that Qdrant raises an error when trying to upsert a vector
    with a dimension that does not match the collection's configured dimension.
    """
    # Create a collection with a dimension that is intentionally wrong.
    wrong_dimension = embed_model.dimension + 1
    qdrant_client.delete_collection(collection_name=collection_name)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=wrong_dimension, distance=models.Distance.COSINE
        ),
    )

    # Generate a vector with the correct dimension.
    # This vector will be invalid for the collection.
    vector_with_correct_dim = [0.1] * embed_model.dimension
    
    # Qdrant server should reject this point. The client raises an exception,
    # often a generic one wrapping a 4xx HTTP error.
    # We catch a broad Exception as the exact type can vary.
    with pytest.raises(Exception) as excinfo:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=str(uuid.uuid4()), vector=vector_with_correct_dim)
            ],
        )

    # Check that the error message indicates a vector dimension mismatch.
    # The exact message can change between qdrant-client versions.
    error_str = str(excinfo.value).lower()
    assert "wrong input: vector inserting error" in error_str or "wrong vector size" in error_str 