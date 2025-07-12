"""
This module contains integration tests for Azure services, such as
Cosmos DB for metadata storage and Storage Queues for message passing.

These tests are marked as 'integration' and 'azure' and will only run if:
1. The necessary Azure SDKs (`azure-cosmos`, `azure-storage-queue`) are installed.
2. The required environment variables (e.g., connection strings) are present
   in a `.env.azure` file or in the system environment.
"""
import os
import sys
import time
import uuid
from typing import Generator

import pytest
from dotenv import load_dotenv

# Add project root to path to allow importing from `root`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Conditionally import Azure modules
try:
    from root.src.azure.cosmos_db import CosmosMetadataStore
    from root.src.azure.storage_queue import AzureStorageQueue
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False
    # Define dummy classes if SDK is not present to avoid runtime errors
    class CosmosMetadataStore: pass
    class AzureStorageQueue: pass

# Load Azure-specific environment variables for integration tests
load_dotenv(".env.azure")


@pytest.fixture(scope="module")
def cosmos_client() -> Generator[CosmosMetadataStore, None, None]:
    """
    Creates a Cosmos DB client for testing, cleaning up afterwards.
    Skips test if connection string is not found.
    """
    connection_string = os.getenv("AZURE_COSMOS_CONNECTION_STRING")
    if not connection_string:
        pytest.skip("AZURE_COSMOS_CONNECTION_STRING not set, skipping Azure tests")

    client = CosmosMetadataStore(
        connection_string=connection_string,
        database_name="sentio-test-db",
        container_name=f"metadata-test-container-{uuid.uuid4().hex}",
    )
    client.initialize()
    yield client
    try:
        client.delete_container()
    except Exception as e:
        print(f"Error cleaning up test container: {e}")


@pytest.fixture(scope="module")
def queue_client() -> Generator[AzureStorageQueue, None, None]:
    """
    Creates an Azure Storage Queue client for testing, cleaning up afterwards.
    Skips test if connection string is not found.
    """
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        pytest.skip("AZURE_STORAGE_CONNECTION_STRING not set, skipping Azure tests")

    client = AzureStorageQueue(
        connection_string=connection_string,
        queue_name=f"test-queue-{uuid.uuid4().hex}",
    )
    client.initialize()
    yield client
    try:
        client.delete_queue()
    except Exception as e:
        print(f"Error cleaning up test queue: {e}")


@pytest.mark.azure
@pytest.mark.integration
@pytest.mark.skipif(not AZURE_SDK_AVAILABLE, reason="Azure SDKs not installed")
def test_cosmos_db_operations(cosmos_client: CosmosMetadataStore):
    """
    Tests basic CRUD operations with Azure Cosmos DB.
    """
    doc_id = f"test-doc-{uuid.uuid4().hex}"
    document = {"id": doc_id, "title": "Test Document"}

    # Create
    result = cosmos_client.create_item(document)
    assert result["id"] == doc_id

    # Read
    retrieved = cosmos_client.get_item(doc_id)
    assert retrieved["id"] == doc_id

    # Update
    document["title"] = "Updated Title"
    updated = cosmos_client.update_item(document)
    assert updated["title"] == "Updated Title"

    # Query
    query_results = cosmos_client.query_items("SELECT * FROM c WHERE c.id = @id", {"@id": doc_id})
    assert len(query_results) == 1

    # Delete
    cosmos_client.delete_item(doc_id)
    retrieved_after_delete = cosmos_client.get_item(doc_id)
    assert retrieved_after_delete is None


@pytest.mark.azure
@pytest.mark.integration
@pytest.mark.skipif(not AZURE_SDK_AVAILABLE, reason="Azure SDKs not installed")
def test_storage_queue_operations(queue_client: AzureStorageQueue):
    """
    Tests sending, receiving, and deleting messages from an Azure Storage Queue.
    """
    message_content = f"Test message {uuid.uuid4().hex}"
    queue_client.send_message(message_content)

    # Allow time for message to be processed
    time.sleep(2)

    messages = queue_client.receive_messages(max_messages=1)
    assert messages
    received_message = messages[0]
    assert received_message.content == message_content

    queue_client.delete_message(received_message) 