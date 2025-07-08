import pytest
import os
import sys
import uuid
import time
from dotenv import load_dotenv

# Add the parent directory to sys.path to import the app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env.azure file
load_dotenv(".env.azure")

# Import Azure modules
from src.azure.cosmos_db import CosmosMetadataStore
from src.azure.storage_queue import AzureStorageQueue
from src.azure.app_insights import ApplicationInsightsLogger


@pytest.fixture
def cosmos_client():
    """Create a Cosmos DB client for testing."""
    connection_string = os.getenv("AZURE_COSMOS_CONNECTION_STRING")
    if not connection_string:
        pytest.skip("AZURE_COSMOS_CONNECTION_STRING not set, skipping Azure integration tests")
    
    client = CosmosMetadataStore(
        connection_string=connection_string,
        database_name="sentio-test",
        container_name="metadata-test"
    )
    
    # Create test container if it doesn't exist
    client.initialize()
    
    yield client
    
    # Clean up test data after tests
    try:
        client.delete_container()
    except Exception as e:
        print(f"Error cleaning up test container: {e}")


@pytest.fixture
def queue_client():
    """Create an Azure Storage Queue client for testing."""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        pytest.skip("AZURE_STORAGE_CONNECTION_STRING not set, skipping Azure integration tests")
    
    client = AzureStorageQueue(
        connection_string=connection_string,
        queue_name="test-queue"
    )
    
    # Create test queue if it doesn't exist
    client.initialize()
    
    yield client
    
    # Clean up test queue after tests
    try:
        client.delete_queue()
    except Exception as e:
        print(f"Error cleaning up test queue: {e}")


@pytest.fixture
def app_insights_client():
    """Create an Application Insights client for testing."""
    connection_string = os.getenv("APPINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        pytest.skip("APPINSIGHTS_CONNECTION_STRING not set, skipping Azure integration tests")
    
    return ApplicationInsightsLogger(connection_string=connection_string)


def test_cosmos_db_operations(cosmos_client):
    """Test basic CRUD operations with Cosmos DB."""
    # Generate a unique document ID for testing
    doc_id = f"test-doc-{uuid.uuid4()}"
    
    # Test document creation
    document = {
        "id": doc_id,
        "title": "Test Document",
        "content": "This is a test document for Cosmos DB integration testing.",
        "created_at": "2025-07-07T12:00:00Z"
    }
    
    # Create document
    result = cosmos_client.create_item(document)
    assert result["id"] == doc_id
    
    # Read document
    retrieved = cosmos_client.get_item(doc_id)
    assert retrieved["id"] == doc_id
    assert retrieved["title"] == "Test Document"
    
    # Update document
    document["title"] = "Updated Test Document"
    updated = cosmos_client.update_item(document)
    assert updated["title"] == "Updated Test Document"
    
    # Query documents
    query_results = cosmos_client.query_items("SELECT * FROM c WHERE c.id = @id", {"@id": doc_id})
    assert len(query_results) == 1
    assert query_results[0]["id"] == doc_id
    
    # Delete document
    cosmos_client.delete_item(doc_id)
    
    # Verify deletion
    query_results = cosmos_client.query_items("SELECT * FROM c WHERE c.id = @id", {"@id": doc_id})
    assert len(query_results) == 0


def test_storage_queue_operations(queue_client):
    """Test Azure Storage Queue operations."""
    # Generate a unique message ID
    message_id = str(uuid.uuid4())
    message_content = f"Test message {message_id}"
    
    # Send message to queue
    queue_client.send_message(message_content)
    
    # Wait a moment for the message to be available
    time.sleep(2)
    
    # Receive message from queue
    messages = queue_client.receive_messages(max_messages=1)
    assert len(messages) > 0
    
    # Check message content
    received_message = messages[0]
    assert message_content in received_message.content
    
    # Delete the message
    queue_client.delete_message(received_message)


def test_app_insights_logging(app_insights_client):
    """Test logging to Application Insights."""
    # Log a test event
    app_insights_client.log_event("test_event", {"test_property": "test_value"})
    
    # Log a test metric
    app_insights_client.log_metric("test_metric", 42.0)
    
    # Log a test exception
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        app_insights_client.log_exception(e)
    
    # No assertions here since we can't easily verify the logs in App Insights
    # This test mainly ensures that the logging calls don't raise exceptions 