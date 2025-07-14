import json
import random
import uuid
from typing import Any, Dict, List

from locust import HttpUser, between, task


# ==== Locust Performance Test Scenarios ==== #
# --► USER BEHAVIOR SIMULATION


class SentioUser(HttpUser):
    """
    Simulates user behavior for the Sentio API to perform load testing.

    This class defines various tasks that a virtual user will perform,
    including chatting, ingesting documents, and searching documents.
    """
    # Wait time between consecutive tasks for each user, in seconds.
    wait_time = between(1, 3)
    
    def on_start(self) -> None:
        """
        Initialize user session.

        This method is called when a new virtual user starts. It can be used
        for authentication or any other setup required at the beginning of
        a user's session.

        Preconditions:
            - The Locust test is running.

        Postconditions:
            - User session is set up (e.g., authenticated) if required.
        """
        # For this example, we'll assume no authentication is required.
        # If authentication is needed, add it here.
        pass
    
    @task(3)
    def chat_with_documents(self) -> None:
        """
        Simulate a user chatting with documents via the /chat endpoint.

        This task generates a random query from a predefined list and sends it
        to the chat endpoint. It verifies that the response has a 200 status
        code and contains an 'answer' field.

        Preconditions:
            - The API server is running and the /chat endpoint is accessible.

        Postconditions:
            - A chat request is sent and its response is validated.
        """
        # Generate a random query from a list of common questions to simulate diverse user input.
        queries: List[str] = [
            "What is the main purpose of this system?",
            "How does the RAG architecture work?",
            "Explain the document indexing process.",
            "What are the key components of the system?",
            "How is vector search implemented?",
            "What is the role of embeddings in the system?",
            "How does the system handle multiple documents?",
            "What security measures are implemented?",
            "How is the system deployed in Azure?",
            "What are the performance characteristics?",
        ]

        query: str = random.choice(queries)

        # Create a unique conversation ID for this session to track distinct interactions.
        conversation_id: str = str(uuid.uuid4())

        # Define HTTP headers for the request.
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        # Construct the payload for the chat request.
        payload: Dict[str, Any] = {
            "query": query,
            "conversation_id": conversation_id,
            "stream": False,
        }

        # Send the chat request and catch the response for validation.
        with self.client.post(
            "/chat", json=payload, headers=headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                # Verify that the response has the expected structure and contains the 'answer' field.
                try:
                    result: Dict[str, Any] = response.json()
                    if "answer" not in result:
                        response.failure("Response missing 'answer' field")
                except json.JSONDecodeError:
                    response.failure("Response was not valid JSON")
            else:
                response.failure(
                    f"Chat request failed with status code: {response.status_code}"
                )
    
    @task(1)
    def ingest_document(self) -> None:
        """
        Simulate document ingestion via the /embed endpoint.

        This task creates a sample document and sends it to the ingestion endpoint.
        It expects a 200 OK status code, indicating that the document
        has been successfully queued or indexed.
        """
        # Create a sample document for ingestion with a unique ID.
        document_id: str = str(uuid.uuid4())

        # Define HTTP headers for the request.
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        # Construct the payload for the document ingestion request.
        payload: Dict[str, Any] = {
            "id": document_id,
            "content": "This is a sample document for load testing. " * 20,
            "metadata": {
                "title": f"Test Document {document_id[:8]}",
                "author": "Load Test",
                "created_at": "2025-07-07T12:00:00Z",
            },
        }

        # Send the ingestion request and catch the response for validation.
        with self.client.post(
            "/embed", json=payload, headers=headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                # 200 OK is the expected response for sync/async processing.
                pass
            else:
                response.failure(
                    f"Document ingestion failed with status code: {response.status_code}"
                )
    
    @task(2)
    def search_documents(self) -> None:
        """
        Simulate searching for documents via the /search endpoint.

        This task generates a random search query from a predefined list and sends it
        to the search endpoint. It verifies that the response has a 200 status
        code and contains a 'results' field.

        Preconditions:
            - The API server is running and the /search endpoint is accessible.
            - (Implicit: There might be some indexed documents for search to return results.)

        Postconditions:
            - A search request is sent and its response is validated.
        """
        # Generate a random search query from a list of common terms.
        search_terms: List[str] = [
            "architecture",
            "deployment",
            "vector",
            "embedding",
            "azure",
            "container",
            "cosmos",
            "queue",
            "processing",
            "security",
        ]

        query: str = random.choice(search_terms)

        # Send the search request and catch the response for validation.
        with self.client.get(
            f"/search?query={query}&limit=5", catch_response=True
        ) as response:
            if response.status_code == 200:
                # Verify that the response has the expected structure and contains the 'results' field.
                try:
                    result: Dict[str, Any] = response.json()
                    if "results" not in result:
                        response.failure("Response missing 'results' field")
                except json.JSONDecodeError:
                    response.failure("Response was not valid JSON")
            else:
                response.failure(
                    f"Search request failed with status code: {response.status_code}"
                )


# To run this Locust file, execute the following command in your terminal:
# locust -f locustfile.py --host=http://localhost:8000
# Replace 'http://localhost:8000' with the actual URL of your API if it's hosted elsewhere.
# After starting Locust, open your web browser and navigate to http://localhost:8089
# to access the Locust web UI and start the load test. 