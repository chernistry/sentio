from locust import HttpUser, task, between
import json
import random
import uuid


class SentioUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session with authentication if needed."""
        # For this example, we'll assume no authentication is required
        # If authentication is needed, add it here
        pass
    
    @task(3)
    def chat_with_documents(self):
        """Simulate a user chatting with documents."""
        # Generate a random query from a list of common questions
        queries = [
            "What is the main purpose of this system?",
            "How does the RAG architecture work?",
            "Explain the document indexing process.",
            "What are the key components of the system?",
            "How is vector search implemented?",
            "What is the role of embeddings in the system?",
            "How does the system handle multiple documents?",
            "What security measures are implemented?",
            "How is the system deployed in Azure?",
            "What are the performance characteristics?"
        ]
        
        query = random.choice(queries)
        
        # Create a unique conversation ID for this session
        conversation_id = str(uuid.uuid4())
        
        # Send the chat request
        headers = {"Content-Type": "application/json"}
        payload = {
            "query": query,
            "conversation_id": conversation_id,
            "stream": False
        }
        
        with self.client.post("/chat", 
                             json=payload, 
                             headers=headers, 
                             catch_response=True) as response:
            if response.status_code == 200:
                # Verify that the response has the expected structure
                try:
                    result = response.json()
                    if "answer" not in result:
                        response.failure("Response missing 'answer' field")
                except json.JSONDecodeError:
                    response.failure("Response was not valid JSON")
            else:
                response.failure(f"Chat request failed with status code: {response.status_code}")
    
    @task(1)
    def ingest_document(self):
        """Simulate document ingestion."""
        # Create a sample document for ingestion
        document_id = str(uuid.uuid4())
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "document_id": document_id,
            "content": "This is a sample document for load testing. " * 20,  # ~200 words
            "metadata": {
                "title": f"Test Document {document_id[:8]}",
                "author": "Load Test",
                "created_at": "2025-07-07T12:00:00Z"
            }
        }
        
        with self.client.post("/ingest", 
                             json=payload, 
                             headers=headers, 
                             catch_response=True) as response:
            if response.status_code == 202:
                # 202 Accepted is the expected response for async processing
                pass
            else:
                response.failure(f"Document ingestion failed with status code: {response.status_code}")
    
    @task(2)
    def search_documents(self):
        """Simulate searching for documents."""
        # Generate a random search query
        search_terms = [
            "architecture",
            "deployment",
            "vector",
            "embedding",
            "azure",
            "container",
            "cosmos",
            "queue",
            "processing",
            "security"
        ]
        
        query = random.choice(search_terms)
        
        with self.client.get(f"/search?query={query}&limit=5", 
                            catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "results" not in result:
                        response.failure("Response missing 'results' field")
                except json.JSONDecodeError:
                    response.failure("Response was not valid JSON")
            else:
                response.failure(f"Search request failed with status code: {response.status_code}")


# To run this locust file:
# locust -f locustfile.py --host=https://your-api-url
# Then open http://localhost:8089 in your browser 