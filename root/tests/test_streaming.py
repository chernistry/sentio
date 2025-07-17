#!/usr/bin/env python3
"""
Tests for the streaming functionality in LangGraph.
"""

import pytest
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Set environment variables for testing
os.environ["USE_LANGGRAPH"] = "true"


class AsyncIteratorMock:
    """Mock for an async iterator that returns a sequence of values."""
    
    def __init__(self, values):
        self.values = values
        self.index = 0
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if self.index < len(self.values):
            value = self.values[self.index]
            self.index += 1
            return value
        raise StopAsyncIteration


@pytest.fixture
def mock_pipeline_with_streaming():
    """Create a mock pipeline with streaming capability."""
    mock_pipeline = MagicMock()
    mock_pipeline.config = MagicMock()
    mock_pipeline.config.top_k_retrieval = 5
    mock_pipeline.config.top_k_final = 2
    mock_pipeline.config.generation_mode = "test_mode"
    
    # Mock regular generation methods
    mock_pipeline.generate = AsyncMock(return_value=MagicMock(
        answer="This is a test answer.",
        sources=[],
        query="test query",
        mode="test_mode",
        total_time=0.2,
        token_count=10,
        timestamp="2023-01-01"
    ))
    
    # Mock streaming generation method
    async def mock_generate_stream(query, context, mode=None):
        tokens = ["This ", "is ", "a ", "test ", "answer", "."]
        for token in tokens:
            yield token
    
    mock_pipeline.generate_stream = mock_generate_stream
    
    # Mock other methods needed for graph execution
    mock_pipeline.retrieve = AsyncMock(return_value=MagicMock(
        documents=[{"text": "Test doc", "score": 0.9}],
        query="test query",
        strategy="test",
        total_time=0.1,
        sources_found=1
    ))
    
    mock_pipeline.rerank = AsyncMock(return_value=[
        {"text": "Test doc", "score": 0.9}
    ])
    
    return mock_pipeline


@pytest.mark.asyncio
async def test_stream_generator_node(mock_pipeline_with_streaming):
    """Test that the streaming generator node yields tokens incrementally."""
    from root.src.core.graph.streaming import stream_generator_node
    from root.src.core.graph import RAGState
    
    # Create a test state
    state = RAGState(
        query="test query",
        retrieved_documents=[{"text": "Test doc", "score": 0.9}],
        metadata={}
    )
    
    # Collect all streamed tokens
    tokens = []
    async for chunk in stream_generator_node.astream(state, mock_pipeline_with_streaming):
        tokens.append(chunk["answer"])
    
    # Check that tokens were streamed correctly
    assert tokens == ["This ", "is ", "a ", "test ", "answer", "."]
    
    # Check that the state was updated with the complete answer
    assert state.answer == "This is a test answer."


@pytest.mark.asyncio
async def test_streaming_wrapper():
    """Test that the StreamingWrapper correctly wraps a graph for streaming."""
    from root.src.core.graph.streaming import StreamingWrapper
    
    # Create a mock graph
    mock_graph = MagicMock()
    mock_graph.graph = MagicMock()
    
    # Create a mock node
    mock_node = MagicMock()
    mock_node.config = {
        "display_name": "generator",
        "func": MagicMock()
    }
    
    # Set up the mock node function to stream tokens
    async def stream_func(state, *args, **kwargs):
        tokens = ["Token1", "Token2", "Token3"]
        for token in tokens:
            yield {"answer": token}
        return state
    
    # Attach streaming method to the mock function
    mock_node.config["func"].astream = AsyncIteratorMock([
        {"answer": "Token1"},
        {"answer": "Token2"},
        {"answer": "Token3"}
    ])
    
    # Set up the graph to return our mock node
    mock_graph.graph.nodes = [mock_node]
    
    # Create the streaming wrapper
    wrapper = StreamingWrapper(mock_graph)
    
    # Set up the graph to immediately return when invoked
    result_state = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=result_state)
    
    # Collect all streamed tokens
    tokens = []
    async for chunk in wrapper.astream({"query": "test query"}):
        tokens.append(chunk.get("answer"))
    
    # Check that tokens were streamed correctly
    assert tokens == ["Token1", "Token2", "Token3"]


@pytest.mark.asyncio
async def test_streaming_api_response():
    """Test that the API streaming response formats chunks correctly."""
    from fastapi.testclient import TestClient
    from root.src.api.routes import chat_stream_endpoint, ChatRequest
    
    # Create a mock StreamingResponse
    async def mock_streaming_wrapper_astream(*args, **kwargs):
        chunks = [
            {"answer": "This "},
            {"answer": "is "},
            {"answer": "a "},
            {"answer": "streaming "},
            {"answer": "test."}
        ]
        for chunk in chunks:
            yield chunk
    
    # Create a mock streaming graph
    mock_streaming_graph = MagicMock()
    mock_streaming_graph.astream = mock_streaming_wrapper_astream
    
    # Patch the get_streaming_graph function
    with patch("root.src.api.routes.get_streaming_graph", return_value=AsyncMock(return_value=mock_streaming_graph)):
        # Create a chat request
        request = ChatRequest(question="test question", stream=True)
        
        # Call the stream endpoint
        response = await chat_stream_endpoint(request)
        
        # Check that we got a StreamingResponse
        assert response.media_type == "text/event-stream"
        
        # Extract and verify the streamed content
        content = []
        async for chunk in response.body_iterator:
            content.append(chunk.decode())
        
        # Each chunk should be a JSON string followed by a newline
        parsed_content = [json.loads(chunk) for chunk in content]
        
        # Check that the chunks have the expected format
        for i, chunk in enumerate(parsed_content[:-1]):
            assert "answer" in chunk
            assert chunk["done"] is False
        
        # Check that the final chunk has done=True
        assert parsed_content[-1]["done"] is True 