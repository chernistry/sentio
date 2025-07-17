#!/usr/bin/env python3
"""
Basic test for LangGraph implementation without dependencies on conftest.py.

This simplified test ensures the LangGraph implementation can be loaded and executed
with minimal dependencies, making it easier to diagnose any import-related issues.
"""

import pytest
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Explicitly set environment variables for testing
os.environ["QDRANT_COLLECTION"] = "test_collection"
os.environ["USE_LANGGRAPH"] = "true"


@pytest.fixture
def mock_pipeline():
    """Create a simple mocked pipeline for testing."""
    mock = MagicMock()
    mock.config = MagicMock()
    mock.config.top_k_retrieval = 5
    mock.config.top_k_final = 2
    
    # Mock retrieve method
    mock.retrieve = AsyncMock(return_value=MagicMock(
        documents=[
            {"text": "Test doc 1", "score": 0.9, "metadata": {"id": "doc1"}},
            {"text": "Test doc 2", "score": 0.8, "metadata": {"id": "doc2"}},
        ],
        query="test query",
        strategy="test",
        total_time=0.1,
        sources_found=2
    ))
    
    # Mock rerank method
    mock.rerank = AsyncMock(return_value=[
        {"text": "Test doc 1", "score": 0.9, "metadata": {"id": "doc1"}},
    ])
    
    # Mock generate method
    mock.generate = AsyncMock(return_value=MagicMock(
        answer="Test answer",
        sources=[{"text": "Test doc 1", "score": 0.9, "metadata": {"id": "doc1"}}],
        query="test query",
        mode="test",
        total_time=0.2,
        token_count=10,
        timestamp="2023-01-01"
    ))
    
    return mock


@pytest.mark.asyncio
async def test_graph_factory_import():
    """Test that the LangGraph factory can be imported without errors."""
    # Simple import test
    from root.src.core.graph import build_basic_graph
    
    # If we get here without errors, the import works
    assert callable(build_basic_graph)


@pytest.mark.asyncio
async def test_graph_construction(mock_pipeline):
    """Test that a graph can be constructed with the factory."""
    from root.src.core.pipeline import PipelineConfig
    from root.src.core.graph import build_basic_graph
    
    # Create a basic config
    config = PipelineConfig()
    
    # Build the graph
    graph = build_basic_graph(config, mock_pipeline)
    
    # The graph should be callable
    assert callable(graph)


@pytest.mark.asyncio
async def test_graph_basic_invocation(mock_pipeline):
    """Test that a graph can be invoked with a simple query."""
    from root.src.core.pipeline import PipelineConfig
    from root.src.core.graph import build_basic_graph
    
    # Create a basic config
    config = PipelineConfig()
    
    # Build the graph
    graph = build_basic_graph(config, mock_pipeline)
    
    # Invoke the graph
    result = await graph.ainvoke({"query": "test query"})
    
    # Basic assertions
    assert result is not None
    assert hasattr(result, "answer")
    assert result.answer == "Test answer"
    
    # Verify mocks were called
    mock_pipeline.retrieve.assert_called_once()
    mock_pipeline.rerank.assert_called_once()
    mock_pipeline.generate.assert_called_once()


if __name__ == "__main__":
    # Run the tests directly for quick debugging
    asyncio.run(test_graph_factory_import())
    print("✓ Import test passed")
    
    mock = mock_pipeline(None)
    asyncio.run(test_graph_construction(mock))
    print("✓ Construction test passed")
    
    asyncio.run(test_graph_basic_invocation(mock))
    print("✓ Invocation test passed")
    
    print("All tests passed!") 