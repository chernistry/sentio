#!/usr/bin/env python3
"""
Tests for the LangGraph implementation of the RAG pipeline.

This module tests the core functionality of the LangGraph-based pipeline,
ensuring it behaves correctly with various inputs and configurations.
"""

import pytest
import os
from unittest.mock import AsyncMock, patch

from root.src.core.graph.graph_factory import build_basic_graph_for_server, RAGState
from root.src.core.pipeline import PipelineConfig, SentioRAGPipeline
from root.src.utils.settings import settings


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline with controlled behavior."""
    pipeline = SentioRAGPipeline()
    pipeline.initialized = True  # Skip initialization
    
    # Mock the core methods
    pipeline.retrieve = AsyncMock(return_value={
        "documents": [
            {"text": "Document 1", "source": "source1", "score": 0.95},
            {"text": "Document 2", "source": "source2", "score": 0.85},
        ],
        "query": "test query",
        "strategy": "hybrid",
        "total_time": 0.1,
        "sources_found": 2
    })
    
    pipeline.rerank = AsyncMock(return_value=[
        {"text": "Document 1", "source": "source1", "score": 0.95},
    ])
    
    pipeline.generate = AsyncMock(return_value={
        "answer": "This is a test answer",
        "sources": [{"text": "Document 1", "source": "source1", "score": 0.95}],
        "query": "test query",
        "mode": "balanced",
        "total_time": 0.2,
        "token_count": 10,
        "timestamp": "2025-07-17T10:00:00Z"
    })
    
    return pipeline


@pytest.mark.asyncio
@patch("root.src.core.graph.graph_factory._get_initialized_pipeline")
async def test_graph_basic_flow(mock_get_pipeline, mock_pipeline):
    """Test the basic flow of the LangGraph pipeline."""
    # Setup the mock to return our controlled pipeline
    mock_get_pipeline.return_value = mock_pipeline
    
    # Create the graph
    graph = build_basic_graph_for_server()
    
    # Execute the graph
    result = await graph.ainvoke({"query": "test query"})
    
    # Verify the result
    assert "answer" in result
    assert result["answer"] == "This is a test answer"
    
    # Check that pipeline methods were called
    mock_pipeline.retrieve.assert_called_once()
    mock_pipeline.rerank.assert_called_once()
    mock_pipeline.generate.assert_called_once()


@pytest.mark.asyncio
@patch("root.src.core.graph.graph_factory._get_initialized_pipeline")
async def test_graph_empty_query(mock_get_pipeline, mock_pipeline):
    """Test that the graph handles empty queries gracefully."""
    # Setup the mock to return our controlled pipeline
    mock_get_pipeline.return_value = mock_pipeline
    
    # Create the graph
    graph = build_basic_graph_for_server()
    
    # Execute the graph with an empty query
    result = await graph.ainvoke({"query": ""})
    
    # Verify the result still contains an answer
    assert "answer" in result
    
    # Check that pipeline methods were still called
    mock_pipeline.retrieve.assert_called_once()
    mock_pipeline.rerank.assert_called_once()
    mock_pipeline.generate.assert_called_once()


@pytest.mark.asyncio
@patch("root.src.core.graph.graph_factory._get_initialized_pipeline")
async def test_graph_no_documents_found(mock_get_pipeline, mock_pipeline):
    """Test that the graph handles the case when no documents are found."""
    # Override the retrieve method to return no documents
    mock_pipeline.retrieve = AsyncMock(return_value={
        "documents": [],
        "query": "test query",
        "strategy": "hybrid",
        "total_time": 0.1,
        "sources_found": 0
    })
    
    # Setup the mock to return our controlled pipeline
    mock_get_pipeline.return_value = mock_pipeline
    
    # Create the graph
    graph = build_basic_graph_for_server()
    
    # Execute the graph
    result = await graph.ainvoke({"query": "test query"})
    
    # Verify the result
    assert "answer" in result
    assert len(result["retrieved_documents"]) == 0
    assert result["metadata"]["sources_found"] == 0
    
    # Check that pipeline methods were called
    mock_pipeline.retrieve.assert_called_once()
    mock_pipeline.rerank.assert_called_once()
    mock_pipeline.generate.assert_called_once() 