#!/usr/bin/env python3
"""
Smoke test for the LangGraph implementation.

This module tests the basic functionality of the LangGraph-based
RAG pipeline against the legacy pipeline to ensure identical output.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from root.src.core.pipeline import (
    SentioRAGPipeline,
    PipelineConfig,
    RetrievalResult,
    GenerationResult,
)
from root.src.core.graph import build_basic_graph

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_pipeline():
    """Create a mocked pipeline with predefined responses."""
    pipeline = MagicMock()
    pipeline.config = PipelineConfig(
        top_k_retrieval=5,
        top_k_final=2,
    )
    
    # Mock the retrieve method
    async def mock_retrieve(query, top_k=None):
        return RetrievalResult(
            documents=[
                {"text": "Document 1", "source": "source1", "score": 0.95, "metadata": {"id": "doc1"}},
                {"text": "Document 2", "source": "source2", "score": 0.85, "metadata": {"id": "doc2"}},
                {"text": "Document 3", "source": "source3", "score": 0.75, "metadata": {"id": "doc3"}},
            ],
            query=query,
            strategy="hybrid",
            total_time=0.1,
            sources_found=3,
        )
    pipeline.retrieve = AsyncMock(side_effect=mock_retrieve)
    
    # Mock the rerank method
    async def mock_rerank(query, documents, top_k=None):
        return documents[:2]  # Return first 2 documents
    pipeline.rerank = AsyncMock(side_effect=mock_rerank)
    
    # Mock the generate method
    async def mock_generate(query, context, mode=None):
        return GenerationResult(
            answer="This is a test answer",
            sources=context,
            query=query,
            mode="balanced",
            total_time=0.2,
            token_count=7,
        )
    pipeline.generate = AsyncMock(side_effect=mock_generate)
    
    return pipeline


@pytest.mark.asyncio
async def test_graph_output_matches_legacy_pipeline(mock_pipeline):
    """Test that the LangGraph output matches the legacy pipeline output."""
    # Setup
    config = PipelineConfig()
    query = "What is the test query?"
    
    # Create the graph using our factory
    graph = build_basic_graph(config, mock_pipeline)
    
    # Execute the graph
    result = await graph.ainvoke({"query": query})
    
    # Execute the legacy pipeline query method
    async def mock_query(question, top_k=None):
        retrieval_result = await mock_pipeline.retrieve(question)
        reranked_docs = await mock_pipeline.rerank(question, retrieval_result.documents)
        generation_result = await mock_pipeline.generate(question, reranked_docs)
        
        return {
            "answer": generation_result.answer,
            "sources": reranked_docs,
            "metadata": {
                "retrieval_time": retrieval_result.total_time,
                "generation_time": generation_result.total_time,
                "query_time": retrieval_result.total_time + generation_result.total_time,
                "sources_found": retrieval_result.sources_found,
                "sources_used": len(reranked_docs),
                "retrieval_strategy": retrieval_result.strategy,
                "generation_mode": generation_result.mode,
            }
        }
    
    mock_pipeline.query = AsyncMock(side_effect=mock_query)
    legacy_result = await mock_pipeline.query(query)
    
    # Assert core outputs match
    assert result.answer == legacy_result["answer"]
    assert len(result.sources) == len(legacy_result["sources"])
    assert result.metadata["retrieval_strategy"] == legacy_result["metadata"]["retrieval_strategy"]
    assert result.metadata["generation_time"] == legacy_result["metadata"]["generation_time"]
    
    # Verify all methods were called correctly
    mock_pipeline.retrieve.assert_called_once()
    mock_pipeline.rerank.assert_called_once()
    mock_pipeline.generate.assert_called_once()


@pytest.mark.asyncio
async def test_graph_handles_empty_results(mock_pipeline):
    """Test that the LangGraph handles empty results gracefully."""
    # Setup
    config = PipelineConfig()
    query = "What is the test query?"
    
    # Override mocks to return empty results
    mock_pipeline.retrieve = AsyncMock(return_value=RetrievalResult(
        documents=[],
        query=query,
        strategy="hybrid",
        total_time=0.05,
        sources_found=0,
    ))
    
    # Create and execute the graph
    graph = build_basic_graph(config, mock_pipeline)
    result = await graph.ainvoke({"query": query})
    
    # Assertions
    assert len(result.retrieved_documents) == 0
    assert result.metadata["sources_found"] == 0
    # The graph should still proceed through all steps
    mock_pipeline.rerank.assert_called_once()
    mock_pipeline.generate.assert_called_once() 