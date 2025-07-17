#!/usr/bin/env python3
"""
Tests for the RAGAS evaluation node in LangGraph.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os

# Set environment variables for testing
os.environ["ENABLE_AUTOMATIC_EVALUATION"] = "true"
os.environ["USE_LANGGRAPH"] = "true"


@pytest.fixture
def mock_pipeline_with_evaluator():
    """Create a mock pipeline with evaluator for testing."""
    mock_pipeline = MagicMock()
    mock_evaluator = MagicMock()
    
    # Setup the evaluator mock
    mock_evaluator._openrouter_evaluation = AsyncMock(return_value={
        "faithfulness": 0.85,
        "answer_relevancy": 0.78,
        "context_relevancy": 0.92
    })
    
    # Attach the evaluator to the pipeline
    mock_pipeline.evaluator = mock_evaluator
    
    return mock_pipeline


@pytest.fixture
def mock_state():
    """Create a mock state for testing."""
    from root.src.core.graph import RAGState
    
    state = RAGState(
        query="What is the capital of France?",
        answer="The capital of France is Paris.",
        sources=[
            {"text": "Paris is the capital and most populous city of France.", "metadata": {"id": "doc1"}},
            {"text": "France is a country in Western Europe with several territories.", "metadata": {"id": "doc2"}}
        ],
        metadata={}
    )
    
    return state


@pytest.mark.asyncio
async def test_ragas_node_evaluation(mock_pipeline_with_evaluator, mock_state):
    """Test that the RAGAS node correctly evaluates and updates the state."""
    from root.src.core.graph import ragas_evaluation_node
    
    # Call the node
    result_state = await ragas_evaluation_node(mock_state, mock_pipeline_with_evaluator)
    
    # Verify the evaluator was called
    mock_pipeline_with_evaluator.evaluator._openrouter_evaluation.assert_called_once()
    
    # Verify the state was updated with evaluation results
    assert "evaluation" in result_state.metadata
    assert "metrics" in result_state.metadata["evaluation"]
    assert "thresholds" in result_state.metadata["evaluation"]
    assert "passed_thresholds" in result_state.metadata["evaluation"]
    
    # Verify the metrics match what the evaluator returned
    assert result_state.metadata["evaluation"]["metrics"]["faithfulness"] == 0.85
    assert result_state.metadata["evaluation"]["metrics"]["answer_relevancy"] == 0.78
    assert result_state.metadata["evaluation"]["metrics"]["context_relevancy"] == 0.92


@pytest.mark.asyncio
async def test_ragas_node_handles_errors(mock_pipeline_with_evaluator, mock_state):
    """Test that the RAGAS node gracefully handles errors."""
    from root.src.core.graph import ragas_evaluation_node
    
    # Make the evaluator raise an exception
    mock_pipeline_with_evaluator.evaluator._openrouter_evaluation = AsyncMock(
        side_effect=ValueError("Test error")
    )
    
    # Call the node
    result_state = await ragas_evaluation_node(mock_state, mock_pipeline_with_evaluator)
    
    # Verify the evaluator was called
    mock_pipeline_with_evaluator.evaluator._openrouter_evaluation.assert_called_once()
    
    # Verify the state was updated with error information
    assert "evaluation_error" in result_state.metadata
    assert "Test error" in result_state.metadata["evaluation_error"]


@pytest.mark.asyncio
async def test_ragas_node_skips_when_no_evaluator(mock_state):
    """Test that the RAGAS node skips evaluation when no evaluator is available."""
    from root.src.core.graph import ragas_evaluation_node
    
    # Create a pipeline with no evaluator
    mock_pipeline = MagicMock()
    
    # Call the node
    result_state = await ragas_evaluation_node(mock_state, mock_pipeline)
    
    # Verify the state was returned unchanged
    assert "evaluation" not in result_state.metadata
    assert "evaluation_error" not in result_state.metadata 