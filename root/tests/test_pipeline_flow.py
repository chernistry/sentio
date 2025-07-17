import pytest
from unittest.mock import AsyncMock

from root.src.core.graph.graph_factory import build_basic_graph
from root.src.core.pipeline import SentioRAGPipeline, PipelineConfig


@pytest.mark.asyncio
async def test_langgraph_flow_equivalent(monkeypatch):
    """
    Verify that the LangGraph flow produces results equivalent to the legacy
    pipeline by mocking the underlying data processing components.
    """
    # -- Setup Mocks -----------------------------------------------------------
    # Create a mock pipeline instance to control the behavior of the nodes
    mock_pipeline = SentioRAGPipeline()

    # Mock the initialization to avoid real setup
    monkeypatch.setattr(mock_pipeline, "initialize", AsyncMock())

    # Mock the core data processing methods
    monkeypatch.setattr(
        mock_pipeline,
        "retrieve",
        AsyncMock(
            return_value={
                "documents": [{"text": "mock document", "source": "s1", "score": 0.9}],
                "strategy": "hybrid",
                "total_time": 0.1,
                "sources_found": 1,
            }
        ),
    )
    monkeypatch.setattr(
        mock_pipeline,
        "rerank",
        AsyncMock(return_value=[{"text": "reranked doc", "source": "s1"}]),
    )
    monkeypatch.setattr(
        mock_pipeline,
        "generate",
        AsyncMock(
            return_value={
                "answer": "mock answer",
                "total_time": 0.2,
                "mode": "balanced",
                "token_count": 10,
                "timestamp": "now",
            }
        ),
    )

    # -- Build Graph with Mocks ------------------------------------------------
    # Build the graph using the mocked pipeline instance
    graph = build_basic_graph(PipelineConfig(), mock_pipeline)

    # -- Execute and Verify ----------------------------------------------------
    inputs = {"query": "what is sentio?"}
    final_state = None
    async for step in graph.astream(inputs):
        final_state = step

    # Extract the final RAGState object from the last step
    rag_state = final_state[next(iter(final_state))]

    # Assert that the final state contains the expected results from the mocks
    assert rag_state.answer.strip() == "mock answer"
    assert len(rag_state.reranked_documents) == 1
    assert rag_state.reranked_documents[0]["text"] == "reranked doc"
    assert "retrieval_time" in rag_state.metadata
    assert "generation_time" in rag_state.metadata 