"""Tests for the chat handler - core RAG functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.api.handlers.chat import ChatHandler
from src.core.models.document import Document


@pytest.fixture
def mock_graph():
    """Mock LangGraph instance."""
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "response": "Test response",
        "selected_documents": [
            Document(
                text="Test document content",
                metadata={"source": "test.pdf", "score": 0.9}
            )
        ],
        "metadata": {"processing_time": 0.5}
    }
    return graph


@pytest.fixture
def mock_metrics():
    """Mock metrics collector."""
    return MagicMock()


@pytest.fixture
def chat_handler(mock_graph, mock_metrics):
    """Create ChatHandler instance with mocked dependencies."""
    handler = ChatHandler()
    # Mock the internal graph after initialization
    handler._graph = mock_graph
    handler._initialized = True
    return handler


@pytest.mark.asyncio
class TestChatHandler:
    """Test ChatHandler functionality."""

    async def test_process_chat_request_success(self, chat_handler, mock_graph):
        """Test successful chat request processing."""
        result = await chat_handler.process_chat_request(
            question="What is machine learning?",
            history=[],
            top_k=3,
            temperature=0.7
        )

        # Verify graph was called with correct parameters
        mock_graph.ainvoke.assert_called_once()
        call_args = mock_graph.ainvoke.call_args[0][0]
        assert call_args["query"] == "What is machine learning?"

        # Verify response structure
        assert "answer" in result
        assert "sources" in result
        assert "metadata" in result
        assert result["answer"] == "Test response"
        assert len(result["sources"]) == 1

    async def test_process_chat_request_with_history(self, chat_handler, mock_graph):
        """Test chat request with conversation history."""
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        result = await chat_handler.process_chat_request(
            question="Follow up question",
            history=history,
            top_k=5,
            temperature=0.3
        )

        # Verify graph was called
        mock_graph.ainvoke.assert_called_once()
        call_args = mock_graph.ainvoke.call_args[0][0]
        assert call_args["query"] == "Follow up question"

        # Verify response
        assert "answer" in result
        assert "sources" in result

    async def test_process_chat_request_empty_response(self, chat_handler, mock_graph):
        """Test handling of empty response from graph."""
        mock_graph.ainvoke.return_value = {
            "response": "",
            "selected_documents": [],
            "metadata": {}
        }

        result = await chat_handler.process_chat_request(
            question="Test question",
            history=[],
            top_k=3,
            temperature=0.7
        )

        assert result["answer"] == ""
        assert result["sources"] == []

    async def test_process_chat_request_graph_error(self, chat_handler, mock_graph):
        """Test error handling when graph fails."""
        mock_graph.ainvoke.side_effect = Exception("Graph processing failed")

        with pytest.raises(Exception, match="Graph processing failed"):
            await chat_handler.process_chat_request(
                question="Test question",
                history=[],
                top_k=3,
                temperature=0.7
            )

    async def test_format_sources(self, chat_handler):
        """Test source formatting."""
        documents = [
            Document(
                text="First document content",
                metadata={"source": "doc1.pdf", "score": 0.9, "page": 1}
            ),
            Document(
                text="Second document content", 
                metadata={"source": "doc2.pdf", "score": 0.8, "page": 2}
            )
        ]

        sources = chat_handler._format_sources(documents)

        assert len(sources) == 2
        assert sources[0]["text"] == "First document content"
        assert sources[0]["source"] == "doc1.pdf"
        assert sources[0]["score"] == 0.9
        assert sources[0]["metadata"]["page"] == 1

    async def test_validate_parameters(self, chat_handler):
        """Test parameter validation."""
        # Valid parameters should not raise
        chat_handler._validate_parameters(
            question="Valid question",
            top_k=5,
            temperature=0.7
        )

        # Invalid question
        with pytest.raises(ValueError, match="Question cannot be empty"):
            chat_handler._validate_parameters(
                question="",
                top_k=5,
                temperature=0.7
            )

        # Invalid top_k
        with pytest.raises(ValueError, match="top_k must be between 1 and 20"):
            chat_handler._validate_parameters(
                question="Valid question",
                top_k=0,
                temperature=0.7
            )

        # Invalid temperature
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            chat_handler._validate_parameters(
                question="Valid question",
                top_k=5,
                temperature=3.0
            )

    async def test_metrics_collection(self, chat_handler, mock_metrics):
        """Test that metrics are collected during processing."""
        await chat_handler.process_chat_request(
            question="Test question",
            history=[],
            top_k=3,
            temperature=0.7
        )

        # Verify metrics were recorded
        assert mock_metrics.record_value.call_count > 0
        
        # Check for specific metrics
        metric_calls = [call[0] for call in mock_metrics.record_value.call_args_list]
        assert any("chat.requests" in str(call) for call in metric_calls)

    async def test_circuit_breaker_integration(self, chat_handler, mock_graph):
        """Test circuit breaker integration."""
        # Simulate circuit breaker open state
        with patch('src.api.handlers.chat.CircuitBreaker') as mock_cb:
            mock_cb_instance = AsyncMock()
            mock_cb.return_value = mock_cb_instance
            mock_cb_instance.__aenter__.side_effect = Exception("Circuit breaker open")

            handler = ChatHandler(graph=mock_graph, metrics_collector=MagicMock())
            
            with pytest.raises(Exception, match="Circuit breaker open"):
                await handler.process_chat_request(
                    question="Test question",
                    history=[],
                    top_k=3,
                    temperature=0.7
                )
