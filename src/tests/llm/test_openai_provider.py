"""Comprehensive tests for OpenAI provider functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.llm.providers.openai import OpenAIProvider
from src.core.models.document import Document


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = AsyncMock()
    
    # Mock successful HTTP response
    mock_response = MagicMock()  # Use MagicMock for sync methods
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()  # Sync method
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Test response from OpenAI"
                }
            }
        ],
        "usage": {
            "total_tokens": 150,
            "prompt_tokens": 100,
            "completion_tokens": 50
        }
    }
    
    client.post.return_value = mock_response
    return client


@pytest.fixture
def openai_provider(mock_openai_client):
    """Create OpenAI provider with mocked client."""
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-3.5-turbo"
    )
    provider._client = mock_openai_client
    return provider


@pytest.mark.asyncio
class TestOpenAIProvider:
    """Test OpenAI provider functionality."""

    async def test_generate_response_success(self, openai_provider, mock_openai_client):
        """Test successful response generation."""
        documents = [
            Document(text="Test document 1", metadata={"source": "doc1.pdf"}),
            Document(text="Test document 2", metadata={"source": "doc2.pdf"})
        ]
        
        # Mock the chat_completion method directly
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Test response from OpenAI"
                    }
                }
            ],
            "usage": {
                "total_tokens": 150,
                "prompt_tokens": 100,
                "completion_tokens": 50
            }
        }
        
        with patch.object(openai_provider, 'chat_completion', return_value=mock_response) as mock_chat:
            response = await openai_provider.generate_response(
                query="What is machine learning?",
                documents=documents,
                history=[]
            )

            # Verify the response
            assert response == "Test response from OpenAI"
            mock_chat.assert_called_once()

    async def test_generate_response_with_history(self, openai_provider, mock_openai_client):
        """Test response generation with conversation history."""
        documents = [Document(text="Test document", metadata={"source": "doc.pdf"})]
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        mock_response = {
            "choices": [{"message": {"content": "Response with history"}}]
        }
        
        with patch.object(openai_provider, 'chat_completion', return_value=mock_response):
            response = await openai_provider.generate_response(
                query="Follow-up question",
                documents=documents,
                history=history
            )

            assert response == "Response with history"

    async def test_generate_response_no_documents(self, openai_provider, mock_openai_client):
        """Test response generation without documents."""
        mock_response = {
            "choices": [{"message": {"content": "Response without documents"}}]
        }
        
        with patch.object(openai_provider, 'chat_completion', return_value=mock_response):
            response = await openai_provider.generate_response(
                query="General question",
                documents=[],
                history=[]
            )

            assert response == "Response without documents"

    async def test_generate_response_api_error(self, openai_provider, mock_openai_client):
        """Test handling of API errors."""
        with patch.object(openai_provider, 'chat_completion', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await openai_provider.generate_response(
                    query="Test query",
                    documents=[],
                    history=[]
                )

    async def test_generate_response_rate_limit(self, openai_provider, mock_openai_client):
        """Test handling of rate limit errors."""
        with patch.object(openai_provider, 'chat_completion', side_effect=Exception("Rate limit exceeded")):
            with pytest.raises(Exception, match="Rate limit exceeded"):
                await openai_provider.generate_response(
                    query="Test query",
                    documents=[],
                    history=[]
                )

    def test_build_prompt_with_documents(self, openai_provider):
        """Test prompt building with documents."""
        documents = [
            Document(text="Document 1 content", metadata={"source": "doc1.pdf"}),
            Document(text="Document 2 content", metadata={"source": "doc2.pdf"})
        ]
        
        messages = openai_provider._build_messages(
            query="Test query",
            documents=documents,
            history=[]
        )

        # Verify message structure
        assert isinstance(messages, list)
        assert len(messages) >= 2  # System + user message
        assert any("Document 1 content" in str(msg) for msg in messages)
        assert any("Document 2 content" in str(msg) for msg in messages)

    def test_build_prompt_token_limit(self, openai_provider):
        """Test prompt building with token limits."""
        # Create a very long document
        long_document = Document(
            text="Very long content " * 1000,
            metadata={"source": "long.pdf"}
        )
        
        messages = openai_provider._build_messages(
            query="Test query",
            documents=[long_document],
            history=[]
        )

        # Should still create valid messages (may truncate content)
        assert isinstance(messages, list)
        assert len(messages) >= 2

    async def test_health_check_success(self, openai_provider, mock_openai_client):
        """Test successful health check."""
        # Mock successful chat completion response
        mock_openai_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "test"}}]
        }
        
        is_healthy = await openai_provider.health_check()
        assert is_healthy is True

    async def test_health_check_failure(self, openai_provider, mock_openai_client):
        """Test health check failure."""
        # Mock failed health check
        mock_openai_client.post.side_effect = Exception("Connection failed")
        
        is_healthy = await openai_provider.health_check()
        assert is_healthy is False

    def test_token_counting(self, openai_provider):
        """Test token counting functionality."""
        text = "This is a test sentence for token counting."
        
        token_count = openai_provider._count_tokens(text)
        
        # Should return a reasonable token count
        assert isinstance(token_count, int)
        assert token_count > 0

    async def test_model_configuration(self, mock_openai_client):
        """Test different model configurations."""
        # Test with different model
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4",
            base_url="https://api.openai.com/v1"
        )
        provider._client = mock_openai_client
        
        mock_response = {
            "choices": [{"message": {"content": "GPT-4 response"}}]
        }
        
        with patch.object(provider, 'chat_completion', return_value=mock_response):
            response = await provider.generate_response(
                query="Test query",
                documents=[],
                history=[]
            )
            
            assert response == "GPT-4 response"

    async def test_error_recovery(self, openai_provider, mock_openai_client):
        """Test error recovery mechanisms."""
        with patch.object(openai_provider, 'chat_completion', side_effect=Exception("Temporary error")):
            with pytest.raises(Exception, match="Temporary error"):
                await openai_provider.generate_response(
                    query="Test query",
                    documents=[],
                    history=[]
                )

    async def test_streaming_response(self, openai_provider, mock_openai_client):
        """Test streaming response handling."""
        # Mock streaming response
        async def mock_stream_generator():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"
        
        with patch.object(openai_provider, 'chat_completion', return_value=mock_stream_generator()):
            response = await openai_provider.generate_response(
                query="Test query",
                documents=[],
                history=[]
            )
            
            # Should handle streaming and return complete response
            assert isinstance(response, str)

    def test_initialization(self):
        """Test provider initialization."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-3.5-turbo",
            base_url="https://api.openai.com/v1"
        )
        
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-3.5-turbo"
        assert provider.base_url == "https://api.openai.com/v1"

    async def test_close_client(self, openai_provider, mock_openai_client):
        """Test client cleanup."""
        await openai_provider.close()
        mock_openai_client.aclose.assert_called_once()

    def test_build_headers(self, openai_provider):
        """Test HTTP header building."""
        headers = openai_provider._build_headers()
        
        assert isinstance(headers, dict)
        assert "Authorization" in headers
        assert "Bearer test-key" in headers["Authorization"]
        assert "Content-Type" in headers

    async def test_concurrent_requests(self, openai_provider):
        """Test concurrent request handling."""
        import asyncio
        
        mock_response = {
            "choices": [{"message": {"content": "Concurrent response"}}]
        }
        
        with patch.object(openai_provider, 'chat_completion', return_value=mock_response):
            # Run concurrent requests
            tasks = [
                openai_provider.generate_response(f"Query {i}", [], [])
                for i in range(5)
            ]
            responses = await asyncio.gather(*tasks)
            
            # All should succeed
            assert len(responses) == 5
            assert all(response == "Concurrent response" for response in responses)

    async def test_large_context_handling(self, openai_provider):
        """Test handling of large context windows."""
        # Create many documents
        documents = [
            Document(text=f"Document {i} content", metadata={"source": f"doc{i}.pdf"})
            for i in range(50)
        ]
        
        mock_response = {
            "choices": [{"message": {"content": "Large context response"}}]
        }
        
        with patch.object(openai_provider, 'chat_completion', return_value=mock_response):
            response = await openai_provider.generate_response(
                query="Query with large context",
                documents=documents,
                history=[]
            )
            
            assert response == "Large context response"

    def test_message_formatting(self, openai_provider):
        """Test message formatting for different scenarios."""
        # Test with empty documents
        messages = openai_provider._build_messages("Query", [], [])
        assert isinstance(messages, list)
        assert len(messages) >= 1
        
        # Test with history
        history = [{"role": "user", "content": "Previous"}]
        messages = openai_provider._build_messages("Query", [], history)
        assert len(messages) >= 2

    async def test_timeout_handling(self, openai_provider, mock_openai_client):
        """Test request timeout handling."""
        # Mock timeout error
        mock_openai_client.post.side_effect = Exception("Request timeout")
        
        # Should return False for health check instead of raising exception
        is_healthy = await openai_provider.health_check()
        assert is_healthy is False

    def test_special_characters_in_query(self, openai_provider):
        """Test handling of special characters in queries."""
        special_query = "Query with special chars: éñ中文🚀"
        
        messages = openai_provider._build_messages(special_query, [], [])
        
        # Should handle special characters gracefully
        assert isinstance(messages, list)
        assert any(special_query in str(msg) for msg in messages)
