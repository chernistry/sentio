"""Tests for OpenAI LLM provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.llm.providers.openai import OpenAIProvider
from src.core.models.document import Document


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = AsyncMock()
    
    # Mock chat completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response from OpenAI"
    mock_response.usage.total_tokens = 150
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    
    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def openai_provider(mock_openai_client):
    """Create OpenAI provider with mocked client."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value = mock_openai_client
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

        response = await openai_provider.generate_response(
            query="What is machine learning?",
            documents=documents,
            history=[]
        )

        # Verify OpenAI client was called
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args

        # Check request parameters
        assert call_args.kwargs["model"] == "gpt-3.5-turbo"
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["max_tokens"] == 1000
        assert len(call_args.kwargs["messages"]) >= 2  # System + user message

        # Check response
        assert response == "Test response from OpenAI"

    async def test_generate_response_with_history(self, openai_provider, mock_openai_client):
        """Test response generation with conversation history."""
        documents = [Document(text="Test document", metadata={"source": "doc.pdf"})]
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        await openai_provider.generate_response(
            query="Follow up question",
            documents=documents,
            history=history
        )

        # Verify history was included in messages
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        # Should have system message + history + current query
        assert len(messages) >= 4
        assert any(msg["content"] == "Previous question" for msg in messages)
        assert any(msg["content"] == "Previous answer" for msg in messages)

    async def test_generate_response_no_documents(self, openai_provider, mock_openai_client):
        """Test response generation with no documents."""
        response = await openai_provider.generate_response(
            query="General question",
            documents=[],
            history=[]
        )

        # Should still generate response
        assert response == "Test response from OpenAI"
        
        # Verify call was made
        mock_openai_client.chat.completions.create.assert_called_once()

    async def test_generate_response_api_error(self, openai_provider, mock_openai_client):
        """Test handling of OpenAI API errors."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await openai_provider.generate_response(
                query="Test query",
                documents=[],
                history=[]
            )

    async def test_generate_response_rate_limit(self, openai_provider, mock_openai_client):
        """Test handling of rate limit errors."""
        from openai import RateLimitError
        
        mock_openai_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(),
            body={}
        )

        with pytest.raises(RateLimitError):
            await openai_provider.generate_response(
                query="Test query",
                documents=[],
                history=[]
            )

    async def test_build_prompt_with_documents(self, openai_provider):
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

        # Should have system message and user message
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        
        # User message should contain documents and query
        user_content = messages[1]["content"]
        assert "Document 1 content" in user_content
        assert "Document 2 content" in user_content
        assert "Test query" in user_content

    async def test_build_prompt_token_limit(self, openai_provider):
        """Test prompt building with token limit considerations."""
        # Create very long documents that would exceed token limit
        long_documents = [
            Document(text="Very long content " * 1000, metadata={"source": f"doc{i}.pdf"})
            for i in range(10)
        ]

        messages = openai_provider._build_messages(
            query="Test query",
            documents=long_documents,
            history=[]
        )

        # Should still build valid messages (truncated if necessary)
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    async def test_health_check_success(self, openai_provider, mock_openai_client):
        """Test successful health check."""
        # Mock a simple completion for health check
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_openai_client.chat.completions.create.return_value = mock_response

        is_healthy = await openai_provider.health_check()

        assert is_healthy is True
        mock_openai_client.chat.completions.create.assert_called_once()

    async def test_health_check_failure(self, openai_provider, mock_openai_client):
        """Test health check failure."""
        mock_openai_client.chat.completions.create.side_effect = Exception("Connection failed")

        is_healthy = await openai_provider.health_check()

        assert is_healthy is False

    async def test_token_counting(self, openai_provider):
        """Test token counting functionality."""
        text = "This is a test message for token counting."
        
        # Mock tiktoken encoder
        with patch('tiktoken.encoding_for_model') as mock_tiktoken:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_tiktoken.return_value = mock_encoder

            token_count = openai_provider._count_tokens(text)
            assert token_count == 5

    async def test_streaming_response(self, openai_provider, mock_openai_client):
        """Test streaming response generation."""
        # Mock streaming response
        async def mock_stream():
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))])
            ]
            for chunk in chunks:
                yield chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        # Test streaming (if implemented)
        if hasattr(openai_provider, 'generate_response_stream'):
            response_chunks = []
            async for chunk in openai_provider.generate_response_stream(
                query="Test query",
                documents=[],
                history=[]
            ):
                response_chunks.append(chunk)

            assert len(response_chunks) == 3
            assert "".join(response_chunks) == "Hello world!"

    async def test_model_configuration(self, mock_openai_client):
        """Test different model configurations."""
        # Test with different models
        models_to_test = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        
        for model in models_to_test:
            with patch('httpx.AsyncClient') as mock_client:
                mock_client.return_value = mock_openai_client
                provider = OpenAIProvider(
                    api_key="test-key",
                    model=model
                )
                provider._client = mock_openai_client

                await provider.generate_response(
                    query="Test query",
                    documents=[],
                    history=[]
                )

                # Verify correct model was used
                call_args = mock_openai_client.chat.completions.create.call_args
                assert call_args.kwargs["model"] == model
                assert call_args.kwargs["temperature"] == 0.5
                assert call_args.kwargs["max_tokens"] == 500

    async def test_error_recovery(self, openai_provider, mock_openai_client):
        """Test error recovery mechanisms."""
        # First call fails, second succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("Temporary error"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Success"))])
        ]

        # If retry logic is implemented
        if hasattr(openai_provider, '_retry_on_error'):
            response = await openai_provider.generate_response(
                query="Test query",
                documents=[],
                history=[]
            )
            assert response == "Success"
        else:
            # Should fail on first attempt
            with pytest.raises(Exception, match="Temporary error"):
                await openai_provider.generate_response(
                    query="Test query",
                    documents=[],
                    history=[]
                )
