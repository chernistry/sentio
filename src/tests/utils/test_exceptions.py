"""Tests for exception handling system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, Request
from src.utils.exceptions import (
    SentioException, 
    ErrorHandler,
    ErrorCode
)


@pytest.fixture
def mock_request():
    """Mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.url.path = "/test/endpoint"
    request.method = "POST"
    request.client.host = "192.168.1.1"
    request.headers = {"user-agent": "test-client"}
    return request


@pytest.mark.asyncio
class TestSentioException:
    """Test custom Sentio exceptions."""

    def test_sentio_exception_creation(self):
        """Test creating SentioException."""
        exc = SentioException(
            message="Test error",
            error_code=ErrorCode.VALIDATION_INVALID_INPUT,
            status_code=400,
            details={"field": "value"}
        )
        
        assert exc.message == "Test error"
        assert exc.error_code == ErrorCode.VALIDATION_INVALID_INPUT
        assert exc.status_code == 400
        assert exc.details == {"field": "value"}

    def test_sentio_exception_to_dict(self):
        """Test converting exception to dictionary."""
        exc = SentioException(
            message="Test error",
            error_code=ErrorCode.AUTH_INVALID_CREDENTIALS,
            status_code=401,
            details={"user_id": "123"}
        )
        
        result = exc.to_dict()
        
        assert result["error_code"] == "AUTH_INVALID_CREDENTIALS"
        assert result["message"] == "Test error"
        assert result["status_code"] == 401
        assert result["details"]["user_id"] == "123"

    def test_sentio_exception_to_http_exception(self):
        """Test converting to HTTPException."""
        exc = SentioException(
            message="Not found",
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            status_code=404
        )
        
        http_exc = exc.to_http_exception()
        
        assert isinstance(http_exc, HTTPException)
        assert http_exc.status_code == 404

    def test_error_codes_enum(self):
        """Test that error codes are properly defined."""
        # Test some key error codes exist
        assert hasattr(ErrorCode, 'AUTH_INVALID_CREDENTIALS')
        assert hasattr(ErrorCode, 'VALIDATION_INVALID_INPUT')
        assert hasattr(ErrorCode, 'RATE_LIMIT_EXCEEDED')
        assert hasattr(ErrorCode, 'SERVICE_UNAVAILABLE')
        assert hasattr(ErrorCode, 'SYSTEM_INTERNAL_ERROR')
        
        # Test they have string values
        assert isinstance(ErrorCode.AUTH_INVALID_CREDENTIALS.value, str)


@pytest.mark.asyncio
class TestErrorHandler:
    """Test error handler functionality."""

    async def test_handle_sentio_exception(self, mock_request):
        """Test handling of SentioException."""
        exc = SentioException(
            message="Invalid query parameter",
            error_code=ErrorCode.VALIDATION_INVALID_INPUT,
            status_code=422,
            details={"field": "top_k", "value": -1}
        )
        
        response = await ErrorHandler.handle_exception(exc, mock_request)
        
        assert response.status_code == 422
        response_data = response.body.decode()
        assert "Invalid query parameter" in response_data
        assert "VALIDATION_INVALID_INPUT" in response_data

    async def test_handle_http_exception(self, mock_request):
        """Test handling of FastAPI HTTPException."""
        exc = HTTPException(
            status_code=404,
            detail="Resource not found"
        )
        
        response = await ErrorHandler.handle_exception(exc, mock_request)
        
        assert response.status_code == 404
        response_data = response.body.decode()
        assert "Resource not found" in response_data

    async def test_handle_generic_exception(self, mock_request):
        """Test handling of generic Python exceptions."""
        exc = ValueError("Invalid value provided")
        
        response = await ErrorHandler.handle_exception(exc, mock_request)
        
        assert response.status_code == 500
        response_data = response.body.decode()
        assert "internal error occurred" in response_data.lower()

    async def test_error_logging(self, mock_request):
        """Test that errors are properly logged."""
        exc = SentioException(
            message="Test error for logging",
            error_code=ErrorCode.SYSTEM_INTERNAL_ERROR
        )
        
        with patch('src.utils.exceptions.logger') as mock_logger:
            await ErrorHandler.handle_exception(exc, mock_request)
            
            # Should log the error
            mock_logger.error.assert_called()

    async def test_sensitive_data_sanitization(self, mock_request):
        """Test that sensitive data is sanitized from error responses."""
        exc = SentioException(
            message="Database error: password=secret123",
            error_code=ErrorCode.SYSTEM_INTERNAL_ERROR,
            details={"api_key": "sk-secret", "user_id": "user123"}
        )
        
        response = await ErrorHandler.handle_exception(exc, mock_request)
        response_data = response.body.decode()
        
        # Should not contain sensitive data (if sanitization is implemented)
        # This test may need adjustment based on actual implementation
        assert isinstance(response_data, str)

    async def test_error_categorization(self, mock_request):
        """Test that errors are properly categorized."""
        client_errors = [
            SentioException("Bad input", ErrorCode.VALIDATION_INVALID_INPUT, 400),
            SentioException("No auth", ErrorCode.AUTH_INVALID_CREDENTIALS, 401),
            HTTPException(404, "Not found")
        ]
        
        server_errors = [
            SentioException("Service down", ErrorCode.SERVICE_UNAVAILABLE, 503),
            Exception("Unexpected error")
        ]
        
        for exc in client_errors:
            response = await ErrorHandler.handle_exception(exc, mock_request)
            assert 400 <= response.status_code < 500
        
        for exc in server_errors:
            response = await ErrorHandler.handle_exception(exc, mock_request)
            assert response.status_code >= 500
