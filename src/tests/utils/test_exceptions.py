"""Tests for exception handling system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, Request
from src.utils.exceptions import (
    SentioException, 
    ErrorHandler,
    ValidationError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError
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
            error_code="TEST_ERROR",
            status_code=400,
            details={"field": "value"}
        )
        
        assert exc.message == "Test error"
        assert exc.error_code == "TEST_ERROR"
        assert exc.status_code == 400
        assert exc.details == {"field": "value"}

    def test_validation_error(self):
        """Test ValidationError subclass."""
        exc = ValidationError(
            message="Invalid input",
            field="email",
            value="invalid-email"
        )
        
        assert exc.status_code == 422
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.details["field"] == "email"
        assert exc.details["value"] == "invalid-email"

    def test_authentication_error(self):
        """Test AuthenticationError subclass."""
        exc = AuthenticationError(message="Invalid token")
        
        assert exc.status_code == 401
        assert exc.error_code == "AUTHENTICATION_ERROR"
        assert exc.message == "Invalid token"

    def test_rate_limit_error(self):
        """Test RateLimitError subclass."""
        exc = RateLimitError(
            message="Rate limit exceeded",
            retry_after=60
        )
        
        assert exc.status_code == 429
        assert exc.error_code == "RATE_LIMIT_ERROR"
        assert exc.details["retry_after"] == 60

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError subclass."""
        exc = ServiceUnavailableError(
            message="Vector store unavailable",
            service="qdrant"
        )
        
        assert exc.status_code == 503
        assert exc.error_code == "SERVICE_UNAVAILABLE"
        assert exc.details["service"] == "qdrant"


@pytest.mark.asyncio
class TestErrorHandler:
    """Test error handler functionality."""

    async def test_handle_sentio_exception(self, mock_request):
        """Test handling of SentioException."""
        exc = ValidationError(
            message="Invalid query parameter",
            field="top_k",
            value=-1
        )
        
        response = await ErrorHandler.handle_exception(exc, mock_request)
        
        assert response.status_code == 422
        response_data = response.body.decode()
        assert "Invalid query parameter" in response_data
        assert "VALIDATION_ERROR" in response_data

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
        assert "Internal server error" in response_data.lower()

    async def test_error_logging(self, mock_request):
        """Test that errors are properly logged."""
        exc = SentioException("Test error for logging")
        
        with patch('src.utils.exceptions.logger') as mock_logger:
            await ErrorHandler.handle_exception(exc, mock_request)
            
            # Should log the error
            mock_logger.error.assert_called()

    async def test_error_metrics_collection(self, mock_request):
        """Test that error metrics are collected."""
        exc = ValidationError("Test validation error")
        
        with patch('src.utils.exceptions.metrics_collector') as mock_metrics:
            await ErrorHandler.handle_exception(exc, mock_request)
            
            # Should record error metrics
            mock_metrics.record_value.assert_called()

    async def test_sensitive_data_sanitization(self, mock_request):
        """Test that sensitive data is sanitized from error responses."""
        exc = SentioException(
            message="Database error: password=secret123",
            details={"api_key": "sk-secret", "user_id": "user123"}
        )
        
        response = await ErrorHandler.handle_exception(exc, mock_request)
        response_data = response.body.decode()
        
        # Should not contain sensitive data
        assert "secret123" not in response_data
        assert "sk-secret" not in response_data

    async def test_error_correlation_id(self, mock_request):
        """Test that error responses include correlation ID."""
        exc = SentioException("Test error with correlation")
        
        response = await ErrorHandler.handle_exception(exc, mock_request)
        
        # Should include correlation ID in headers or response
        assert "X-Correlation-ID" in response.headers or \
               "correlation_id" in response.body.decode()

    async def test_error_context_preservation(self, mock_request):
        """Test that error context is preserved."""
        exc = ValidationError(
            message="Field validation failed",
            field="email",
            value="invalid@"
        )
        
        response = await ErrorHandler.handle_exception(exc, mock_request)
        response_data = response.body.decode()
        
        # Should preserve validation context
        assert "email" in response_data
        assert "VALIDATION_ERROR" in response_data

    async def test_nested_exception_handling(self, mock_request):
        """Test handling of nested exceptions."""
        try:
            try:
                raise ValueError("Inner exception")
            except ValueError as e:
                raise SentioException("Outer exception") from e
        except SentioException as exc:
            response = await ErrorHandler.handle_exception(exc, mock_request)
            
            assert response.status_code == 500
            # Should handle nested exception gracefully

    async def test_concurrent_error_handling(self, mock_request):
        """Test concurrent error handling."""
        import asyncio
        
        exceptions = [
            ValidationError("Error 1"),
            AuthenticationError("Error 2"),
            RateLimitError("Error 3")
        ]
        
        tasks = [
            ErrorHandler.handle_exception(exc, mock_request)
            for exc in exceptions
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All should be handled successfully
        assert len(responses) == 3
        assert all(hasattr(r, 'status_code') for r in responses)

    async def test_error_recovery_suggestions(self, mock_request):
        """Test that error responses include recovery suggestions."""
        exc = RateLimitError(
            message="Too many requests",
            retry_after=60
        )
        
        response = await ErrorHandler.handle_exception(exc, mock_request)
        response_data = response.body.decode()
        
        # Should include recovery suggestion
        assert "retry" in response_data.lower() or \
               "wait" in response_data.lower()

    async def test_error_categorization(self, mock_request):
        """Test that errors are properly categorized."""
        client_errors = [
            ValidationError("Bad input"),
            AuthenticationError("No auth"),
            HTTPException(404, "Not found")
        ]
        
        server_errors = [
            ServiceUnavailableError("DB down"),
            Exception("Unexpected error")
        ]
        
        for exc in client_errors:
            response = await ErrorHandler.handle_exception(exc, mock_request)
            assert 400 <= response.status_code < 500
        
        for exc in server_errors:
            response = await ErrorHandler.handle_exception(exc, mock_request)
            assert response.status_code >= 500
