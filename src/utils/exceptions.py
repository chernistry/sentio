"""Custom exception classes and error handling utilities.

This module provides a comprehensive error handling system with:
- Custom exception hierarchy
- Error logging and monitoring
- User-friendly error responses
- Security-conscious error handling
"""

import logging
import traceback
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Standard error codes for consistent error handling."""

    # Authentication & Authorization
    AUTH_INVALID_CREDENTIALS = "AUTH_INVALID_CREDENTIALS"
    AUTH_TOKEN_EXPIRED = "AUTH_TOKEN_EXPIRED"
    AUTH_TOKEN_INVALID = "AUTH_TOKEN_INVALID"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_INSUFFICIENT_PERMISSIONS"
    AUTH_ACCOUNT_LOCKED = "AUTH_ACCOUNT_LOCKED"

    # Input Validation
    VALIDATION_INVALID_INPUT = "VALIDATION_INVALID_INPUT"
    VALIDATION_MISSING_FIELD = "VALIDATION_MISSING_FIELD"
    VALIDATION_INVALID_FORMAT = "VALIDATION_INVALID_FORMAT"
    VALIDATION_VALUE_TOO_LONG = "VALIDATION_VALUE_TOO_LONG"
    VALIDATION_INVALID_CONTENT = "VALIDATION_INVALID_CONTENT"

    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    RATE_LIMIT_QUOTA_EXCEEDED = "RATE_LIMIT_QUOTA_EXCEEDED"

    # Resource Management
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"

    # External Services
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    SERVICE_AUTHENTICATION_FAILED = "SERVICE_AUTHENTICATION_FAILED"
    SERVICE_QUOTA_EXCEEDED = "SERVICE_QUOTA_EXCEEDED"

    # Processing Errors
    PROCESSING_FAILED = "PROCESSING_FAILED"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    PROCESSING_INVALID_FORMAT = "PROCESSING_INVALID_FORMAT"

    # System Errors
    SYSTEM_INTERNAL_ERROR = "SYSTEM_INTERNAL_ERROR"
    SYSTEM_CONFIGURATION_ERROR = "SYSTEM_CONFIGURATION_ERROR"
    SYSTEM_DEPENDENCY_ERROR = "SYSTEM_DEPENDENCY_ERROR"


class SentioException(Exception):
    """Base exception class for the Sentio RAG system.
    
    Provides structured error handling with error codes, user messages,
    and optional details for debugging.
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
        user_message: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.user_message = user_message or message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_code": self.error_code.value,
            "message": self.user_message,
            "status_code": self.status_code,
            "details": self.details,
        }

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        return HTTPException(
            status_code=self.status_code,
            detail=self.to_dict(),
        )


class AuthenticationError(SentioException):
    """Authentication-related errors."""

    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: ErrorCode = ErrorCode.AUTH_INVALID_CREDENTIALS,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
            user_message="Authentication failed. Please check your credentials.",
        )


class AuthorizationError(SentioException):
    """Authorization-related errors."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        error_code: ErrorCode = ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
            user_message="You don't have permission to access this resource.",
        )


class ValidationError(SentioException):
    """Input validation errors."""

    def __init__(
        self,
        message: str = "Invalid input",
        error_code: ErrorCode = ErrorCode.VALIDATION_INVALID_INPUT,
        details: dict[str, Any] | None = None,
        field: str | None = None,
    ):
        if field:
            details = details or {}
            details["field"] = field

        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
            user_message="Invalid input provided. Please check your request.",
        )


class RateLimitError(SentioException):
    """Rate limiting errors."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        error_code: ErrorCode = ErrorCode.RATE_LIMIT_EXCEEDED,
        details: dict[str, Any] | None = None,
        retry_after: int | None = None,
    ):
        if retry_after:
            details = details or {}
            details["retry_after"] = retry_after

        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
            user_message="Rate limit exceeded. Please try again later.",
        )


class ResourceError(SentioException):
    """Resource-related errors."""

    def __init__(
        self,
        message: str = "Resource error",
        error_code: ErrorCode = ErrorCode.RESOURCE_NOT_FOUND,
        status_code: int = status.HTTP_404_NOT_FOUND,
        details: dict[str, Any] | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ):
        if resource_type or resource_id:
            details = details or {}
            if resource_type:
                details["resource_type"] = resource_type
            if resource_id:
                details["resource_id"] = resource_id

        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
            user_message="The requested resource could not be found.",
        )


class ServiceError(SentioException):
    """External service errors."""

    def __init__(
        self,
        message: str = "External service error",
        error_code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE,
        details: dict[str, Any] | None = None,
        service_name: str | None = None,
    ):
        if service_name:
            details = details or {}
            details["service_name"] = service_name

        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details,
            user_message="External service is temporarily unavailable. Please try again later.",
        )


class ProcessingError(SentioException):
    """Processing-related errors."""

    def __init__(
        self,
        message: str = "Processing failed",
        error_code: ErrorCode = ErrorCode.PROCESSING_FAILED,
        details: dict[str, Any] | None = None,
        operation: str | None = None,
    ):
        if operation:
            details = details or {}
            details["operation"] = operation

        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
            user_message="Processing failed. Please try again.",
        )


class SystemError(SentioException):
    """System-level errors."""

    def __init__(
        self,
        message: str = "System error",
        error_code: ErrorCode = ErrorCode.SYSTEM_INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
        component: str | None = None,
    ):
        if component:
            details = details or {}
            details["component"] = component

        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
            user_message="A system error occurred. Please try again later.",
        )


class ErrorHandler:
    """Centralized error handling and logging.
    
    Provides methods for consistent error logging, monitoring,
    and response formatting.
    """

    @staticmethod
    def log_error(
        error: Exception,
        request: Request | None = None,
        user_id: str | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> None:
        """Log error with context information.
        
        Args:
            error: Exception that occurred
            request: FastAPI request object
            user_id: User ID if available
            additional_context: Additional context information
        """
        context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "user_id": user_id,
        }

        if request:
            context.update({
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "client_host": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            })

        if additional_context:
            context.update(additional_context)

        # Log with appropriate level
        if isinstance(error, (AuthenticationError, AuthorizationError, ValidationError)):
            logger.warning(f"Client error: {error}", extra=context)
        elif isinstance(error, (RateLimitError, ResourceError)):
            logger.info(f"Client error: {error}", extra=context)
        elif isinstance(error, ServiceError):
            logger.error(f"Service error: {error}", extra=context)
        else:
            logger.error(f"System error: {error}", extra=context, exc_info=True)

    @staticmethod
    def create_error_response(
        error: Exception,
        request: Request | None = None,
        include_traceback: bool = False,
    ) -> JSONResponse:
        """Create standardized error response.
        
        Args:
            error: Exception that occurred
            request: FastAPI request object
            include_traceback: Whether to include traceback (dev only)
            
        Returns:
            JSON response with error details
        """
        if isinstance(error, SentioException):
            response_data = error.to_dict()
            status_code = error.status_code
        elif isinstance(error, HTTPException):
            # Handle FastAPI HTTPException
            response_data = {
                "error_code": f"HTTP_{error.status_code}",
                "message": str(error.detail),
                "status_code": error.status_code,
                "details": {},
            }
            status_code = error.status_code
        else:
            # Handle unexpected errors
            response_data = {
                "error_code": ErrorCode.SYSTEM_INTERNAL_ERROR.value,
                "message": "An internal error occurred",
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "details": {},
            }
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        # Add request ID if available
        if request and hasattr(request.state, "request_id"):
            response_data["request_id"] = request.state.request_id

        # Add traceback in development
        if include_traceback and not isinstance(error, SentioException):
            response_data["traceback"] = traceback.format_exc()

        return JSONResponse(
            status_code=status_code,
            content=response_data,
        )

    @staticmethod
    async def handle_exception(
        error: Exception,
        request: Request | None = None,
        user_id: str | None = None,
    ) -> JSONResponse:
        """Handle exception with logging and response creation.
        
        Args:
            error: Exception that occurred
            request: FastAPI request object
            user_id: User ID if available
            
        Returns:
            JSON error response
        """
        # Log the error
        ErrorHandler.log_error(error, request, user_id)

        # Create response
        return ErrorHandler.create_error_response(
            error,
            request,
            include_traceback=False,  # Never include in production
        )


# Error response models for OpenAPI documentation
class ErrorResponse:
    """Standard error response schema."""

    error_code: str
    message: str
    status_code: int
    details: dict[str, Any]
    request_id: str | None = None
