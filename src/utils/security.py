"""Security utilities for production deployment.

This module provides security hardening features including:
- API key sanitization for logs
- Input validation
- Security headers
- Rate limiting helpers
"""

import hashlib
import html
import ipaddress
import logging
import re
import secrets
import urllib.parse
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class LogSanitizer:
    """Sanitizes sensitive information from logs and debug output.
    
    Removes API keys, tokens, passwords, and other sensitive data
    from dictionaries, strings, and log messages.
    """

    # Patterns for sensitive data detection
    SENSITIVE_PATTERNS = [
        # API Keys
        r'(?i)(api[_-]?key|token|secret|password)["\']?\s*[:=]\s*["\']?([^\s"\']+)',
        # Authorization headers
        r"(?i)(authorization|bearer)\s+([^\s]+)",
        # OpenAI API keys
        r"sk-[a-zA-Z0-9]{20,}",
        # Jina API keys
        r"jina_[a-zA-Z0-9]{32,}",
        # Generic keys/tokens
        r"[a-zA-Z0-9]{32,}",
    ]

    SENSITIVE_KEYS = {
        "api_key", "token", "secret", "password", "key", "authorization",
        "bearer", "credential", "auth", "openai_api_key", "jina_api_key",
        "embedding_model_api_key", "chat_llm_api_key", "qdrant_api_key"
    }

    @classmethod
    def sanitize_string(cls, text: str) -> str:
        """Sanitize sensitive information from a string.
        
        Args:
            text: String potentially containing sensitive data
            
        Returns:
            String with sensitive data replaced
        """
        if not isinstance(text, str):
            return str(text)

        sanitized = text
        for pattern in cls.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, r"\1=***REDACTED***", sanitized)

        return sanitized

    @classmethod
    def sanitize_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively sanitize dictionary values.
        
        Args:
            data: Dictionary potentially containing sensitive data
            
        Returns:
            Dictionary with sensitive data replaced
        """
        if not isinstance(data, dict):
            return data

        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()

            # Check if key indicates sensitive data
            if any(sensitive_key in key_lower for sensitive_key in cls.SENSITIVE_KEYS):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    cls.sanitize_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, str) and len(value) > 20:
                # Check if value looks like an API key
                if any(pattern in value for pattern in ["sk-", "jina_", "Bearer"]):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = cls.sanitize_string(value)
            else:
                sanitized[key] = value

        return sanitized

    @classmethod
    def sanitize_log_record(cls, record: logging.LogRecord) -> logging.LogRecord:
        """Sanitize a log record before output.
        
        Args:
            record: Log record to sanitize
            
        Returns:
            Sanitized log record
        """
        if hasattr(record, "getMessage"):
            # Sanitize the message
            original_msg = record.getMessage()
            record.msg = cls.sanitize_string(original_msg)
            record.args = ()  # Clear args since we've already formatted

        return record


class InputValidator:
    """Validates and sanitizes user inputs.
    
    Prevents injection attacks and validates input formats.
    """

    # Maximum lengths for various inputs
    MAX_QUERY_LENGTH = 2000
    MAX_DOCUMENT_LENGTH = 50000
    MAX_METADATA_SIZE = 1000

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
        r"(--|\#|\/\*|\*\/)",
        r"(\bor\b\s*\d+\s*=\s*\d+)",
        r"(\band\b\s*\d+\s*=\s*\d+)",
        r"(;\s*(drop|delete|truncate|update)\s)",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`\$\(\)<>]",
        r"\b(nc|netcat|wget|curl|bash|sh|cmd|powershell)\b",
    ]

    @classmethod
    def validate_query(cls, query: str) -> str:
        """Validate and sanitize a user query.
        
        Args:
            query: User query string
            
        Returns:
            Sanitized query string
            
        Raises:
            ValueError: If query is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()

        if len(query) > cls.MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long (max {cls.MAX_QUERY_LENGTH} chars)")

        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValueError("Query contains potentially dangerous SQL patterns")

        # Check for XSS patterns
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValueError("Query contains potentially dangerous XSS patterns")

        # HTML escape the query
        query = html.escape(query)

        # URL encode special characters
        query = urllib.parse.quote(query, safe=" ")
        query = urllib.parse.unquote(query)

        return query

    @classmethod
    def validate_document_content(cls, content: str) -> str:
        """Validate document content for ingestion.
        
        Args:
            content: Document content
            
        Returns:
            Validated content
            
        Raises:
            ValueError: If content is invalid
        """
        if not content or not content.strip():
            raise ValueError("Document content cannot be empty")

        if len(content) > cls.MAX_DOCUMENT_LENGTH:
            raise ValueError(f"Document too long (max {cls.MAX_DOCUMENT_LENGTH} chars)")

        return content.strip()

    @classmethod
    def validate_metadata(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        """Validate metadata dictionary.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Validated metadata
            
        Raises:
            ValueError: If metadata is invalid
        """
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        # Check total size
        import json
        metadata_size = len(json.dumps(metadata))
        if metadata_size > cls.MAX_METADATA_SIZE:
            raise ValueError(f"Metadata too large (max {cls.MAX_METADATA_SIZE} chars)")

        # Sanitize values
        sanitized = {}
        for key, value in metadata.items():
            # Validate key
            if not isinstance(key, str) or len(key) > 100:
                continue

            # Sanitize value
            if isinstance(value, str):
                value = re.sub(r"[<>{}]", "", value)[:500]
            elif isinstance(value, (int, float, bool)):
                pass  # Keep as-is
            else:
                value = str(value)[:500]

            sanitized[key] = value

        return sanitized


class SecurityHeaders:
    """Security headers for HTTP responses.
    
    Provides standard security headers for API responses.
    """

    @staticmethod
    def get_security_headers() -> dict[str, str]:
        """Get standard security headers.
        
        Returns:
            Dictionary of security headers
        """
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "X-API-Version": "3.0.0",
        }


class TokenGenerator:
    """Secure token generation for API keys and session tokens.
    """

    @staticmethod
    def generate_api_token(length: int = 32) -> str:
        """Generate a secure API token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Hex-encoded secure token
        """
        return secrets.token_hex(length)

    @staticmethod
    def generate_session_id() -> str:
        """Generate a secure session ID.
        
        Returns:
            Session ID string
        """
        return secrets.token_urlsafe(32)

    @staticmethod
    def hash_token(token: str, salt: str | None = None) -> str:
        """Hash a token for secure storage.
        
        Args:
            token: Token to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Hashed token
        """
        if salt is None:
            salt = secrets.token_hex(16)

        return hashlib.pbkdf2_hmac("sha256", token.encode(), salt.encode(), 100000).hex()


class IPValidator:
    """IP address validation and blocking utilities.
    """

    # Common malicious IP ranges and known bad actors
    BLOCKED_IP_RANGES = [
        "127.0.0.1/32",  # Localhost (if not expected)
        "169.254.0.0/16",  # Link-local
        "224.0.0.0/4",  # Multicast
    ]

    # Rate limiting by IP
    RATE_LIMIT_BY_IP = {}

    @classmethod
    def is_valid_ip(cls, ip_address: str) -> bool:
        """Check if IP address is valid and not blocked.
        
        Args:
            ip_address: IP address to validate
            
        Returns:
            True if IP is valid and allowed
        """
        try:
            ip = ipaddress.ip_address(ip_address)

            # Check against blocked ranges
            for blocked_range in cls.BLOCKED_IP_RANGES:
                if ip in ipaddress.ip_network(blocked_range):
                    return False

            return True

        except ValueError:
            return False

    @classmethod
    def is_rate_limited(cls, ip_address: str, max_requests: int = 100, window_minutes: int = 5) -> bool:
        """Check if IP is rate limited.
        
        Args:
            ip_address: IP address to check
            max_requests: Maximum requests allowed
            window_minutes: Time window in minutes
            
        Returns:
            True if rate limited
        """
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)

        # Clean old entries
        if ip_address in cls.RATE_LIMIT_BY_IP:
            cls.RATE_LIMIT_BY_IP[ip_address] = [
                timestamp for timestamp in cls.RATE_LIMIT_BY_IP[ip_address]
                if timestamp > window_start
            ]
        else:
            cls.RATE_LIMIT_BY_IP[ip_address] = []

        # Check if over limit
        if len(cls.RATE_LIMIT_BY_IP[ip_address]) >= max_requests:
            return True

        # Add current request
        cls.RATE_LIMIT_BY_IP[ip_address].append(now)
        return False


class CSRFProtection:
    """Cross-Site Request Forgery protection utilities.
    """

    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token.
        
        Returns:
            CSRF token string
        """
        return secrets.token_urlsafe(32)

    @staticmethod
    def validate_csrf_token(token: str, expected_token: str) -> bool:
        """Validate CSRF token using constant-time comparison.
        
        Args:
            token: Provided token
            expected_token: Expected token
            
        Returns:
            True if tokens match
        """
        if not token or not expected_token:
            return False

        return secrets.compare_digest(token, expected_token)


class XSSProtection:
    """Cross-Site Scripting protection utilities.
    """

    # HTML entities to escape
    HTML_ENTITIES = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
        "/": "&#x2F;",
    }

    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """Sanitize HTML content to prevent XSS.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return text

        # Escape HTML entities
        for char, entity in cls.HTML_ENTITIES.items():
            text = text.replace(char, entity)

        return text

    @classmethod
    def validate_content_type(cls, content_type: str) -> bool:
        """Validate content type header.
        
        Args:
            content_type: Content type to validate
            
        Returns:
            True if content type is safe
        """
        safe_types = [
            "application/json",
            "text/plain",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        ]

        return any(content_type.startswith(safe_type) for safe_type in safe_types)


class RateLimitConfig:
    """Enhanced configuration for rate limiting with adaptive limits.
    """

    # Default rate limits per endpoint
    CHAT_ENDPOINT = "10/minute"
    EMBED_ENDPOINT = "50/minute"
    DEFAULT_LIMIT = "100/minute"

    # Burst limits
    CHAT_BURST = "20/minute"
    EMBED_BURST = "100/minute"

    # Adaptive limits based on user role
    ROLE_LIMITS = {
        "admin": {"chat": "100/minute", "embed": "500/minute"},
        "premium": {"chat": "50/minute", "embed": "200/minute"},
        "user": {"chat": "10/minute", "embed": "50/minute"},
        "trial": {"chat": "5/minute", "embed": "10/minute"},
    }

    @classmethod
    def get_limit_for_endpoint(cls, endpoint: str, user_role: str = "user") -> str:
        """Get rate limit for specific endpoint and user role.
        
        Args:
            endpoint: Endpoint name
            user_role: User role for adaptive limiting
            
        Returns:
            Rate limit string
        """
        # Get role-specific limits
        role_limits = cls.ROLE_LIMITS.get(user_role, cls.ROLE_LIMITS["user"])

        # Map endpoints to role limits
        endpoint_mapping = {
            "/chat": role_limits.get("chat", cls.CHAT_ENDPOINT),
            "/embed": role_limits.get("embed", cls.EMBED_ENDPOINT),
            "/health": "1000/minute",  # High limit for health checks
            "/metrics": "100/minute",
        }

        return endpoint_mapping.get(endpoint, cls.DEFAULT_LIMIT)

    @classmethod
    def get_adaptive_limit(cls, base_limit: str, load_factor: float) -> str:
        """Adjust rate limit based on system load.
        
        Args:
            base_limit: Base rate limit
            load_factor: System load factor (0.0 to 1.0)
            
        Returns:
            Adjusted rate limit
        """
        # Parse base limit
        parts = base_limit.split("/")
        if len(parts) != 2:
            return base_limit

        try:
            count = int(parts[0])
            period = parts[1]

            # Reduce limit if system is under high load
            if load_factor > 0.8:
                count = int(count * 0.5)  # Reduce by 50%
            elif load_factor > 0.6:
                count = int(count * 0.7)  # Reduce by 30%

            return f"{count}/{period}"

        except ValueError:
            return base_limit


# Configure log sanitization
class SanitizingFilter(logging.Filter):
    """Logging filter that sanitizes sensitive information."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and sanitize log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to keep the record
        """
        try:
            LogSanitizer.sanitize_log_record(record)
        except Exception:
            # If sanitization fails, don't block the log
            pass
        return True


# Apply sanitizing filter to all loggers
def setup_log_sanitization():
    """Set up log sanitization for security."""
    sanitizing_filter = SanitizingFilter()

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(sanitizing_filter)

    # Add to commonly used loggers
    for logger_name in ["src", "uvicorn", "fastapi"]:
        logger = logging.getLogger(logger_name)
        logger.addFilter(sanitizing_filter)
