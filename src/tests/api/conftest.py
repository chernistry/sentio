"""Test configuration and fixtures for API tests.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


# Mock classes for testing
class MockLimiter:
    """Mock rate limiter that passes through without restrictions."""

    def limit(self, rate_limit_string):
        def decorator(func):
            return func
        return decorator


class MockState:
    """Mock application state for testing."""
    def __init__(self):
        self.chat_handler = AsyncMock()
        self.health_handler = AsyncMock()


async def mock_token_data():
    """Mock token data that bypasses authentication."""
    from src.utils.auth import AuthScope, TokenData
    return TokenData(
        user_id="test_user",
        username="test_user",
        scopes=[AuthScope.CHAT, AuthScope.EMBED, AuthScope.METRICS]
    )

async def mock_auth_manager():
    """Mock authentication manager."""
    mock_manager = AsyncMock()
    # Mock require_scopes to return a dependency that returns mock token data
    mock_manager.require_scopes.return_value = mock_token_data
    mock_manager.log_security_event = AsyncMock()
    return mock_manager

# Global mocks for sharing between tests
_mock_chat_handler = AsyncMock()
_mock_health_handler = AsyncMock()
_mock_ingestor = AsyncMock()

async def mock_chat_handler():
    """Mock chat handler."""
    handler = AsyncMock()
    handler.process_chat_request.return_value = {
        "answer": "This is a test answer",
        "sources": [
            {
                "text": "Test source content",
                "source": "test.pdf",
                "score": 0.9,
                "metadata": {}
            }
        ],
        "metadata": {
            "query_id": "test-123",
            "processing_time": 0.5
        }
    }
    return handler

async def mock_health_handler():
    """Mock health handler."""
    handler = AsyncMock()
    handler.basic_health_check.return_value = {
        "status": "healthy",
        "timestamp": 1234567890.0,
        "version": "3.0.0"
    }
    handler.detailed_health_check.return_value = {
        "status": "healthy",
        "timestamp": 1234567890.0,
        "version": "3.0.0",
        "components": {}
    }
    handler.readiness_check.return_value = {
        "status": "ready",
        "timestamp": 1234567890.0
    }
    handler.liveness_check.return_value = {
        "status": "alive",
        "timestamp": 1234567890.0
    }
    return handler

async def mock_ingestor():
    """Mock document ingestor."""
    ingestor = AsyncMock()
    ingestor.chunker = AsyncMock()
    ingestor._generate_embeddings.return_value = {"test-id": [0.1, 0.2, 0.3]}
    ingestor._store_chunks_with_embeddings = AsyncMock()
    return ingestor

@pytest.fixture(scope="function")
def test_client():
    """Create a test client with all dependencies mocked.
    """
    # Import app first
    from src.api.app import app
    from src.core.dependencies import (
        get_auth_manager_dep,
        get_chat_handler,
        get_health_handler,
        get_ingestor,
    )

    # Simple auth override that returns token data for any auth requirement
    def override_any_auth_dependency():
        return mock_token_data

    # Override dependencies - first patch at route level, then dependency level
    app.dependency_overrides[get_auth_manager_dep] = mock_auth_manager
    app.dependency_overrides[get_chat_handler] = mock_chat_handler
    app.dependency_overrides[get_health_handler] = mock_health_handler
    app.dependency_overrides[get_ingestor] = mock_ingestor

    # Try to override the function that gets returned by require_scopes
    app.dependency_overrides[mock_token_data] = override_any_auth_dependency

    # Mock rate limiter
    rate_limiter_patcher = patch("src.api.app.rate_limiter")
    mock_rate_limiter = rate_limiter_patcher.start()
    mock_rate_limiter.is_allowed = AsyncMock(return_value=True)

    # Mock auth_manager instance used in routes
    auth_manager_patcher = patch("src.api.app.auth_manager")
    mock_auth_manager_instance = auth_manager_patcher.start()
    mock_auth_manager_instance.require_scopes.return_value = mock_token_data
    mock_auth_manager_instance.log_security_event = AsyncMock()

    # Mock other dependencies
    patches = [rate_limiter_patcher, auth_manager_patcher]

    other_modules = [
        "src.api.app.resource_monitor",
        "src.api.app.performance_monitor",
        "src.observability.metrics.metrics_collector",
        "src.core.dependencies.check_dependency_health",
    ]

    for module in other_modules:
        patcher = patch(module)
        mock_obj = patcher.start()
        patches.append(patcher)

        if "metrics_collector" in module:
            mock_obj.get_metrics_export.return_value = "test_metric 1.0"
        elif "resource_monitor" in module or "performance_monitor" in module:
            mock_obj.get_resource_summary.return_value = {"test": "data"}
            mock_obj.check_resource_health.return_value = {"status": "healthy"}
            if hasattr(mock_obj, "get_all_metrics_summary"):
                mock_obj.get_all_metrics_summary.return_value = {"perf": "data"}
            if hasattr(mock_obj, "get_resource_trends"):
                mock_obj.get_resource_trends.return_value = {"trends": "data"}
        elif "check_dependency_health" in module:
            mock_obj.return_value = AsyncMock(return_value={"test_service": "healthy"})

    try:
        client = TestClient(app)

        # Make mocks accessible for tests
        class MockState:
            chat_handler = _mock_chat_handler
            health_handler = _mock_health_handler
            ingestor = _mock_ingestor

        client.mock_state = MockState()

        yield client

    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()

        # Clean up patches
        for patcher in patches:
            patcher.stop()
