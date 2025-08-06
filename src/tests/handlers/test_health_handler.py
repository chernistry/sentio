"""Tests for the health handler - system monitoring."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.api.handlers.health import HealthHandler


@pytest.fixture
def mock_dependencies():
    """Mock system dependencies."""
    return {
        "vector_store": AsyncMock(),
        "embedder": AsyncMock(),
        "llm_provider": AsyncMock(),
        "cache": AsyncMock()
    }


@pytest.fixture
def health_handler(mock_dependencies):
    """Create HealthHandler instance with mocked dependencies."""
    return HealthHandler(dependencies=mock_dependencies)


@pytest.mark.asyncio
class TestHealthHandler:
    """Test HealthHandler functionality."""

    async def test_basic_health_check_healthy(self, health_handler):
        """Test basic health check when all systems are healthy."""
        result = await health_handler.basic_health_check()

        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "version" in result
        assert isinstance(result["timestamp"], float)

    async def test_detailed_health_check_all_healthy(self, health_handler, mock_dependencies):
        """Test detailed health check when all components are healthy."""
        # Mock all dependencies as healthy
        for dep in mock_dependencies.values():
            dep.health_check.return_value = True

        result = await health_handler.detailed_health_check()

        assert result["status"] == "healthy"
        assert "checks" in result
        assert len(result["checks"]) == len(mock_dependencies)
        
        for component_name in mock_dependencies.keys():
            assert result["checks"][component_name]["healthy"] is True
            assert "response_time" in result["checks"][component_name]

    async def test_detailed_health_check_some_unhealthy(self, health_handler, mock_dependencies):
        """Test detailed health check when some components are unhealthy."""
        # Make vector_store unhealthy
        mock_dependencies["vector_store"].health_check.return_value = False
        mock_dependencies["embedder"].health_check.return_value = True
        mock_dependencies["llm_provider"].health_check.return_value = True
        mock_dependencies["cache"].health_check.return_value = True

        result = await health_handler.detailed_health_check()

        assert result["status"] == "degraded"
        assert result["checks"]["vector_store"]["healthy"] is False
        assert result["checks"]["embedder"]["healthy"] is True

    async def test_detailed_health_check_component_error(self, health_handler, mock_dependencies):
        """Test detailed health check when component check raises exception."""
        # Make vector_store raise exception
        mock_dependencies["vector_store"].health_check.side_effect = Exception("Connection failed")
        mock_dependencies["embedder"].health_check.return_value = True

        result = await health_handler.detailed_health_check()

        assert result["status"] == "degraded"
        assert result["checks"]["vector_store"]["healthy"] is False
        assert "error" in result["checks"]["vector_store"]
        assert "Connection failed" in result["checks"]["vector_store"]["error"]

    async def test_readiness_check_ready(self, health_handler, mock_dependencies):
        """Test readiness check when system is ready."""
        # Mock all critical dependencies as healthy
        for dep in mock_dependencies.values():
            dep.health_check.return_value = True

        result = await health_handler.readiness_check()

        assert result["ready"] is True
        assert result["status"] == "ready"
        assert "timestamp" in result

    async def test_readiness_check_not_ready(self, health_handler, mock_dependencies):
        """Test readiness check when critical component is down."""
        # Make vector_store (critical component) unhealthy
        mock_dependencies["vector_store"].health_check.return_value = False
        mock_dependencies["embedder"].health_check.return_value = True

        result = await health_handler.readiness_check()

        assert result["ready"] is False
        assert result["status"] == "not_ready"
        assert "failed_checks" in result

    async def test_liveness_check_alive(self, health_handler):
        """Test liveness check when application is alive."""
        result = await health_handler.liveness_check()

        assert result["alive"] is True
        assert result["status"] == "alive"
        assert "timestamp" in result

    async def test_liveness_check_memory_pressure(self, health_handler):
        """Test liveness check under memory pressure."""
        with patch('psutil.virtual_memory') as mock_memory:
            # Simulate high memory usage (95%)
            mock_memory.return_value.percent = 95

            result = await health_handler.liveness_check()

            assert result["alive"] is True  # Still alive but with warning
            assert "memory_warning" in result

    async def test_component_health_check_timeout(self, health_handler, mock_dependencies):
        """Test component health check with timeout."""
        import asyncio
        
        # Make embedder hang
        async def slow_health_check():
            await asyncio.sleep(10)  # Longer than timeout
            return True
            
        mock_dependencies["embedder"].health_check = slow_health_check

        result = await health_handler.detailed_health_check()

        # Should handle timeout gracefully
        assert result["checks"]["embedder"]["healthy"] is False
        assert "timeout" in result["checks"]["embedder"]["error"].lower()

    async def test_dependency_chain_check(self, health_handler, mock_dependencies):
        """Test checking dependency chains."""
        # Vector store depends on network, embedder depends on API
        mock_dependencies["vector_store"].health_check.return_value = True
        mock_dependencies["embedder"].health_check.return_value = False

        result = await health_handler.detailed_health_check()

        # Should identify which dependencies are affected
        assert result["status"] == "degraded"
        assert len([check for check in result["checks"].values() if not check["healthy"]]) == 1

    async def test_health_metrics_collection(self, health_handler):
        """Test that health metrics are collected."""
        with patch('src.api.handlers.health.metrics_collector') as mock_metrics:
            await health_handler.basic_health_check()

            # Verify health metrics were recorded
            mock_metrics.record_value.assert_called()

    async def test_startup_health_check(self, health_handler, mock_dependencies):
        """Test health check during application startup."""
        # Simulate startup scenario where some components are still initializing
        mock_dependencies["vector_store"].health_check.return_value = True
        mock_dependencies["embedder"].health_check.side_effect = Exception("Still initializing")

        result = await health_handler.readiness_check()

        assert result["ready"] is False
        assert "initialization" in result.get("message", "").lower() or \
               "initializing" in str(result.get("failed_checks", "")).lower()

    async def test_health_check_caching(self, health_handler):
        """Test that health check results are cached appropriately."""
        # First call
        result1 = await health_handler.basic_health_check()
        timestamp1 = result1["timestamp"]

        # Second call immediately after (should be cached)
        result2 = await health_handler.basic_health_check()
        timestamp2 = result2["timestamp"]

        # Timestamps should be very close (cached result)
        assert abs(timestamp2 - timestamp1) < 1.0  # Less than 1 second difference
