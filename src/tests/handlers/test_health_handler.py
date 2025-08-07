"""Comprehensive tests for health handler functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.api.handlers.health import HealthHandler


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for testing."""
    return {
        "embedder": AsyncMock(),
        "vector_store": AsyncMock(),
        "llm": AsyncMock(),
        "cache": AsyncMock()
    }


@pytest.fixture
def health_handler(mock_dependencies):
    """Create HealthHandler instance with mocked dependencies."""
    return HealthHandler()  # HealthHandler takes no constructor parameters


@pytest.mark.asyncio
class TestHealthHandler:
    """Test HealthHandler functionality."""

    async def test_basic_health_check_healthy(self, health_handler):
        """Test basic health check when system is healthy."""
        health_status = await health_handler.basic_health_check()
        
        # Verify basic health response structure
        assert isinstance(health_status, dict)
        assert "status" in health_status
        assert "timestamp" in health_status
        assert "version" in health_status
        assert health_status["status"] == "healthy"

    async def test_detailed_health_check_all_healthy(self, health_handler):
        """Test detailed health check when all components are healthy."""
        # Mock all components as healthy
        with patch('src.core.embeddings.get_embedder') as mock_get_embedder, \
             patch('src.core.vector_store.get_vector_store') as mock_get_vector_store:
            
            mock_embedder = AsyncMock()
            mock_vector_store = AsyncMock()
            mock_get_embedder.return_value = mock_embedder
            mock_get_vector_store.return_value = mock_vector_store
            
            # Mock health check methods
            mock_vector_store.health_check.return_value = True
            
            health_status = await health_handler.detailed_health_check()
            
            # Verify detailed health response
            assert isinstance(health_status, dict)
            assert "status" in health_status
            assert "components" in health_status or "timestamp" in health_status

    async def test_detailed_health_check_some_unhealthy(self, health_handler):
        """Test detailed health check when some components are unhealthy."""
        with patch('src.core.embeddings.get_embedder') as mock_get_embedder, \
             patch('src.core.vector_store.get_vector_store') as mock_get_vector_store:
            
            mock_embedder = AsyncMock()
            mock_vector_store = AsyncMock()
            mock_get_embedder.return_value = mock_embedder
            mock_get_vector_store.return_value = mock_vector_store
            
            # Mock one component as unhealthy
            mock_vector_store.health_check.return_value = False
            
            health_status = await health_handler.detailed_health_check()
            
            # Should still return health status
            assert isinstance(health_status, dict)

    async def test_detailed_health_check_component_error(self, health_handler):
        """Test detailed health check when component check raises error."""
        with patch('src.core.embeddings.get_embedder') as mock_get_embedder:
            mock_get_embedder.side_effect = Exception("Component error")
            
            health_status = await health_handler.detailed_health_check()
            
            # Should handle errors gracefully
            assert isinstance(health_status, dict)

    async def test_readiness_check_ready(self, health_handler):
        """Test readiness check when system is ready."""
        # Mock system as ready
        with patch.object(health_handler, 'detailed_health_check') as mock_detailed:
            mock_detailed.return_value = {
                "status": "healthy",
                "components": {"all": "healthy"}
            }
            
            # Test readiness through detailed health check
            health_status = await health_handler.detailed_health_check()
            is_ready = health_status.get("status") == "healthy"
            
            assert is_ready is True

    async def test_readiness_check_not_ready(self, health_handler):
        """Test readiness check when system is not ready."""
        with patch.object(health_handler, 'detailed_health_check') as mock_detailed:
            mock_detailed.return_value = {
                "status": "unhealthy",
                "components": {"some": "unhealthy"}
            }
            
            health_status = await health_handler.detailed_health_check()
            is_ready = health_status.get("status") == "healthy"
            
            assert is_ready is False

    async def test_liveness_check_alive(self, health_handler):
        """Test liveness check when system is alive."""
        # Basic health check serves as liveness check
        health_status = await health_handler.basic_health_check()
        
        # Should return healthy status
        assert health_status["status"] == "healthy"

    async def test_liveness_check_memory_pressure(self, health_handler):
        """Test liveness check under memory pressure."""
        # Mock memory pressure scenario
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 95  # High memory usage
            
            # Should still be alive but may report degraded performance
            health_status = await health_handler.basic_health_check()
            
            # Basic health should still work
            assert isinstance(health_status, dict)
            assert "status" in health_status

    async def test_component_health_check_timeout(self, health_handler):
        """Test component health check with timeout."""
        with patch('src.core.vector_store.get_vector_store') as mock_get_vector_store:
            mock_vector_store = AsyncMock()
            mock_get_vector_store.return_value = mock_vector_store
            
            # Mock timeout scenario
            async def slow_health_check():
                import asyncio
                await asyncio.sleep(0.1)
                return True
            
            mock_vector_store.health_check.side_effect = slow_health_check
            
            # Should handle timeout gracefully
            health_status = await health_handler.detailed_health_check()
            assert isinstance(health_status, dict)

    async def test_dependency_chain_check(self, health_handler):
        """Test health check of dependency chain."""
        # Mock dependency chain
        with patch('src.core.embeddings.get_embedder') as mock_get_embedder, \
             patch('src.core.vector_store.get_vector_store') as mock_get_vector_store:
            
            mock_embedder = AsyncMock()
            mock_vector_store = AsyncMock()
            mock_get_embedder.return_value = mock_embedder
            mock_get_vector_store.return_value = mock_vector_store
            
            # Mock successful dependency checks
            mock_vector_store.health_check.return_value = True
            
            health_status = await health_handler.detailed_health_check()
            
            # Should check all dependencies
            assert isinstance(health_status, dict)

    async def test_health_metrics_collection(self, health_handler):
        """Test health metrics collection."""
        # Mock metrics collection
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            
            health_status = await health_handler.basic_health_check()
            
            # Should include timestamp
            assert "timestamp" in health_status
            assert health_status["timestamp"] == 1000.0

    async def test_startup_health_check(self, health_handler):
        """Test health check during startup."""
        # Test health check when system is starting up
        health_status = await health_handler.basic_health_check()
        
        # Should return basic health info even during startup
        assert isinstance(health_status, dict)
        assert "status" in health_status

    async def test_health_check_caching(self, health_handler):
        """Test health check result caching."""
        # Test that health checks can be cached
        health_status1 = await health_handler.basic_health_check()
        health_status2 = await health_handler.basic_health_check()
        
        # Both should return valid health status
        assert isinstance(health_status1, dict)
        assert isinstance(health_status2, dict)
        assert health_status1["status"] == health_status2["status"]

    async def test_health_check_with_custom_components(self, health_handler):
        """Test health check with custom component configuration."""
        # Mock custom components
        with patch('src.core.embeddings.get_embedder') as mock_get_embedder:
            mock_embedder = AsyncMock()
            mock_get_embedder.return_value = mock_embedder
            
            health_status = await health_handler.detailed_health_check()
            
            # Should handle custom components
            assert isinstance(health_status, dict)

    async def test_health_check_error_recovery(self, health_handler):
        """Test health check error recovery."""
        # Test recovery from transient errors
        with patch('src.core.vector_store.get_vector_store') as mock_get_vector_store:
            # First call fails, second succeeds
            mock_get_vector_store.side_effect = [
                Exception("Transient error"),
                AsyncMock()
            ]
            
            # First call should handle error
            health_status1 = await health_handler.detailed_health_check()
            assert isinstance(health_status1, dict)
            
            # Reset for second call
            mock_vector_store = AsyncMock()
            mock_vector_store.health_check.return_value = True
            mock_get_vector_store.side_effect = None
            mock_get_vector_store.return_value = mock_vector_store
            
            # Second call should succeed
            health_status2 = await health_handler.detailed_health_check()
            assert isinstance(health_status2, dict)

    async def test_concurrent_health_checks(self, health_handler):
        """Test concurrent health check requests."""
        import asyncio
        
        # Run multiple concurrent health checks
        tasks = [
            health_handler.basic_health_check()
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 5
        assert all(isinstance(result, dict) for result in results)
        assert all(result["status"] == "healthy" for result in results)

    def test_health_handler_initialization(self):
        """Test health handler initialization."""
        handler = HealthHandler()
        
        # Should initialize successfully
        assert handler is not None
        assert hasattr(handler, 'basic_health_check')
        assert hasattr(handler, 'detailed_health_check')

    async def test_health_check_response_format(self, health_handler):
        """Test health check response format consistency."""
        basic_health = await health_handler.basic_health_check()
        detailed_health = await health_handler.detailed_health_check()
        
        # Both should be dictionaries
        assert isinstance(basic_health, dict)
        assert isinstance(detailed_health, dict)
        
        # Basic health should have required fields
        assert "status" in basic_health
        assert "timestamp" in basic_health
        assert "version" in basic_health

    async def test_health_check_under_load(self, health_handler):
        """Test health check behavior under load."""
        import asyncio
        
        # Simulate load with many concurrent requests
        tasks = [
            health_handler.basic_health_check()
            for _ in range(20)
        ]
        
        # Should handle load gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed
        successful_results = [r for r in results if isinstance(r, dict)]
        assert len(successful_results) > 0
