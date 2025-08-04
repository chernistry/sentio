"""Observability package for production monitoring.

This package provides comprehensive monitoring, metrics collection,
distributed tracing, and performance analysis capabilities.
"""

from .metrics import (
    MetricsCollector,
    track_embedding_metrics,
    track_llm_metrics,
    track_request_metrics,
    track_retrieval_metrics,
)
from .monitoring import PerformanceMonitor, ResourceMonitor
from .tracing import (
    instrument_fastapi,
    instrument_http_clients,
    setup_tracing,
    trace_function,
)

# Global instances for monitoring
performance_monitor = PerformanceMonitor()
resource_monitor = ResourceMonitor()

__all__ = [
    "MetricsCollector",
    "PerformanceMonitor",
    "ResourceMonitor",
    "instrument_fastapi",
    "instrument_http_clients",
    "performance_monitor",
    "resource_monitor",
    "setup_tracing",
    "trace_function",
    "track_embedding_metrics",
    "track_llm_metrics",
    "track_request_metrics",
    "track_retrieval_metrics",
]
