"""Comprehensive metrics collection for production monitoring.

This module provides Prometheus-compatible metrics for all system components
including request processing, embedding generation, retrieval operations,
and LLM interactions.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        generate_latest,
        multiprocess,
        start_http_server,
        values,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """Container for metric data points."""
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class InMemoryMetrics:
    """Fallback metrics collection when Prometheus is not available."""

    def __init__(self):
        self.counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.histograms: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        self.gauges: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._lock = threading.Lock()

    def increment_counter(self, name: str, labels: dict[str, str], amount: float = 1.0):
        """Increment a counter metric."""
        with self._lock:
            label_key = self._labels_to_key(labels)
            self.counters[name][label_key] += amount

    def observe_histogram(self, name: str, labels: dict[str, str], value: float):
        """Record a histogram observation."""
        with self._lock:
            label_key = self._labels_to_key(labels)
            self.histograms[name][label_key].append(value)

    def set_gauge(self, name: str, labels: dict[str, str], value: float):
        """Set a gauge value."""
        with self._lock:
            label_key = self._labels_to_key(labels)
            self.gauges[name][label_key] = value

    def _labels_to_key(self, labels: dict[str, str]) -> str:
        """Convert labels dict to string key."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "histograms": {
                    name: {
                        key: {
                            "count": len(values),
                            "sum": sum(values),
                            "avg": sum(values) / len(values) if values else 0,
                            "min": min(values) if values else 0,
                            "max": max(values) if values else 0,
                        }
                        for key, values in hist.items()
                    }
                    for name, hist in self.histograms.items()
                },
                "gauges": dict(self.gauges),
            }


class MetricsCollector:
    """Central metrics collector with Prometheus integration.
    
    Collects metrics for all system components with automatic
    fallback to in-memory collection when Prometheus is unavailable.
    """

    def __init__(self, enable_prometheus: bool = True, registry: CollectorRegistry | None = None):
        self.enable_prometheus = enable_prometheus and HAS_PROMETHEUS
        self.registry = registry
        self.fallback_metrics = InMemoryMetrics()

        if self.enable_prometheus:
            self._setup_prometheus_metrics()

        logger.info(f"Metrics collector initialized (Prometheus: {self.enable_prometheus})")

    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        registry = self.registry

        # Request metrics
        self.request_count = Counter(
            "rag_requests_total",
            "Total number of requests",
            ["endpoint", "method", "status"],
            registry=registry
        )

        self.request_duration = Histogram(
            "rag_request_duration_seconds",
            "Request processing duration",
            ["endpoint", "method"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0),
            registry=registry
        )

        self.request_size = Histogram(
            "rag_request_size_bytes",
            "Request size in bytes",
            ["endpoint"],
            buckets=(100, 1000, 10000, 100000, 1000000),
            registry=registry
        )

        # Embedding metrics
        self.embedding_requests = Counter(
            "embedding_requests_total",
            "Total embedding requests",
            ["provider", "model", "status"],
            registry=registry
        )

        self.embedding_duration = Histogram(
            "embedding_duration_seconds",
            "Embedding generation duration",
            ["provider", "model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=registry
        )

        self.embedding_tokens = Histogram(
            "embedding_tokens_processed",
            "Number of tokens processed for embedding",
            ["provider", "model"],
            buckets=(10, 50, 100, 500, 1000, 5000, 10000),
            registry=registry
        )

        # Retrieval metrics
        self.retrieval_requests = Counter(
            "retrieval_requests_total",
            "Total retrieval requests",
            ["strategy", "status"],
            registry=registry
        )

        self.retrieval_duration = Histogram(
            "retrieval_duration_seconds",
            "Document retrieval duration",
            ["strategy"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=registry
        )

        self.retrieval_documents = Histogram(
            "retrieval_documents_returned",
            "Number of documents returned by retrieval",
            ["strategy"],
            buckets=(1, 5, 10, 20, 50, 100),
            registry=registry
        )

        # Vector store metrics
        self.vector_store_operations = Counter(
            "vector_store_operations_total",
            "Total vector store operations",
            ["operation", "status"],
            registry=registry
        )

        self.vector_store_duration = Histogram(
            "vector_store_duration_seconds",
            "Vector store operation duration",
            ["operation"],
            buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0),
            registry=registry
        )

        # LLM metrics
        self.llm_requests = Counter(
            "llm_requests_total",
            "Total LLM requests",
            ["provider", "model", "status"],
            registry=registry
        )

        self.llm_duration = Histogram(
            "llm_duration_seconds",
            "LLM request duration",
            ["provider", "model"],
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0),
            registry=registry
        )

        self.llm_tokens = Counter(
            "llm_tokens_total",
            "Total LLM tokens used",
            ["provider", "model", "type"],  # types: prompt, completion
            registry=registry
        )

        # System metrics
        self.active_connections = Gauge(
            "active_connections",
            "Number of active connections",
            registry=registry
        )

        self.memory_usage = Gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],  # rss, vms, shared
            registry=registry
        )

        self.cpu_usage = Gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
            registry=registry
        )

        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            "circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half-open)",
            ["service"],
            registry=registry
        )

        self.circuit_breaker_failures = Counter(
            "circuit_breaker_failures_total",
            "Total circuit breaker failures",
            ["service"],
            registry=registry
        )

    @contextmanager
    def track_request_metrics(self, endpoint: str, method: str = "POST"):
        """Context manager for tracking request metrics."""
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time

            if self.enable_prometheus:
                self.request_count.labels(
                    endpoint=endpoint,
                    method=method,
                    status=status
                ).inc()
                self.request_duration.labels(
                    endpoint=endpoint,
                    method=method
                ).observe(duration)
            else:
                self.fallback_metrics.increment_counter(
                    "request_count",
                    {"endpoint": endpoint, "method": method, "status": status}
                )
                self.fallback_metrics.observe_histogram(
                    "request_duration",
                    {"endpoint": endpoint, "method": method},
                    duration
                )

    @contextmanager
    def track_embedding_metrics(self, provider: str, model: str):
        """Context manager for tracking embedding metrics."""
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time

            if self.enable_prometheus:
                self.embedding_requests.labels(
                    provider=provider,
                    model=model,
                    status=status
                ).inc()
                self.embedding_duration.labels(
                    provider=provider,
                    model=model
                ).observe(duration)
            else:
                self.fallback_metrics.increment_counter(
                    "embedding_requests",
                    {"provider": provider, "model": model, "status": status}
                )
                self.fallback_metrics.observe_histogram(
                    "embedding_duration",
                    {"provider": provider, "model": model},
                    duration
                )

    @contextmanager
    def track_retrieval_metrics(self, strategy: str):
        """Context manager for tracking retrieval metrics."""
        start_time = time.time()
        status = "success"
        doc_count = 0

        try:
            result = yield
            if hasattr(result, "__len__"):
                doc_count = len(result)
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time

            if self.enable_prometheus:
                self.retrieval_requests.labels(
                    strategy=strategy,
                    status=status
                ).inc()
                self.retrieval_duration.labels(strategy=strategy).observe(duration)
                if doc_count > 0:
                    self.retrieval_documents.labels(strategy=strategy).observe(doc_count)
            else:
                self.fallback_metrics.increment_counter(
                    "retrieval_requests",
                    {"strategy": strategy, "status": status}
                )
                self.fallback_metrics.observe_histogram(
                    "retrieval_duration",
                    {"strategy": strategy},
                    duration
                )

    @contextmanager
    def track_llm_metrics(self, provider: str, model: str):
        """Context manager for tracking LLM metrics."""
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time

            if self.enable_prometheus:
                self.llm_requests.labels(
                    provider=provider,
                    model=model,
                    status=status
                ).inc()
                self.llm_duration.labels(
                    provider=provider,
                    model=model
                ).observe(duration)
            else:
                self.fallback_metrics.increment_counter(
                    "llm_requests",
                    {"provider": provider, "model": model, "status": status}
                )
                self.fallback_metrics.observe_histogram(
                    "llm_duration",
                    {"provider": provider, "model": model},
                    duration
                )

    def record_llm_tokens(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int):
        """Record LLM token usage."""
        if self.enable_prometheus:
            self.llm_tokens.labels(
                provider=provider,
                model=model,
                type="prompt"
            ).inc(prompt_tokens)
            self.llm_tokens.labels(
                provider=provider,
                model=model,
                type="completion"
            ).inc(completion_tokens)
        else:
            self.fallback_metrics.increment_counter(
                "llm_tokens",
                {"provider": provider, "model": model, "type": "prompt"},
                prompt_tokens
            )
            self.fallback_metrics.increment_counter(
                "llm_tokens",
                {"provider": provider, "model": model, "type": "completion"},
                completion_tokens
            )

    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            if self.enable_prometheus:
                self.memory_usage.labels(type="rss").set(memory_info.rss)
                self.memory_usage.labels(type="vms").set(memory_info.vms)
                self.cpu_usage.set(process.cpu_percent())
            else:
                self.fallback_metrics.set_gauge(
                    "memory_usage", {"type": "rss"}, memory_info.rss
                )
                self.fallback_metrics.set_gauge(
                    "memory_usage", {"type": "vms"}, memory_info.vms
                )
                self.fallback_metrics.set_gauge(
                    "cpu_usage", {}, process.cpu_percent()
                )

        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def update_circuit_breaker_metrics(self, service: str, state: int, failure_count: int = 0):
        """Update circuit breaker metrics."""
        if self.enable_prometheus:
            self.circuit_breaker_state.labels(service=service).set(state)
            if failure_count > 0:
                self.circuit_breaker_failures.labels(service=service).inc(failure_count)
        else:
            self.fallback_metrics.set_gauge(
                "circuit_breaker_state", {"service": service}, state
            )
            if failure_count > 0:
                self.fallback_metrics.increment_counter(
                    "circuit_breaker_failures", {"service": service}, failure_count
                )

    def get_metrics_export(self) -> str:
        """Get metrics in Prometheus format."""
        if self.enable_prometheus:
            if self.registry:
                return generate_latest(self.registry)
            return generate_latest()
        # Return JSON format for fallback metrics
        import json
        return json.dumps(self.fallback_metrics.get_metrics_summary(), indent=2)

    def start_metrics_server(self, port: int = 8080):
        """Start Prometheus metrics server."""
        if self.enable_prometheus:
            start_http_server(port, registry=self.registry)
            logger.info(f"Metrics server started on port {port}")
        else:
            logger.warning("Prometheus not available, metrics server not started")


# Global metrics collector instance
metrics_collector = MetricsCollector()

# Convenience functions
def track_request_metrics(endpoint: str, method: str = "POST"):
    """Decorator/context manager for request metrics."""
    return metrics_collector.track_request_metrics(endpoint, method)

def track_embedding_metrics(provider: str, model: str):
    """Decorator/context manager for embedding metrics."""
    return metrics_collector.track_embedding_metrics(provider, model)

def track_retrieval_metrics(strategy: str):
    """Decorator/context manager for retrieval metrics."""
    return metrics_collector.track_retrieval_metrics(strategy)

def track_llm_metrics(provider: str, model: str):
    """Decorator/context manager for LLM metrics."""
    return metrics_collector.track_llm_metrics(provider, model)
