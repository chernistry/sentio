"""Distributed tracing for RAG pipeline monitoring.

This module provides OpenTelemetry-based distributed tracing
for tracking requests across the entire RAG pipeline.
"""

import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace.status import Status, StatusCode
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class MockTracer:
    """Mock tracer for when OpenTelemetry is not available."""

    def start_span(self, name: str, **kwargs):
        return MockSpan(name)

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs):
        yield MockSpan(name)


class MockSpan:
    """Mock span for when OpenTelemetry is not available."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()

    def set_attribute(self, key: str, value: Any):
        pass

    def set_status(self, status):
        pass

    def record_exception(self, exception):
        pass

    def end(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TracingManager:
    """Manages distributed tracing configuration and setup.
    
    Provides centralized tracing configuration with support for
    multiple exporters (Jaeger, OTLP, Console).
    """

    def __init__(self):
        self.tracer = None
        self.enabled = HAS_OPENTELEMETRY
        self._configured = False

        if not self.enabled:
            logger.warning("OpenTelemetry not available, using mock tracer")
            self.tracer = MockTracer()

    def setup_tracing(
        self,
        service_name: str = "sentio-rag",
        service_version: str = "3.0.0",
        jaeger_endpoint: str | None = None,
        otlp_endpoint: str | None = None,
        console_export: bool = False,
        sample_rate: float = 1.0,
    ):
        """Configure distributed tracing.
        
        Args:
            service_name: Name of the service for tracing
            service_version: Version of the service
            jaeger_endpoint: Jaeger collector endpoint
            otlp_endpoint: OTLP collector endpoint
            console_export: Whether to export to console
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        if not self.enabled:
            logger.info("Tracing setup skipped (OpenTelemetry not available)")
            return

        if self._configured:
            logger.info("Tracing already configured")
            return

        # Create resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": service_version,
        })

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Set up exporters
        exporters = []

        if jaeger_endpoint:
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=jaeger_endpoint.split(":")[0],
                    agent_port=int(jaeger_endpoint.split(":")[1]) if ":" in jaeger_endpoint else 6831,
                )
                exporters.append(jaeger_exporter)
                logger.info(f"Jaeger exporter configured: {jaeger_endpoint}")
            except Exception as e:
                logger.error(f"Failed to configure Jaeger exporter: {e}")

        if otlp_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                exporters.append(otlp_exporter)
                logger.info(f"OTLP exporter configured: {otlp_endpoint}")
            except Exception as e:
                logger.error(f"Failed to configure OTLP exporter: {e}")

        if console_export:
            console_exporter = ConsoleSpanExporter()
            exporters.append(console_exporter)
            logger.info("Console exporter configured")

        # Add span processors
        for exporter in exporters:
            span_processor = BatchSpanProcessor(exporter)
            tracer_provider.add_span_processor(span_processor)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        self._configured = True

        logger.info("Distributed tracing configured successfully")

    def instrument_fastapi(self, app):
        """Instrument FastAPI application for automatic tracing."""
        if self.enabled and self._configured:
            try:
                FastAPIInstrumentor.instrument_app(app)
                logger.info("FastAPI instrumentation enabled")
            except Exception as e:
                logger.error(f"Failed to instrument FastAPI: {e}")

    def instrument_http_clients(self):
        """Instrument HTTP clients for automatic tracing."""
        if self.enabled and self._configured:
            try:
                HTTPXClientInstrumentor().instrument()
                RequestsInstrumentor().instrument()
                logger.info("HTTP client instrumentation enabled")
            except Exception as e:
                logger.error(f"Failed to instrument HTTP clients: {e}")

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: dict[str, Any] | None = None,
        record_exception: bool = True,
    ):
        """Context manager for tracing operations.
        
        Args:
            operation_name: Name of the operation being traced
            attributes: Additional attributes to add to the span
            record_exception: Whether to record exceptions in the span
        """
        if not self.tracer:
            yield MockSpan(operation_name)
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                if record_exception:
                    span.record_exception(e)
                raise

    def trace_function(
        self,
        operation_name: str | None = None,
        attributes: dict[str, Any] | None = None,
        record_args: bool = False,
        record_result: bool = False,
    ):
        """Decorator for tracing function calls.
        
        Args:
            operation_name: Custom operation name (defaults to function name)
            attributes: Additional attributes to add to the span
            record_args: Whether to record function arguments
            record_result: Whether to record function result
        """
        def decorator(func: F) -> F:
            span_name = operation_name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.trace_operation(span_name, attributes) as span:
                    if record_args:
                        span.set_attribute("function.args", str(args))
                        span.set_attribute("function.kwargs", str(kwargs))

                    result = await func(*args, **kwargs)

                    if record_result:
                        span.set_attribute("function.result", str(result))

                    return result

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.trace_operation(span_name, attributes) as span:
                    if record_args:
                        span.set_attribute("function.args", str(args))
                        span.set_attribute("function.kwargs", str(kwargs))

                    result = func(*args, **kwargs)

                    if record_result:
                        span.set_attribute("function.result", str(result))

                    return result

            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator


# Global tracing manager
tracing_manager = TracingManager()


def setup_tracing(
    service_name: str = "sentio-rag",
    service_version: str = "3.0.0",
    jaeger_endpoint: str | None = None,
    otlp_endpoint: str | None = None,
    console_export: bool = False,
):
    """Set up distributed tracing.
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        jaeger_endpoint: Jaeger endpoint (e.g., "localhost:6831")
        otlp_endpoint: OTLP endpoint (e.g., "http://localhost:4317")
        console_export: Export traces to console
    """
    tracing_manager.setup_tracing(
        service_name=service_name,
        service_version=service_version,
        jaeger_endpoint=jaeger_endpoint,
        otlp_endpoint=otlp_endpoint,
        console_export=console_export,
    )


def trace_function(
    operation_name: str | None = None,
    attributes: dict[str, Any] | None = None,
    record_args: bool = False,
    record_result: bool = False,
):
    """Decorator for tracing function calls.
    
    Args:
        operation_name: Custom operation name
        attributes: Additional span attributes
        record_args: Record function arguments
        record_result: Record function result
    """
    return tracing_manager.trace_function(
        operation_name=operation_name,
        attributes=attributes,
        record_args=record_args,
        record_result=record_result,
    )


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
):
    """Context manager for tracing operations.
    
    Args:
        operation_name: Name of the operation
        attributes: Additional span attributes
        record_exception: Record exceptions in span
    """
    with tracing_manager.trace_operation(
        operation_name=operation_name,
        attributes=attributes,
        record_exception=record_exception,
    ) as span:
        yield span


def instrument_fastapi(app):
    """Instrument FastAPI application."""
    tracing_manager.instrument_fastapi(app)


def instrument_http_clients():
    """Instrument HTTP clients."""
    tracing_manager.instrument_http_clients()
