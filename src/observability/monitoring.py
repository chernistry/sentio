"""Performance and resource monitoring for production systems.

This module provides comprehensive monitoring of system resources,
performance metrics, and operational health indicators.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Container for performance metric data."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric_name: str
    threshold_value: float
    operator: str = "gt"  # gt, lt, eq, ge, le
    duration_seconds: float = 60.0
    severity: str = "warning"  # info, warning, error, critical


class PerformanceMonitor:
    """Monitors application performance and tracks key metrics.
    
    Provides real-time performance monitoring with configurable
    alerting and historical data retention.
    """

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.alert_thresholds: list[AlertThreshold] = []
        self.alert_callbacks: list[Callable[[str, PerformanceMetric], None]] = []
        self._lock = threading.Lock()
        self._monitoring_active = False
        self._monitoring_task: asyncio.Task | None = None

    def add_alert_threshold(self, threshold: AlertThreshold):
        """Add an alert threshold configuration."""
        with self._lock:
            self.alert_thresholds.append(threshold)
        logger.info(f"Added alert threshold: {threshold.metric_name} {threshold.operator} {threshold.threshold_value}")

    def add_alert_callback(self, callback: Callable[[str, PerformanceMetric], None]):
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)

    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        with self._lock:
            self.metrics_history[metric.name].append(metric)

        # Check for alerts
        self._check_alerts(metric)

    def record_value(self, name: str, value: float, tags: dict[str, str] | None = None):
        """Record a metric value."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            tags=tags or {}
        )
        self.record_metric(metric)

    def get_metric_history(self, metric_name: str, duration_seconds: float | None = None) -> list[PerformanceMetric]:
        """Get historical data for a metric."""
        with self._lock:
            history = list(self.metrics_history[metric_name])

        if duration_seconds is None:
            return history

        cutoff_time = time.time() - duration_seconds
        return [m for m in history if m.timestamp >= cutoff_time]

    def get_metric_stats(self, metric_name: str, duration_seconds: float | None = None) -> dict[str, Any]:
        """Get statistical summary for a metric."""
        history = self.get_metric_history(metric_name, duration_seconds)

        if not history:
            return {"count": 0}

        values = [m.value for m in history]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "timestamp": history[-1].timestamp if history else None,
        }

    def _check_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any alerts."""
        for threshold in self.alert_thresholds:
            if threshold.metric_name != metric.name:
                continue

            # Check threshold condition
            triggered = False
            if (threshold.operator == "gt" and metric.value > threshold.threshold_value) or (threshold.operator == "lt" and metric.value < threshold.threshold_value) or (threshold.operator == "eq" and metric.value == threshold.threshold_value) or (threshold.operator == "ge" and metric.value >= threshold.threshold_value) or (threshold.operator == "le" and metric.value <= threshold.threshold_value):
                triggered = True

            if triggered:
                alert_message = (
                    f"Alert: {metric.name} = {metric.value} "
                    f"({threshold.operator} {threshold.threshold_value})"
                )
                logger.warning(alert_message)

                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert_message, metric)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")

    def get_all_metrics_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of all tracked metrics."""
        summary = {}
        with self._lock:
            for metric_name in self.metrics_history.keys():
                summary[metric_name] = self.get_metric_stats(metric_name)
        return summary

    async def start_monitoring(self, interval_seconds: float = 10.0):
        """Start continuous monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self, interval_seconds: float):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)

    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())

            # Memory metrics
            memory_info = process.memory_info()
            self.record_value("memory.rss_bytes", memory_info.rss)
            self.record_value("memory.vms_bytes", memory_info.vms)

            # CPU metrics
            self.record_value("cpu.usage_percent", process.cpu_percent())

            # Thread count
            self.record_value("threads.count", process.num_threads())

            # File descriptors
            self.record_value("files.open_count", len(process.open_files()))

            # System-wide metrics
            self.record_value("system.cpu_percent", psutil.cpu_percent())
            system_memory = psutil.virtual_memory()
            self.record_value("system.memory_percent", system_memory.percent)

        except ImportError:
            logger.debug("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class ResourceMonitor:
    """Monitors system resources and provides resource usage analytics.
    
    Tracks CPU, memory, disk, and network usage with trend analysis
    and capacity planning insights.
    """

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self._setup_default_alerts()

    def _setup_default_alerts(self):
        """Set up default alert thresholds."""
        default_thresholds = [
            AlertThreshold("memory.rss_bytes", 1024 * 1024 * 1024, "gt", 60.0, "warning"),  # 1GB
            AlertThreshold("cpu.usage_percent", 80.0, "gt", 120.0, "warning"),
            AlertThreshold("system.memory_percent", 90.0, "gt", 60.0, "error"),
            AlertThreshold("files.open_count", 1000, "gt", 60.0, "warning"),
        ]

        for threshold in default_thresholds:
            self.performance_monitor.add_alert_threshold(threshold)

    async def start_monitoring(self, interval_seconds: float = 10.0):
        """Start resource monitoring."""
        await self.performance_monitor.start_monitoring(interval_seconds)

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        await self.performance_monitor.stop_monitoring()

    def get_resource_summary(self) -> dict[str, Any]:
        """Get current resource usage summary."""
        return self.performance_monitor.get_all_metrics_summary()

    def get_resource_trends(self, duration_minutes: int = 60) -> dict[str, Any]:
        """Get resource usage trends over time."""
        duration_seconds = duration_minutes * 60
        trends = {}

        key_metrics = [
            "memory.rss_bytes",
            "cpu.usage_percent",
            "system.memory_percent",
            "files.open_count"
        ]

        for metric in key_metrics:
            history = self.performance_monitor.get_metric_history(metric, duration_seconds)
            if len(history) < 2:
                continue

            values = [m.value for m in history]

            # Calculate trend
            if len(values) >= 5:
                # Simple linear trend calculation
                x = list(range(len(values)))
                n = len(values)
                sum_x = sum(x)
                sum_y = sum(values)
                sum_xy = sum(xi * yi for xi, yi in zip(x, values, strict=False))
                sum_x2 = sum(xi * xi for xi in x)

                # Linear regression slope
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

                trends[metric] = {
                    "current": values[-1],
                    "trend_slope": slope,
                    "trend_direction": "increasing" if slope > 0 else "decreasing",
                    "change_rate": abs(slope),
                    "samples": len(values),
                }

        return trends

    def check_resource_health(self) -> dict[str, Any]:
        """Check overall resource health status."""
        summary = self.get_resource_summary()
        trends = self.get_resource_trends(30)  # 30-minute trends

        health_status = {
            "status": "healthy",
            "issues": [],
            "warnings": [],
            "recommendations": [],
        }

        # Check memory usage
        memory_stats = summary.get("memory.rss_bytes", {})
        if memory_stats.get("latest", 0) > 2 * 1024 * 1024 * 1024:  # 2GB
            health_status["issues"].append("High memory usage detected")
            health_status["status"] = "degraded"

        # Check CPU usage trends
        cpu_trend = trends.get("cpu.usage_percent", {})
        if cpu_trend.get("current", 0) > 80:
            health_status["warnings"].append("High CPU usage")

        if cpu_trend.get("trend_direction") == "increasing" and cpu_trend.get("change_rate", 0) > 5:
            health_status["warnings"].append("CPU usage trend increasing")

        # Check system memory
        system_memory_stats = summary.get("system.memory_percent", {})
        if system_memory_stats.get("latest", 0) > 85:
            health_status["issues"].append("System memory usage critical")
            health_status["status"] = "unhealthy"

        # Generate recommendations
        if len(health_status["issues"]) > 0 or len(health_status["warnings"]) > 0:
            health_status["recommendations"].append("Consider scaling resources")
            health_status["recommendations"].append("Review application memory usage patterns")

        return health_status

    def record_custom_metric(self, name: str, value: float, tags: dict[str, str] | None = None):
        """Record a custom application metric."""
        self.performance_monitor.record_value(name, value, tags)

    def add_custom_alert(self, threshold: AlertThreshold):
        """Add a custom alert threshold."""
        self.performance_monitor.add_alert_threshold(threshold)


# Global monitor instances
performance_monitor = PerformanceMonitor()
resource_monitor = ResourceMonitor()
