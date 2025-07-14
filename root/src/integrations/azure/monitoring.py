#!/usr/bin/env python3
"""
Azure Application Insights Integration Module

Provides functionality for integrating with Azure Application Insights
for telemetry, monitoring, and logging.
"""

import os
import logging
from typing import Dict, Any

# Azure Application Insights
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure import metrics_exporter
from opencensus.stats import stats as stats_module
from opencensus.trace import config_integration
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer

logger = logging.getLogger(__name__)

def configure_azure_app_insights():
    """
    Configure Azure Application Insights for telemetry and logging.
    
    Integrates OpenCensus with Azure Application Insights to provide:
    - Automatic logging
    - Request telemetry
    - Dependency tracking
    - Custom metrics
    
    Returns:
        bool: True if configuration was successful, False otherwise
    """
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        logger.warning("APPLICATIONINSIGHTS_CONNECTION_STRING not set, App Insights integration disabled")
        return False
    
    # Add Azure Log Handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(AzureLogHandler(connection_string=connection_string))
    
    # Configure metrics exporter
    exporter = metrics_exporter.new_metrics_exporter(
        connection_string=connection_string
    )
    stats = stats_module.stats
    view_manager = stats.view_manager
    view_manager.register_exporter(exporter)
    
    # Configure trace integration
    config_integration.trace_integrations(["logging", "requests", "sqlalchemy", "fastapi"])
    
    logger.info("Azure Application Insights integration configured")
    return True

def get_tracer():
    """
    Get a configured Azure Application Insights tracer.
    
    Returns:
        Tracer: OpenCensus tracer configured for Azure App Insights
    """
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        # Return a no-op tracer if App Insights is not configured
        logger.warning("App Insights not configured, returning no-op tracer")
        return Tracer()
    
    return Tracer(
        sampler=ProbabilitySampler(1.0),
        exporter=AzureLogHandler(connection_string=connection_string)
    )

def track_event(name: str, properties: Dict[str, Any] = None):
    """
    Track a custom event in Application Insights.
    
    Args:
        name: Event name
        properties: Optional event properties
    """
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        logger.debug(f"App Insights not configured, skipping event tracking: {name}")
        return
    
    # Using tracer to track custom event
    with get_tracer().span(name) as span:
        if properties:
            for k, v in properties.items():
                span.add_attribute(k, v)
        logger.info(f"Tracked event: {name}")

def track_metric(name: str, value: float, properties: Dict[str, Any] = None):
    """
    Track a custom metric in Application Insights.
    
    Args:
        name: Metric name
        value: Metric value
        properties: Optional metric dimensions
    """
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        logger.debug(f"App Insights not configured, skipping metric: {name}={value}")
        return
    
    # Create and record the metric
    stats = stats_module.stats
    stats.stats.record(name, value, properties or {}) 