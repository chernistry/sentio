"""Health check utilities for Sentio RAG System Streamlit UI

This module provides health check functionality that can be used independently
from the main Streamlit application.
"""

import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_system_info(backend_url: str) -> dict:
    """Get system information from the backend."""
    try:
        url = f"{backend_url}/info"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("System info request timed out after 30s. Backend URL: %s", backend_url)
        return {}
    except Exception as e:
        logger.error("System info request failed. URL: %s, Error: %s", backend_url, str(e))
        return {}


def check_health(backend_url: str) -> dict:
    """Check system health."""
    try:
        url = f"{backend_url}/health"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("Health check timed out after 10s. Backend URL: %s", backend_url)
        return {"status": "timeout", "services": {}}
    except Exception as e:
        logger.error("Health check failed. URL: %s, Error: %s", backend_url, str(e))
        return {"status": "unknown", "services": {}}


def render_health_panel(backend_url: str) -> None:
    """Render health check panel in Streamlit sidebar.
    
    This function can be imported and used in the main Streamlit app
    when health check functionality is needed.
    """
    import streamlit as st
    
    # System information
    st.subheader("System Info")
    info = get_system_info(backend_url)
    health = check_health(backend_url)

    if info:
        st.info(f"Version: {info.get('version', 'Unknown')}")

        config = info.get("configuration", {})
        st.write("Configuration:")
        st.json(config, expanded=False)

    if health:
        st.subheader("Health Status")
        status = health.get("status", "unknown")
        status_color = {
            "healthy": "green",
            "degraded": "orange",
            "unhealthy": "red",
            "unknown": "gray"
        }.get(status, "gray")

        st.markdown(f"Status: :{status_color}[{status}]")

        # Show services health
        services = health.get("services", {})
        if services:
            for service, service_status in services.items():
                service_color = {
                    "healthy": "green",
                    "unavailable": "gray",
                    "unhealthy": "red",
                    "unknown": "yellow"
                }.get(service_status, "gray")
                st.markdown(f"- {service}: :{service_color}[{service_status}]") 