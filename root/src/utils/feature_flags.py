#!/usr/bin/env python3
"""
Feature flags for Sentio RAG.

This module provides a centralized place to manage feature flags
that control behavior across the application.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Default feature flag values
_DEFAULT_FLAGS = {
    "USE_LANGGRAPH": True,  # Use LangGraph instead of legacy Pipeline
    "ENABLE_HYDE": False,   # Enable Hypothetical Document Embeddings
    "ENABLE_STREAMING": True,  # Enable streaming responses
    "ENABLE_RAGAS": True,   # Enable RAGAS evaluation
}


def get_feature_flag(flag_name: str, default_value: Any = None) -> Any:
    """
    Get the value of a feature flag from environment or defaults.
    
    Args:
        flag_name: The name of the feature flag
        default_value: Default value if not found in environment or defaults
        
    Returns:
        The value of the feature flag
    """
    # Check environment first
    env_value = os.environ.get(flag_name)
    
    if env_value is not None:
        # Convert string environment values to appropriate types
        if env_value.lower() in ("true", "1", "yes", "y", "on"):
            return True
        elif env_value.lower() in ("false", "0", "no", "n", "off"):
            return False
        elif env_value.isdigit():
            return int(env_value)
        else:
            return env_value
    
    # Fall back to defaults if defined
    if flag_name in _DEFAULT_FLAGS:
        return _DEFAULT_FLAGS[flag_name]
    
    # Use provided default or None
    return default_value


def get_all_feature_flags() -> Dict[str, Any]:
    """
    Get all feature flags with their current values.
    
    Returns:
        Dictionary of all feature flags and their values
    """
    flags = {}
    
    # Start with defaults
    for flag_name, default_value in _DEFAULT_FLAGS.items():
        flags[flag_name] = get_feature_flag(flag_name, default_value)
    
    return flags


# Convenience accessors for commonly used flags
def use_langgraph() -> bool:
    """Check if LangGraph should be used instead of legacy Pipeline."""
    return get_feature_flag("USE_LANGGRAPH", True)


def enable_hyde() -> bool:
    """Check if HyDE is enabled."""
    return get_feature_flag("ENABLE_HYDE", False)


def enable_streaming() -> bool:
    """Check if streaming responses are enabled."""
    return get_feature_flag("ENABLE_STREAMING", True)


def enable_ragas() -> bool:
    """Check if RAGAS evaluation is enabled."""
    return get_feature_flag("ENABLE_RAGAS", True)


# Log feature flag status at module import
logger.info(f"Feature flags: {get_all_feature_flags()}") 