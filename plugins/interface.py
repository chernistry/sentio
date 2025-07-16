"""
Interface definitions for Sentio plugins.

This module defines the base classes and interfaces that all Sentio plugins
must implement to be compatible with the plugin system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SentioPlugin(ABC):
    """Base class for all Sentio plugins."""

    name: str = "base_plugin"
    plugin_type: str = "generic"
    version: str = "0.1.0"
    description: str = "Base plugin interface"

    @abstractmethod
    def register(self, pipeline: Any) -> None:
        """
        Register the plugin with a pipeline or application.
        
        This method is called when the plugin is loaded and should
        attach any necessary components or monkey-patch the pipeline.
        
        Args:
            pipeline: The pipeline or application object to register with.
        """
        pass
