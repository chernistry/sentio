"""
Compatibility wrapper for HyDE plugin.

This module provides backward compatibility with the old plugin location.
It simply re-exports the functionality from the new location.
"""

from .hyde_expander import (
    expand_query_hyde,
    expand_query_hyde_async,
    HyDEPlugin,
    hyde_plugin,
)


def get_plugin():
    """Return plugin instance for backward compatibility."""
    return hyde_plugin 