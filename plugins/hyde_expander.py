"""
Hypothetical Document Embeddings (HyDE) query expansion module.

This is a compatibility wrapper that imports from the new location.
The actual implementation has been moved to root.src.core.plugins.hyde_expander.
"""

import logging
from typing import Any

from root.src.core.plugins.hyde_expander import (
    expand_query_hyde,
    expand_query_hyde_async,
    HyDEPlugin as CoreHyDEPlugin,
)

logger = logging.getLogger(__name__)

# Log a deprecation warning
logger.warning(
    "Importing HyDE plugin from plugins/hyde_expander.py is deprecated. "
    "Use root.src.core.plugins.hyde_expander instead."
)


class HyDEPlugin(CoreHyDEPlugin):
    """Legacy plugin class for backward compatibility."""
    
    def __init__(self):
        """Initialize with deprecation warning."""
        logger.warning(
            "Using HyDEPlugin from plugins/hyde_expander.py is deprecated. "
            "Use root.src.core.plugins.hyde_expander.HyDEPlugin instead."
        )


def get_plugin() -> Any:
    """Return plugin instance for backward compatibility."""
    return HyDEPlugin()
