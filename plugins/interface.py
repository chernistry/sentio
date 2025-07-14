from __future__ import annotations

"""Base plugin interface for Sentio."""

from abc import ABC, abstractmethod
from typing import Any


class SentioPlugin(ABC):
    """Abstract base class for all plugins.

    Attributes:
        name: Human-readable plugin name used for discovery.
        plugin_type: Functional area the plugin extends (e.g., *reranker*, *embedding*).
        version: Semantic version string so the host can manage compatibility.
        description: Short one-line description visible in plugin listings.
    """

    # ---------------- Mandatory metadata ---------------- #
    name: str
    plugin_type: str

    # ---------------- Optional metadata ----------------- #
    version: str = "0.1.0"
    description: str = ""

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def register(self, pipeline: Any) -> None:  # noqa: D401 – imperative mood
        """Attach plugin to the *pipeline* instance.

        Implementations should mutate the *pipeline* by adding or replacing
        components.  They **must not** perform heavyweight initialisation in
        this method – do that in ``__init__`` so failures surface early.
        """

    # New but optional: provide safe default so existing plugins remain valid.
    def unregister(self, pipeline: Any) -> None:  # noqa: D401 – imperative mood
        """Detach plugin from *pipeline*.

        This enables hot-swapping at runtime.  Default implementation is a
        no-op to preserve backward compatibility.
        """
        # Default no-op – override if cleanup is required.
        return
