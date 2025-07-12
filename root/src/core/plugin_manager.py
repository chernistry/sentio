from __future__ import annotations

"""Plugin loader and registry for Sentio."""

import importlib
import logging
import os
from typing import List, Optional, Dict

from plugins.interface import SentioPlugin
from pathlib import Path
from types import ModuleType
import sys

logger = logging.getLogger(__name__)


class PluginManager:
    """Load and register plugins."""

    def __init__(self) -> None:
        # Keep both list (ordering) and dict (lookup by name) for convenience
        self._plugins: List[SentioPlugin] = []
        self._plugin_map: Dict[str, SentioPlugin] = {}
        self._modules: Dict[str, ModuleType] = {}

    def load_from_env(self) -> None:
        """Load plugins listed in ``SENTIO_PLUGINS`` env var."""
        names = os.getenv("SENTIO_PLUGINS", "")
        if not names:
            return
        for name in [n.strip() for n in names.split(",") if n.strip()]:
            self.load_plugin(name)

    # ------------------------------------------------------------------
    # Bulk loading helpers
    # ------------------------------------------------------------------

    def load_all(self, package: str = "plugins") -> None:
        """Discover and load **all** plugin modules in *package*."""
        for name in self.discover(package):
            self.load_plugin(name)

    def load_plugin(self, name: str) -> None:
        """Load a plugin by module name."""
        try:
            module = importlib.import_module(f"plugins.{name}")
            plugin = getattr(module, "get_plugin")()

            if not isinstance(plugin, SentioPlugin):
                logger.warning("%s is not a valid plugin", name)
                return

            # If plugin already loaded, skip
            if name in self._plugin_map:
                logger.debug("Plugin %s already loaded", name)
                return

            self._plugins.append(plugin)
            self._plugin_map[name] = plugin
            self._modules[name] = module
            logger.info("✓ Loaded plugin: %s v%s", name, getattr(plugin, "version", "unknown"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load plugin %s: %s", name, exc)

    def register_all(self, pipeline: object) -> None:
        """Register loaded plugins with the pipeline."""
        for plugin in self._plugins:
            try:
                plugin.register(pipeline)
                logger.info("✓ Registered plugin: %s", plugin.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Plugin %s failed to register: %s", plugin.name, exc)

    # ------------------------------------------------------------------
    # Discovery & Hot swapping
    # ------------------------------------------------------------------

    def discover(self, package: str = "plugins") -> List[str]:
        """Return a list of importable plugin module names in *package* dir."""
        from importlib import util as _iu

        # Resolve package location even for namespace packages (no __init__.py)
        try:
            spec = _iu.find_spec(package)
            if spec and spec.submodule_search_locations:
                package_dir = Path(next(iter(spec.submodule_search_locations)))
            else:
                package_dir = Path(package)
        except Exception:  # noqa: BLE001
            package_dir = Path(package)

        candidates: List[str] = []
        for py in package_dir.glob("*.py"):
            if py.stem.startswith("__"):
                continue
            candidates.append(py.stem)
        return sorted(candidates)

    def unload_plugin(self, name: str, pipeline: object | None = None) -> None:
        """Unload *name* plugin and optionally detach from *pipeline*."""
        plugin = self._plugin_map.get(name)
        if not plugin:
            logger.debug("Plugin %s not loaded", name)
            return

        # Attempt to call unregister hook
        try:
            if pipeline is not None and hasattr(plugin, "unregister"):
                plugin.unregister(pipeline)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during plugin %s unregister: %s", name, exc)

        # Remove from internal registries
        self._plugins.remove(plugin)
        self._plugin_map.pop(name, None)

        # Remove module if present to allow reload
        module = self._modules.pop(name, None)
        if module and module.__name__ in sys.modules:
            del sys.modules[module.__name__]

        logger.info("✗ Unloaded plugin: %s", name)

    def reload_plugin(self, name: str, pipeline: object | None = None) -> None:
        """Hot-reload a plugin by *name* (unload → import → register)."""
        self.unload_plugin(name, pipeline)
        self.load_plugin(name)
        if pipeline is not None:
            plugin = self._plugin_map.get(name)
            if plugin:
                try:
                    plugin.register(pipeline)
                    logger.info("↻ Re-registered plugin: %s", name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to re-register plugin %s: %s", name, exc)

    def get_first(self, plugin_type: str) -> Optional[SentioPlugin]:
        """Return the first plugin of ``plugin_type``."""
        for plugin in self._plugins:
            if plugin.plugin_type == plugin_type:
                return plugin
        return None
