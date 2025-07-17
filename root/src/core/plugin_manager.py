from __future__ import annotations

"""Plugin loader and registry for Sentio."""

import importlib
import logging
import os
from typing import List, Optional, Dict, Any, Callable

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
        
        # LangGraph node registry
        self._graph_nodes: Dict[str, Dict[str, Callable]] = {}

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

    def register_plugin(self, plugin: SentioPlugin, pipeline: object = None) -> None:
        """Register a single plugin instance with the pipeline.
        
        Args:
            plugin: Plugin instance to register
            pipeline: Optional pipeline to register with
        """
        if not isinstance(plugin, SentioPlugin):
            logger.warning("%s is not a valid plugin", plugin)
            return
            
        # Add to plugin registry
        self._plugins.append(plugin)
        self._plugin_map[plugin.name] = plugin
        
        # Register with pipeline if provided
        if pipeline:
            try:
                logger.info(f"Registering plugin {plugin.name} with pipeline")
                logger.info(f"Plugin has register method: {hasattr(plugin, 'register')}")
                
                # Call the plugin's register method
                plugin.register(pipeline)
                
                # Check if the plugin successfully added evaluator to the pipeline
                if hasattr(pipeline, "evaluator"):
                    logger.info(f"✅ Plugin {plugin.name} successfully registered evaluator")
                    if hasattr(pipeline.evaluator, "get_evaluation_history"):
                        logger.info("✅ get_evaluation_history method available")
                    else:
                        logger.warning("⚠️ get_evaluation_history method NOT available")
                        
                    if hasattr(pipeline.evaluator, "get_average_metrics"):
                        logger.info("✅ get_average_metrics method available")
                    else:
                        logger.warning("⚠️ get_average_metrics method NOT available")
                else:
                    logger.warning(f"⚠️ Plugin {plugin.name} failed to register evaluator")
                
                logger.info(f"✅ Registered plugin: {plugin.name}")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Plugin {plugin.name} failed to register: {exc}", exc_info=True)

    # ------------------------------------------------------------------
    # LangGraph node registration
    # ------------------------------------------------------------------
    
    def register_graph_node(self, graph_type: str, node_name: str, node_func: Callable) -> None:
        """
        Register a custom node function for a specific graph type.
        
        Args:
            graph_type: Type of graph (e.g., 'basic', 'streaming')
            node_name: Name of the node in the graph
            node_func: Node function to register
        """
        if graph_type not in self._graph_nodes:
            self._graph_nodes[graph_type] = {}
        
        self._graph_nodes[graph_type][node_name] = node_func
        logger.info(f"✅ Registered graph node: {node_name} for {graph_type} graph")
    
    def get_graph_nodes(self, graph_type: str) -> Dict[str, Callable]:
        """
        Get all registered node functions for a specific graph type.
        
        Args:
            graph_type: Type of graph (e.g., 'basic', 'streaming')
            
        Returns:
            Dictionary of node names to node functions
        """
        return self._graph_nodes.get(graph_type, {})
    
    def has_graph_node(self, graph_type: str, node_name: str) -> bool:
        """
        Check if a node is registered for a specific graph type.
        
        Args:
            graph_type: Type of graph (e.g., 'basic', 'streaming')
            node_name: Name of the node in the graph
            
        Returns:
            True if node exists, False otherwise
        """
        return graph_type in self._graph_nodes and node_name in self._graph_nodes[graph_type]
    
    def get_graph_node(self, graph_type: str, node_name: str) -> Optional[Callable]:
        """
        Get a specific node function for a graph type.
        
        Args:
            graph_type: Type of graph (e.g., 'basic', 'streaming')
            node_name: Name of the node in the graph
            
        Returns:
            Node function if found, None otherwise
        """
        if not self.has_graph_node(graph_type, node_name):
            return None
        return self._graph_nodes[graph_type][node_name]

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
