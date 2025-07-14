import sys
import types

import pytest

from plugins.interface import SentioPlugin
from root.src.core.plugin_manager import PluginManager


class _StubPipeline:  # pylint: disable=too-few-public-methods
    """Lightweight object to verify plugin registration hooks."""


class _DummyPlugin(SentioPlugin):
    """Minimal in-memory plugin implementation for testing."""

    name = "dummy"
    plugin_type = "test"

    def __init__(self) -> None:  # noqa: D401 – imperative mood
        self.register_called: bool = False
        self.unregister_called: bool = False

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def register(self, pipeline: _StubPipeline) -> None:  # type: ignore[override]
        """Mark *pipeline* object so we can assert hook execution."""
        self.register_called = True
        setattr(pipeline, "_dummy_registered", True)

    def unregister(self, pipeline: _StubPipeline) -> None:  # type: ignore[override]
        """Opposite of :meth:`register`."""
        self.unregister_called = True
        setattr(pipeline, "_dummy_registered", False)


@pytest.fixture()
def dummy_plugin_module() -> types.ModuleType:
    """Provide a temporary *plugins.dummy_test_plugin* module in *sys.modules*."""
    module_name = "plugins.dummy_test_plugin"
    module = types.ModuleType(module_name)

    # Expose factory required by *PluginManager.load_plugin*
    def get_plugin() -> _DummyPlugin:  # noqa: D401 – imperative mood
        return _DummyPlugin()

    module.get_plugin = get_plugin  # type: ignore[attr-defined]

    # Install into *sys.modules* and yield to caller.
    sys.modules[module_name] = module
    try:
        yield module
    finally:
        # Cleanup so other tests are unaffected.
        sys.modules.pop(module_name, None)


def test_plugin_lifecycle(dummy_plugin_module: types.ModuleType) -> None:  # noqa: D401 – imperative mood
    """End-to-end coverage of load → register → unload → reload flow."""
    manager = PluginManager()

    # ------------------------------------------------------------------
    # Load & register
    # ------------------------------------------------------------------
    manager.load_plugin("dummy_test_plugin")
    assert manager.get_first("test") is not None, "Plugin should be present after load()"

    pipeline = _StubPipeline()
    manager.register_all(pipeline)

    plugin = manager.get_first("test")
    assert plugin is not None and getattr(plugin, "register_called"), "register() hook should run"
    assert getattr(pipeline, "_dummy_registered", False) is True, "Pipeline flag set by plugin"

    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------
    manager.unload_plugin("dummy_test_plugin", pipeline)
    assert manager.get_first("test") is None, "Plugin removed after unload()"
    assert getattr(pipeline, "_dummy_registered", False) is False, "unregister() hook should run"

    # ------------------------------------------------------------------
    # Reload
    # ------------------------------------------------------------------
    # Re-add module so *importlib* can find it again after previous unload
    sys.modules["plugins.dummy_test_plugin"] = dummy_plugin_module
    manager.reload_plugin("dummy_test_plugin", pipeline)
    plugin = manager.get_first("test")
    assert plugin is not None, "Plugin reloaded"
    assert getattr(plugin, "register_called"), "register() invoked on reload"
    assert getattr(pipeline, "_dummy_registered", False) is True, "Pipeline updated again" 