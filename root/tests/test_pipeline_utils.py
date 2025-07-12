import os
from types import ModuleType

import pytest

from root.src.core.pipeline import SentioRAGPipeline, PipelineConfig, GenerationMode
from root.src.core.plugin_manager import PluginManager

pytestmark = pytest.mark.unit


@pytest.fixture(scope="function")
def pipeline_instance() -> SentioRAGPipeline:
    """Return a minimal, uninitialised pipeline for utility testing."""
    # Disable potentially expensive components via env vars.
    os.environ.setdefault("CHAT_PROVIDER", "mock")
    os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
    os.environ.setdefault("RERANKER_PROVIDER", "mock")
    return SentioRAGPipeline(config=PipelineConfig(cache_enabled=False))


def test_build_context_string_and_prompt(pipeline_instance: SentioRAGPipeline):
    """_build_context_string and _build_prompt should format data correctly."""
    docs = [
        {"text": "Vector DBs like Qdrant are fast.", "source": "doc1", "score": 0.95},
        {"text": "Another doc.", "source": "doc2", "score": 0.75},
    ]

    context = pipeline_instance._build_context_string(docs)
    assert "[Source 1: doc1" in context
    assert context.count("---") == 1  # Delimiter between two context parts

    fast_cfg = pipeline_instance._generation_configs[GenerationMode.FAST]
    prompt = pipeline_instance._build_prompt("What is Qdrant?", context, fast_cfg)

    assert "Question: What is Qdrant?" in prompt
    assert "concise, direct answer" in prompt.lower()


# ---------------------------------------------------------------------------
# PluginManager behaviour
# ---------------------------------------------------------------------------


def test_plugin_manager_load_and_unload(monkeypatch):
    """PluginManager should load, register and unload a dynamically injected plugin."""
    import sys
    import types

    # Dynamically create a module `plugins.dummy` to avoid touching filesystem.
    module_name = "plugins.dummy"
    dummy_module = types.ModuleType(module_name)

    from plugins.interface import SentioPlugin  # Import inside function to avoid top-level cost.

    class DummyPlugin(SentioPlugin):  # type: ignore[misc]
        name = "dummy"
        version = "0.1"
        plugin_type = "test"

        def register(self, pipeline):  # noqa: D401, ANN001
            pipeline._dummy_registered = True

    def get_plugin():  # noqa: D401, ANN001
        return DummyPlugin()

    dummy_module.get_plugin = get_plugin  # type: ignore[attr-defined]

    # Inject into sys.modules so that importlib can resolve it.
    sys.modules[module_name] = dummy_module

    pm = PluginManager()
    pm.load_plugin("dummy")

    plugin = pm.get_first("test")
    assert plugin is not None and plugin.name == "dummy"

    # Register the plugin and then unload to test full lifecycle.
    class FakePipeline:  # Minimal stand-in
        pass

    pipeline = FakePipeline()
    pm.register_all(pipeline)
    assert getattr(pipeline, "_dummy_registered", False)

    pm.unload_plugin("dummy", pipeline)
    assert pm.get_first("test") is None 