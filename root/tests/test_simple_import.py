#!/usr/bin/env python3
"""
Simple test for importing modules.
"""

import pytest


def test_import_langchain_core():
    """Test that langchain_core can be imported."""
    import langchain_core
    assert langchain_core.__name__ == "langchain_core"


def test_import_langchain_core_runnables():
    """Test that langchain_core.runnables can be imported."""
    from langchain_core import runnables
    assert runnables.__name__ == "langchain_core.runnables"


def test_import_runnable():
    """Test that Runnable can be imported from langchain_core.runnables."""
    from langchain_core.runnables import Runnable
    assert Runnable.__name__ == "Runnable"


def test_import_langgraph():
    """Test that langgraph can be imported."""
    import langgraph
    assert langgraph.__name__ == "langgraph"


def test_import_langgraph_graph():
    """Test that langgraph.graph can be imported."""
    from langgraph import graph
    assert graph.__name__ == "langgraph.graph" 