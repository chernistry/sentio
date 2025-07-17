#!/usr/bin/env python3
"""
Simple test for LangGraph functionality without project dependencies.
"""

import pytest
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class SimpleState(BaseModel):
    """Simple state for testing."""
    query: str = Field(description="User query")
    result: str = Field(default="", description="Result")


def test_create_simple_graph():
    """Test that a simple graph can be created with LangGraph."""
    try:
        from langgraph.graph import StateGraph, END
        
        # Create a simple graph
        graph = StateGraph(SimpleState)
        
        # Define a simple node
        def simple_node(state: SimpleState) -> SimpleState:
            state.result = f"Processed: {state.query}"
            return state
        
        # Add the node to the graph
        graph.add_node("processor", simple_node)
        
        # Connect the node to the end
        graph.add_edge("processor", END)
        
        # Set the entry point
        graph.set_entry_point("processor")
        
        # Compile the graph
        compiled_graph = graph.compile()
        
        # Just check that compilation succeeded
        assert compiled_graph is not None
    except ImportError as e:
        pytest.skip(f"LangGraph not properly installed: {e}")


@pytest.mark.asyncio
async def test_invoke_simple_graph():
    """Test that a simple graph can be invoked."""
    try:
        from langgraph.graph import StateGraph, END
        
        # Create a simple graph
        graph = StateGraph(SimpleState)
        
        # Define a simple node
        def simple_node(state: Dict) -> Dict:
            # Handle both dict and SimpleState
            if isinstance(state, dict):
                return {"query": state.get("query", ""), "result": f"Processed: {state.get('query', '')}"}
            else:
                state.result = f"Processed: {state.query}"
                return state
        
        # Add the node to the graph
        graph.add_node("processor", simple_node)
        
        # Connect the node to the end
        graph.add_edge("processor", END)
        
        # Set the entry point
        graph.set_entry_point("processor")
        
        # Compile the graph
        compiled_graph = graph.compile()
        
        # Invoke the graph
        result = await compiled_graph.ainvoke({"query": "test query"})
        
        # Verify the result - handle both dict and SimpleState return types
        if isinstance(result, dict):
            assert result.get("result") == "Processed: test query"
        else:
            assert result.result == "Processed: test query"
    except ImportError as e:
        pytest.skip(f"LangGraph not properly installed: {e}") 