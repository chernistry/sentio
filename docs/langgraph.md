# LangGraph Integration

## Overview

Sentio RAG now uses [LangGraph](https://github.com/langchain-ai/langgraph) as the underlying framework for building and executing the RAG pipeline. LangGraph provides a flexible, composable approach to building complex LLM-powered applications with clear state management and visualization capabilities.

## Key Benefits

- **Visualization**: Debug and monitor your RAG pipeline in a visual interface
- **Modularity**: Easily add, remove, or modify pipeline components
- **Streaming**: Native support for token-by-token streaming responses
- **Extensibility**: Plugin system for adding custom nodes
- **Traceability**: Integration with LangSmith for tracking and evaluation

## Architecture

The LangGraph implementation follows this architecture:

```
                   ┌───────────────┐
                   │ Input Query   │
                   └───────┬───────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Input Normalizer  │
                 └─────────┬─────────┘
                           │
                           ▼
                  ┌─────────────────┐
          ┌──────►│ HyDE Expansion  │───────┐
          │       └─────────────────┘       │
          │                                 │
          │                                 ▼
┌─────────┴──────────┐             ┌────────────────┐
│                    │             │                │
│  (if HYDE enabled) │             │    Retriever   │
│                    │             │                │
└────────────────────┘             └────────┬───────┘
                                            │
                                            ▼
                                   ┌────────────────┐
                                   │    Reranker    │
                                   └────────┬───────┘
                                            │
                                            ▼
                                   ┌────────────────┐
                                   │    Generator   │
                                   └────────┬───────┘
                                            │
                                            ▼
                                   ┌────────────────┐
                                   │ Post-processor │
                                   └────────┬───────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                          ┌──────►│ RAGAS Evaluator  │───────┐
                          │       └──────────────────┘       │
                          │                                  │
┌─────────────────────┐   │                                  │   ┌─────────────────────┐
│                     │   │                                  │   │                     │
│ (if RAGAS disabled) │◄──┘                                  └──►│  (if RAGAS enabled) │
│                     │                                          │                     │
└─────────────────────┘                                          └─────────────────────┘
```

## Using LangGraph

### Basic Usage

The LangGraph implementation is now the default in Sentio RAG. You can use it through the existing API endpoints without any changes to your code:

```python
from root.src.api.client import SentioClient

client = SentioClient("http://localhost:8000")
response = await client.query("What is Retrieval-Augmented Generation?")
print(response.answer)
```

### Visualization with LangGraph Server

To visualize the RAG pipeline in action, you can use the LangGraph Server:

1. Start the server:
   ```bash
   python scripts/langgraph_server.py
   ```

2. Open your browser at http://localhost:2024

3. You'll see the graph visualization and can trace execution in real-time

### Switching Between Pipeline and LangGraph

You can switch between the legacy Pipeline and LangGraph implementations using the `USE_LANGGRAPH` environment variable:

```bash
# Use LangGraph (default)
export USE_LANGGRAPH=1

# Use legacy Pipeline
export USE_LANGGRAPH=0
```

Or programmatically:

```python
import os
os.environ["USE_LANGGRAPH"] = "0"  # Use legacy Pipeline
```

## Extending the Graph

### Adding Custom Nodes

You can add custom nodes to the graph using the plugin system:

```python
from root.src.core.plugin_manager import plugin_manager
from root.src.core.graph.graph_factory import RAGState

async def my_custom_node(state: RAGState) -> RAGState:
    """A custom node that modifies the state."""
    # Modify the state as needed
    state.metadata["custom_field"] = "custom value"
    return state

# Register the node for both graph types
plugin_manager.register_graph_node("basic", "my_custom_node", my_custom_node)
plugin_manager.register_graph_node("streaming", "my_custom_node", my_custom_node)
```

### Modifying the Graph Structure

To modify the graph structure, you can create a custom graph factory:

```python
from root.src.core.graph.graph_factory import _build_graph_from_nodes
from langgraph.graph import StateGraph

def build_custom_graph(config):
    """Build a custom graph with a modified structure."""
    graph = _build_graph_from_nodes("basic", config)
    
    # Add a custom edge
    graph.add_edge("input_normalizer", "my_custom_node")
    graph.add_edge("my_custom_node", "retriever")
    
    return graph.compile()
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| USE_LANGGRAPH | Use LangGraph instead of legacy Pipeline | 1 (true) |
| ENABLE_HYDE | Enable Hypothetical Document Embeddings | 0 (false) |
| ENABLE_STREAMING | Enable streaming responses | 1 (true) |
| ENABLE_RAGAS | Enable RAGAS evaluation | 1 (true) |
| LANGSMITH_API_KEY | API key for LangSmith integration | None |
| LANGCHAIN_PROJECT | Project name for LangSmith | sense-rag |

### Feature Flags

You can also use the feature flags module to check and set flags programmatically:

```python
from root.src.utils.feature_flags import use_langgraph, enable_hyde

if use_langgraph():
    # Use LangGraph
    ...

if enable_hyde():
    # Use HyDE expansion
    ...
```

## Troubleshooting

### Common Issues

1. **Pipeline not initialized error**:
   - Make sure the pipeline is properly initialized before using the graph
   - Check that all required environment variables are set

2. **Blocking calls in async context**:
   - Use the `--allow-blocking` flag when starting the LangGraph Server
   - Or set `BG_JOB_ISOLATED_LOOPS=true` in your environment

3. **Missing dependencies**:
   - Ensure you have installed all required dependencies:
     ```bash
     pip install langgraph>=0.5.3 langgraph-api>=0.2.86 aiofiles>=23.2.1
     ```

### Getting Help

If you encounter any issues, please:
1. Check the logs for detailed error messages
2. Refer to the [LangGraph documentation](https://github.com/langchain-ai/langgraph)
3. Open an issue in the project repository 