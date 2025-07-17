# LangGraph Migration Project

This document combines the migration checklist, architecture decision record, and integration guide for the LangGraph migration project.

## Migration Checklist

This step-by-step plan integrates LangGraph as the backbone of the LLM pipeline and
cleans up the repository structure. Tick the boxes as you progress.

### 0 · Preparation

- [x] **Create migration branch** `feat/langgraph-migration`.
- [x] **Baseline assurance** – run `pytest -q`, save coverage report.
- [x] **Add dependencies** – append `langgraph>=0.0.35` (or latest) and
      `langchain>=0.1` to *requirements*. Build Docker locally to confirm.
- [x] **Freeze versions** with `pip-compile` or similar.

### 1 · Prototype Graph Backbone

- [x] Create package `src/core/graph/` with `__init__.py`.
- [x] Implement `graph_factory.py` that converts legacy `Pipeline` steps into
      LangGraph nodes:
  - `InputNormalizer`
  - `Retriever`
  - `Reranker`
  - `Generator`
  - `PostProcessor`
- [x] Build linear graph mirroring current flow; expose
      `build_basic_graph(settings: Settings) -> Graph`.
- [x] Add smoke test `tests/test_graph_smoke.py` ensuring identical output to
      legacy pipeline on a fixed prompt.

### 2 · API Switching

- [x] Inject the graph into FastAPI: adapt `src/api/routes.py` to call
      `graph_factory.build_basic_graph()` instead of `Pipeline`.
- [x] Toggle via `settings.USE_LANGGRAPH` env-flag; default **on** after parity.

### 3 · Incremental Feature Ports

- [x] HYDE expansion branch (`plugins.hyde_expander`) → optional LangGraph path.
- [x] RAGAS evaluation node (`core.llm.ragas`) after answer generation.
- [x] Streaming support – wrap generator node with async iterator.
- [x] Plugin Manager hooks – allow nodes to be dynamically registered.

### 4 · Repository Restructure (after graph stable)

```
sense/
├── api/               # FastAPI (routers, deps, schemas)
├── core/
│   ├── graph/         # LangGraph graphs + nodes
│   ├── embeddings/
│   ├── llm/
│   ├── retrievers/
│   ├── rerankers/
│   └── plugins/       # lightweight hooks, moved from top-level
├── utils/
├── cli/
├── tests/
└── infra/             # docker, k8s, azure, etc.
```

- [x] Move `plugins/*.py` under `core/plugins/`, keep thin wrappers importing old
      paths for backward compatibility.
- [ ] Delete `core/pipeline.py` once all tests use graph.

### 5 · CI + Docker

- [ ] Extend CI matrix to run both legacy + graph tests until cut-over.
- [ ] Update `docker-compose.yml` and Dockerfiles to use slim images & install
      LangGraph.
- [x] Add `pre-commit` hooks: *ruff*, *black* (88 chars), *mypy* strict.

### 6 · Documentation

- [x] Replace old *pipeline* diagram with LangGraph DAG in `README.md`.
- [x] Add ADR (Architecture Decision Record) describing LangGraph adoption.

### 7 · Clean-up & De-risk

- [ ] Remove deprecated code paths and TODO comments.
- [ ] Run `pytest --cov`, aim for ≥ 90 %.
- [ ] Tag release `v0.2.0-LangGraph`.

### Acceptance Criteria

- All tests green.
- API returns same or better latency & accuracy.
- New nodes can be added by creating a single file & registering via Plugin
  Manager without touching core graph logic.

## Architecture Decision Record: LangGraph Adoption

### Status

Accepted

### Context

The Sentio RAG system was initially built with a custom pipeline architecture that handled the flow of data through various components (input normalization, retrieval, reranking, generation, etc.). While this approach worked well for our initial needs, several challenges emerged as the system grew:

1. **Complexity Management**: As we added more features and components, the pipeline code became increasingly complex and difficult to maintain.
2. **Extensibility**: Adding new components or alternative paths required significant changes to the core pipeline code.
3. **Visualization**: Debugging and understanding the flow of data through the pipeline was challenging without proper visualization tools.
4. **Streaming**: Implementing efficient streaming responses required custom code that was difficult to maintain.
5. **Testing**: Testing individual components and their interactions was cumbersome.

LangGraph, a framework for building stateful, multi-actor applications with LLMs, emerged as a potential solution to these challenges. It provides a structured way to define and execute computational graphs with LLMs, offering features like state management, visualization, and streaming capabilities.

### Decision

We have decided to adopt LangGraph as the backbone of our RAG pipeline for the following reasons:

1. **Structured Graph Representation**: LangGraph provides a clear way to represent our pipeline as a directed graph, making the flow of data and control explicit.
2. **State Management**: LangGraph's state management simplifies tracking and manipulating data as it flows through the pipeline.
3. **Visualization**: LangGraph Studio offers built-in visualization tools for debugging and monitoring.
4. **Streaming Support**: LangGraph has native support for streaming responses, simplifying our implementation.
5. **Extensibility**: Adding new nodes or alternative paths is straightforward with LangGraph's graph structure.
6. **Community Support**: As part of the LangChain ecosystem, LangGraph has active development and community support.

### Implementation

The implementation strategy involves:

1. **Incremental Migration**: We'll start by creating a parallel LangGraph implementation that mirrors our existing pipeline, allowing us to compare results and ensure correctness.
2. **Feature Parity**: We'll ensure all existing features are supported in the LangGraph implementation.
3. **Enhanced Features**: We'll leverage LangGraph's capabilities to add new features like visualization and improved streaming.
4. **Plugin System**: We'll adapt our plugin system to work with LangGraph, allowing for dynamic registration of nodes.
5. **Testing**: We'll develop comprehensive tests to ensure the LangGraph implementation behaves as expected.
6. **Documentation**: We'll update our documentation to reflect the new architecture and provide guidance for developers.

### Consequences

#### Positive

- Clearer representation of the pipeline flow
- Easier debugging and visualization through LangGraph Studio
- Simplified extension of the pipeline with new components
- Better streaming support
- Alignment with industry standards and practices

#### Negative

- Learning curve for developers unfamiliar with LangGraph
- Temporary maintenance of two parallel implementations during migration
- Potential performance overhead from the additional abstraction layer

#### Neutral

- Need to adapt existing plugins to the new architecture
- Refactoring of some code to fit the LangGraph paradigm

### References

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangGraph GitHub Repository](https://github.com/langchain-ai/langgraph)
- [LangGraph Studio](https://smith.langchain.com/studio)

## LangGraph Integration Guide

This section provides an overview of how LangGraph is integrated into the Sentio RAG system and how to work with it.

### Overview

LangGraph is a framework for building stateful, multi-actor applications with LLMs. In Sentio, we use LangGraph to represent our RAG pipeline as a directed graph, with nodes for each component (input normalization, retrieval, reranking, generation, etc.) and edges representing the flow of data between them.

### Architecture

#### RAG State

The state of the graph is represented by the `RAGState` class in `graph_factory.py`:

```python
class RAGState(BaseModel):
    """State container for RAG graph execution."""
    
    # Input state
    query: str = Field(description="User query")
    
    # Processing state
    normalized_query: Optional[str] = Field(None, description="Normalized query after preprocessing")
    retrieved_documents: List[Dict] = Field(default_factory=list, description="Documents retrieved from vector store")
    reranked_documents: List[Dict] = Field(default_factory=list, description="Reranked documents")
    context: Optional[str] = Field(None, description="Formatted context string")
    
    # Output state
    answer: Optional[str] = Field(None, description="Generated answer")
    sources: List[Dict] = Field(default_factory=list, description="Source documents used for generation")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
```

#### Graph Nodes

The following nodes are implemented in the graph:

1. **input_normalizer**: Preprocesses the input query
2. **hyde_expander** (optional): Expands the query using Hypothetical Document Embeddings
3. **retriever**: Retrieves relevant documents from the vector store
4. **reranker**: Reranks the retrieved documents
5. **generator**: Generates an answer based on the query and context
6. **post_processor**: Applies post-processing to the generated answer
7. **ragas_evaluator** (optional): Evaluates the quality of the generated answer

#### Graph Factory

Two main graph types are provided:

1. **Basic Graph**: A standard RAG pipeline with all nodes
2. **Streaming Graph**: A streaming-enabled version of the RAG pipeline that supports token-by-token streaming

### Using LangGraph

#### Basic Usage

To use the LangGraph implementation, ensure that the `USE_LANGGRAPH` environment variable is set to `true` (default). The API will automatically use the LangGraph implementation.

```python
from root.src.core.graph import build_basic_graph
from root.src.core.pipeline import PipelineConfig

# Create a pipeline configuration
config = PipelineConfig(
    collection_name="my_collection",
    top_k_retrieval=5,
    top_k_final=3
)

# Build a graph
graph = build_basic_graph(config)

# Execute the graph
result = graph.invoke({"query": "What is RAG?"})

# Access the result
answer = result.answer
sources = result.sources
metadata = result.metadata
```

#### Streaming Usage

For streaming responses, use the streaming graph:

```python
from root.src.core.graph import build_streaming_graph
from root.src.core.pipeline import PipelineConfig

# Create a pipeline configuration
config = PipelineConfig(
    collection_name="my_collection",
    top_k_retrieval=5,
    top_k_final=3
)

# Build a streaming graph
streaming_graph = build_streaming_graph(config)

# Stream the response
async for chunk in streaming_graph.astream({"query": "What is RAG?"}):
    print(chunk["answer"], end="", flush=True)
```

### Extending the Graph

#### Adding Custom Nodes

You can add custom nodes to the graph using the plugin manager:

```python
from root.src.core.plugin_manager import plugin_manager

# Define a custom node function
async def my_custom_node(state, pipeline):
    # Process the state
    # ...
    return state

# Register the node with the plugin manager
plugin_manager.register_graph_node("basic", "my_custom_node", my_custom_node)
plugin_manager.register_graph_node("streaming", "my_custom_node", my_custom_node)
```

#### Modifying the Graph Structure

To modify the graph structure, you can create a custom graph factory function:

```python
from langgraph.graph import StateGraph, END
from root.src.core.graph import RAGState, plugin_manager

def build_custom_graph(config, pipeline=None):
    # Create the graph
    graph = StateGraph(RAGState)
    
    # Get registered nodes
    nodes = plugin_manager.get_graph_nodes("basic")
    
    # Add nodes
    graph.add_node("input_normalizer", lambda state: nodes["input_normalizer"](state))
    graph.add_node("my_custom_node", lambda state: nodes["my_custom_node"](state, pipeline))
    graph.add_node("retriever", lambda state: nodes["retriever"](state, pipeline))
    # ... add other nodes
    
    # Define edges
    graph.add_edge("input_normalizer", "my_custom_node")
    graph.add_edge("my_custom_node", "retriever")
    # ... add other edges
    
    # Set entry point
    graph.set_entry_point("input_normalizer")
    
    return graph.compile()
```

### Visualization with LangGraph Studio

LangGraph Studio provides a visual interface for debugging and monitoring your graph. To use it:

1. Start the LangGraph Server:
   ```bash
   python scripts/langgraph_server.py
   ```

2. Connect to LangGraph Studio through your browser:
   ```
   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
   ```

3. Select one of the available graphs and start testing with sample queries.

### Best Practices

1. **State Management**: Keep the state object clean and well-structured. Add new fields to `RAGState` as needed.
2. **Error Handling**: Implement proper error handling in your nodes to prevent the graph from crashing.
3. **Logging**: Use the logger to provide useful information about the execution of your nodes.
4. **Testing**: Write tests for your custom nodes and graph configurations.
5. **Documentation**: Document your custom nodes and graph configurations.

### Troubleshooting

#### Common Issues

1. **Graph not executing**: Ensure that all nodes are properly registered and that the graph is compiled.
2. **Node errors**: Check the logs for error messages from specific nodes.
3. **Streaming issues**: Ensure that the streaming graph is properly configured and that the client supports streaming responses.
4. **Visualization not working**: Check that the LangGraph Server is running and that the LangSmith API key is properly configured.

#### Debugging Tips

1. Use LangGraph Studio to visualize the graph execution and inspect the state at each node.
2. Add logging statements to your nodes to track the flow of data.
3. Use the `debug` mode in the LangGraph Server to get more detailed logs.
