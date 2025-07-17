#!/usr/bin/env python3
"""
LangGraph Factory for RAG Pipeline.

This module provides factory methods to build LangGraph nodes and graphs that
replicate the functionality of the existing SentioRAGPipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Callable
import inspect
from functools import partial

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from root.src.core.pipeline import PipelineConfig, RetrievalStrategy, GenerationMode, SentioRAGPipeline
from root.src.core.plugin_manager import PluginManager
from root.src.utils.settings import settings
from .ragas_node import ragas_evaluation_node
from .streaming import stream_generator_node, StreamingWrapper
from .hyde_node import hyde_expansion_node


# Configure logging
logger = logging.getLogger(__name__)

# Check if HyDE is enabled from environment
HYDE_ENABLED = os.environ.get("ENABLE_HYDE", "0") == "1"

# Initialize plugin manager
plugin_manager = PluginManager()


# ==== STATE DEFINITIONS ==== #

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


# ==== NODE DEFINITIONS ==== #

def input_normalizer_node(state: RAGState) -> RAGState:
    """
    Normalize the input query.
    
    This node preprocesses the input query to enhance retrieval quality.
    """
    logger.debug(f"Normalizing query: {state.query}")
    
    # Simple normalization (can be expanded with more preprocessing steps)
    normalized = state.query.strip()
    
    # Update state
    state.normalized_query = normalized
    return state


async def retriever_node(state: RAGState, *, pipeline) -> RAGState:
    """
    Retrieve relevant documents for the query.
    
    This node performs document retrieval using the existing pipeline retriever.
    """
    query = state.normalized_query or state.query
    logger.debug(f"Retrieving documents for query: {query}")
    
    # Use the existing pipeline retrieval functionality
    retrieval_result = await pipeline.retrieve(
        query, 
        top_k=pipeline.config.top_k_retrieval
    )
    
    # Update state
    state.retrieved_documents = retrieval_result.documents
    state.metadata["retrieval_strategy"] = retrieval_result.strategy
    state.metadata["retrieval_time"] = retrieval_result.total_time
    state.metadata["sources_found"] = retrieval_result.sources_found
    
    return state


async def reranker_node(state: RAGState, *, pipeline) -> RAGState:
    """
    Rerank retrieved documents.
    
    This node uses the existing pipeline reranker to improve document ordering.
    """
    query = state.normalized_query or state.query
    logger.debug(f"Reranking {len(state.retrieved_documents)} documents")
    
    # Use the existing pipeline reranking functionality
    reranked_docs = await pipeline.rerank(
        query, 
        state.retrieved_documents,
        top_k=pipeline.config.top_k_final
    )
    
    # Update state
    state.reranked_documents = reranked_docs
    state.sources = reranked_docs
    state.metadata["sources_used"] = len(reranked_docs)
    
    return state


async def generator_node(state: RAGState, *, pipeline) -> RAGState:
    """
    Generate an answer based on the query and context.
    
    This node uses the existing pipeline generator for answer generation.
    """
    query = state.normalized_query or state.query
    logger.debug(f"Generating answer for query: {query}")
    
    # Format context string from reranked documents
    context_docs = state.reranked_documents or state.retrieved_documents
    
    # Use the existing pipeline generation functionality
    generation_result = await pipeline.generate(
        query,
        context_docs,
        mode=pipeline.config.generation_mode
    )
    
    # Update state
    state.answer = generation_result.answer
    state.metadata["generation_time"] = generation_result.total_time
    state.metadata["generation_mode"] = generation_result.mode
    state.metadata["token_count"] = generation_result.token_count
    state.metadata["timestamp"] = generation_result.timestamp
    
    return state


def post_processor_node(state: RAGState) -> RAGState:
    """
    Apply post-processing to the generated answer.
    
    This node applies any necessary post-processing to the answer before returning.
    """
    if not state.answer:
        return state
        
    # For now, simple cleanup (can be expanded with more post-processing)
    state.answer = state.answer.strip()
    
    # Calculate total query time
    if "retrieval_time" in state.metadata and "generation_time" in state.metadata:
        state.metadata["query_time"] = (
            state.metadata.get("retrieval_time", 0) + 
            state.metadata.get("generation_time", 0)
        )
    
    return state


# Register built-in nodes with the plugin manager
def register_builtin_nodes():
    """Register built-in nodes with the plugin manager."""
    plugin_manager.register_graph_node("basic", "input_normalizer", input_normalizer_node)
    plugin_manager.register_graph_node("basic", "retriever", retriever_node)
    plugin_manager.register_graph_node("basic", "reranker", reranker_node)
    plugin_manager.register_graph_node("basic", "generator", generator_node)
    plugin_manager.register_graph_node("basic", "post_processor", post_processor_node)
    plugin_manager.register_graph_node("basic", "hyde_expander", hyde_expansion_node)
    plugin_manager.register_graph_node("basic", "ragas_evaluator", ragas_evaluation_node)
    
    plugin_manager.register_graph_node("streaming", "input_normalizer", input_normalizer_node)
    plugin_manager.register_graph_node("streaming", "retriever", retriever_node)
    plugin_manager.register_graph_node("streaming", "reranker", reranker_node)
    plugin_manager.register_graph_node("streaming", "generator", stream_generator_node)
    plugin_manager.register_graph_node("streaming", "post_processor", post_processor_node)
    plugin_manager.register_graph_node("streaming", "hyde_expander", hyde_expansion_node)
    plugin_manager.register_graph_node("streaming", "ragas_evaluator", ragas_evaluation_node)


# Register built-in nodes on module import
register_builtin_nodes()


# ==== GRAPH FACTORY ==== #

def build_basic_graph(settings: PipelineConfig, pipeline = None) -> StateGraph:
    """
    Build a basic linear RAG graph.
    
    This factory method constructs a simple linear LangGraph that replicates
    the existing Pipeline functionality.
    
    Args:
        settings: Pipeline configuration.
        pipeline: Optional existing pipeline instance for component reuse.
        
    Returns:
        StateGraph: Configured LangGraph.
    """
    # Create the graph with the defined state schema
    graph = StateGraph(RAGState)
    
    # Get all registered nodes for the basic graph type
    registered_nodes = plugin_manager.get_graph_nodes("basic")
    
    # Add all nodes to the graph
    graph.add_node("input_normalizer", registered_nodes["input_normalizer"])
    
    # Add HyDE expansion node if enabled
    if HYDE_ENABLED and pipeline is not None:
        logger.info("Adding HyDE expansion node to the graph")
        node_func = registered_nodes["hyde_expander"]
        graph.add_node("hyde_expander", partial(node_func, pipeline=pipeline))
    
    node_func = registered_nodes["retriever"]
    graph.add_node("retriever", partial(node_func, pipeline=pipeline))
    
    node_func = registered_nodes["reranker"]
    graph.add_node("reranker", partial(node_func, pipeline=pipeline))

    node_func = registered_nodes["generator"]
    graph.add_node("generator", partial(node_func, pipeline=pipeline))

    graph.add_node("post_processor", registered_nodes["post_processor"])
    
    # Add RAGAS evaluation node if automatic evaluation is enabled
    if settings.enable_automatic_evaluation and pipeline is not None:
        logger.info("Adding RAGAS evaluation node to the graph")
        node_func = registered_nodes["ragas_evaluator"]
        graph.add_node("ragas_evaluator", partial(node_func, pipeline=pipeline))
    
    # Add custom nodes from plugins
    for node_name, node_func in registered_nodes.items():
        if node_name not in [
            "input_normalizer", "retriever", "reranker", 
            "generator", "post_processor", "hyde_expander", 
            "ragas_evaluator"
        ]:
            logger.info(f"Adding custom node {node_name} to the graph")
            sig = inspect.signature(node_func)
            if 'pipeline' in sig.parameters:
                graph.add_node(node_name, partial(node_func, pipeline=pipeline))
            else:
                graph.add_node(node_name, node_func)

    # Define the edges in the graph
    graph.add_edge("input_normalizer", "hyde_expander" if HYDE_ENABLED and pipeline is not None else "retriever")
    
    # Add HyDE edge if enabled
    if HYDE_ENABLED and pipeline is not None:
        graph.add_edge("hyde_expander", "retriever")
    
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")
    graph.add_edge("generator", "post_processor")
    
    # Add RAGAS edge if enabled
    if settings.enable_automatic_evaluation and pipeline is not None:
        graph.add_edge("post_processor", "ragas_evaluator")
        graph.add_edge("ragas_evaluator", END)
    else:
        graph.add_edge("post_processor", END)
    
    # Set the entry point
    graph.set_entry_point("input_normalizer")
    
    logger.info("RAG graph built successfully")
    return graph.compile()


def build_streaming_graph(settings: PipelineConfig, pipeline = None) -> StreamingWrapper:
    """
    Build a streaming-capable RAG graph.
    
    This factory method constructs a LangGraph that supports streaming responses,
    with the generator node adapted for token-by-token streaming.
    
    Args:
        settings: Pipeline configuration.
        pipeline: Optional existing pipeline instance for component reuse.
        
    Returns:
        StreamingWrapper: Wrapped graph with streaming capabilities.
    """
    # Create the graph with the defined state schema
    graph = StateGraph(RAGState)
    
    # Get all registered nodes for the streaming graph type
    registered_nodes = plugin_manager.get_graph_nodes("streaming")
    
    # Add all nodes to the graph
    graph.add_node("input_normalizer", registered_nodes["input_normalizer"])
    
    # Add HyDE expansion node if enabled
    if HYDE_ENABLED and pipeline is not None:
        logger.info("Adding HyDE expansion node to the streaming graph")
        node_func = registered_nodes["hyde_expander"]
        graph.add_node("hyde_expander", partial(node_func, pipeline=pipeline))
    
    node_func = registered_nodes["retriever"]
    graph.add_node("retriever", partial(node_func, pipeline=pipeline))

    node_func = registered_nodes["reranker"]
    graph.add_node("reranker", partial(node_func, pipeline=pipeline))
    
    # Use the streaming generator node
    node_func = registered_nodes["generator"]
    graph.add_node("generator", partial(node_func, pipeline=pipeline))
    
    graph.add_node("post_processor", registered_nodes["post_processor"])
    
    # Add RAGAS evaluation node if automatic evaluation is enabled
    if settings.enable_automatic_evaluation and pipeline is not None:
        logger.info("Adding RAGAS evaluation node to the streaming graph")
        node_func = registered_nodes["ragas_evaluator"]
        graph.add_node("ragas_evaluator", partial(node_func, pipeline=pipeline))
    
    # Add custom nodes from plugins
    for node_name, node_func in registered_nodes.items():
        if node_name not in [
            "input_normalizer", "retriever", "reranker", 
            "generator", "post_processor", "hyde_expander", 
            "ragas_evaluator"
        ]:
            logger.info(f"Adding custom node {node_name} to the streaming graph")
            sig = inspect.signature(node_func)
            if 'pipeline' in sig.parameters:
                graph.add_node(node_name, partial(node_func, pipeline=pipeline))
            else:
                graph.add_node(node_name, node_func)
    
    # Define the edges in the graph
    graph.add_edge("input_normalizer", "hyde_expander" if HYDE_ENABLED and pipeline is not None else "retriever")
    
    # Add HyDE edge if enabled
    if HYDE_ENABLED and pipeline is not None:
        graph.add_edge("hyde_expander", "retriever")
    
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")
    graph.add_edge("generator", "post_processor")
    
    # Add RAGAS edge if enabled
    if settings.enable_automatic_evaluation and pipeline is not None:
        graph.add_edge("post_processor", "ragas_evaluator")
        graph.add_edge("ragas_evaluator", END)
    else:
        graph.add_edge("post_processor", END)
    
    # Set the entry point
    graph.set_entry_point("input_normalizer")
    
    logger.info("Streaming RAG graph built successfully")
    compiled_graph = graph.compile()
    
    # Wrap the graph with the streaming wrapper
    return StreamingWrapper(compiled_graph)


# ==== SERVER ENTRYPOINTS ==== #

def build_basic_graph_for_server(config: Optional[RunnableConfig] = None) -> StateGraph:
    """
    Entrypoint for LangGraph server to build the basic RAG graph.
    
    This function initializes the pipeline and settings required by the
    graph factory and is compatible with `langgraph dev`.
    """
    pipeline_instance = SentioRAGPipeline()
    return build_basic_graph(settings, pipeline_instance)


def build_streaming_graph_for_server(config: Optional[RunnableConfig] = None) -> StreamingWrapper:
    """
    Entrypoint for LangGraph server to build the streaming RAG graph.
    
    This function initializes the pipeline and settings required by the
    graph factory and is compatible with `langgraph dev`.
    """
    pipeline_instance = SentioRAGPipeline()
    return build_streaming_graph(settings, pipeline_instance) 