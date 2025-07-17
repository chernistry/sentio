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
import asyncio

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

# Global pipeline instance and lock for thread-safe initialization
_pipeline_instance: Optional[SentioRAGPipeline] = None
_init_lock = asyncio.Lock()


async def get_pipeline() -> SentioRAGPipeline:
    """
    Get a singleton, initialized SentioRAGPipeline instance.
    This function ensures that the pipeline is initialized asynchronously
    and only once, handling concurrent access safely.
    """
    global _pipeline_instance
    if not (_pipeline_instance and _pipeline_instance.initialized):
        async with _init_lock:
            # Double-check inside the lock
            if not (_pipeline_instance and _pipeline_instance.initialized):
                logger.info("Initializing RAG pipeline for LangGraph server...")
                # Create instance if it doesn't exist or is not initialized
                if _pipeline_instance is None:
                    _pipeline_instance = SentioRAGPipeline()
                
                await _pipeline_instance.initialize()
                logger.info("RAG pipeline initialized successfully.")
    return _pipeline_instance


async def run_with_pipeline(node_func: Callable, state: "RAGState") -> "RAGState":
    """
    Wrapper that ensures the pipeline is initialized before executing a graph node.
    """
    pipeline = await get_pipeline()
    return await node_func(state, pipeline=pipeline)


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


async def retriever_node(state: RAGState, *, pipeline: SentioRAGPipeline) -> RAGState:
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


async def reranker_node(state: RAGState, *, pipeline: SentioRAGPipeline) -> RAGState:
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


async def generator_node(state: RAGState, *, pipeline: SentioRAGPipeline) -> RAGState:
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

def _build_graph_from_nodes(graph_type: str, config: PipelineConfig) -> StateGraph:
    """A unified factory to construct a graph from registered nodes."""
    graph = StateGraph(RAGState)
    registered_nodes = plugin_manager.get_graph_nodes(graph_type)

    nodes_requiring_pipeline = {
        "retriever", "reranker", "generator", "hyde_expander", "ragas_evaluator"
    }

    for name, node_func in registered_nodes.items():
        if name in nodes_requiring_pipeline:
            # Wrap node to ensure pipeline is initialized and passed
            graph.add_node(name, partial(run_with_pipeline, node_func))
        else:
            graph.add_node(name, node_func)

    # Define graph topology
    entry_point = "input_normalizer"
    graph.set_entry_point(entry_point)

    # Conditional HyDE path
    if HYDE_ENABLED:
        graph.add_edge("input_normalizer", "hyde_expander")
        graph.add_edge("hyde_expander", "retriever")
    else:
        graph.add_edge("input_normalizer", "retriever")

    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")
    graph.add_edge("generator", "post_processor")

    # Conditional RAGAS path
    if config.enable_automatic_evaluation:
        graph.add_edge("post_processor", "ragas_evaluator")
        graph.add_edge("ragas_evaluator", END)
    else:
        graph.add_edge("post_processor", END)

    logger.info(f"{graph_type.capitalize()} RAG graph built successfully")
    return graph


def build_basic_graph_for_server(config: Optional[RunnableConfig] = None) -> StateGraph:
    """
    Entrypoint for LangGraph server to build the basic RAG graph.
    The pipeline is initialized lazily on the first request.
    """
    graph = _build_graph_from_nodes("basic", settings)
    return graph.compile()


def build_streaming_graph_for_server(config: Optional[RunnableConfig] = None) -> StreamingWrapper:
    """
    Entrypoint for LangGraph server to build the streaming RAG graph.
    The pipeline is initialized lazily on the first request.
    """
    graph = _build_graph_from_nodes("streaming", settings)
    compiled_graph = graph.compile()
    return StreamingWrapper(compiled_graph)