#!/usr/bin/env python
"""
Script to run the LangGraph RAG pipeline locally.

This script demonstrates how to create and use the LangGraph RAG pipeline.
It sets up the necessary components, builds the graph, and runs a sample query.
"""

import argparse
import logging
import sys
import os
from typing import Optional, Dict, Any

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.graph import build_basic_graph
from src.core.graph.state import RAGState
from src.core.retrievers import get_retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def setup_langgraph_studio(port: int = 8001) -> None:
    """
    Setup and launch LangGraph Studio for visualization.
    
    Args:
        port: Port number to run the Studio server on
    """
    try:
        from langgraph.ui import create_ui
        create_ui()
        logger.info(f"LangGraph Studio should be available at http://localhost:{port}")
    except ImportError:
        logger.warning("LangGraph UI not available. Install with 'pip install langserve'.")


def run_rag_graph(query: str, config: Optional[Dict[str, Any]] = None) -> RAGState:
    """
    Run the RAG graph with the given query.
    
    Args:
        query: User query to process
        config: Optional configuration for the graph
        
    Returns:
        Final RAG state with response
    """
    config = config or {}
    
    # Set mock mode for local execution
    if 'mode' not in config:
        config['mode'] = 'mock'
    
    # Build the graph
    graph = build_basic_graph(config=config)
    logger.info("Created LangGraph RAG pipeline")
    
    # Create initial state
    state = RAGState(query=query)
    
    # Run the graph
    logger.info("Running graph with query: %s", query)
    raw_result = graph.invoke(state)
    
    # Convert result to RAGState if it's a dictionary
    if isinstance(raw_result, dict):
        result = RAGState(query=query)
        for key, value in raw_result.items():
            setattr(result, key, value)
    else:
        result = raw_result
    
    return result


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run LangGraph RAG pipeline")
    parser.add_argument("--query", type=str, default="What is LangGraph?", 
                        help="Query to process")
    parser.add_argument("--studio", action="store_true", 
                        help="Launch LangGraph Studio")
    parser.add_argument("--studio-port", type=int, default=8001,
                        help="Port for LangGraph Studio")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Run in mock mode without real retrievers")
    args = parser.parse_args()
    
    if args.studio:
        setup_langgraph_studio(port=args.studio_port)
    
    # Run the graph
    config = {'mode': 'mock'} if args.mock else {}
    result = run_rag_graph(args.query, config=config)
    
    # Print results
    print("\n" + "=" * 40)
    print(f"Query: {result.query}")
    print("-" * 40)
    print(f"Retrieved: {len(result.retrieved_documents)} docs")
    print(f"Reranked: {len(result.reranked_documents)} docs")
    print("-" * 40)
    print(f"Response: {result.response}")
    print("=" * 40)


if __name__ == "__main__":
    main() 