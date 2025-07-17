#!/usr/bin/env python3
"""
Streaming Support for LangGraph.

This module provides utilities to support streaming responses in LangGraph nodes,
particularly for the generator node to stream tokens incrementally.
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import logging

from langgraph.graph import StateGraph

logger = logging.getLogger(__name__)


class StreamingWrapper:
    """
    Wrapper around a LangGraph to enable streaming of generated tokens.
    
    This wrapper allows the generator node to emit incremental tokens while
    preserving the final state for subsequent nodes in the graph.
    """
    
    def __init__(self, graph: StateGraph):
        """
        Initialize the streaming wrapper.
        
        Args:
            graph: The compiled LangGraph to wrap
        """
        self.graph = graph
    
    async def astream(
        self,
        inputs: Dict[str, Any],
        stream_node: str = "generator",
        chunk_field: str = "answer"
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream the response from a specified node in the graph.
        
        Args:
            inputs: The inputs to pass to the graph
            stream_node: The name of the node to stream from
            chunk_field: The field in the state to stream
            
        Yields:
            Incremental tokens from the specified node
        """
        # We'll use internal APIs from langgraph to get a reference to the generator node
        # This is a bit of a hack, but allows us to intercept the output
        config = self.graph.graph
        
        # Find the node with the specified name
        streaming_node = None
        for node in config.nodes:
            if node.config["display_name"] == stream_node:
                streaming_node = node
                break
        
        if streaming_node is None:
            logger.error(f"Node '{stream_node}' not found in graph")
            raise ValueError(f"Node '{stream_node}' not found in graph")
        
        # Create a queue to receive streamed chunks
        queue = asyncio.Queue()
        
        # Create a wrapper around the original node function
        original_func = streaming_node.config["func"]
        
        async def wrapped_func(state, *args, **kwargs):
            """Wrapper around the original node function to capture chunks."""
            # Call the original function which should return an AsyncIterator
            if hasattr(original_func, "astream"):
                # If the function has streaming capability
                async for chunk in original_func.astream(state, *args, **kwargs):
                    # Put the chunk in the queue
                    await queue.put(chunk)
                    
                # After streaming, return the final state
                return state
            else:
                # If the function doesn't have streaming capability, call it normally
                final_state = await original_func(state, *args, **kwargs)
                
                # Put the final state in the queue
                await queue.put({chunk_field: getattr(final_state, chunk_field, "")})
                
                return final_state
        
        # Replace the original function with our wrapper
        streaming_node.config["func"] = wrapped_func
        
        # Start the graph execution in a separate task
        graph_task = asyncio.create_task(self.graph.ainvoke(inputs))
        
        try:
            # Stream chunks from the queue until the graph task completes
            while True:
                # Use asyncio.wait to wait for either queue item or graph completion
                done, pending = await asyncio.wait(
                    [graph_task, asyncio.create_task(queue.get())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Check if the graph task completed
                if graph_task in done:
                    # Graph completed, yield any remaining chunks
                    while not queue.empty():
                        chunk = await queue.get()
                        yield chunk
                    break
                
                # Get the chunk from the queue
                for task in done:
                    if task != graph_task:
                        chunk = task.result()
                        yield chunk
        finally:
            # Restore the original function
            streaming_node.config["func"] = original_func
            
            # Cancel the graph task if it's still running
            if not graph_task.done():
                graph_task.cancel()
            
            # Wait for the task to finish if needed
            try:
                await graph_task
            except asyncio.CancelledError:
                pass


async def stream_generator_node(state, pipeline):
    """
    Streaming version of the generator node.
    
    This is a drop-in replacement for the regular generator_node that supports
    streaming token output.
    
    Args:
        state: The current graph state
        pipeline: The pipeline instance with generation capabilities
        
    Yields:
        Incremental tokens as they're generated
        
    Returns:
        Final state with complete answer
    """
    query = state.normalized_query or state.query
    logger.debug(f"Generating answer (streaming) for query: {query}")
    
    # Format context string from reranked documents
    context_docs = state.reranked_documents or state.retrieved_documents
    
    # Check if pipeline has streaming capability
    if hasattr(pipeline, "generate_stream"):
        # Use the streaming generation method
        async for token in pipeline.generate_stream(
            query,
            context_docs,
            mode=pipeline.config.generation_mode
        ):
            # Yield each token as it arrives
            yield {"answer": token}
            
            # Update the state incrementally
            if not hasattr(state, "answer") or state.answer is None:
                state.answer = token
            else:
                state.answer += token
    else:
        # Fall back to non-streaming generation
        generation_result = await pipeline.generate(
            query,
            context_docs,
            mode=pipeline.config.generation_mode
        )
        
        # Update state
        state.answer = generation_result.answer
        
        # Yield the full answer at once
        yield {"answer": state.answer}
    
    # Update metadata at the end
    state.metadata["generation_time"] = generation_result.total_time if 'generation_result' in locals() else 0
    state.metadata["generation_mode"] = generation_result.mode if 'generation_result' in locals() else pipeline.config.generation_mode
    state.metadata["token_count"] = generation_result.token_count if 'generation_result' in locals() else len(state.answer.split())
    state.metadata["timestamp"] = generation_result.timestamp if 'generation_result' in locals() else ""
    
    return state

# Attach streaming method to the node function
stream_generator_node.astream = stream_generator_node 