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
        for node in config.nodes.values():
            if node.name == stream_node:
                streaming_node = node
                break
        
        if streaming_node is None:
            logger.error(f"Node '{stream_node}' not found in graph")
            raise ValueError(f"Node '{stream_node}' not found in graph")
        
        # Create a queue to receive streamed chunks
        queue = asyncio.Queue()
        
        # Create a wrapper around the original node function
        original_runnable = streaming_node.runnable
        
        async def wrapped_runnable(state, *args, **kwargs):
            """Wrapper around the original node function to capture chunks."""
            # Call the original function which should return an AsyncIterator
            if hasattr(original_runnable, "astream"):
                # If the function has streaming capability
                async for chunk in original_runnable.astream(state, *args, **kwargs):
                    # Put the chunk in the queue
                    await queue.put(chunk)
                    
                # After streaming, return the final state
                return state
            else:
                # If the function doesn't have streaming capability, call it normally
                final_state = await original_runnable.ainvoke(state, *args, **kwargs)
                
                # Put the final state in the queue
                await queue.put({chunk_field: getattr(final_state, chunk_field, "")})
                
                return final_state
        
        # Replace the original function with our wrapper
        streaming_node.runnable = wrapped_runnable
        
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
            streaming_node.runnable = original_runnable
            
            # Cancel the graph task if it's still running
            if not graph_task.done():
                graph_task.cancel()
            
            # Wait for the task to finish if needed
            try:
                await graph_task
            except asyncio.CancelledError:
                pass


async def stream_generator_node(state, *, pipeline):
    """Non-streaming fallback for the generator node.

    This function is invoked by LangGraph during regular graph execution and **must**
    return the final state object so that downstream nodes receive the updated
    answer and metadata.  A dedicated async generator (attached via the
    ``.astream`` attribute) handles incremental token streaming for the wrapper
    in :class:`StreamingWrapper`.
    """

    query: str = state.normalized_query or state.query
    logger.debug("Generating answer (non-streaming) for query: %s", query)

    # Compose context from already-retrieved or reranked documents.
    context_docs = state.reranked_documents or state.retrieved_documents

    generation_result = await pipeline.generate(
        query,
        context_docs,
        mode=pipeline.config.generation_mode,
    )

    # Persist answer and metadata on the state object.
    state.answer = generation_result.answer
    state.metadata.update(
        {
            "generation_time": generation_result.total_time,
            "generation_mode": generation_result.mode,
            "token_count": generation_result.token_count,
            "timestamp": generation_result.timestamp,
        }
    )

    return state


# === Streaming companion =====================================================

async def _stream_generator_node_astream(state, *, pipeline):
    """Async generator that yields answer chunks for real-time streaming.

    The implementation mirrors :func:`stream_generator_node` but emits tokens as
    soon as they are produced by ``pipeline.generate_stream`` (if available). If
    the pipeline lacks native streaming support we gracefully degrade to the
    non-streaming ``pipeline.generate`` method and yield the full answer once.
    """

    query: str = state.normalized_query or state.query
    logger.debug("Generating answer (streaming) for query: %s", query)

    context_docs = state.reranked_documents or state.retrieved_documents

    if hasattr(pipeline, "generate_stream"):
        async for token in pipeline.generate_stream(
            query,
            context_docs,
            mode=pipeline.config.generation_mode,
        ):
            # Send incremental chunk to the StreamingWrapper.
            yield {"answer": token}

            # Accumulate token into state for downstream consumers.
            state.answer = (state.answer or "") + token
    else:
        # Fallback to single-shot generation when streaming is unavailable.
        generation_result = await pipeline.generate(
            query,
            context_docs,
            mode=pipeline.config.generation_mode,
        )

        state.answer = generation_result.answer
        yield {"answer": generation_result.answer}

    # NOTE: Metadata is populated by the non-streaming function or by downstream
    # post-processing nodes, so we do not duplicate that work here.


# Expose the async generator for the StreamingWrapper to detect.
stream_generator_node.astream = _stream_generator_node_astream 