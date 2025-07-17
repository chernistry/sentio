#!/usr/bin/env python3
"""
HyDE Expansion Node for LangGraph.

This module provides a LangGraph node that implements Hypothetical Document Embeddings
(HyDE) query expansion to improve retrieval performance.
"""

import logging
import os
from typing import Dict, Any, Optional

# Import the existing HyDE implementation
from plugins.hyde_expander import expand_query_hyde, expand_query_hyde_async

logger = logging.getLogger(__name__)

# Check if HyDE is enabled from environment
HYDE_ENABLED = os.environ.get("ENABLE_HYDE", "0") == "1"


async def hyde_expansion_node(state, pipeline) -> Dict[str, Any]:
    """
    Expand the query using Hypothetical Document Embeddings (HyDE).
    
    This node generates a hypothetical document that answers the query and
    uses it to enhance retrieval performance.
    
    Args:
        state: The current graph state containing the query
        pipeline: The main pipeline instance
        
    Returns:
        Updated state with expanded query
    """
    # Skip if HyDE is not enabled
    if not HYDE_ENABLED:
        logger.debug("HyDE expansion disabled, skipping")
        return state
    
    query = state.query
    logger.info(f"Running HyDE expansion for query: {query}")
    
    try:
        # Use the async version if available in the pipeline
        if hasattr(pipeline, "hyde_expand"):
            hypothetical_doc = await pipeline.hyde_expand(query)
        else:
            # Fall back to synchronous version
            hypothetical_doc = expand_query_hyde(query)
        
        if hypothetical_doc:
            # Store the original query
            state.metadata["original_query"] = query
            
            # Create an expanded query that combines the original and the hypothetical document
            # This approach preserves the original query intent while adding context
            expanded_query = f"{query}\n\nContext: {hypothetical_doc}"
            
            # Update the normalized query with the expanded version
            state.normalized_query = expanded_query
            
            # Add metadata about the expansion
            state.metadata["query_expansion"] = "hyde"
            state.metadata["hyde_doc_length"] = len(hypothetical_doc)
            
            logger.info(f"HyDE expansion successful, expanded query length: {len(expanded_query)}")
        else:
            logger.warning("HyDE expansion failed, using original query")
            
    except Exception as e:
        logger.error(f"Error in HyDE expansion: {e}", exc_info=True)
        # Don't fail the pipeline on expansion error
        
    return state 