from __future__ import annotations

"""LangGraph pipeline nodes for RAG.

This module defines the basic nodes for a LangGraph-based RAG pipeline.
Each node is a function that takes a RAGState and returns an updated RAGState.
"""

import logging
from collections.abc import Callable
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.core.graph.state import RAGState
from src.core.rerankers.base import Reranker
from src.core.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)

# Type variable for node functions
T = TypeVar("T")
NodeFunction = Callable[[T], T]


def create_retriever_node(
    retriever: BaseRetriever,
    top_k: int = 10,
) -> NodeFunction[RAGState]:
    """Create a retriever node for the RAG pipeline.
    
    Args:
        retriever: The retriever to use
        top_k: Number of documents to retrieve
        
    Returns:
        A function that takes a RAGState and returns an updated RAGState
    """

    def retrieve_node(state: RAGState) -> RAGState:
        """Retrieve documents from the vector store.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with retrieved documents
        """
        logger.info("Retrieving documents for query: %s", state.query)

        try:
            # Retrieve documents
            docs = retriever.retrieve(state.query, top_k=top_k)

            # Update state
            state.add_retrieved_documents(docs)
            state.add_metadata("retriever_type", type(retriever).__name__)
            state.add_metadata("retrieved_count", len(docs))

            logger.info("Retrieved %d documents", len(docs))
        except Exception as e:
            logger.error("Error retrieving documents: %s", e)
            state.add_metadata("retriever_error", str(e))

        return state

    return retrieve_node


def create_reranker_node(
    reranker: Reranker,
    top_k: int = 5,
) -> NodeFunction[RAGState]:
    """Create a reranker node for the RAG pipeline.
    
    Args:
        reranker: The reranker to use
        top_k: Number of documents to return after reranking
        
    Returns:
        A function that takes a RAGState and returns an updated RAGState
    """

    def rerank_node(state: RAGState) -> RAGState:
        """Rerank retrieved documents.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with reranked documents
        """
        if not state.retrieved_documents:
            logger.warning("No documents to rerank")
            return state

        logger.info("Reranking %d documents", len(state.retrieved_documents))

        try:
            # Rerank documents
            reranked_docs = reranker.rerank(
                query=state.query,
                docs=state.retrieved_documents,
                top_k=top_k,
            )

            # Update state
            state.add_reranked_documents(reranked_docs)
            state.add_metadata("reranker_type", type(reranker).__name__)
            state.add_metadata("reranked_count", len(reranked_docs))

            logger.info("Reranked to %d documents", len(reranked_docs))
        except Exception as e:
            logger.error("Error reranking documents: %s", e)
            state.add_metadata("reranker_error", str(e))
            # Fall back to retrieved documents
            state.add_reranked_documents(state.retrieved_documents[:top_k])

        return state

    return rerank_node


def create_document_selector_node(
    top_k: int = 3,
    max_tokens: int = 2000,
) -> NodeFunction[RAGState]:
    """Create a document selector node for the RAG pipeline.
    
    This node selects documents from the reranked documents based on
    token count and other criteria.
    
    Args:
        top_k: Maximum number of documents to select
        max_tokens: Maximum total tokens across all selected documents
        
    Returns:
        A function that takes a RAGState and returns an updated RAGState
    """

    def select_documents_node(state: RAGState) -> RAGState:
        """Select documents for context generation.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with selected documents
        """
        # Use reranked documents if available, otherwise use retrieved documents
        candidate_docs = state.reranked_documents or state.retrieved_documents

        if not candidate_docs:
            logger.warning("No documents to select")
            return state

        logger.info("Selecting documents from %d candidates", len(candidate_docs))

        try:
            # Simple token counting (approximate)
            selected_docs = []
            total_tokens = 0

            for doc in candidate_docs[:top_k]:
                # Approximate token count (4 chars ≈ 1 token)
                doc_tokens = len(doc.text) // 4

                if total_tokens + doc_tokens <= max_tokens:
                    selected_docs.append(doc)
                    total_tokens += doc_tokens
                else:
                    # If we can't fit the whole document, we're done
                    break

            # Update state
            state.add_selected_documents(selected_docs)
            state.add_metadata("selected_count", len(selected_docs))
            state.add_metadata("selected_tokens", total_tokens)

            logger.info("Selected %d documents (%d tokens)", len(selected_docs), total_tokens)
        except Exception as e:
            logger.error("Error selecting documents: %s", e)
            state.add_metadata("selector_error", str(e))
            # Fall back to top documents
            state.add_selected_documents(candidate_docs[:min(top_k, len(candidate_docs))])

        return state

    return select_documents_node


def create_generator_node(
    llm: BaseChatModel | None = None,
    prompt_template: ChatPromptTemplate | None = None,
    mode: str = "balanced",
    max_tokens: int = 1024,
) -> NodeFunction[RAGState]:
    """Create a generator node for the RAG pipeline.
    
    Args:
        llm: The language model to use (legacy LangChain support)
        prompt_template: Optional custom prompt template (legacy LangChain support)
        mode: Generation mode (fast, balanced, quality, creative)
        max_tokens: Maximum tokens to generate
        
    Returns:
        A function that takes a RAGState and returns an updated RAGState
    """
    # Import here to avoid circular imports
    from src.core.llm.factory import create_generator

    # Create LLM generator
    generator = create_generator(mode=mode, max_tokens=max_tokens)

    async def generate_response_node(state: RAGState) -> RAGState:
        """Generate a response based on the selected documents.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with generated response
        """
        if not state.selected_documents:
            logger.warning("No documents selected for generation")
            state.set_response("I don't have enough information to answer that question.")
            return state

        logger.info("Generating response for query: %s", state.query)

        try:
            # Use our LLM generator
            return await generator.generate_for_state(state)
        except Exception as e:
            logger.error("Error generating response: %s", e)
            state.add_metadata("generator_error", str(e))
            state.set_response("I encountered an error while generating a response.")

        return state

    return generate_response_node
