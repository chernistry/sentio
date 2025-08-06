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

            # Normalize document content with fallback and detailed logging
            normalized_docs = []
            for i, doc in enumerate(docs):
                original_text = doc.text
                fallback_content = doc.metadata.get('content', '') if doc.metadata else ''
                
                # Apply fallback: use metadata.content if doc.text is empty
                if not doc.text and fallback_content:
                    doc.text = fallback_content
                    logger.info(f"Retriever - Doc {i}: Applied fallback from metadata.content")
                
                # Log diagnostic details (truncated for safety)
                logger.info(f"Retriever - Doc {i}: original_text='{original_text[:100]}...', "
                           f"fallback_content='{fallback_content[:100]}...', "
                           f"final_text='{doc.text[:100]}...', "
                           f"metadata_keys={list(doc.metadata.keys()) if doc.metadata else []}")
                
                normalized_docs.append(doc)

            # Update state
            state.add_retrieved_documents(normalized_docs)
            state.add_metadata("retriever_type", type(retriever).__name__)
            state.add_metadata("retrieved_count", len(normalized_docs))

            logger.info("Retrieved %d documents with content normalization", len(normalized_docs))
            
            # Log summary of content availability
            docs_with_text = sum(1 for doc in normalized_docs if doc.text.strip())
            docs_with_fallback = sum(1 for doc in normalized_docs if not doc.text.strip() and doc.metadata.get('content', '').strip())
            logger.info(f"Content summary: {docs_with_text} docs with text, {docs_with_fallback} docs needing fallback")
            
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
            # Apply content fallback and log incoming documents before reranking
            prepared_docs = []
            for i, doc in enumerate(state.retrieved_documents):
                original_text = doc.text
                fallback_content = doc.metadata.get('content', '') if doc.metadata else ''
                
                # Apply fallback: use metadata.content if doc.text is empty
                if not doc.text and fallback_content:
                    doc.text = fallback_content
                    logger.info(f"Reranker - Doc {i}: Applied fallback from metadata.content")
                
                # Log incoming document details (truncated for safety)
                logger.info(f"Reranker - Input Doc {i}: original_text='{original_text[:100]}...', "
                           f"fallback_content='{fallback_content[:100]}...', "
                           f"final_text='{doc.text[:100]}...', "
                           f"has_content={bool(doc.text.strip())}")
                
                prepared_docs.append(doc)

            # Rerank documents with prepared content
            reranked_docs = reranker.rerank(
                query=state.query,
                docs=prepared_docs,
                top_k=top_k,
            )

            # Log reranked results
            for i, doc in enumerate(reranked_docs):
                logger.info(f"Reranker - Output Doc {i}: text='{doc.text[:100]}...', "
                           f"score={doc.metadata.get('score', 'N/A')}")

            # Update state
            state.add_reranked_documents(reranked_docs)
            state.add_metadata("reranker_type", type(reranker).__name__)
            state.add_metadata("reranked_count", len(reranked_docs))

            logger.info("Reranked to %d documents", len(reranked_docs))
            
            # Log content availability after reranking
            docs_with_content = sum(1 for doc in reranked_docs if doc.text.strip())
            logger.info(f"Reranker output: {docs_with_content}/{len(reranked_docs)} docs have content")
            
        except Exception as e:
            logger.error("Error reranking documents: %s", e)
            state.add_metadata("reranker_error", str(e))
            # Fall back to retrieved documents with content normalization
            fallback_docs = []
            for doc in state.retrieved_documents[:top_k]:
                if not doc.text and doc.metadata.get('content'):
                    doc.text = doc.metadata['content']
                fallback_docs.append(doc)
            state.add_reranked_documents(fallback_docs)

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
        logger.info("Selected documents: %d", len(state.selected_documents))
        for i, doc in enumerate(state.selected_documents):
            logger.info(f"Selected doc {i}: text='{doc.text[:100]}...'")

        try:
            # Use our LLM generator
            return await generator.generate_for_state(state)
        except Exception as e:
            logger.error("Error generating response: %s", e)
            state.add_metadata("generator_error", str(e))
            state.set_response("I encountered an error while generating a response.")

        return state

    return generate_response_node
