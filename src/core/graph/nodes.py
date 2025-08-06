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

from src.core.graph.state import RAGState, add_retrieved_documents, add_metadata, add_reranked_documents, add_selected_documents, set_response
from src.core.models.document import Document
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
        logger.info("Retrieving documents for query: %s", state["query"])

        try:
            # Retrieve documents
            docs = retriever.retrieve(state["query"], top_k=top_k)

            # Create new document instances with proper text content
            normalized_docs = []
            for i, doc in enumerate(docs):
                # Determine the text to use
                text_to_use = doc.text
                if not text_to_use and doc.metadata and 'content' in doc.metadata:
                    text_to_use = doc.metadata['content']
                    logger.info(f"Retriever - Doc {i}: Using fallback content from metadata")
                
                # Create a new document instance with proper text
                normalized_doc = Document(
                    id=doc.id,
                    text=text_to_use,
                    metadata=doc.metadata.copy() if doc.metadata else {}
                )
                
                # Log diagnostic details (truncated for safety)
                logger.info(f"Retriever - Doc {i}: original_text='{doc.text[:100]}...', "
                           f"fallback_content='{doc.metadata.get('content', '')[:100] if doc.metadata else ''}...', "
                           f"final_text='{normalized_doc.text[:100]}...', "
                           f"metadata_keys={list(doc.metadata.keys()) if doc.metadata else []}")
                
                normalized_docs.append(normalized_doc)

            # Update state
            add_retrieved_documents(state, normalized_docs)
            add_metadata(state, "retriever_type", type(retriever).__name__)
            add_metadata(state, "retrieved_count", len(normalized_docs))

            logger.info("Retrieved %d documents with content normalization", len(normalized_docs))
            
            # Log summary of content availability
            docs_with_text = sum(1 for doc in normalized_docs if doc.text.strip())
            logger.info(f"Content summary: {docs_with_text}/{len(normalized_docs)} docs have content after normalization")
            
            # CRITICAL DEBUG: Verify state was updated correctly
            logger.info(f"CRITICAL DEBUG - State after retrieval: retrieved_docs={len(state['retrieved_documents'])}")
            logger.info(f"CRITICAL DEBUG - State object ID: {id(state)}")
            
            # Verify documents are actually in the state
            if len(state['retrieved_documents']) != len(normalized_docs):
                logger.error(f"CRITICAL ERROR - Document count mismatch! Expected {len(normalized_docs)}, got {len(state['retrieved_documents'])}")
            
            # Sample first document to verify content
            if state['retrieved_documents']:
                first_doc = state['retrieved_documents'][0]
                logger.info(f"CRITICAL DEBUG - First doc in state: text_length={len(first_doc.text)}, text_preview='{first_doc.text[:50]}...'")
            else:
                logger.error("CRITICAL ERROR - No documents in state after adding them!")
            
        except Exception as e:
            logger.error("Error retrieving documents: %s", e)
            add_metadata(state, "retriever_error", str(e))

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
        if not state["retrieved_documents"]:
            logger.warning("No documents to rerank")
            logger.warning(f"CRITICAL DEBUG - State object ID: {id(state)}")
            logger.warning(f"CRITICAL DEBUG - State type: {type(state)}")
            logger.warning(f"CRITICAL DEBUG - State dict keys: {list(state.keys())}")
            logger.warning(f"CRITICAL DEBUG - Retrieved docs type: {type(state['retrieved_documents'])}")
            logger.warning(f"CRITICAL DEBUG - Retrieved docs length: {len(state['retrieved_documents'])}")
            logger.warning(f"CRITICAL DEBUG - Query: {state['query']}")
            return state

        logger.info("Reranking %d documents", len(state["retrieved_documents"]))

        try:
            # Create new document instances with proper text content for reranking
            prepared_docs = []
            for i, doc in enumerate(state["retrieved_documents"]):
                # Determine the text to use
                text_to_use = doc.text
                if not text_to_use and doc.metadata and 'content' in doc.metadata:
                    text_to_use = doc.metadata['content']
                    logger.info(f"Reranker - Doc {i}: Using fallback content from metadata")
                
                # Create a new document instance with proper text
                prepared_doc = Document(
                    id=doc.id,
                    text=text_to_use,
                    metadata=doc.metadata.copy() if doc.metadata else {}
                )
                
                # Log document details for reranking process
                logger.info(f"Reranker - Input Doc {i}: original_text='{doc.text[:100]}...', "
                           f"fallback_content='{doc.metadata.get('content', '')[:100] if doc.metadata else ''}...', "
                           f"final_text='{prepared_doc.text[:100]}...', "
                           f"has_content={bool(prepared_doc.text.strip())}")
                
                prepared_docs.append(prepared_doc)

            # Rerank documents with prepared content
            reranked_docs = reranker.rerank(
                query=state["query"],
                docs=prepared_docs,
                top_k=top_k,
            )

            # Log reranked results
            for i, doc in enumerate(reranked_docs):
                logger.info(f"Reranker - Output Doc {i}: text='{doc.text[:100]}...', "
                           f"score={doc.metadata.get('score', 'N/A')}")

            # Update state
            add_reranked_documents(state, reranked_docs)
            add_metadata(state, "reranker_type", type(reranker).__name__)
            add_metadata(state, "reranked_count", len(reranked_docs))

            logger.info("Reranked to %d documents", len(reranked_docs))
            
            # Log content availability after reranking
            docs_with_content = sum(1 for doc in reranked_docs if doc.text.strip())
            logger.info(f"Reranker output: {docs_with_content}/{len(reranked_docs)} docs have content")
            
        except Exception as e:
            logger.error("Error reranking documents: %s", e)
            add_metadata(state, "reranker_error", str(e))
            # Fall back to retrieved documents with content normalization
            fallback_docs = []
            for doc in state["retrieved_documents"][:top_k]:
                text_to_use = doc.text
                if not text_to_use and doc.metadata and 'content' in doc.metadata:
                    text_to_use = doc.metadata['content']
                
                fallback_doc = Document(
                    id=doc.id,
                    text=text_to_use,
                    metadata=doc.metadata.copy() if doc.metadata else {}
                )
                fallback_docs.append(fallback_doc)
            
            add_reranked_documents(state, fallback_docs)

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
        candidate_docs = state["reranked_documents"] or state["retrieved_documents"]

        if not candidate_docs:
            logger.warning("No documents to select")
            logger.warning(f"CRITICAL DEBUG - State object ID: {id(state)}")
            logger.warning(f"CRITICAL DEBUG - Retrieved docs: {len(state['retrieved_documents'])}")
            logger.warning(f"CRITICAL DEBUG - Reranked docs: {len(state['reranked_documents'])}")
            logger.warning(f"CRITICAL DEBUG - Query: {state['query']}")
            return state

        logger.info("Selecting documents from %d candidates", len(candidate_docs))

        try:
            # Token counting and document selection with proper text content
            selected_docs = []
            total_tokens = 0

            for i, doc in enumerate(candidate_docs[:top_k]):
                # Determine the text to use for token counting and final selection
                text_to_use = doc.text
                if not text_to_use and doc.metadata and 'content' in doc.metadata:
                    text_to_use = doc.metadata['content']
                    logger.info(f"Selector - Doc {i}: Using fallback content from metadata")
                
                # Create a new document instance with proper text
                selected_doc = Document(
                    id=doc.id,
                    text=text_to_use,
                    metadata=doc.metadata.copy() if doc.metadata else {}
                )
                
                # Approximate token count (4 chars ≈ 1 token)
                doc_tokens = len(text_to_use) // 4

                logger.info(f"Selector - Doc {i}: original_text='{doc.text[:100]}...', "
                           f"fallback_content='{doc.metadata.get('content', '')[:100] if doc.metadata else ''}...', "
                           f"final_text='{selected_doc.text[:100]}...', "
                           f"text_length={len(text_to_use)}, "
                           f"estimated_tokens={doc_tokens}, "
                           f"current_total={total_tokens}")

                if total_tokens + doc_tokens <= max_tokens:
                    selected_docs.append(selected_doc)
                    total_tokens += doc_tokens
                    logger.info(f"Selector - Doc {i}: SELECTED (new_total={total_tokens})")
                else:
                    # If we can't fit the whole document, we're done
                    logger.info(f"Selector - Doc {i}: REJECTED (would exceed max_tokens={max_tokens})")
                    break

            # Update state
            add_selected_documents(state, selected_docs)
            add_metadata(state, "selected_count", len(selected_docs))
            add_metadata(state, "selected_tokens", total_tokens)

            logger.info("Selected %d documents (%d tokens)", len(selected_docs), total_tokens)
            
            # Log final selection summary
            docs_with_content = sum(1 for doc in selected_docs if doc.text.strip())
            logger.info(f"Selection summary: {docs_with_content}/{len(selected_docs)} selected docs have content")
            
            # Log each selected document's content snippet
            for i, doc in enumerate(selected_docs):
                logger.info(f"Selected Doc {i}: text='{doc.text[:100]}...', "
                           f"metadata_keys={list(doc.metadata.keys()) if doc.metadata else []}")
                           
        except Exception as e:
            logger.error("Error selecting documents: %s", e)
            add_metadata(state, "selector_error", str(e))
            # Fall back to top documents with content normalization
            fallback_docs = []
            for doc in candidate_docs[:min(top_k, len(candidate_docs))]:
                text_to_use = doc.text
                if not text_to_use and doc.metadata and 'content' in doc.metadata:
                    text_to_use = doc.metadata['content']
                
                fallback_doc = Document(
                    id=doc.id,
                    text=text_to_use,
                    metadata=doc.metadata.copy() if doc.metadata else {}
                )
                fallback_docs.append(fallback_doc)
            
            add_selected_documents(state, fallback_docs)

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
        if not state["selected_documents"]:
            logger.warning("No documents selected for generation")
            logger.warning(f"CRITICAL DEBUG - State object ID: {id(state)}")
            logger.warning(f"CRITICAL DEBUG - Retrieved docs: {len(state['retrieved_documents'])}")
            logger.warning(f"CRITICAL DEBUG - Reranked docs: {len(state['reranked_documents'])}")
            logger.warning(f"CRITICAL DEBUG - Selected docs: {len(state['selected_documents'])}")
            logger.warning(f"CRITICAL DEBUG - Query: {state['query']}")
            set_response(state, "I don't have enough information to answer that question.")
            return state

        logger.info("Generating response for query: %s", state["query"])
        logger.info("Selected documents: %d", len(state["selected_documents"]))
        for i, doc in enumerate(state["selected_documents"]):
            logger.info(f"Selected doc {i}: text='{doc.text[:100]}...'")

        try:
            # Use our LLM generator
            return await generator.generate_for_state(state)
        except Exception as e:
            logger.error("Error generating response: %s", e)
            add_metadata(state, "generator_error", str(e))
            set_response(state, "I encountered an error while generating a response.")

        return state

    return generate_response_node
