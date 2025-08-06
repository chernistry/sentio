from __future__ import annotations

"""State definitions for LangGraph pipelines."""

from typing import Any, TypedDict

from src.core.models.document import Document


class RAGState(TypedDict):
    """State for the RAG pipeline.
    
    This class represents the state that flows through the RAG pipeline nodes.
    It contains the query, retrieved documents, reranked documents, generated
    response, and other metadata.
    
    Attributes:
        query: The user's query string
        retrieved_documents: Documents retrieved from the vector store
        reranked_documents: Documents after reranking
        selected_documents: Final documents selected for generation
        response: Generated response text
        metadata: Additional metadata about the RAG process
        evaluation: Evaluation metrics for the RAG process
    """

    query: str
    retrieved_documents: list[Document]
    reranked_documents: list[Document]
    selected_documents: list[Document]
    response: str
    metadata: dict[str, Any]
    evaluation: dict[str, float]


def create_initial_state(query: str) -> RAGState:
    """Create an initial RAG state with default values.
    
    Args:
        query: The user's query string
        
    Returns:
        Initial RAG state
    """
    return RAGState(
        query=query,
        retrieved_documents=[],
        reranked_documents=[],
        selected_documents=[],
        response="",
        metadata={},
        evaluation={}
    )


def add_retrieved_documents(state: RAGState, documents: list[Document]) -> RAGState:
    """Add documents to the retrieved documents list.
    
    Args:
        state: Current RAG state
        documents: Documents to add
        
    Returns:
        Updated RAG state
    """
    state["retrieved_documents"].extend(documents)
    return state


def add_reranked_documents(state: RAGState, documents: list[Document]) -> RAGState:
    """Add documents to the reranked documents list.
    
    Args:
        state: Current RAG state
        documents: Documents to add
        
    Returns:
        Updated RAG state
    """
    state["reranked_documents"].extend(documents)
    return state


def add_selected_documents(state: RAGState, documents: list[Document]) -> RAGState:
    """Add documents to the selected documents list.
    
    Args:
        state: Current RAG state
        documents: Documents to add
        
    Returns:
        Updated RAG state
    """
    state["selected_documents"].extend(documents)
    return state


def set_response(state: RAGState, response: str) -> RAGState:
    """Set the generated response.
    
    Args:
        state: Current RAG state
        response: Generated response text
        
    Returns:
        Updated RAG state
    """
    state["response"] = response
    return state


def add_metadata(state: RAGState, key: str, value: Any) -> RAGState:
    """Add metadata to the state.
    
    Args:
        state: Current RAG state
        key: Metadata key
        value: Metadata value
        
    Returns:
        Updated RAG state
    """
    state["metadata"][key] = value
    return state


def add_evaluation_metric(state: RAGState, key: str, value: float) -> RAGState:
    """Add evaluation metric to the state.
    
    Args:
        state: Current RAG state
        key: Metric name
        value: Metric value
        
    Returns:
        Updated RAG state
    """
    state["evaluation"][key] = value
    return state
