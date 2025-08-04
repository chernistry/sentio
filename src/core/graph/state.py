from __future__ import annotations

"""State definitions for LangGraph pipelines."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from src.core.models.document import Document


class RAGState(BaseModel):
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
    """

    # Input
    query: str = Field(
        default="",
        description="User query string"
    )

    # Retrieval stage
    retrieved_documents: list[Document] = Field(
        default_factory=list,
        description="Documents retrieved from the vector store",
    )

    # Reranking stage
    reranked_documents: list[Document] = Field(
        default_factory=list,
        description="Documents after reranking",
    )

    # Selection stage
    selected_documents: list[Document] = Field(
        default_factory=list,
        description="Final documents selected for generation",
    )

    # Generation stage
    response: str = Field(
        default="",
        description="Generated response text",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the RAG process",
    )

    # Evaluation metrics
    evaluation: dict[str, float] = Field(
        default_factory=dict,
        description="Evaluation metrics for the RAG process",
    )

    def add_retrieved_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to the retrieved documents list.
        
        Args:
            documents: Documents to add
        """
        self.retrieved_documents.extend(documents)

    def add_reranked_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to the reranked documents list.
        
        Args:
            documents: Documents to add
        """
        self.reranked_documents.extend(documents)

    def add_selected_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to the selected documents list.
        
        Args:
            documents: Documents to add
        """
        self.selected_documents.extend(documents)

    def set_response(self, response: str) -> None:
        """Set the generated response.
        
        Args:
            response: Generated response text
        """
        self.response = response

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the state.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def add_evaluation_metric(self, key: str, value: float) -> None:
        """Add evaluation metric to the state.
        
        Args:
            key: Metric name
            value: Metric value
        """
        self.evaluation[key] = value
