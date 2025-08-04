"""Document ingestion module for Sentio RAG system.

This module provides functionality to ingest documents into the vector database
for retrieval in the RAG pipeline.
"""

from src.core.ingest.ingest import DocumentIngestor, ingest_directory

__all__ = ["DocumentIngestor", "ingest_directory"]
