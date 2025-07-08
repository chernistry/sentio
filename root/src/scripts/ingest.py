#!/usr/bin/env python3
"""
Enhanced document ingestion script for Sentio RAG system.

This script provides a robust, configurable interface for ingesting documents
into the Qdrant vector database with proper error handling and logging.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Rich-based logging for polished CLI output
try:
    from rich.logging import RichHandler
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
except ImportError:  # Fallback if Rich not installed
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.core.chunking import TextChunker
from src.core.embeddings import EmbeddingModel


# Re-obtain logger after handlers are configured
logger = logging.getLogger(__name__)


class DocumentIngestor:
    """High-performance document ingestion with error handling and monitoring."""
    
    def __init__(
        self, 
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "Sentio_docs_v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.client: Optional[QdrantClient] = None
        self.embed_model: Optional[EmbeddingModel] = None
        self.chunker: Optional[TextChunker] = None
        
    async def initialize(self) -> None:
        """Initialize all components with proper error handling."""
        try:
            logger.info("Initializing document ingestor...")
            
            # Initialize embedding model
            self.embed_model = EmbeddingModel()
            logger.info("✓ Embedding model initialized")
            
            # Initialize text chunker
            self.chunker = TextChunker(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            logger.info("✓ Text chunker initialized")
            
            # Initialize Qdrant client
            self.client = QdrantClient(url=self.qdrant_url)
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"✓ Connected to Qdrant at {self.qdrant_url}")
            logger.info(f"  Found {len(collections.collections)} existing collections")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            if self.client.collection_exists(collection_name=self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
                
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embed_model.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"✓ Created new collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    def _load_documents(self, data_dir: Path) -> List:
        """Load documents from directory with validation."""
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        if not data_dir.is_dir():
            raise ValueError(f"Path is not a directory: {data_dir}")
        
        # Load documents
        reader = SimpleDirectoryReader(str(data_dir))
        docs = reader.load_data()
        
        if len(docs) == 0:
            raise ValueError(f"No documents found in directory: {data_dir}")
        
        logger.info(f"✓ Loaded {len(docs)} documents from {data_dir}")
        return docs
    
    async def ingest_documents(self, data_dir: Path) -> None:
        """Main ingestion workflow with comprehensive error handling."""
        try:
            # Load documents
            docs = self._load_documents(data_dir)
            
            # Ensure collection exists
            self._ensure_collection_exists()
            
            # Chunk documents
            logger.info("Chunking documents...")
            nodes = self.chunker.split(docs)
            logger.info(f"✓ Created {len(nodes)} text chunks")
            
            # Set up vector store
            vector_store = QdrantVectorStore(
                client=self.client, 
                collection_name=self.collection_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Build and persist index
            logger.info("Building vector index and ingesting to Qdrant...")
            index = VectorStoreIndex(
                nodes, 
                storage_context=storage_context, 
                embed_model=self.embed_model._model,
                show_progress=True
            )
            
            # Verify ingestion
            collection_info = self.client.get_collection(self.collection_name)
            point_count = collection_info.points_count or 0
            
            logger.info(f"✅ Successfully ingested {len(nodes)} chunks")
            logger.info(f"   Total points in collection: {point_count}")
            logger.info(f"   Collection: {self.collection_name}")
            logger.info(f"   Vector dimension: {self.embed_model.dimension}")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise


async def main() -> None:
    """Main entry point with argument parsing and configuration."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.scripts.ingest --data_dir ./data/raw
  python -m src.scripts.ingest --collection my_docs --chunk_size 1024
  QDRANT_URL=http://remote:6333 python -m src.scripts.ingest
        """
    )
    
    parser.add_argument(
        "--data_dir", 
        type=Path, 
        default=Path("data/raw"),
        help="Directory containing documents to ingest (default: data/raw)"
    )
    
    parser.add_argument(
        "--collection", 
        type=str, 
        default="Sentio_docs_v2",
        help="Qdrant collection name (default: Sentio_docs_v2)"
    )
    
    parser.add_argument(
        "--qdrant_url",
        type=str,
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant server URL (default: http://localhost:6333)"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Text chunk size in tokens (default: 512)"
    )
    
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=64,
        help="Overlap between chunks in tokens (default: 64)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Create and run ingestor
    try:
        ingestor = DocumentIngestor(
            qdrant_url=args.qdrant_url,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        await ingestor.initialize()
        await ingestor.ingest_documents(args.data_dir)
        
        logger.info("🎉 Ingestion completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Ingestion cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 