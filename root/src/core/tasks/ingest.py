#!/usr/bin/env python3
# ==== ENHANCED DOCUMENT INGESTION SCRIPT FOR SENTIO RAG SYSTEM ==== #


"""
Enhanced document ingestion script for Sentio RAG system.

This script provides a robust, configurable interface for ingesting documents
into the Qdrant vector database with proper error handling and logging.
"""


# ==== IMPORTS & DEPENDENCIES ================================================= #

import argparse
import asyncio
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import qdrant_client
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance
from packaging import version
from importlib import metadata as importlib_metadata

from root.src.core.tasks.embeddings import EmbeddingModel
from root.src.core.tasks.chunking import TextChunker
from root.src.utils.settings import settings


# ==== LOGGING CONFIGURATION ================================================== #

try:
    from rich.logging import RichHandler
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.info,
        format=LOG_FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
except ImportError:
    logging.basicConfig(
        level=logging.info,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.storage import StorageContext
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from llama_index.vector_stores.qdrant import QdrantVectorStore


# ==== ENVIRONMENT VARIABLE LOADING =========================================== #

try:
    from dotenv import load_dotenv, find_dotenv
    _env_path = find_dotenv(usecwd=True)
    if _env_path:
        load_dotenv(_env_path, override=False)
except ImportError:
    pass


# ==== LOGGER INITIALIZATION ================================================== #

logger = logging.getLogger(__name__)


# ==== CORE PROCESSING MODULE ================================================= #

TEXT_VECTOR_NAME: str = os.getenv("QDRANT_VECTOR_NAME", "text-dense")


class DocumentIngestor:
    """
    High-performance document ingestion with error handling and monitoring.

    Args:
        qdrant_url (str): Qdrant server URL.
        collection_name (str): Name of the Qdrant collection.
        chunk_size (int): Size of text chunks in tokens.
        chunk_overlap (int): Overlap between chunks in tokens.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "Sentio_docs",
        chunk_size: int = 1024,
        chunk_overlap: int = 128
    ) -> None:
        """
        Initialize the document ingestor with specified parameters.
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model = None
        self.chunker = None
        self.client = None
        self.aclient = None
        
        # Check Qdrant client version
        try:
            # Newer versions of qdrant-client expose __version__
            self.qdrant_version = qdrant_client.__version__  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback to importlib.metadata (PEP 566)
            try:
                self.qdrant_version = importlib_metadata.version("qdrant-client")
            except Exception:
                # As a last resort, set to 0.0.0 to avoid crashes
                self.qdrant_version = "0.0.0"
        logger.info(f"Using Qdrant client version: {self.qdrant_version}")
        
        # Parse version for comparison
        try:
            self.qdrant_version_parsed = version.parse(self.qdrant_version)
        except Exception:
            logger.warning(f"Could not parse Qdrant version: {self.qdrant_version}, assuming 0.0.0")
            self.qdrant_version_parsed = version.parse("0.0.0")

        logger.info("Document ingestor initialized with:")
        logger.info(f"  - Collection: {collection_name}")
        logger.info(f"  - Chunk size: {chunk_size}")
        logger.info(f"  - Chunk overlap: {chunk_overlap}")


    async def initialize(self) -> None:
        """
        Initialize all components with proper error handling.

        Raises:
            Exception: If any component fails to initialize.
        """
        try:
            logger.info("Initializing document ingestor...")

            # Create an instance of the embedding model
            embedding_timeout = int(os.getenv("EMBEDDING_TIMEOUT", "60"))
            embed_model = EmbeddingModel(timeout=embedding_timeout)
            
            # Create an adapter for compatibility with LlamaIndex
            class EmbeddingAdapter:
                def __init__(self, model):
                    # Store the underlying embedding model in a private attr to
                    # avoid name clashes with the ``model`` property below.
                    self._model = model
                    self.callback_manager = None
                    
                def get_text_embedding(self, text, **kwargs):
                    return self._model.get_text_embedding(text, **kwargs)
                    
                def get_text_embedding_batch(self, texts, **kwargs):
                    return self._model.get_text_embedding_batch(texts, **kwargs)
                    
                async def aget_text_embedding(self, text, **kwargs):
                    return await self._model.aget_text_embedding(text, **kwargs)
                    
                async def aget_text_embedding_batch(self, texts, **kwargs):
                    return await self._model.aget_text_embedding_batch(texts, **kwargs)
                    
                @property
                def dimension(self):
                    return self._model.dimension
                
                @property
                def model(self):
                    """Return the underlying embedding model object for internal use."""
                    return self._model
            
                def __str__(self) -> str:  # noqa: D401 – imperative mood
                    if hasattr(self._model, "model_name"):
                        return f"{self._model.__class__.__name__}(model_name='{self._model.model_name}')"
                    return repr(self._model)
            
            self.embed_model = EmbeddingAdapter(embed_model)
            logger.info(f"✓ Embedding model initialized: {self.embed_model.model}")
            
            # Verify embedding dimension is valid
            if self.embed_model.dimension <= 0:
                logger.warning("Embedding model initialized with invalid dimension: 0")
                logger.info("Attempting to get embedding dimension by running a test embedding...")
                test_text = "This is a test text for embedding dimension verification."
                try:
                    # Force synchronous execution to ensure dimension is set before proceeding
                    if hasattr(self.embed_model, "get_text_embedding"):
                        test_embedding = self.embed_model.get_text_embedding(test_text)
                    else:
                        # Fallback to async method run synchronously
                        loop = asyncio.new_event_loop()
                        try:
                            test_embedding = loop.run_until_complete(
                                self.embed_model.aget_text_embedding(test_text)
                            )
                        finally:
                            loop.close()
                    
                    if test_embedding and len(test_embedding) > 0:
                        logger.info(f"Successfully obtained embedding dimension: {len(test_embedding)}")
                    else:
                        logger.error("Failed to determine embedding dimension")
                        # Set a default dimension as fallback
                        logger.warning("Setting default embedding dimension to 1024")
                except Exception as e:
                    logger.error(f"Error getting test embedding: {e}")
                    # Set a default dimension as fallback
                    logger.warning("Setting default embedding dimension to 1024")

            logger.info(f"Using embedding dimension: {self.embed_model.dimension}")

            self.chunker = TextChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info("✓ Text chunker initialized")

            # Check for API key for Qdrant Cloud
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            if "cloud.qdrant.io" in self.qdrant_url and not qdrant_api_key:
                logger.warning("Qdrant Cloud URL detected, but QDRANT_API_KEY is not set")
            # Initialize Qdrant clients with API key if available
            client_params = {"url": self.qdrant_url}
            if qdrant_api_key:
                client_params["api_key"] = qdrant_api_key
                logger.info("✓ Using API key for Qdrant")
                
            self.client = QdrantClient(**client_params)
            self.aclient = AsyncQdrantClient(**client_params)

            collections = self.client.get_collections()
            logger.info(f"✓ Connected to Qdrant at {self.qdrant_url}")
            logger.info(f"  Found {len(collections.collections)} existing collections")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise


    def _ensure_collection_exists(self) -> None:
        """
        Create the Qdrant collection if it does not exist, or validate dimensions.

        Raises:
            ValueError: If collection exists but has a dimension mismatch.
        """
        # Check if the collection exists
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Collection {self.collection_name} does not exist, creating it...")

            # Ensure we have a valid vector size
            vector_size = self.embed_model.dimension
            if vector_size <= 0:
                # If the dimension is still invalid, try to get the embedding again
                test_text = "This is a test text for embedding dimension verification."
                try:
                    if hasattr(self.embed_model, "get_text_embedding"):
                        test_embedding = self.embed_model.get_text_embedding(test_text)
                    else:
                        # Fallback to async method run synchronously
                        loop = asyncio.new_event_loop()
                        try:
                            test_embedding = loop.run_until_complete(
                                self.embed_model.aget_text_embedding(test_text)
                            )
                        finally:
                            loop.close()
                    
                    vector_size = len(test_embedding)
                    logger.info(f"Successfully obtained embedding dimension: {vector_size}")
                except Exception:
                    # If dimension is still invalid, use a safe default
                    vector_size = 1024
                    logger.warning(f"Invalid embedding dimension detected. Using default: {vector_size}")
                
            logger.info(f"Creating collection with embedding dimension: {vector_size}")
            logger.info(f"Embedding model being used: {self.embed_model.model}")

            # Attempt to create a collection using different formats based on Qdrant version
            success = False
            
            # Define possible formats for collection creation
            formats = []
            
            # For newer Qdrant versions (>= 1.1.0), prefer named vectors
            if self.qdrant_version_parsed >= version.parse("1.1.0"):
                formats = [
                    {
                        "name": "named vectors",
                        "create": lambda: self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config={
                                TEXT_VECTOR_NAME: models.VectorParams(
                                    size=vector_size,
                                    distance=models.Distance.COSINE
                                )
                            }
                        )
                    },
                    {
                        "name": "direct VectorParams",
                        "create": lambda: self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=models.VectorParams(
                                size=vector_size,
                                distance=models.Distance.COSINE
                            )
                        )
                    },
                    {
                        "name": "dict format",
                        "create": lambda: self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config={"size": vector_size, "distance": "Cosine"}
                        )
                    }
                ]
            else:
                # For older versions, prefer direct format
                formats = [
                    {
                        "name": "direct VectorParams",
                        "create": lambda: self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=models.VectorParams(
                                size=vector_size,
                                distance=models.Distance.COSINE
                            )
                        )
                    },
                    {
                        "name": "dict format",
                        "create": lambda: self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config={"size": vector_size, "distance": "Cosine"}
                        )
                    },
                    {
                        "name": "named vectors",
                        "create": lambda: self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config={
                                TEXT_VECTOR_NAME: models.VectorParams(
                                    size=vector_size,
                                    distance=models.Distance.COSINE
                                )
                            }
                        )
                    }
                ]
            
            # Try each format until one succeeds
            for fmt in formats:
                try:
                    logger.info(f"Trying {fmt['name']} format...")
                    fmt["create"]()
                    logger.info(f"✓ Collection {self.collection_name} created with {fmt['name']} format")
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Failed with {fmt['name']} format: {e}")
            
            # If all formats fail, attempt to delete the collection (if partially created) and recreate it
            if not success:
                logger.warning("All collection creation attempts failed, trying to recreate collection")
                try:
                    # Check if the collection was partially created
                    if self.client.collection_exists(self.collection_name):
                        logger.warning(f"Collection {self.collection_name} exists but may be corrupt, deleting it")
                        self.client.delete_collection(self.collection_name)
                    
                    # Try creating with the most basic format
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            "size": vector_size,
                            "distance": "Cosine"
                        }
                    )
                    logger.info(f"✓ Collection {self.collection_name} created with fallback format")
                    success = True
                except Exception as e:
                    logger.error(f"Final attempt to create collection failed: {e}")
                    raise ValueError(f"Could not create collection: {e}")

        else:
            # Collection exists, check dimension consistency
            try:
                collection_info = self.client.get_collection(self.collection_name)
                
                # Attempt to get vector size from collection info
                # Format may vary depending on Qdrant version
                existing_size = None
                
                # Check different paths to vector size
                if hasattr(collection_info.config.params, 'vectors'):
                    vectors_info = collection_info.config.params.vectors
                    if isinstance(vectors_info, dict) and TEXT_VECTOR_NAME in vectors_info:
                        existing_size = vectors_info[TEXT_VECTOR_NAME].size
                    elif hasattr(vectors_info, 'size'):
                        existing_size = vectors_info.size
                # Old format
                elif hasattr(collection_info.config.params, 'vector_size'):
                    existing_size = collection_info.config.params.vector_size
                
                # If unable to determine size, use model's value
                if existing_size is None:
                    logger.warning("Could not determine collection vector size, using model dimension")
                    existing_size = self.embed_model.dimension
                
                model_size = self.embed_model.dimension

                logger.info(f"Collection {self.collection_name} already exists with vector size {existing_size}")
                logger.info(f"Current embedding model dimension: {model_size}")

                # ------------------------------------------------------------------
                # Ensure the collection has the expected *named* vector.  If it was
                # created with an *unnamed* vector (legacy Qdrant behaviour), any
                # upsert/search operation that specifies `vector_name` will fail
                # with "Not existing vector name".  To provide a consistent API
                # across the codebase we *recreate* the collection with the
                # required named vector when it is safe to do so (i.e. when the
                # collection is empty).  Otherwise we abort and instruct the user
                # to migrate their data or set QDRANT_VECTOR_NAME accordingly.
                # ------------------------------------------------------------------
                has_expected_vector: bool = False
                try:
                    if hasattr(collection_info.config.params, "vectors"):
                        vectors_cfg = collection_info.config.params.vectors
                        if isinstance(vectors_cfg, dict):
                            has_expected_vector = TEXT_VECTOR_NAME in vectors_cfg
                        else:
                            # Single unnamed vector (VectorParams)
                            has_expected_vector = False
                except Exception:
                    # Any failure defaults to False – we will handle below
                    has_expected_vector = False

                if not has_expected_vector:
                    logger.warning(
                        f"Collection '{self.collection_name}' does not contain expected vector name "
                        f"'{TEXT_VECTOR_NAME}'."
                    )

                    # Determine whether the collection contains data
                    point_count: int = 0
                    try:
                        point_count = collection_info.points_count or 0
                    except Exception:
                        # If we cannot determine point count, assume non-zero to be safe
                        point_count = 1

                    if point_count == 0:
                        logger.info(
                            "Collection is empty – recreating it with the correct named vector configuration…"
                        )

                        try:
                            self.client.delete_collection(self.collection_name)
                            logger.debug("Deleted legacy collection with unnamed vector")

                            self.client.create_collection(
                                collection_name=self.collection_name,
                                vectors_config={
                                    TEXT_VECTOR_NAME: models.VectorParams(
                                        size=existing_size,
                                        distance=models.Distance.COSINE,
                                    )
                                },
                            )
                            logger.info(
                                f"✓ Recreated collection '{self.collection_name}' with named vector "
                                f"'{TEXT_VECTOR_NAME}' (dim={existing_size})"
                            )
                            # Refresh collection_info for any downstream checks
                            collection_info = self.client.get_collection(self.collection_name)
                            has_expected_vector = True
                        except Exception as e:
                            logger.error(
                                f"Failed to recreate collection with named vector: {e}"
                            )
                            raise
                    else:
                        error_msg = (
                            f"Collection '{self.collection_name}' already contains {point_count} points "
                            "but lacks the expected named vector. Automatic migration is disabled to "
                            "avoid data loss. Either migrate your collection manually or set the "
                            "environment variable QDRANT_VECTOR_NAME to the existing vector name."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                # If the model dimension does not match the collection dimension, adapt the model
                if model_size != existing_size:
                    logger.warning(
                        f"Model dimension {model_size} doesn't match collection dimension {existing_size}. "
                        f"Adapting to collection dimension."
                    )
                    # Set the model dimension to match the collection
                    object.__setattr__(self.embed_model, "_dimension", existing_size)
                    
            except Exception as e:
                logger.warning(f"Error checking collection dimensions: {e}")
                # Continue operation even if dimension check fails


    def _load_documents(self, data_dir: Path) -> List[Document]:
        """
        Load documents from directory with proper error handling.

        Args:
            data_dir (Path): Directory containing documents.

        Returns:
            List: Loaded documents.

        Raises:
            ValueError: If no files or valid documents are found.
        """
        try:
            if not any(data_dir.iterdir()):
                raise ValueError(f"No files found in data directory: {data_dir}")

            logger.info(f"Loading documents from {data_dir}")
            docs = SimpleDirectoryReader(str(data_dir)).load_data()

            if not docs:
                raise ValueError(f"No valid documents found in {data_dir}")

            logger.info(f"✓ Loaded {len(docs)} documents")
            return docs

        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise


    async def ingest_documents(self, data_dir: Path) -> None:
        """
        Ingest documents from a directory into Qdrant with chunking and embedding.

        Args:
            data_dir (Path): Directory containing documents to ingest.

        Raises:
            Exception: If ingestion fails at any step.
        """
        logger.info(f"🔍 Ingesting documents from: {data_dir}")

        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        try:
            docs = self._load_documents(data_dir)

            self._ensure_collection_exists()

            logger.info("Chunking documents...")
            nodes = self.chunker.split(docs)
            logger.info(f"✓ Created {len(nodes)} text chunks")

            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                aclient=self.aclient,
                vector_name=TEXT_VECTOR_NAME,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            logger.info("Building vector index and ingesting to Qdrant...")
            logger.info(f"Embedding model: {self.embed_model.model}, dimension: {self.embed_model.dimension}")
            # Check embedding dimension before indexing
            test_text = "This is a test text for embedding dimension verification."
            test_embedding = await self.embed_model.aget_text_embedding(test_text)
            logger.info(f"Test embedding dimension: {len(test_embedding)}")

            # Retrieve collection information to verify dimension
            try:
                collection_info = self.client.get_collection(self.collection_name)
                
                # Attempt to get vector dimension from collection information
                collection_dim = None
                
                # Check different paths to vector dimension
                if hasattr(collection_info.config.params, 'vectors'):
                    vectors_info = collection_info.config.params.vectors
                    if isinstance(vectors_info, dict) and TEXT_VECTOR_NAME in vectors_info:
                        collection_dim = vectors_info[TEXT_VECTOR_NAME].size
                    elif hasattr(vectors_info, 'size'):
                        collection_dim = vectors_info.size
                # Old format
                elif hasattr(collection_info.config.params, 'vector_size'):
                    collection_dim = collection_info.config.params.vector_size
                
                if collection_dim is None:
                    logger.warning("Could not determine collection vector dimension, using test embedding dimension")
                    collection_dim = len(test_embedding)
                
                logger.info(f"Collection vector dimension: {collection_dim}")

                # Check dimension match
                if len(test_embedding) != collection_dim:
                    logger.error(
                        f"⚠️ CRITICAL DIMENSION MISMATCH: Collection expects {collection_dim}, "
                        f"but embedding produces {len(test_embedding)}"
                    )
                    
                    # Attempt to fix mismatch
                    logger.warning("Attempting to fix dimension mismatch by recreating collection...")
                    
                    # Delete existing collection
                    self.client.delete_collection(self.collection_name)
                    logger.info(f"Deleted collection {self.collection_name}")
                    
                    # Create collection with correct dimension
                    vector_size = len(test_embedding)
                    
                    # Create collection with new dimension
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            TEXT_VECTOR_NAME: models.VectorParams(
                                size=vector_size,
                                distance=models.Distance.COSINE
                            )
                        }
                    )
                    logger.info(f"✓ Recreated collection {self.collection_name} with correct dimension: {vector_size}")
                    
                    # Recreate vector_store with new collection
                    vector_store = QdrantVectorStore(
                        client=self.client,
                        collection_name=self.collection_name,
                        aclient=self.aclient,
                        vector_name=TEXT_VECTOR_NAME,
                    )
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
            except Exception as e:
                logger.error(f"Error checking collection dimensions: {e}")
                # Continue even if dimension check fails

            def _create_index_sync() -> VectorStoreIndex:
                """Sync wrapper for index creation to run in a separate thread."""
                return VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    use_async=True,
                    show_progress=True,
                )

            index = await asyncio.to_thread(_create_index_sync)

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
    """
    Main entry point with argument parsing and configuration.

    Raises:
        SystemExit: If ingestion is cancelled or fails.
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python -m root.src.etl.ingest --data_dir ./data/raw
              python -m root.src.etl.ingest --collection my_docs --chunk_size 1024
              QDRANT_URL=http://remote:6333 python -m root.src.etl.ingest
        """)
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("root/data/raw"),
        help="Directory containing documents to ingest (default: root/data/raw)"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="Sentio_docs",
        help="Qdrant collection name (default: Sentio_docs)"
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
        default=1024,
        help="Text chunk size in tokens (default: 1024)"
    )

    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=128,
        help="Overlap between chunks in tokens (default: 128)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

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