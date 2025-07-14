#!/usr/bin/env python3
"""
Build or rebuild search indexes with the latest models.

This script rebuilds both dense (Qdrant) and sparse (BM25) indexes
using the latest embedding models and configurations.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import tqdm
from dotenv import load_dotenv

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.core.embeddings import EmbeddingModel
from src.core.retrievers.sparse import BM25Retriever, Document

# Configure logging
logging.basicConfig(
    level=logging.info,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("index-builder")


def load_documents(source_file: str) -> List[Dict[str, Any]]:
    """Load documents from source file (JSON or JSONL)."""
    logger.info(f"Loading documents from {source_file}")
    
    docs = []
    with open(source_file, "r") as f:
        if source_file.endswith(".jsonl"):
            # JSONL format - one JSON object per line
            for line in f:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line: {line[:100]}...")
        else:
            # Assume single JSON array
            docs = json.load(f)
    
    logger.info(f"Loaded {len(docs)} documents")
    return docs


def build_qdrant_index(
    docs: List[Dict[str, Any]], 
    collection_name: str, 
    embedding_model: EmbeddingModel,
    batch_size: int = 100,
) -> None:
    """Build or rebuild the Qdrant vector database."""
    from qdrant_client import QdrantClient, models

    logger.info(f"Building Qdrant index for collection '{collection_name}'")
    
    # Initialize Qdrant client
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    
    # Get embedding dimension from model
    vector_size = embedding_model.dimension
    
    # Check if collection exists and recreate if requested
    if client.collection_exists(collection_name):
        logger.info(f"Collection {collection_name} exists")
        should_recreate = input("Do you want to recreate it? (y/N): ").lower() == "y"
        
        if should_recreate:
            logger.info(f"Deleting collection {collection_name}")
            client.delete_collection(collection_name=collection_name)
        else:
            logger.info("Keeping existing collection")
    
    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        logger.info(f"Creating collection {collection_name} with vector size {vector_size}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
    
    # Process documents in batches
    total_docs = len(docs)
    start_time = time.time()
    
    for i in tqdm.tqdm(range(0, total_docs, batch_size)):
        batch = docs[i:i+batch_size]
        
        # Extract texts and IDs
        texts = [doc.get("text", "") for doc in batch]
        ids = [doc.get("id", str(i + idx)) for idx, doc in enumerate(batch)]
        
        # Generate embeddings
        embeddings = embedding_model.embed(texts)
        
        # Prepare points for Qdrant
        points = [
            models.PointStruct(
                id=doc_id,
                vector=embedding,
                payload=doc
            )
            for doc_id, embedding, doc in zip(ids, embeddings, batch)
        ]
        
        # Upsert points
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    elapsed = time.time() - start_time
    logger.info(f"Indexed {total_docs} documents in {elapsed:.2f} seconds")


def build_bm25_index(docs: List[Dict[str, Any]], output_path: str) -> None:
    """Build BM25 sparse index and save to disk."""
    logger.info(f"Building BM25 index")
    
    # Convert to Document objects
    bm25_docs = [
        Document(
            id=str(doc.get("id", i)), 
            text=doc.get("text", ""),
            metadata=doc
        )
        for i, doc in enumerate(docs)
    ]
    
    # Create and populate the BM25 index
    start_time = time.time()
    retriever = BM25Retriever(bm25_docs)
    
    # Save to disk
    retriever.save(output_path)
    
    elapsed = time.time() - start_time
    logger.info(f"BM25 indexing completed in {elapsed:.2f} seconds")


def main():
    """Main entrypoint for indexing script."""
    parser = argparse.ArgumentParser(description="Build search indexes")
    parser.add_argument("--source", "-s", type=str, required=True, help="Source documents file (JSON or JSONL)")
    parser.add_argument("--collection", "-c", type=str, help="Qdrant collection name (defaults to env var)")
    parser.add_argument("--bm25-path", "-b", type=str, help="BM25 index output path")
    parser.add_argument("--skip-dense", action="store_true", help="Skip building dense index")
    parser.add_argument("--skip-sparse", action="store_true", help="Skip building sparse index")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set defaults from environment if not provided
    collection_name = args.collection or os.environ.get("COLLECTION_NAME", "Sentio_docs")
    bm25_path = args.bm25_path or os.path.join(root_dir, ".sparse_cache", "bm25_index.pkl")
    
    # Load documents
    docs = load_documents(args.source)
    
    if not docs:
        logger.error("No documents to index! Exiting.")
        return
    
    # Initialize embedding model
    embedding_model = EmbeddingModel()
    
    # Build dense index (Qdrant)
    if not args.skip_dense:
        build_qdrant_index(docs, collection_name, embedding_model)
    else:
        logger.info("Skipping dense index build")
    
    # Build sparse index (BM25)
    if not args.skip_sparse:
        build_bm25_index(docs, bm25_path)
    else:
        logger.info("Skipping sparse index build")
    
    logger.info("Indexing complete!")


if __name__ == "__main__":
    main() 