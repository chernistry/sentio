#!/usr/bin/env python3
"""Test script for demonstrating the chunking functionality on actual documents.
"""
import asyncio
import sys
import time
from pathlib import Path

# Add project src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.chunking.text_splitter import TextChunker
from src.core.models.document import Document


def read_text_file(file_path: str) -> str:
    """Read text file content."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


async def main():
    """Main function to demonstrate chunking functionality."""
    # Path to test document
    test_file_path = "src/tests/chunking/sample_data/python_for_osint.txt"

    try:
        # Read document
        print(f"Reading document: {test_file_path}")
        text = read_text_file(test_file_path)
        print(f"Document size: {len(text)} characters")

        # Create document object
        doc = Document(
            text=text,
            metadata={"source": test_file_path, "type": "text/plain"}
        )
        print(f"Created document with ID: {doc.id}")

        # Initialize TextChunker
        print("\nInitializing TextChunker...")
        chunker = await TextChunker.create(
            strategy="recursive",
            chunk_size=512,
            chunk_overlap=64
        )

        # Time the chunking operation
        start_time = time.time()
        print("Splitting document into chunks...")

        # Create chunks
        chunks = chunker.split([doc])

        end_time = time.time()
        duration = end_time - start_time

        # Print results
        print(f"\nChunking completed in {duration:.2f} seconds")
        print(f"Created {len(chunks)} chunks from document")

        # Show some sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}/{len(chunks)}")
            print(f"Chunk ID: {chunk.id}")
            print(f"Parent ID: {chunk.metadata.get('parent_id')}")
            print(f"Source: {chunk.metadata.get('source')}")
            print("Text preview (first 100 chars):", chunk.text[:100] + "...")
            print(f"Chunk length: {len(chunk.text)} chars")

        # Calculate average chunk size
        avg_size = sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0
        print(f"\nAverage chunk size: {avg_size:.2f} characters")

        # Verify content integrity by checking if the total length of chunks
        # is close to the original text length. This is an approximation
        # due to overlap and splitting logic.
        original_length = len(doc.text)
        chunked_length = sum(len(c.text) for c in chunks)
        # A simple heuristic: total chunk length should be at least the original,
        # and not excessively larger.
        content_integrity_ok = chunked_length >= original_length

        print(f"\nContent integrity check (heuristic): {'PASSED' if content_integrity_ok else 'FAILED'}")
        print(f"Original length: {original_length}, Combined chunk length: {chunked_length}")


    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
