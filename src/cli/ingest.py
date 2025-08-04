"""Ingestion CLI for Sentio RAG system.

This module provides the command-line interface for ingesting documents into 
the Sentio RAG system. It uses the DocumentIngestor from src.core.ingest to process
and store documents in the vector database.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated

import typer

from src.core.ingest import ingest_directory
from src.utils.settings import settings

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create Typer CLI app
ingest_app = typer.Typer(help="Document ingestion commands")


@ingest_app.command("directory")
def ingest_directory_command(
    directory: Annotated[Path | None, typer.Argument(help="Directory containing documents to ingest")] = None,
    collection: Annotated[str, typer.Option("--collection", "-c", help="Vector store collection name")] = settings.collection_name,
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="Size of text chunks in tokens")] = settings.chunk_size,
    chunk_overlap: Annotated[int, typer.Option("--overlap", help="Overlap between chunks in tokens")] = settings.chunk_overlap,
    strategy: Annotated[str, typer.Option("--strategy", help="Chunking strategy to use")] = settings.chunking_strategy,
    embedder: Annotated[str, typer.Option("--embedder", help="Embedding model to use")] = settings.embedder_name,
    vector_store: Annotated[str, typer.Option("--vector-store", help="Vector store to use")] = settings.vector_store_name,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
):
    """Ingest documents from a directory into the vector database.
    
    This command recursively processes all supported documents in the specified
    directory, chunks them according to the configured strategy, generates
    embeddings, and stores them in the vector database.
    """
    # Configure logging based on verbosity
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    # Resolve input directory
    if directory is None:
        env_dir = os.getenv("SAMPLE_DOCS_FOLDER")
        if env_dir:
            directory = Path(env_dir)
        else:
            directory = Path("data/raw")
            typer.echo(f"No directory specified, using default: {directory}")

    if not directory.exists():
        typer.echo(f"Error: Directory '{directory}' does not exist", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Starting document ingestion from {directory}")

    try:
        # Run the ingestion using asyncio
        stats = asyncio.run(ingest_directory(
            data_dir=directory,
            collection_name=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=strategy,
            embedder_name=embedder,
            vector_store_name=vector_store,
        ))

        typer.echo("🎉 Ingestion completed successfully!")
        typer.echo(f"   Documents processed: {stats['documents_processed']}")
        typer.echo(f"   Chunks created: {stats['chunks_created']}")
        typer.echo(f"   Embeddings generated: {stats['embeddings_generated']}")
        typer.echo(f"   Bytes processed: {stats['bytes_processed']}")

    except KeyboardInterrupt:
        typer.echo("Ingestion cancelled by user")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during ingestion: {e}", err=True)
        raise typer.Exit(code=1)


@ingest_app.command("file")
def ingest_file_command(
    file_path: Annotated[Path, typer.Argument(help="Path to file to ingest")],
    collection: Annotated[str, typer.Option("--collection", "-c", help="Vector store collection name")] = settings.collection_name,
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="Size of text chunks in tokens")] = settings.chunk_size,
    chunk_overlap: Annotated[int, typer.Option("--overlap", help="Overlap between chunks in tokens")] = settings.chunk_overlap,
    strategy: Annotated[str, typer.Option("--strategy", help="Chunking strategy to use")] = settings.chunking_strategy,
    embedder: Annotated[str, typer.Option("--embedder", help="Embedding model to use")] = settings.embedder_name,
    vector_store: Annotated[str, typer.Option("--vector-store", help="Vector store to use")] = settings.vector_store_name,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
):
    """Ingest a single file into the vector database.
    
    This command processes a single document, chunks it according to the configured strategy,
    generates embeddings, and stores them in the vector database.
    """
    # Configure logging based on verbosity
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    if not file_path.exists():
        typer.echo(f"Error: File '{file_path}' does not exist", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Processing file: {file_path}")

    # Create a directory containing just this file
    temp_dir = Path(f"temp_{file_path.stem}")
    try:
        # Create temp directory
        temp_dir.mkdir(exist_ok=True)

        # Create a symlink to the original file
        link_path = temp_dir / file_path.name
        if os.name == "posix":  # Unix-like systems
            if link_path.exists():
                link_path.unlink()
            link_path.symlink_to(file_path.resolve())
        else:  # Windows or other systems - copy the file
            import shutil
            shutil.copy2(file_path, link_path)

        # Use the directory ingest function
        stats = asyncio.run(ingest_directory(
            data_dir=temp_dir,
            collection_name=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=strategy,
            embedder_name=embedder,
            vector_store_name=vector_store,
        ))

        typer.echo("🎉 Ingestion completed successfully!")
        typer.echo(f"   Documents processed: {stats['documents_processed']}")
        typer.echo(f"   Chunks created: {stats['chunks_created']}")
        typer.echo(f"   Embeddings generated: {stats['embeddings_generated']}")
        typer.echo(f"   Bytes processed: {stats['bytes_processed']}")

    except KeyboardInterrupt:
        typer.echo("Ingestion cancelled by user")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during ingestion: {e}", err=True)
        raise typer.Exit(code=1)
    finally:
        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            typer.echo(f"Warning: Failed to clean up temporary directory: {e}", err=True)


# Default command for backwards compatibility with "sentio ingest" without subcommands
@ingest_app.callback(invoke_without_command=True)
def ingest_command(ctx: typer.Context):
    """Ingest documents into the RAG system.
    """
    if ctx.invoked_subcommand is None:
        # Default to directory command if no subcommand specified
        typer.echo("No subcommand specified. Use 'directory' or 'file' subcommand.")
        typer.echo("Example: sentio ingest directory ./my_documents")
        typer.echo("Example: sentio ingest file ./my_document.pdf")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    ingest_app()
