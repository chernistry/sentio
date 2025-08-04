"""API server launcher for Sentio RAG system.

This module provides commands to start and manage the API server.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from src.utils.settings import settings

# Setup logging
logger = logging.getLogger(__name__)

# Create Typer app
api_app = typer.Typer(help="API server commands")


@api_app.command("start")
def start_api(
    host: Annotated[str, typer.Option("--host", help="Host to bind the API server to")] = settings.host,
    port: Annotated[int, typer.Option("--port", "-p", help="Port for API server")] = settings.port,
    reload: Annotated[bool, typer.Option("--reload", "-r", help="Enable auto-reload for development")] = False,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Logging level")] = settings.log_level.lower(),
):
    """Start the Sentio RAG API server.
    
    This command launches the FastAPI-based API server that provides endpoints
    for document ingestion and querying.
    """
    api_dir = Path(__file__).parent.parent / "api"
    api_file = api_dir / "app.py"

    if not api_file.exists():
        typer.echo(f"Error: API app file not found at {api_file}", err=True)
        raise typer.Exit(code=1)

    # Convert app.py path to module path
    module_path = "src.api.app:app"

    typer.echo(f"Starting API server on http://{host}:{port}")
    try:
        uvicorn.run(
            module_path,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level.lower(),
        )
    except KeyboardInterrupt:
        typer.echo("API server stopped by user")
    except Exception as e:
        typer.echo(f"Error starting API server: {e}", err=True)
        raise typer.Exit(code=1)


@api_app.command("install-deps")
def install_api_dependencies():
    """Install API server dependencies.
    
    This command installs the necessary dependencies for running the API server.
    """
    try:
        typer.echo("Installing API server dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "fastapi>=0.100.0", "uvicorn>=0.22.0", "httpx>=0.24.1"
        ], check=True)
        typer.echo("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error installing dependencies: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    api_app()
