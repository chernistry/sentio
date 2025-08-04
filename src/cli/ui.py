"""Streamlit UI launcher for Sentio RAG system.

This module provides a command to launch the Streamlit-based user interface
for the Sentio RAG system.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

# Setup logging
logger = logging.getLogger(__name__)

# Create Typer app
ui_app = typer.Typer(help="User interface commands")


@ui_app.command("start")
def start_ui(
    port: Annotated[int, typer.Option("--port", "-p", help="Port for Streamlit UI")] = 8501,
    api_url: Annotated[str, typer.Option("--api-url", "-a", help="URL of the API server")] = "http://localhost:8000",
):
    """Start the Streamlit UI for Sentio RAG.
    
    This command launches the Streamlit-based user interface that connects
    to the Sentio RAG API server.
    """
    # Find the streamlit_app.py file
    ui_dir = Path(__file__).parent.parent / "ui"
    streamlit_file = ui_dir / "streamlit_app.py"

    if not streamlit_file.exists():
        typer.echo(f"Error: Streamlit app file not found at {streamlit_file}", err=True)
        raise typer.Exit(code=1)

    # Set environment variable for API URL
    os.environ["SENTIO_BACKEND_URL"] = api_url
    typer.echo(f"Setting API URL: {api_url}")

    # Launch Streamlit
    typer.echo(f"Starting Streamlit UI on port {port}...")
    try:
        cmd = [
            "streamlit", "run",
            str(streamlit_file),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.serverAddress", "localhost"
        ]
        typer.echo(f"Running: {' '.join(cmd)}")

        # Execute streamlit with the current environment
        process = subprocess.run(cmd, env=os.environ, check=False)

        if process.returncode != 0:
            typer.echo(f"Streamlit exited with code {process.returncode}", err=True)
            raise typer.Exit(code=process.returncode)
    except FileNotFoundError:
        typer.echo("Error: Streamlit not found. Please install it with: pip install streamlit", err=True)
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        typer.echo("Streamlit UI stopped by user")
    except Exception as e:
        typer.echo(f"Error starting Streamlit UI: {e}", err=True)
        raise typer.Exit(code=1)


@ui_app.command("install-deps")
def install_ui_dependencies():
    """Install UI dependencies (Streamlit and PyPDF2).
    
    This command installs the necessary dependencies for running the UI.
    """
    try:
        typer.echo("Installing Streamlit UI dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "streamlit>=1.24.0", "PyPDF2>=3.0.0"
        ], check=True)
        typer.echo("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error installing dependencies: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    ui_app()
