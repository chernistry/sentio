"""Run Sentio RAG system with multiple components.

This module provides commands to start multiple components of the Sentio RAG system
together for convenience.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from typing import Annotated

import typer

# Setup logging
logger = logging.getLogger(__name__)

# Create Typer app
run_app = typer.Typer(help="Run Sentio RAG system")


def run_process(cmd, name, env=None):
    """Run a process and return a handle to it."""
    typer.echo(f"Starting {name}...")
    if env is None:
        env = os.environ.copy()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )

        # Start a thread to handle the output
        def print_output(process, name):
            for line in iter(process.stdout.readline, ""):
                typer.echo(f"[{name}] {line.rstrip()}")

        threading.Thread(target=print_output, args=(process, name), daemon=True).start()

        return process
    except Exception as e:
        typer.echo(f"Error starting {name}: {e}", err=True)
        return None


@run_app.command("all")
def run_all(
    api_host: Annotated[str, typer.Option("--api-host", help="Host for API server")] = "0.0.0.0",
    api_port: Annotated[int, typer.Option("--api-port", help="Port for API server")] = 8000,
    ui_port: Annotated[int, typer.Option("--ui-port", help="Port for UI server")] = 8501,
    vector_store: Annotated[str, typer.Option("--vector-store", help="Vector store to use")] = "qdrant",
):
    """Run the complete Sentio RAG system.
    
    This command starts both the API server and the UI, allowing for a single
    command to launch the complete system.
    """
    # Check if we need to start Qdrant
    qdrant_process = None
    if vector_store == "qdrant":
        # Check if Docker is available
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)

            # Check if Qdrant container is running
            qdrant_check = subprocess.run(
                ["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Names}}"],
                capture_output=True, text=True, check=False
            )

            if "qdrant" not in qdrant_check.stdout:
                typer.echo("Qdrant container not running. Starting it...")
                qdrant_process = run_process(
                    ["docker", "run", "-d", "--name", "qdrant", "-p", "6333:6333", "-p", "6334:6334", "qdrant/qdrant"],
                    "qdrant"
                )
                # Wait for Qdrant to start
                typer.echo("Waiting for Qdrant to start...")
                time.sleep(5)
        except subprocess.CalledProcessError:
            typer.echo("Docker not available. Cannot start Qdrant container.", err=True)
            typer.echo("Please start Qdrant manually or use another vector store.")

    # Start API
    python_executable = sys.executable
    api_cmd = [
        python_executable, "-m", "src.cli.api", "start",
        "--host", api_host, "--port", str(api_port)
    ]
    api_process = run_process(api_cmd, "API")
    if api_process is None:
        typer.echo("Failed to start API server", err=True)
        return

    # Wait for API to start
    typer.echo("Waiting for API to start...")
    time.sleep(3)

    # Start UI with API URL
    env = os.environ.copy()
    env["SENTIO_BACKEND_URL"] = f"http://{api_host if api_host != '0.0.0.0' else 'localhost'}:{api_port}"
    ui_cmd = [
        python_executable, "-m", "src.cli.ui", "start",
        "--port", str(ui_port),
        "--api-url", env["SENTIO_BACKEND_URL"]
    ]
    ui_process = run_process(ui_cmd, "UI", env=env)
    if ui_process is None:
        typer.echo("Failed to start UI", err=True)
        api_process.terminate()
        return

    # Wait for processes
    typer.echo("🚀 Sentio RAG system is running!")
    typer.echo(f"   API: http://{api_host if api_host != '0.0.0.0' else 'localhost'}:{api_port}")
    typer.echo(f"   UI:  http://localhost:{ui_port}")
    typer.echo("\nPress Ctrl+C to stop all services...")

    try:
        while True:
            # Check if processes are still running
            if api_process.poll() is not None:
                typer.echo("API process exited unexpectedly", err=True)
                break
            if ui_process.poll() is not None:
                typer.echo("UI process exited unexpectedly", err=True)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        typer.echo("\nStopping services...")
    finally:
        # Clean up processes
        for process, name in [(ui_process, "UI"), (api_process, "API")]:
            if process and process.poll() is None:
                typer.echo(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    typer.echo(f"{name} did not terminate gracefully, killing...")
                    process.kill()

        # Stop Qdrant if we started it
        if qdrant_process and qdrant_process.poll() is None:
            typer.echo("Stopping Qdrant...")
            subprocess.run(["docker", "stop", "qdrant"], check=False)
            subprocess.run(["docker", "rm", "qdrant"], check=False)

    typer.echo("✅ All services stopped")


if __name__ == "__main__":
    run_app()
