"""Sentio unified CLI.

This consolidates scattered maintenance/debug scripts into a single
entry-point for easier discoverability and reuse. Exposed commands cover
environment checks, index building and test runs.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

import typer
from typing_extensions import Annotated
import re

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent
cli = typer.Typer(add_completion=False, help="Sentio unified command-line tool")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _run(cmd: list[str] | str, cwd: Optional[Path] = None) -> None:
    """Run a shell command and stream its output.

    Args:
        cmd: Command and arguments.
        cwd: Directory to run the command from.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{APP_DIR}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"

    if isinstance(cmd, str):
        # Use shell=True so that redirections (&, >, | …) are honoured.
        subprocess.run(cmd, cwd=cwd or APP_DIR, check=True, shell=True, env=env)
    else:
        subprocess.run(cmd, cwd=cwd or APP_DIR, check=True, env=env)


def _load_module_from_path(module_path: str, module_name: str) -> Any:
    """Dynamically load a module from a file path.
    
    Args:
        module_path: Path to the Python file.
        module_name: Name to give the module.
        
    Returns:
        The loaded module or None if loading failed.
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


# Try to load check_env from various locations
_check_env_module = None
_check_env_paths = [
    os.path.join(APP_DIR, "check_env.py"),
    os.path.join(APP_DIR.parent, ".archive", "scripts", "check_env.py"),
    os.path.join(APP_DIR.parent, "debug", "scripts", "check_env.py"),
]

for path in _check_env_paths:
    if os.path.exists(path):
        _check_env_module = _load_module_from_path(path, "check_env")
        if _check_env_module:
            break

_check_env: Optional[Callable[[], None]] = getattr(_check_env_module, "check_env", None)





# ---------------------------------------------------------------------------
# Stack management commands
# ---------------------------------------------------------------------------

@cli.command("start")
def cmd_start(
    mode: Annotated[str, typer.Option("--mode", "-m", help="Mode to start in")] = "dev",
) -> None:
    """Start the Sentio stack in the specified mode (dev or prod)."""
    if mode not in ["dev", "prod"]:
        typer.secho(f"Invalid mode: {mode}. Use 'dev' or 'prod'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    typer.secho(f"Starting Sentio in {mode} mode...", fg=typer.colors.GREEN)
    
    # Check embedding provider
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
    # Check if we're using cloud mode for beam
    beam_mode = os.environ.get("BEAM_MODE", "local")
    # Don't start Qdrant locally when using cloud services
    use_cloud = (embedding_provider == "jina" or beam_mode == "cloud")
    
    # Use the setup-env.sh script to start docker-compose
    setup_env_script = os.path.join(REPO_ROOT, "root", "devops", "setup-env.sh")
    if os.path.exists(setup_env_script):
        if mode == "dev":
            if use_cloud:
                # Don't start Qdrant locally when using cloud services
                typer.secho("Cloud services detected, not starting local Qdrant", fg=typer.colors.BLUE)
                _run(f"{setup_env_script} up -d api ui", cwd=REPO_ROOT)
            else:
                # Only start Docker services for development
                _run(f"{setup_env_script} up -d", cwd=REPO_ROOT)
        else:
            # Start full stack for production
            _run(f"{setup_env_script} up -d", cwd=REPO_ROOT)
    else:
        # Fallback to direct docker-compose if script not found
        if mode == "dev":
            if use_cloud:
                # Don't start Qdrant locally when using cloud services
                typer.secho("Cloud services detected, not starting local Qdrant", fg=typer.colors.BLUE)
                _run("docker compose up -d api ui", cwd=REPO_ROOT)
            else:
                _run("docker compose up -d", cwd=REPO_ROOT)
        else:
            _run("docker compose up -d", cwd=REPO_ROOT)
    
    typer.secho(f"Sentio {mode} stack started successfully!", fg=typer.colors.GREEN)


@cli.command("stop")
def cmd_stop() -> None:
    """Stop the Sentio stack."""
    typer.secho("Stopping Sentio stack...", fg=typer.colors.YELLOW)
    
    # Use the setup-env.sh script to start docker-compose
    setup_env_script = os.path.join(REPO_ROOT, "root", "devops", "setup-env.sh")
    if os.path.exists(setup_env_script):
        _run(f"{setup_env_script} down", cwd=REPO_ROOT)
    else:
        # Fallback to direct docker-compose if script not found
        _run("docker compose down", cwd=REPO_ROOT)

    # Reload environment variables from .env file
    load_env()
    
    # Check embedding provider and Qdrant settings
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
    qdrant_url = os.environ.get("QDRANT_URL", "")
    
    # Always show cloud Qdrant status
    if qdrant_url:
        typer.secho(f"Qdrant: using cloud instance at {qdrant_url}", fg=typer.colors.GREEN)
    else:
        typer.secho("Qdrant URL not configured!", fg=typer.colors.RED)
        
    typer.secho("Sentio stack stopped.", fg=typer.colors.GREEN)


@cli.command("restart")
def cmd_restart(
    mode: Annotated[str, typer.Option("--mode", "-m", help="Mode to restart in")] = "dev",
) -> None:
    """Restart the Sentio stack in the specified mode."""
    cmd_stop()
    cmd_start(mode)


@cli.command("status")
def cmd_status() -> None:
    """Show the status of the Sentio stack."""
    typer.secho("Docker services:", fg=typer.colors.BLUE)
    _run("docker compose ps", cwd=REPO_ROOT)
    
    # Reload environment variables from .env file
    load_env()
    
    # Check embedding provider and Qdrant settings
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
    qdrant_url = os.environ.get("QDRANT_URL", "")
    
    # Always show cloud Qdrant status
    if qdrant_url:
        typer.secho(f"Qdrant: using cloud instance at {qdrant_url}", fg=typer.colors.GREEN)
    else:
        typer.secho("Qdrant URL not configured!", fg=typer.colors.RED)


@cli.command("logs")
def cmd_logs(
    service: Annotated[str, typer.Argument(help="Service name (or all if omitted)")] = "",
    lines: Annotated[int, typer.Option("--lines", "-n", help="Number of lines to show")] = 100,
) -> None:
    """Show logs for the specified service or all services."""
    if service:
        # Normalize container name (e.g., api-1 → api)
        normalized_service = service
        match = re.match(r"^([a-zA-Z0-9_.-]+?)-\d+$", service)
        if match:
            normalized_service = match.group(1)
        typer.secho(f"Showing logs for {normalized_service}...", fg=typer.colors.BLUE)
        _run(f"docker compose logs -f --tail={lines} {normalized_service}", cwd=REPO_ROOT)
    else:
        typer.secho("Showing logs for all services...", fg=typer.colors.BLUE)
        _run(f"docker compose logs -f --tail={lines}", cwd=REPO_ROOT)


# ---------------------------------------------------------------------------
# Environment and diagnostics commands
# ---------------------------------------------------------------------------

@cli.command("env")
def cmd_env() -> None:  # noqa: D401 – imperative mood is fine for CLI verbs
    """Print environment diagnostics."""
    if _check_env is None:
        typer.secho("check_env module not available", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    _check_env()





# ---------------------------------------------------------------------------
# Data management commands
# ---------------------------------------------------------------------------

@cli.command("ingest")
def cmd_ingest(
    path: Annotated[Path, typer.Argument(help="Path to file or directory to ingest")] = None,
) -> None:
    """Ingest documents into the system."""
    typer.secho("Running document ingestion...", fg=typer.colors.GREEN)
    
    # Check embedding provider
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
    
    cmd = ["docker", "compose", "run", "--rm"]
    
    # If using jina or cloud mode, pass Qdrant API key to container
    if embedding_provider == "jina" or os.environ.get("BEAM_MODE", "local") == "cloud":
        if qdrant_api_key:
            cmd.extend(["-e", f"QDRANT_API_KEY={qdrant_api_key}"])
            cmd.extend(["-e", f"QDRANT_API_KEY_HEADER=api-key"])
    
    cmd.append("ingest")
    
    if path:
        # If path is provided, mount it to the container
        volume_arg = f"{path.resolve()}:/data/input"
        cmd.extend(["-v", volume_arg])
    
    _run(cmd, cwd=REPO_ROOT)


@cli.command("build-index")
def cmd_build_index(
    source: Path = typer.Argument(..., exists=True, readable=True),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    bm25_path: Optional[Path] = typer.Option(None, "--bm25-path", "-b"),
    skip_dense: bool = typer.Option(False, help="Skip dense (Qdrant) index"),
    skip_sparse: bool = typer.Option(False, help="Skip sparse (BM25) index"),
) -> None:
    """Build search indexes from source documents."""
    rel = source.resolve()
    cmd = [
        sys.executable,
        "-m",
        "cli.scripts.build_index",
        "--source",
        str(rel),
    ]
    if collection:
        cmd += ["--collection", collection]
    if bm25_path:
        cmd += ["--bm25-path", str(bm25_path)]
    if skip_dense:
        cmd.append("--skip-dense")
    if skip_sparse:
        cmd.append("--skip-sparse")
    _run(cmd, cwd=APP_DIR)


# ---------------------------------------------------------------------------
# Testing commands
# ---------------------------------------------------------------------------

@cli.command("tests")
def cmd_tests(
    scope: str = typer.Option("local", "--scope", "-s", help="Test scope: local or cloud"),
    markers: Optional[str] = typer.Option(None, "-m", help="Additional pytest markers expression"),
    verbose: bool = typer.Option(False, "-v", help="Verbose pytest output"),
) -> None:
    """Execute the pytest suite.

    Args:
        scope: "local" (default) runs only fast, offline-safe tests; "cloud" runs only cloud-tagged tests.
        markers: Extra marker expression to pass through.
        verbose: Enable -vv output.
    """

    if scope not in {"local", "cloud"}:
        typer.secho("Scope must be 'local' or 'cloud'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    marker_expr = "cloud" if scope == "cloud" else "not cloud"
    if markers:
        marker_expr = f"({marker_expr}) and ({markers})"

    args = [
        "pytest",
        "-m",
        marker_expr,
        "--cov=src",
        "--cov-report=term-missing",
    ]

    if scope == "local":
        args += ["--ignore", "tests/test_azure_integration.py"]

    if verbose:
        args.append("-vv")

    _run(args, cwd=APP_DIR)


@cli.command("chat-test")
def cmd_chat_test(
    preset: Annotated[str, typer.Option("--preset", "-p", help="Preset question to run")] = "all",
    verbose: bool = typer.Option(False, "-v", help="Verbose output"),
) -> None:
    """Run sample chat queries against the API."""
    # Ensure the API is running
    if not _is_api_running():
        typer.secho("API is not running. Starting dev stack...", fg=typer.colors.YELLOW)
        cmd_start("dev")
    
    typer.secho("Running chat tests...", fg=typer.colors.GREEN)
    # Run pytest with the `e2e` marker instead of the deprecated test_chat script.
    cmd = [sys.executable, "-m", "pytest", "-m", "e2e"]

    # Preserve compatibility with the legacy --preset flag (ignored by current tests).
    if preset and preset != "all":
        os.environ["CHAT_TEST_PRESET"] = preset

    if verbose:
        cmd.append("-vv")

    _run(cmd, cwd=APP_DIR)


@cli.command("locust")
def cmd_locust(
    host: str = typer.Option("http://localhost:8000", "--host", help="API host"),
    web_port: int = typer.Option(8089, "--port", "-p", help="Locust web UI port"),
) -> None:
    """Start Locust load tests with sensible defaults."""
    locustfile = APP_DIR / "tests" / "locustfile.py"
    cmd = [
        "locust",
        "-f",
        str(locustfile),
        f"--host={host}",
        f"--web-port={web_port}",
    ]
    _run(cmd)


# ---------------------------------------------------------------------------
# Deployment commands
# ---------------------------------------------------------------------------

@cli.command("azure-prepare")
def cmd_azure_prepare(version: str = typer.Option("v2.1.0", "--version", "-v")) -> None:
    """Prepare images for Azure deployment."""
    script_paths = [
        REPO_ROOT / "prepare_for_azure.sh",
        REPO_ROOT / "debug" / "prepare_for_azure.sh",
    ]
    
    for script in script_paths:
        if script.exists():
            _run(["bash", str(script)], cwd=REPO_ROOT)
            return
            
    typer.secho("prepare_for_azure.sh not found", fg=typer.colors.RED)
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Maintenance commands
# ---------------------------------------------------------------------------


@cli.command("flush")
def cmd_flush(force: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")) -> None:
    """Delete all Qdrant collections to start from a clean slate."""

    # Ensure .env variables are loaded so that the command also works when
    # invoked directly via ``run.sh flush`` before any stack is running.
    load_env()

    # Get Qdrant URL and API key from environment
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    
    if not qdrant_url:
        typer.secho("QDRANT_URL environment variable is not set", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    if not force and not typer.confirm(f"Delete ALL Qdrant collections at {qdrant_url}? This action is irreversible."):
        raise typer.Exit()

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        typer.secho("The 'qdrant-client' package is required (pip install qdrant-client).", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Connect to Qdrant with or without API key
    try:
        if qdrant_api_key:
            typer.secho(f"Connecting to Qdrant at {qdrant_url} with API key", fg=typer.colors.BLUE)
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            typer.secho(f"Connecting to Qdrant at {qdrant_url} without API key", fg=typer.colors.BLUE)
            client = QdrantClient(url=qdrant_url)
            
        collections = [c.name for c in client.get_collections().collections]
        if not collections:
            typer.secho("No collections found – nothing to flush.", fg=typer.colors.YELLOW)
            return

        for coll in collections:
            typer.secho(f"Deleting collection '{coll}'...", fg=typer.colors.YELLOW)
            client.delete_collection(coll)

        typer.secho("All collections deleted.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error connecting to Qdrant: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@cli.command("rebuild")
def cmd_rebuild() -> None:
    """Rebuild Docker images for all services (use after code changes)."""
    typer.secho("Rebuilding Docker images...", fg=typer.colors.YELLOW)
    _run("docker compose build --pull", cwd=REPO_ROOT)
    typer.secho("Rebuild completed. Run 'run.sh restart' to apply the new images.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------



def _is_api_running() -> bool:
    """Check if the API is running."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:8000/health"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------- Extra helpers ---------------- #


def load_env() -> None:
    """Load environment variables from .env file."""
    env_path = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(env_path):
        try:
            # Try to use python-dotenv if available
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                return
            except ImportError:
                pass
                
            # Fallback to manual parsing
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        except Exception as e:
            typer.secho(f"Error loading .env file: {e}", fg=typer.colors.RED)
    else:
        typer.secho(".env file not found", fg=typer.colors.YELLOW)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@cli.command("test")
def cmd_test(
    scope: str = typer.Option("local", "--scope", "-s", help="Test scope: local or cloud"),
    markers: Optional[str] = typer.Option(None, "-m", help="Additional pytest markers expression"),
    verbose: bool = typer.Option(False, "-v", help="Verbose pytest output"),
) -> None:
    """Alias for the *tests* command so that `sentio_cli test` feels natural."""
    cmd_tests(scope=scope, markers=markers, verbose=verbose)


# ---------------------------------------------------------------------------
# Convenience alias for build-index → index
# ---------------------------------------------------------------------------

@cli.command("index")
def cmd_index(
    source: Path = typer.Argument(..., exists=True, readable=True),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    bm25_path: Optional[Path] = typer.Option(None, "--bm25-path", "-b"),
    skip_dense: bool = typer.Option(False, help="Skip dense (Qdrant) index"),
    skip_sparse: bool = typer.Option(False, help="Skip sparse (BM25) index"),
) -> None:
    """Alias for the *build-index* command."""
    cmd_build_index(
        source=source,
        collection=collection,
        bm25_path=bm25_path,
        skip_dense=skip_dense,
        skip_sparse=skip_sparse,
    )

def main() -> None:
    """CLI entry point."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main() 