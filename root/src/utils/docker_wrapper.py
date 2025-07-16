"""Sentio ETL Docker wrapper.

This tiny module exists solely as the Docker entry-point for the *ingest*
service declared in *docker-compose.yml*:

```
entrypoint: ["python", "-m", "root.src.utils.docker_wrapper", "--data_dir", ...]
```

It forwards the CLI arguments unmodified to *root.src.core.tasks.ingest.main* so we can
reuse the richer argument parser already implemented there.

The wrapper keeps the surface small on purpose – any real logic lives in
*root.src.core.tasks.ingest*.
"""

from __future__ import annotations

import asyncio
import sys
import os
from types import ModuleType
from typing import Any
from pathlib import Path

# Lazy import to avoid unnecessary startup cost if the module graph changes.

def _fix_import_paths() -> None:
    """
    Fix Python import paths to ensure modules can be found regardless of CWD.
    
    This function adds the project root and plugins directory to sys.path
    to make imports work correctly in Docker containers.
    """
    # Get the absolute path to the project root (two levels up from this file)
    current_file = Path(__file__).resolve()
    utils_dir = current_file.parent
    src_dir = utils_dir.parent
    root_dir = src_dir.parent
    project_root = root_dir.parent
    
    # Add project root to sys.path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"[docker_wrapper] Added {project_root} to sys.path")
    
    # Check if plugins directory exists and add it to sys.path
    plugins_dir = project_root / "plugins"
    if plugins_dir.exists() and str(plugins_dir) not in sys.path:
        sys.path.insert(0, str(plugins_dir))
        print(f"[docker_wrapper] Added {plugins_dir} to sys.path")
    
    # Check if .env file exists in project root and load it
    env_file = project_root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_file)
            print(f"[docker_wrapper] Loaded environment from {env_file}")
        except ImportError:
            print("[docker_wrapper] python-dotenv not available, skipping .env loading")

def _import_ingest() -> ModuleType:  # noqa: D401 – imperative form is OK
    """Import the *src.etl.ingest* module safely.

    Returns:
        The imported module.

    Raises:
        ImportError: If the target module cannot be found.
    """
    import importlib

    module_name = "root.src.core.tasks.ingest"
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover – surfaced in container logs
        msg = (
            f"Failed to import '{module_name}'. The Docker image may be out of "
            "sync with the source tree. Rebuild the image or ensure the module "
            "exists."
        )
        raise ImportError(msg) from exc


def _run() -> None:  # noqa: D401
    """Entrypoint helper that executes *ingest.main* asynchronously."""
    # Fix import paths before attempting any imports
    _fix_import_paths()
    
    ingest_mod: ModuleType = _import_ingest()

    # *ingest.main* is an *async def* that expects *sys.argv* to already contain
    # any CLI arguments provided via *docker-compose*. We therefore forward the
    # current process argv untouched.
    main_fn: Any = getattr(ingest_mod, "main", None)
    if main_fn is None or not callable(main_fn):
        raise AttributeError(
            "The target module 'root.src.core.tasks.ingest' does not expose an async 'main' "
            "function."
        )

    # Ensure we are dealing with an async coroutine function.
    if not asyncio.iscoroutinefunction(main_fn):
        raise TypeError("ingest.main must be an async function")

    # Run inside a fresh event loop to avoid nested-loop issues.
    asyncio.run(main_fn())


if __name__ == "__main__":  # pragma: no cover
    try:
        _run()
    except Exception as exc:  # noqa: BLE001 – visible in container output
        # Print to stderr then exit with non-zero status for Docker to report.
        sys.stderr.write(f"[docker_wrapper] Fatal error: {exc}\n")
        sys.exit(1) 