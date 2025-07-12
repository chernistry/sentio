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
from types import ModuleType
from typing import Any

# Lazy import to avoid unnecessary startup cost if the module graph changes.

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