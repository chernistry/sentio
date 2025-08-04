"""Dependency checker for Sentio vNext.

This module checks for the presence of required dependencies and
provides helpful error messages when they're missing.
"""

from __future__ import annotations

import importlib.util
import logging
import sys

logger = logging.getLogger(__name__)

# Required dependencies by component
DEPENDENCIES: dict[str, list[str]] = {
    "core": ["httpx", "requests"],
    "embeddings": ["httpx", "requests"],
    "ingest": ["pandas", "typer"],
    "api": ["fastapi", "uvicorn", "pydantic"],
    "vector_store": ["qdrant-client"],
    "retrieval": ["rank_bm25", "qdrant-client"],
    "rerank": ["sentence-transformers"],
}


def is_package_installed(package_name: str) -> bool:
    """Return *True* when package importable."""
    return importlib.util.find_spec(package_name) is not None


def check_dependencies(components: list[str] | None = None) -> dict[str, set[str]]:
    """Return mapping of *components* to missing dependencies."""
    components = components or list(DEPENDENCIES.keys())
    missing_deps: dict[str, set[str]] = {}
    for component in components:
        if component not in DEPENDENCIES:
            logger.warning("Unknown component: %s", component)
            continue
        missing = {dep for dep in DEPENDENCIES[component] if not is_package_installed(dep)}
        if missing:
            missing_deps[component] = missing
    return missing_deps


def print_dependency_report(missing_deps: dict[str, set[str]]) -> bool:
    """Print human-readable dependency report. Return *True* if all good."""
    if not missing_deps:
        print("✅ All dependencies installed!")
        return True

    print("❌ Missing dependencies:")
    for component, deps in missing_deps.items():
        print(f"  - {component}: {', '.join(deps)}")

    # Consolidated install hint
    all_missing: set[str] = set().union(*missing_deps.values())
    print("\nInstall missing dependencies with:")
    print(f"pip install {' '.join(sorted(all_missing))}")
    return False


if __name__ == "__main__":
    missing = check_dependencies()
    if not print_dependency_report(missing):
        sys.exit(1)
