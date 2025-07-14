"""
Tests to verify that all import paths are correct and modules can be imported.
This helps catch issues with module structure and import paths early.
"""

import importlib
import pytest
from types import ModuleType


def test_core_modules_importable():
    """Test that core modules can be imported correctly."""
    modules = [
        "root.src.core.tasks.ingest",
        "root.src.utils.docker_wrapper",
    ]

    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


def test_docker_wrapper_imports_ingest():
    """Test that docker_wrapper can correctly import the ingest module."""
    from root.src.utils.docker_wrapper import _import_ingest

    ingest_module = _import_ingest()
    assert hasattr(ingest_module, "main"), "Ingest module should have a main function"
    assert callable(getattr(ingest_module, "main")), "main should be callable"


def test_entrypoint_paths():
    """
    Test that entrypoint paths in docker-compose.yml match actual module locations.
    
    This is a simple check to ensure that the entrypoint paths in docker-compose.yml
    match the actual module locations in the codebase.
    """
    import os
    import yaml
    
    # Path to docker-compose.yml relative to project root
    docker_compose_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      "docker-compose.yml")
    
    with open(docker_compose_path, "r") as f:
        docker_compose = yaml.safe_load(f)
    
    # Check ingest service entrypoint
    ingest_entrypoint = docker_compose["services"]["ingest"]["entrypoint"][1]
    assert ingest_entrypoint == "-m", "Second element should be -m flag"
    
    module_path = docker_compose["services"]["ingest"]["entrypoint"][2]
    # The path should be to docker_wrapper, not to ingest
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        pytest.fail(f"Entrypoint module {module_path} cannot be imported: {e}") 