"""Beam Cloud runtime wrapper for Sentio.

This module provides utilities for creating and managing Beam Cloud resources
like Images, Volumes, and task queues.
"""

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast
from pathlib import Path

from root.src.utils.settings import settings

# Import Beam SDK conditionally to avoid hard dependency.  Some environments
# may have the *beam* package installed but mis-configured (e.g. missing
# ~/.beam/config).  Any exception during import should therefore trigger the
# fallback stub definitions.
try:
    from beam import Image, Volume, task_queue, env, Output  # type: ignore
    BEAM_AVAILABLE = True
except Exception:  # pragma: no cover – broad catch is intentional here
    # Create stub classes/functions for type checking when Beam is not available
    class Image:  # type: ignore
        def __init__(self, python_version: str = "python3.10", 
                     python_packages: str = "requirements.txt") -> None:
            self.python_version = python_version
            self.python_packages = python_packages

    class Volume:  # type: ignore
        def __init__(self, name: str, mount_path: str) -> None:
            self.name = name
            self.mount_path = mount_path

    class Output:  # type: ignore
        pass

    def task_queue(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator

    class env:  # type: ignore
        @staticmethod
        def is_remote() -> bool:
            return False

    BEAM_AVAILABLE = False


F = TypeVar('F', bound=Callable[..., Any])


class BeamRuntime:
    """Wrapper for Beam Cloud runtime operations."""

    @staticmethod
    def is_remote() -> bool:
        """Check if code is running in Beam Cloud environment.

        Returns:
            bool: True if running in Beam Cloud, False otherwise.
        """
        if not BEAM_AVAILABLE:
            return False
        return env.is_remote()

    @staticmethod
    def create_image(
        python_version: str = "python3.10",
        packages_file: str | None = "requirements.txt",
    ) -> Image:
        """Create a Beam Image configuration.

        Args:
            python_version: Python version to use.
            packages_file: Path to requirements file.

        Returns:
            Image: Beam Image configuration.
        """
        # ------------------------------------------------------------------
        # Resolve *python_packages* argument:
        # Beam SDK accepts either a *list[str]* **or** a path to a
        # requirements file.  When running *beam deploy* from nested folders
        # the default "requirements.txt" may not exist → we fall back to an
        # empty list to avoid FileNotFoundError.
        # ------------------------------------------------------------------
        python_packages: str | list[str]
        if packages_file and Path(packages_file).expanduser().exists():
            python_packages = packages_file
        else:
            python_packages = []  # handled later via Poetry/lockfile layer

        return Image(
            python_version=python_version,
            python_packages=python_packages,
        )

    @staticmethod
    def create_volume(name: Optional[str] = None, 
                      mount_path: str = "./models") -> Volume:
        """Create a Beam Volume configuration.

        Args:
            name: Volume name, defaults to settings.beam_volume.
            mount_path: Path where volume will be mounted.

        Returns:
            Volume: Beam Volume configuration.
        """
        volume_name = name or settings.beam_volume
        return Volume(name=volume_name, mount_path=mount_path)

    @staticmethod
    def task_decorator(
        name: str,
        cpu: Optional[int] = None,
        memory: Optional[str] = None,
        gpu: Optional[str] = None,
        volumes: Optional[List[Volume]] = None,
        image: Optional[Image] = None,
    ) -> Callable[[F], F]:
        """Decorator for Beam task queues with settings integration.

        Args:
            name: Task name.
            cpu: CPU cores, defaults to settings.beam_cpu.
            memory: Memory allocation, defaults to settings.beam_memory.
            gpu: GPU type, defaults to settings.beam_gpu.
            volumes: List of volumes, defaults to [BeamRuntime.create_volume()].
            image: Image configuration, defaults to BeamRuntime.create_image().

        Returns:
            Callable: Decorator function.
        """
        if not BEAM_AVAILABLE:
            def local_decorator(func: F) -> F:
                @wraps(func)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    return func(*args, **kwargs)
                return cast(F, wrapper)
            return local_decorator

        # Use settings as defaults
        _cpu = cpu or settings.beam_cpu
        _memory = memory or settings.beam_memory
        _gpu = gpu or settings.beam_gpu
        _volumes = volumes or [BeamRuntime.create_volume()]
        _image = image or BeamRuntime.create_image()

        return task_queue(
            name=name,
            cpu=_cpu,
            memory=_memory,
            gpu=_gpu,
            volumes=_volumes,
            image=_image,
        )


# Convenience decorator for local and remote tasks
def local_task(
    name: Optional[str] = None,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Decorator for tasks that work both locally and on Beam.

    Args:
        name: Task name, defaults to function name.
        **kwargs: Additional arguments for BeamRuntime.task_decorator.

    Returns:
        Callable: Decorated function that works locally and on Beam.
    """
    def decorator(func: F) -> F:
        task_name = name or func.__name__
        
        if not BEAM_AVAILABLE:
            return func
            
        return BeamRuntime.task_decorator(name=task_name, **kwargs)(func)
        
    return decorator 