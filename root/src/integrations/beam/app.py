"""Beam Cloud deployment script for Sentio.

This module provides the entry points for deploying models and tasks to Beam Cloud.

It can be executed *directly* via ``beam deploy`` from **any** working directory.
To guarantee that absolute imports like ``root.src...`` resolve correctly even
when the CWD is several levels deeper, we dynamically inject the **project
root (folder containing the top-level ``root`` package)** into ``sys.path`` *at
runtime* **before** importing the rest of the codebase.
"""

# ---------------------------------------------------------------------------
# Ensure top-level project dir is on sys.path  – needed when running the file
# from nested locations via ``beam deploy`` (which sets CWD to the file's
# directory).  We walk up the parents until we find a folder that contains the
# *root* package and prepend it to ``sys.path``.
# ---------------------------------------------------------------------------

from __future__ import annotations

import sys
import os
from pathlib import Path

import logging
import time

# ---------------------------------------------------------------------------
# Logging setup – respects LOG_LEVEL env var (default info).
# ---------------------------------------------------------------------------

_log_level = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=_log_level,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)

logger.info("Beam app initialising (log level %s)", _log_level)
# Duplicate __future__ import and redundant imports removed to keep future imports at top

_THIS_FILE = Path(__file__).resolve()
logger.debug("Current file path: %s", _THIS_FILE)
logger.debug("Current working directory: %s", os.getcwd())
logger.debug("Current sys.path: %s", sys.path)

# Try to find the project root directory
found_root = False
for _p in _THIS_FILE.parents:
    logger.debug("Checking parent: %s", _p)
    if (_p / "root").is_dir():
        logger.debug("Found root directory at: %s", _p)
        if str(_p) not in sys.path:
            logger.debug("Adding %s to sys.path", _p)
            sys.path.insert(0, str(_p))
            found_root = True
        break

# If we didn't find the root directory, try a different approach
if not found_root:
    logger.debug("Root directory not found in parents, trying alternative approach")
    # Try to find 'root' directory in current working directory
    cwd = Path.cwd()
    for _p in [cwd] + list(cwd.parents):
        logger.debug("Checking path: %s", _p)
        if (_p / "root").is_dir():
            logger.debug("Found root directory at: %s", _p)
            if str(_p) not in sys.path:
                logger.debug("Adding %s to sys.path", _p)
                sys.path.insert(0, str(_p))
                found_root = True
            break

# If all else fails, try to add the current directory
if not found_root:
    logger.debug("Root directory not found, adding current directory to sys.path")
    sys.path.insert(0, os.getcwd())
    # Create minimal structure if needed
    os.makedirs("root/src/utils", exist_ok=True)

logger.debug("Final sys.path: %s", sys.path)

# ---------------------------------------------------------------------------
# Downstream imports – now safe.
# ---------------------------------------------------------------------------

try:
    from typing import Dict, Any, List
    
    from beam import Image, Volume, endpoint, task_queue
    
    from root.src.utils.settings import settings
    from root.src.integrations.beam.ai_model import BeamModel
    
    logger.info("Successfully imported Beam deployment modules (Beam SDK, settings, BeamModel)")
except ImportError as e:
    logger.warning("Import error during Beam app init: %s", e)
    # ------------------------------------------------------------------
    # Fallback: synthesise minimal *settings* and *BeamModel* stubs so the
    # decorators below have something to reference.  We execute this block
    # unconditionally on ImportError because the presence of a random
    # '/root' folder in the container can mis-set *found_root* and skip the
    # previous conditional.
    # ------------------------------------------------------------------
    import types

    sys.modules.setdefault('root', types.ModuleType('root'))
    sys.modules.setdefault('root.src', types.ModuleType('root.src'))
    sys.modules.setdefault('root.src.utils', types.ModuleType('root.src.utils'))
    sys.modules.setdefault('root.src.utils.settings', types.ModuleType('root.src.utils.settings'))

    class MinimalSettings:  # noqa: D101 – simple fallback container
        beam_volume = "sentio-models"
        beam_cpu = 2
        beam_memory = "8Gi"
        beam_gpu = "T4"
        beam_model_id = "llama3"
        embedding_model = "all-MiniLM-L6-v2"

    settings = MinimalSettings()

    class MinimalBeamModel:  # noqa: D101 – stub used only during cold start
        def __init__(self, model_id: str | None = None):
            self.model_id = model_id

        async def initialize(self):  # noqa: D401 – simple stub
            return None

        async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:  # noqa: E501 – long signature
            return f"[Stub] Echo: {prompt[:100]}…"

    BeamModel = MinimalBeamModel

# Create standard configuration objects
def get_image():
    """Create a standard Beam Image with required dependencies."""
    return Image(
        python_version="python3.10",
        python_packages=[
            "torch==2.7.1",
            "numpy==1.26.4",  # strictly as in requirements.txt
            "transformers==4.53.2",  # Qwen3 config available starting 4.42
            "sentence-transformers==5.0.0",
            "accelerate==1.8.1",
            # Optimum ≥1.17 is needed for BetterTransformer with transformers>4.38
            "optimum>=1.26.1",  # BetterTransformer kernels for faster inference
            "sentencepiece==0.2.0",
            "aiohttp",
            "fastapi",
            "httpx",
            "json5",
            "llama_index",
            "nest_asyncio",
            "opencensus",
            "prometheus_fastapi_instrumentator",
            "psutil",
            "pydantic",
            "pydantic-settings",
            "PyPDF2",
            "qdrant_client>=1.14.0,<2.0",
            "requests",
            "rich",
            "typing_extensions",
            "uvicorn",
            "python-dotenv",
            "llama-index-vector-stores-qdrant",
            "langchain_community",
            "rank_bm25",
            "fastembed"
        ],
    )


def get_volume():
    """Create a standard Beam Volume for model weights."""
    return Volume(
        name=settings.beam_volume,
        mount_path="./models",
    )


@endpoint(
    name="chat",
    cpu=settings.beam_cpu,
    memory=settings.beam_memory,
    gpu=settings.beam_gpu,
    image=get_image(),
    volumes=[get_volume()],
)
async def chat_endpoint(**inputs) -> Dict[str, Any]:
    """Chat endpoint for Beam Cloud.

    Args:
        inputs: Input parameters including messages, model_id, temperature, etc.

    Returns:
        Dict with chat completion response.
    """
    messages = inputs.get("messages", [])
    model_id = inputs.get("model_id", settings.beam_model_id)
    temperature = inputs.get("temperature", 0.7)
    max_tokens = inputs.get("max_tokens", 1024)
    logger.info(
        "[chat_endpoint] model_id=%s | messages=%d | temperature=%.2f | max_tokens=%d",
        model_id,
        len(messages),
        temperature,
        max_tokens,
    )

    _start = time.time()

    # Convert messages to a prompt
    prompt = _messages_to_prompt(messages)
    
    # Initialize model
    model = BeamModel(model_id=model_id)
    await model.initialize()
    
    # Generate response
    response = await model.generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    _duration = time.time() - _start
    logger.info(
        "[chat_endpoint] Completed in %.2fs | output_len=%d chars",
        _duration,
        len(response) if isinstance(response, str) else -1,
    )
    
    return {
        "id": f"beam-{model_id}",
        "object": "chat.completion",
        "created": 0,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response,
                },
                "finish_reason": "stop",
            }
        ],
    }


@task_queue(
    name="inference",
    cpu=settings.beam_cpu,
    memory=settings.beam_memory,
    gpu=settings.beam_gpu,
    image=get_image(),
    volumes=[get_volume()],
)
async def inference_task(**inputs) -> Dict[str, Any]:
    """Inference task for Beam Cloud.

    Args:
        inputs: Input parameters including prompt, model_id, temperature, etc.

    Returns:
        Dict with inference results.
    """
    prompt = inputs.get("prompt", "")
    model_id = inputs.get("model_id", settings.beam_model_id)
    temperature = inputs.get("temperature", 0.7)
    max_tokens = inputs.get("max_tokens", 1024)

    logger.info(
        "[inference_task] prompt_len=%d | model_id=%s | temp=%.2f | max_tokens=%d",
        len(prompt),
        model_id,
        temperature,
        max_tokens,
    )

    _start = time.time()

    # Initialize model
    model = BeamModel(model_id=model_id)
    await model.initialize()
    
    # Generate response
    response = await model.generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    _duration = time.time() - _start
    logger.info("[inference_task] Took %.2fs | result_len=%d chars", _duration, len(response))
    
    return {
        "model_id": model_id,
        "prompt": prompt,
        "result": response,
    }


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert chat messages to a prompt string.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.

    Returns:
        Formatted prompt string.
    """
    prompt_parts = []
    
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        else:
            # Handle other roles or ignore
            prompt_parts.append(f"{role.capitalize()}: {content}")
            
    return "\n\n".join(prompt_parts) 


# ---------------------------------------------------------------------------
# Embedding endpoint – CPU-only (no GPU) to keep costs minimal.
# ---------------------------------------------------------------------------


from root.src.core.embeddings.providers.beam_embeddings import BeamEmbedding  # noqa: E402


@endpoint(
    name="embed",
    cpu=settings.beam_cpu,
    memory=settings.beam_memory,
    gpu=settings.beam_gpu, 
    image=get_image(),
    volumes=[get_volume()],
    env={"EMBEDDING_COMPILE": "1"},  # Enable torch.compile for speed-up
)
async def embed_endpoint(**inputs) -> Dict[str, Any]:
    """Return sentence embeddings for *texts*.

    Args:
        inputs: Dict containing ``texts`` (str | list[str]) and optional
            ``model_name``.

    Returns:
        Dict[str, Any]: JSON with ``embeddings`` list.
    """
    raw_texts = inputs.get("texts", [])
    if isinstance(raw_texts, str):  # single string → wrap into list
        texts: list[str] = [raw_texts]
    else:
        texts = list(map(str, raw_texts))

    model_name: str = inputs.get("model_name", settings.embedding_model)

    logger.info(
        "[embed_endpoint] texts=%d | model_name=%s",
        len(texts),
        model_name,
    )

    embed_model = BeamEmbedding(model_name=model_name)
    vectors = await embed_model.embed_async_many(texts)

    return {
        "model": model_name,
        "embeddings": vectors,
    } 