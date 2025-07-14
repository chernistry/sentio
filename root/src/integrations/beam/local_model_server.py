from __future__ import annotations

"""Local embedding server for development.

This module starts a FastAPI app that exposes a single **/embed** endpoint
compatible with the production Beam embedding API.  It loads any
`sentence-transformers` model specified in the ``EMBEDDING_MODEL`` env var and
listens on the host/port derived from ``BEAM_EMBEDDING_BASE_LOCAL_URL``.

Run:
    python -m root.src.integrations.beam.local_model_server

The server detects Mac M-series chips and transparently switches to the MPS
backend if available (requires PyTorch ≥ 2.1 compiled with Metal).
"""

from pathlib import Path
from typing import List
from urllib.parse import urlparse

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv, find_dotenv

# ---------------------------------------------------------------------------
# Load .env early so subsequent imports (incl. settings) see the variables
# ---------------------------------------------------------------------------
_env_path = find_dotenv(usecwd=True)
if _env_path:
    load_dotenv(_env_path, override=False)

from root.src.utils.settings import settings  # must come after dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer  # type: ignore
import torch

# ---------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.info, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment & configuration -------------------------------------------------
# ---------------------------------------------------------------------------

# Model name to load – falls back to sensible default
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", settings.embedding_model or "sentence-transformers/all-MiniLM-L6-v2")

# Optional HuggingFace token – used for gated/private models
HF_TOKEN = os.getenv("HF_TOKEN") or None

# Derive host, port and path from BEAM_EMBEDDING_BASE_LOCAL_URL; defaults keep
# backward-compat if var is missing.
_parsed = urlparse(os.environ.get("BEAM_EMBEDDING_BASE_LOCAL_URL", "http://0.0.0.0:8003/embed"))
# Always use 0.0.0.0 for listening to ensure the server is accessible both locally and for Docker
HOST = "0.0.0.0"
PORT = _parsed.port or 8003
API_PATH = (_parsed.path or "/embed").rstrip("/")

# Cache folder for model weights – inside project to speed up reloads.
CACHE_DIR = Path(os.environ.get("LOCAL_MODEL_CACHE_DIR", "./models")).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Using model '%s' (cache_dir=%s)", MODEL_NAME, CACHE_DIR)
logger.info("Listening on http://%s:%s%s", HOST, PORT, API_PATH)
logger.info("Docker access URL: http://host.docker.internal:%s%s", PORT, API_PATH)

# ---------------------------------------------------------------------------
# Device selection -----------------------------------------------------------
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
logger.info("Selected device: %s", DEVICE)

# ---------------------------------------------------------------------------
# Load model (lazy; we load inside a function to avoid blocking import time) ---
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None  # loaded on first request


def _get_model() -> SentenceTransformer:  # noqa: D401
    """Return a cached :class:`SentenceTransformer` instance."""
    global _model  # noqa: PLW0603 – intentional module-level cache
    if _model is None:
        logger.info("Loading SentenceTransformer '%s' (device=%s) – this may take a while...", MODEL_NAME, DEVICE)
        _model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(CACHE_DIR),
            device=DEVICE,
            use_auth_token=HF_TOKEN,
        )
        logger.info("Model loaded (embedding dim=%s)", _model.get_sentence_embedding_dimension())
    return _model


# ---------------------------------------------------------------------------
# FastAPI schema --------------------------------------------------------------
# ---------------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    texts: List[str] | str


class EmbeddingResponse(BaseModel):
    model: str
    embeddings: List[List[float]]


app = FastAPI(title="Local Beam Embedding Server", version="1.0.0")


@app.post(API_PATH, response_model=EmbeddingResponse)
async def embed(req: EmbeddingRequest) -> EmbeddingResponse:  # noqa: D401
    """Return sentence embeddings for the supplied texts."""
    texts_raw = req.texts
    texts: List[str] = [texts_raw] if isinstance(texts_raw, str) else list(map(str, texts_raw))

    if not texts:
        raise HTTPException(status_code=400, detail="'texts' cannot be empty")

    model = _get_model()

    # Determine batch size – default 16 (matches .env), can be overridden via ENV.
    max_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    
    # For MPS (Apple Silicon GPU), use smaller batch size to avoid OOM errors
    if DEVICE == "mps":
        # Limit batch size for MPS to 4 to avoid memory shortage errors
        max_batch_size = min(max_batch_size, 4)
        logger.info(f"Using reduced batch size for MPS device: {max_batch_size}")
        
    batch_size: int = min(max_batch_size, len(texts))

    # PyTorch ops are blocking – run in a thread pool so we don't block the event loop.
    loop = asyncio.get_event_loop()
    vectors: List[List[float]] = await loop.run_in_executor(
        None,  # default ThreadPoolExecutor
        lambda: model.encode(  # type: ignore[arg-type]
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
        ).tolist(),
    )

    return EmbeddingResponse(model=MODEL_NAME, embeddings=vectors)


# ---------------------------------------------------------------------------
# Entrypoint ------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    # Run uvicorn programmatically; extra lifespan tasks can be added later.
    try:
        import uvicorn  # noqa: WPS433 – runtime import
    except ImportError as exc:
        logger.error("uvicorn is required to run the server: pip install 'uvicorn[standard]'")
        sys.exit(1)

    uvicorn.run(
        "root.src.integrations.beam.local_model_server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
