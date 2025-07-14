import asyncio
import logging
from functools import partial
from typing import Any, List
import contextlib  # needed for nullcontext in CPU path
import time  # runtime profiling

import httpx
import os

# GPU acceleration
try:
    import torch  # noqa: F401 – optional runtime dep
    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover – torch optional
    _HAS_CUDA = False

# Optional heavy deps – imported lazily (only when using local model)
SentenceTransformer: Any | None  # noqa: N815 – keep camel-case alias
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover – optional dep in CI
    SentenceTransformer = None  # type: ignore

from root.src.core.tasks.embeddings import BaseEmbeddingModel, EmbeddingError
from root.src.integrations.beam.runtime import BeamRuntime
from root.src.utils.settings import settings

logger = logging.getLogger(__name__)


class BeamEmbedding(BaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = "Qwen3-Embedding-0.6B",
        cache_enabled: bool = True,
        cache_size: int = 10_000,
        cache_ttl: int = 3_600,
        remote_base_url: str | None = None,
        timeout: int = 180,
        batch_size: int = 256,
        **kwargs: Any,
    ) -> None:  # noqa: D401 – descriptive
        """Initialize Beam embedding provider.

        In *remote* Beam containers we load the model locally (shared volume).
        During local development we prefer to call the already-deployed Beam
        endpoint to avoid the heavyweight SentenceTransformer dependency.
        """

        # Initialize with a default dimension that matches the model
        # This prevents errors when creating collections before first embedding
        self._model_name = model_name  # keep reference for reloading on fallback

        if "Qwen3-Embedding-0.6B" in model_name:
            self._dimension = 1024  # Known dimension
        else:
            # Default to 1024 for most modern embedding models
            self._dimension = 1024
        
        logger.info(f"Pre-configured embedding dimension: {self._dimension}")

        # Determine remote base URL based on runtime mode ---------------------------------
        logger.info(
            "[BeamEmbedding] Initialising model '%s' (provider=beam) – cache_enabled=%s",
            model_name,
            cache_enabled,
        )

        _init_start = time.time()

        if "Qwen3-Embedding-0.6B" in model_name:
            self._dimension = 1024  # Known dimension
        else:
            # Default to 1024 for most modern embedding models
            self._dimension = 1024
        
        logger.info(f"Pre-configured embedding dimension: {self._dimension}")

        # Determine remote base URL based on runtime mode ---------------------------------
        if remote_base_url is not None:
            _base_url = remote_base_url
        else:
            # If beam_embedding_base_url is not set, determine it based on BEAM_MODE
            if settings.beam_embedding_base_url:
                _base_url = settings.beam_embedding_base_url
            elif settings.beam_mode == "local" and settings.beam_embedding_base_local_url:
                _base_url = settings.beam_embedding_base_local_url
                logger.info(f"Using local embedding URL from settings: {_base_url}")
            elif settings.beam_mode == "cloud" and settings.BEAM_EMBEDDING_BASE_CLOUD_URL:
                _base_url = settings.BEAM_EMBEDDING_BASE_CLOUD_URL
                logger.info(f"Using cloud embedding URL from settings: {_base_url}")
            else:
                _base_url = None
                logger.warning(
                    "No embedding URL found in settings. Set BEAM_EMBEDDING_BASE_LOCAL_URL for local mode "
                    "or BEAM_EMBEDDING_BASE_CLOUD_URL for cloud mode."
                )

        # Check that the URL does not contain extra spaces
        if _base_url and isinstance(_base_url, str):
            _base_url = _base_url.strip()
            logger.info(f"Using cleaned embedding URL: {_base_url}")

        self._remote_base_url: str | None = _base_url

        # ------------------------------------------------------------------------------
        # Get timeout from environment variable or use default value
        self._timeout = int(os.getenv("EMBEDDING_TIMEOUT", str(timeout)))
        logger.info(f"Using embedding timeout: {self._timeout}s")
        
        # Get batch size from environment variable or use default value
        # On GPUs like A10G we allow large initial batch sizes for maximum throughput.
        # The adaptive OOM logic below will automatically down-scale if memory is
        # insufficient, so we no longer pre-cap the value to 16.
        _env_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(batch_size)))
        self._batch_size = _env_batch_size

        if _HAS_CUDA:
            logger.info(
                "GPU detected – initial embedding batch size set to %s (auto-tuned on OOM)",
                self._batch_size,
            )
        else:
            logger.info("CPU mode – embedding batch size: %s", self._batch_size)

        # Call the parent constructor BEFORE accessing self.cache
        super().__init__(
            model_name=model_name,
            cache_enabled=cache_enabled,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            _skip_dimension_check=True,  # Skip dimension check in parent constructor
            **kwargs,
        )
        
        # Detect execution context --------------------------------------------------
        self._use_remote_http = not BeamRuntime.is_remote() and self._remote_base_url
        self._cache_enabled = cache_enabled
        
        logger.info(f"Remote base URL: {self._remote_base_url}")
        logger.info(f"BEAM_MODE: {settings.beam_mode}")
        logger.info(f"Using remote HTTP: {self._use_remote_http}")
        
        if self._use_remote_http:
            logger.info(
                "BeamEmbedding using remote HTTP endpoint at %s",
                self._remote_base_url,
            )
            self._st_model = None  # type: ignore[attr-defined]
            
            # Immediately try to get a test embedding to validate dimension
            try:
                test_text = "This is a test text for embedding dimension verification."
                test_embedding = self.get_text_embedding(test_text)
                if test_embedding and len(test_embedding) > 0:
                    self._dimension = len(test_embedding)
                    logger.info(f"Validated embedding dimension via test request: {self._dimension}")
            except Exception as e:
                # Do not fail hard here – server might need more time to load the model.
                # We keep the pre-configured dimension and continue; any real embedding
                # request will update it automatically once the server responds.
                logger.warning(f"Skipping dimension validation: {e}")
                
        else:
            # Only allow fallback to local model in cloud mode
            if settings.beam_mode == "local" and not BeamRuntime.is_remote():
                raise EmbeddingError(
                    f"Embedding server not found at {self._remote_base_url}. "
                    f"Make sure to start the local model server with: "
                    f"python -m root.src.integrations.beam.local_model_server"
                )
                
            if SentenceTransformer is None:
                raise EmbeddingError(
                    "sentence-transformers not installed – add it to requirements",
                )

            cache_dir = "./models"
            if BeamRuntime.is_remote():
                logger.info(
                    "Running inside Beam – using volume path '%s'", cache_dir,
                )

            # ------------------------------------------------------------------
            # Prefer GPU when available inside Beam container – gives 10-20× speed-up
            # ------------------------------------------------------------------
            _device = "cuda" if _HAS_CUDA else "cpu"
            logger.info(
                "Loading SentenceTransformer on %s (CUDA available: %s)",
                _device,
                _HAS_CUDA,
            )

            self._st_model = SentenceTransformer(  # type: ignore[attr-defined]
                model_name,
                cache_folder=cache_dir,
                device=_device,
            )
            self._device = _device  # track current device
            self._dimension = int(self._st_model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]

            # --------------------------------------------------------------
            # 🏎  Speed-ups (Beam remote only)
            # --------------------------------------------------------------
            if BeamRuntime.is_remote():
                # 1) BetterTransformer – fused kernels & pad-skipping
                try:
                    from optimum.bettertransformer import BetterTransformer  # noqa: WPS433 – optional heavy dep

                    base_model = self._st_model._first_module()  # SentenceTransformer helper
                    if not getattr(base_model, "_bettertransformer_transform", False):
                        bt_model = BetterTransformer.transform(base_model, keep_original_model=False)
                        # Replace the original module inside the SentenceTransformer stack
                        self._st_model._modules[self._st_model._first_module_name] = bt_model  # type: ignore[index]
                        logger.info("✓ BetterTransformer optimisation applied – fused attention kernels enabled")
                except Exception as exc:  # pragma: no cover – non-fatal
                    logger.warning("BetterTransformer optimisation skipped: %s", exc)

                # 2) torch.compile (opt-in via env flag) – further graph fusion
                if os.getenv("EMBEDDING_COMPILE", "0") == "1":
                    try:
                        import torch

                        compiled_model = torch.compile(
                            self._st_model._first_module(),  # type: ignore[arg-type]
                            mode="reduce-overhead",
                            dynamic=True,
                        )
                        self._st_model._modules[self._st_model._first_module_name] = compiled_model  # type: ignore[index]
                        logger.info("✓ torch.compile optimisation applied – dynamic batches supported")
                    except Exception as exc:  # pragma: no cover – non-fatal
                        logger.warning("torch.compile optimisation skipped: %s", exc)
 
            logger.info(
                "[BeamEmbedding] Model loaded on %s, dim=%d, load_time=%.2fs",
                self._device,
                self._dimension,
                time.time() - _init_start,
            )

            try:
                import torch
                logger.debug(
                    "torch version=%s, cudnn=%s, cuda available=%s, device capability=%s",
                    torch.__version__,
                    torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                    torch.cuda.is_available(),
                    torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
                )
            except Exception:  # pragma: no cover – safety
                pass

    def _get_embedding_dimension(self) -> int:  # noqa: D401 – interface impl
        """Return the embedding dimension.
        
        This method is called by the parent class constructor, so we need to ensure
        that self._dimension is already initialized before this method is called.
        """
        # Safely return the dimension, even if the attribute is not yet created.
        if not hasattr(self, "_dimension") or self.__dict__.get("_dimension") is None:
            # Create and return default (1024) to avoid AttributeError
            self._dimension = 1024  # type: ignore[attr-defined]
            return 1024
        try:
            return int(self._dimension)  # type: ignore[arg-type]
        except Exception:
            # Last resort: ensure an integer
            return 1024

    async def embed_async_single(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        cached = self._check_cache(text)
        if cached is not None:
            self._update_stats(hit=True)
            return cached
            
        try:
            self._update_stats(hit=False)
            if self._use_remote_http:
                logger.info(f"Using remote embedding endpoint: {self._remote_base_url}")
                vectors = await self._embed_remote([text])
                vector = vectors[0]

                # Validate vector dimension matches expected dimension
                if len(vector) != self._dimension:
                    logger.info(f"Updating embedding dimension from {self._dimension} to {len(vector)}")
                    self._dimension = len(vector)

                self._store_cache(text, vector)
                return vector

            loop = asyncio.get_running_loop()
            _start = time.time()

            def _encode_one(txt: str):  # local helper to enable autocast
                if _HAS_CUDA and self._device == "cuda":
                    import torch
                    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                        return self._st_model.encode(  # type: ignore[attr-defined]
                            txt,
                            normalize_embeddings=True,
                            convert_to_numpy=False,
                        )
                return self._st_model.encode(  # type: ignore[attr-defined]
                    txt,
                    normalize_embeddings=True,
                    convert_to_numpy=False,
                )

            vector: List[float] = await loop.run_in_executor(None, partial(_encode_one, text))  # type: ignore[type-var]

            logger.debug(
                "[BeamEmbedding] embed_async_single took %.3fs (device=%s)",
                time.time() - _start,
                self._device,
            )

            if hasattr(vector, "tolist"):
                vector = vector.tolist()  # type: ignore[assignment]

            # Validate vector
            if not vector or not isinstance(vector, list):
                logger.error(f"Invalid vector returned: {type(vector)}")
                # Return a safe default vector with the expected dimension
                if self._dimension > 0:
                    return [0.0] * self._dimension
                else:
                    # If dimension is not set, use a reasonable default
                    self._dimension = 1024
                    return [0.0] * 1024

            # Update dimension if needed
            if len(vector) != self._dimension:
                logger.info(f"Updating embedding dimension from {self._dimension} to {len(vector)}")
                self._dimension = len(vector)

            self._store_cache(text, vector)  # type: ignore[arg-type]
            return vector  # type: ignore[return-value]
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            self._update_stats(error=True)
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    async def embed_async_many(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Check if all texts are in cache
        all_vectors = []
        missing_indices = []
        missing_texts = []
        
        # First try to get from cache
        for i, text in enumerate(texts):
            cached = self._check_cache(text)
            if cached is not None:
                all_vectors.append(cached)
                self._update_stats(hit=True)
            else:
                all_vectors.append(None)
                missing_indices.append(i)
                missing_texts.append(text)
                self._update_stats(hit=False)
                
        # If all found in cache, return early
        if not missing_texts:
            return all_vectors

        # ------------------------------------------------------------------
        # Optimisation: sort texts by length (desc) before batching to reduce
        # padding waste and improve GPU utilisation.  We keep a mapping so we
        # can restore the original order when writing back to *all_vectors*.
        # This change is entirely internal to BeamEmbedding and therefore
        # affects only Beam-based deployments.
        # ------------------------------------------------------------------
        _sorted_pairs = sorted(enumerate(missing_texts), key=lambda p: len(p[1]), reverse=True)
        _orig_pos_map = [p[0] for p in _sorted_pairs]          # idx in missing_texts → sorted position
        missing_texts_sorted = [p[1] for p in _sorted_pairs]   # texts ordered by length
        # Replace for downstream code paths
        missing_texts = missing_texts_sorted
        # NB: *missing_indices* stays unchanged – we will translate using
        # *_orig_pos_map* when mapping vectors back.

        try:
            # Get embeddings for missing texts
            if self._use_remote_http:
                # Get batch size from settings or use default value
                batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
                # Limit the number of concurrent tasks
                max_concurrent_tasks = min(8, len(missing_texts)) 
                
                logger.info(f"Processing {len(missing_texts)} texts in batches of {batch_size} with up to {max_concurrent_tasks} concurrent tasks")
                
                # Split into batches
                batches = []
                for i in range(0, len(missing_texts), batch_size):
                    batch = missing_texts[i:i + batch_size]
                    batches.append(batch)
                
                # Process batches sequentially in groups of max_concurrent_tasks
                flattened_vectors = []
                for i in range(0, len(batches), max_concurrent_tasks):
                    current_batches = batches[i:i + max_concurrent_tasks]
                    tasks = [asyncio.create_task(self._embed_remote(batch)) for batch in current_batches]
                    
                    try:
                        batch_results = await asyncio.gather(*tasks)
                        # Flatten results from this group of batches
                        for batch_result in batch_results:
                            flattened_vectors.extend(batch_result)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
                        raise

                if len(flattened_vectors) != len(missing_indices):
                    raise EmbeddingError(
                        f"Mismatch between requested texts ({len(missing_indices)}) and returned embeddings ({len(flattened_vectors)})"
                    )

                # Re-map vectors back to original positions
                for sorted_idx, vector in enumerate(flattened_vectors):
                    _missing_list_idx = _orig_pos_map[sorted_idx]
                    _all_vec_idx = missing_indices[_missing_list_idx]
                    all_vectors[_all_vec_idx] = vector
                    self._store_cache(texts[_all_vec_idx], vector)

                return all_vectors
            else:
                # Local model flow
                loop = asyncio.get_running_loop()

                def _encode_cpu(texts_batch: List[str]) -> List[List[float]]:  # sync helper
                    with torch.no_grad() if _HAS_CUDA else contextlib.nullcontext():  # type: ignore[name-defined]
                        return self._st_model.encode(  # type: ignore[attr-defined]
                            texts_batch,
                            normalize_embeddings=True,
                            convert_to_numpy=False,
                            batch_size=self._batch_size,
                        )

                try:
                    # ------------------------------------------------------
                    # Adaptive GPU batching with AMP: progressively reduce
                    # batch size on CUDA OOM. Uses float16 autocast for speed.
                    # ------------------------------------------------------

                    def _encode_gpu(texts: List[str], bs: int) -> List[List[float]]:  # sync helper
                        if _HAS_CUDA and self._device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                                return self._st_model.encode(  # type: ignore[attr-defined]
                                    texts,
                                    normalize_embeddings=True,
                                    convert_to_numpy=False,
                                    batch_size=bs,
                                    show_progress_bar=False,
                                )
                        # CPU or no AMP
                        return self._st_model.encode(  # type: ignore[attr-defined]
                            texts,
                            normalize_embeddings=True,
                            convert_to_numpy=False,
                            batch_size=bs,
                            show_progress_bar=False,
                        )

                    current_bs = self._batch_size
                    last_exc: Exception | None = None
                    while current_bs >= 4:
                        try:
                            _gpu_batch_start = time.time()
                            vectors: List[List[float]] = await loop.run_in_executor(  # type: ignore[type-var]
                                None,
                                partial(_encode_gpu, missing_texts, current_bs),
                            )
                            logger.debug(
                                "[BeamEmbedding] GPU batch encode size=%d took %.3fs (bs=%d, device=%s)",
                                len(missing_texts),
                                time.time() - _gpu_batch_start,
                                current_bs,
                                self._device,
                            )
                            break  # success
                        except RuntimeError as gpu_e:
                            last_exc = gpu_e
                            if "out of memory" in str(gpu_e).lower():
                                logger.warning("GPU OOM at batch_size=%s – reducing.", current_bs)
                                import gc
                                if _HAS_CUDA:
                                    torch.cuda.empty_cache()
                                gc.collect()
                                current_bs //= 2
                                continue
                            raise
                    else:
                        # Give up and fallback – handled below
                        raise last_exc or RuntimeError("Unknown encode error")

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.warning(
                            "CUDA OOM during embedding generation – falling back to CPU with smaller batch size",
                        )
                        import gc
                        if _HAS_CUDA:
                            torch.cuda.empty_cache()
                        gc.collect()

                        # Reload model on CPU if it was on CUDA
                        if getattr(self, "_device", "cpu") == "cuda":
                            self._st_model = SentenceTransformer(  # type: ignore[attr-defined]
                                self._model_name,
                                cache_folder="./models",
                                device="cpu",
                            )
                            self._device = "cpu"

                        # Reduce batch size and retry on CPU
                        self._batch_size = max(4, self._batch_size // 2)
                        logger.info(
                            "Retrying encode with batch_size=%s on CPU", self._batch_size,
                        )

                        vectors = await loop.run_in_executor(  # type: ignore[type-var]
                            None,
                            partial(
                                self._st_model.encode,  # type: ignore[attr-defined]
                                missing_texts,
                                normalize_embeddings=True,
                                convert_to_numpy=False,
                                batch_size=self._batch_size,
                            ),
                        )
                    else:
                        raise

                out: List[List[float]] = []
                for v in vectors:
                    if hasattr(v, "tolist"):
                        out.append(v.tolist())  # type: ignore[arg-type]
                    else:
                        out.append(list(v))

                # Fill in missing vectors and update cache
                for sorted_idx, vector in enumerate(out):
                    _missing_list_idx = _orig_pos_map[sorted_idx]
                    _all_vec_idx = missing_indices[_missing_list_idx]
                    all_vectors[_all_vec_idx] = vector
                    self._store_cache(texts[_all_vec_idx], vector)
                    
                return all_vectors
                
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            self._update_stats(error=True)
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}") from e

    # ------------------------------------------------------------------
    # Remote HTTP helper
    # ------------------------------------------------------------------

    async def _embed_remote(self, texts: List[str]) -> List[List[float]]:
        """Call remote Beam embedding endpoint."""
        if not self._remote_base_url:
            raise EmbeddingError("Remote base URL not configured for Beam embeddings")

        logger.info(f"Embedding request to URL: {self._remote_base_url}")
        
        headers = {"Content-Type": "application/json"}
        if settings.beam_api_token:
            headers["Authorization"] = f"Bearer {settings.beam_api_token}"

        payload = {"texts": texts}

        try:
            # Increase timeout for large text batches
            timeout = self._timeout * max(1, len(texts) // 4)
            logger.info(f"Using timeout of {timeout}s for {len(texts)} texts")
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.debug(f"Calling remote embedding API at {self._remote_base_url}")
                resp = await client.post(self._remote_base_url, json=payload, headers=headers)
                resp.raise_for_status()
                
                # Check the response content before parsing JSON
                content = resp.content.decode('utf-8').strip()
                if not content:
                    raise EmbeddingError("Empty response received from API")
                    
                # Check for authorization error message
                if content.startswith('{"message":"Unauthorized"}'):
                    raise EmbeddingError("API authorization failed: Unauthorized access")
                
                data = resp.json()

            vectors: List[List[float]] = data.get("embeddings", [])
            
            if not vectors:
                raise EmbeddingError("No embeddings returned from remote API")
                
            # Validate all vectors have the same dimension
            dimensions = [len(v) for v in vectors]
            if len(set(dimensions)) > 1:
                raise EmbeddingError(f"Inconsistent vector dimensions returned: {dimensions}")

            # Update dimension if needed
            if vectors and len(vectors[0]) != self._dimension:
                old_dim = self._dimension
                self._dimension = len(vectors[0])
                logger.info(f"Vector dimension updated: {old_dim} → {self._dimension}")

            return vectors
            
        except httpx.TimeoutException as e:
            logger.error(f"Timeout error when calling embedding API: {e}")
            if settings.beam_mode == "local":
                raise EmbeddingError(
                    f"Timeout connecting to local embedding server: {e}. "
                    f"Consider increasing EMBEDDING_TIMEOUT in .env file."
                ) from e
            else:
                raise EmbeddingError(f"Timeout error when calling embedding API: {e}. Consider increasing EMBEDDING_TIMEOUT in .env file.") from e
        except httpx.HTTPError as e:
            logger.error(f"HTTP error when calling embedding API: {e}")
            if settings.beam_mode == "local":
                raise EmbeddingError(
                    f"Failed to connect to local embedding server: {e}. "
                    f"Make sure the local model server is running with: "
                    f"python -m root.src.integrations.beam.local_model_server"
                ) from e
            else:
                raise EmbeddingError(f"HTTP error when calling embedding API: {e}") from e
        except Exception as e:
            logger.error(f"Error calling remote embedding API: {e}")
            raise EmbeddingError(f"Failed to get embeddings from remote API: {e}") from e

    # Add synchronous method to get vector dimension
    def get_text_embedding(self, text: str) -> List[float]:
        """Synchronous version to get embedding for a single text string."""
        cached = self._check_cache(text)
        if cached is not None:
            self._update_stats(hit=True)
            return cached
            
        self._update_stats(hit=False)
        try:
            if self._use_remote_http:
                # Run asynchronous method synchronously
                try:
                    # Check if the event loop is already running
                    try:
                        loop = asyncio.get_running_loop()
                        # If we are here, the event loop is already running
                        # Cannot create a new loop, use httpx with sync API directly
                        logger.info("Using synchronous HTTP request because event loop is already running")
                        headers = {"Content-Type": "application/json"}
                        if settings.beam_api_token:
                            headers["Authorization"] = f"Bearer {settings.beam_api_token}"
                        
                        # Use increased timeout for synchronous requests
                        with httpx.Client(timeout=self._timeout) as client:
                            logger.info(f"Sending synchronous request to {self._remote_base_url}")
                            resp = client.post(
                                self._remote_base_url,
                                json={"texts": [text]},
                                headers=headers
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            vectors = data.get("embeddings", [])
                            if not vectors:
                                raise EmbeddingError("No embeddings returned from remote API")
                            vector = vectors[0]
                    except RuntimeError:
                        # If the event loop is not running, use a new one
                        logger.info("Creating new event loop for embedding request")
                        loop = asyncio.new_event_loop()
                        vectors = loop.run_until_complete(self._embed_remote([text]))
                        vector = vectors[0]
                        loop.close()
                    except httpx.TimeoutException as e:
                        logger.error(f"Timeout error in synchronous request: {e}")
                        raise EmbeddingError(f"Timeout error when calling embedding API: {e}. Consider increasing EMBEDDING_TIMEOUT in .env file.")
                    except httpx.HTTPError as e:
                        logger.error(f"HTTP error in synchronous request: {e}")
                        raise EmbeddingError(f"HTTP error when calling embedding API: {e}")
                except Exception as e:
                    # In local mode, we should not fall back to local model download
                    if settings.beam_mode == "local":
                        logger.error(f"Failed to connect to local embedding server: {e}")
                        raise EmbeddingError(
                            f"Failed to connect to embedding server at {self._remote_base_url}. "
                            f"Make sure to start the local model server with: "
                            f"python -m root.src.integrations.beam.local_model_server"
                        ) from e
                    else:
                        # For cloud mode, re-raise the exception
                        raise
                    
                # Update dimension if needed
                if len(vector) != self._dimension:
                    logger.info(f"Updating embedding dimension from {self._dimension} to {len(vector)}")
                    self._dimension = len(vector)
                    
                self._store_cache(text, vector)
                return vector
            
            # Local model
            if self._st_model:
                vector = self._st_model.encode(  # type: ignore[attr-defined]
                    text, 
                    normalize_embeddings=True,
                    convert_to_numpy=False
                )
                
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()  # type: ignore[assignment]
                
                # Update dimension if needed
                if len(vector) != self._dimension:
                    logger.info(f"Updating embedding dimension from {self._dimension} to {len(vector)}")
                    self._dimension = len(vector)
                    
                self._store_cache(text, vector)  # type: ignore[arg-type]
                return vector  # type: ignore[return-value]
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            self._update_stats(error=True)
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e
            
        # This code should never be reached with the raised exceptions above
        raise EmbeddingError("Failed to generate embedding and no fallback available")
