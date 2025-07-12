import asyncio
import logging
from functools import partial
from typing import Any, List

import httpx

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
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
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
        if "Qwen3-Embedding-0.6B" in model_name:
            self._dimension = 1024  # Known dimension for Qwen3-Embedding-0.6B
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

        # Check that the URL does not contain an outdated version (v18, v19, v20)
        # and update to the current version (v21)
        if _base_url and "embed-240a6d4-v" in _base_url:
            # Extract the base URL without the version
            base_parts = _base_url.split("embed-240a6d4-v")
            if len(base_parts) == 2:
                version_parts = base_parts[1].split(".", 1)
                if len(version_parts) > 0:
                    # Update the URL to the latest version v21
                    _base_url = f"{base_parts[0]}embed-240a6d4-v21"
                    if len(version_parts) > 1:
                        _base_url += f".{version_parts[1]}"
                    logger.info(f"Updated embedding URL to latest version: {_base_url}")

        self._remote_base_url: str | None = _base_url

        # ------------------------------------------------------------------------------
        self._timeout = timeout
        self._batch_size = batch_size

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

            self._st_model = SentenceTransformer(  # type: ignore[attr-defined]
                model_name,
                cache_folder=cache_dir,
                device="cpu",
            )
            self._dimension = int(self._st_model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]

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
            vector: List[float] = await loop.run_in_executor(  # type: ignore[type-var]
                None,  # default executor
                partial(
                    self._st_model.encode,  # type: ignore[attr-defined]
                    text,
                    normalize_embeddings=True,
                    convert_to_numpy=False,
                ),
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
            
        try:
            # Get embeddings for missing texts
            if self._use_remote_http:
                # Split into batches and process them in parallel
                max_concurrent_tasks = 16  # Maximum number of concurrent tasks
                batch_size = min(32, len(missing_texts))  # Size of one batch
                
                logger.info(f"Processing {len(missing_texts)} texts in batches of {batch_size} with up to {max_concurrent_tasks} concurrent tasks")
                
                # Split into batches
                batches = []
                for i in range(0, len(missing_texts), batch_size):
                    batch = missing_texts[i:i + batch_size]
                    batches.append(batch)
                
                # Run all batches in parallel using asyncio.create_task.
                # Keep the order of results to correctly match them with the original texts.

                tasks = [asyncio.create_task(self._embed_remote(batch)) for batch in batches]

                try:
                    batch_results: List[List[List[float]]] = await asyncio.gather(*tasks)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    raise

                # Results come in the same order as batches
                flattened_vectors: List[List[float]] = [vec for batch in batch_results for vec in batch]

                if len(flattened_vectors) != len(missing_indices):
                    raise EmbeddingError(
                        "Mismatch between requested texts and returned embeddings"
                    )

                for idx, vector in zip(missing_indices, flattened_vectors):
                    all_vectors[idx] = vector
                    self._store_cache(texts[idx], vector)

                return all_vectors
            else:
                # Local model flow
                loop = asyncio.get_running_loop()
                vectors: List[List[float]] = await loop.run_in_executor(  # type: ignore[type-var]
                    None,
                    partial(
                        self._st_model.encode,  # type: ignore[attr-defined]
                        missing_texts,
                        normalize_embeddings=True,
                        convert_to_numpy=False,
                        batch_size=self._batch_size,
                    ),
                )

                out: List[List[float]] = []
                for v in vectors:
                    if hasattr(v, "tolist"):
                        out.append(v.tolist())  # type: ignore[arg-type]
                    else:
                        out.append(list(v))

                # Fill in missing vectors and update cache
                for i, (idx, vector) in enumerate(zip(missing_indices, out)):
                    all_vectors[idx] = vector
                    self._store_cache(missing_texts[i], vector)
                    
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
            async with httpx.AsyncClient(timeout=self._timeout) as client:
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
                        
                        with httpx.Client(timeout=self._timeout) as client:
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
                        loop = asyncio.new_event_loop()
                        vectors = loop.run_until_complete(self._embed_remote([text]))
                        vector = vectors[0]
                        loop.close()
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
