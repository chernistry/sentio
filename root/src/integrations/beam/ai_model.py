"""AI model runner for Beam Cloud.

This module provides a wrapper for running AI models on Beam Cloud,
handling model loading, caching, and inference.
"""

import os
import json
import asyncio
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from root.src.utils.settings import settings
from root.src.integrations.beam.runtime import BeamRuntime, local_task

# Import optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BeamModel:
    """Wrapper for AI models running on Beam Cloud."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        volume_path: Optional[str] = None,
        cache_dir: str = "./model_cache",
    ) -> None:
        """Initialize BeamModel.

        Args:
            model_id: Model identifier, defaults to settings.beam_model_id.
            volume_path: Path to Beam volume, defaults to "./models".
            cache_dir: Directory for caching models.
        """
        self.model_id = model_id or settings.beam_model_id
        self.volume_path = volume_path or "./models"
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the model (load weights, prepare cache).

        This method should be called before first inference.
        """
        if self._is_initialized:
            return

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load model (implementation depends on model type)
        await self._load_model()
        self._is_initialized = True

    async def _load_model(self) -> None:
        """Load model weights and tokenizer.

        This method should be overridden by subclasses for specific model types.
        """
        # ------------------------------------------------------------------
        # Local (dev) vs Beam (remote) handling
        # ------------------------------------------------------------------
        if not BeamRuntime.is_remote():
            # Local mode – keep lightweight stub but still prefer GPU if present
            print(f"[LOCAL] Loading huggingface model '{self.model_id}' for debugging")
        
        model_path = os.path.join(self.volume_path, self.model_id)

        # ------------------------------------------------------------------
        # GPU detection
        # ------------------------------------------------------------------
        _device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        print(f"Loading model '{self.model_id}' on {_device} from {model_path}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # local lazy import
        except ImportError as exc:  # pragma: no cover – optional heavy dep
            raise RuntimeError(
                "transformers not installed – add it to requirements or bake into Beam image",
            ) from exc

        # HuggingFace loading with device placement
        dtype = torch.float16 if _device == "cuda" else torch.float32 if HAS_TORCH else None

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,  # type: ignore[arg-type]
            device_map="cuda" if _device == "cuda" else None,
        )

        if _device == "cuda":
            self._model.to("cuda")  # type: ignore[call-arg]

        # Small warm-up forward pass – avoids first-request latency spike
        _ = self._model.generate(
            **self._tokenizer("GPU warm-up", return_tensors="pt").to(_device),
            max_new_tokens=1,
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text from the model.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stream: Whether to stream the output.
            **kwargs: Additional model-specific parameters.

        Returns:
            Generated text or async generator for streaming.
        """
        if not self._is_initialized:
            await self.initialize()

        if not BeamRuntime.is_remote():
            # Local mode - stub implementation
            if stream:
                async def _mock_stream() -> AsyncGenerator[str, None]:
                    for i in range(5):
                        await asyncio.sleep(0.2)
                        yield f"Mock output {i} for: {prompt[:20]}...\n"
                return _mock_stream()
            return f"Mock output for: {prompt[:50]}..."

        # ------------------------------------------------------------------
        # Actual generation when model is loaded
        # ------------------------------------------------------------------
        if self._model is None or self._tokenizer is None:
            return "[ERROR] Model not initialized"

        if stream:
            async def _stream_output() -> AsyncGenerator[str, None]:
                input_ids = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
                generated_ids = self._model.generate(
                    **input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                )
                text = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                yield text
            return _stream_output()

        # Non-streaming path
        input_ids = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
        )
        text = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return text

    @classmethod
    def get_instance(
        cls, model_id: Optional[str] = None
    ) -> "BeamModel":
        """Get or create a cached model instance.

        Args:
            model_id: Model identifier.

        Returns:
            BeamModel instance.
        """
        return _get_model_instance(model_id or settings.beam_model_id)


@lru_cache(maxsize=4)
def _get_model_instance(model_id: str) -> BeamModel:
    """Cache and return model instances (up to 4 different models).

    Args:
        model_id: Model identifier.

    Returns:
        BeamModel instance.
    """
    return BeamModel(model_id=model_id)


@local_task(name="beam_model_inference")
async def run_inference(
    prompt: str,
    model_id: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stream: bool = False,
) -> Dict[str, Any]:
    """Run inference on Beam Cloud.

    Args:
        prompt: Input text prompt.
        model_id: Model identifier.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        stream: Whether to stream the output.

    Returns:
        Dict with inference results.
    """
    model = BeamModel.get_instance(model_id)
    await model.initialize()
    
    if stream:
        chunks = []
        async for chunk in await model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            chunks.append(chunk)
        result = "".join(chunks)
    else:
        result = await model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
    
    return {
        "model_id": model.model_id,
        "prompt": prompt,
        "result": result,
    }
