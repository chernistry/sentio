from __future__ import annotations

"""HuggingFace Transformers-based reranker using Qwen3-Reranker.

The implementation mirrors the method proposed in the Qwen2-Reranker model card:
we compute the *yes/no* classification probability for each *query-document*
pair and use the probability of *"yes"* as a relevance score.

This implementation processes documents in batches for efficiency.

Notes
-----
* This wrapper keeps a single global model and tokenizer instance to avoid
  repeated loading. The instances are thread-safe and can be shared.
* The class is synchronous. In async contexts call it via
  ``asyncio.to_thread`` to off-load blocking execution.
"""

import os
import threading
import time
import logging
from typing import Any, Dict, List, Optional

from plugins.interface import SentioPlugin

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

__all__ = ["TransformersReranker"]

_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


class TransformersReranker:  # noqa: D101 – simple wrapper
    _model: Optional[AutoModelForCausalLM] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _device: Optional[str] = None
    _token_true_id: Optional[int] = None
    _token_false_id: Optional[int] = None
    _prefix_tokens: Optional[List[int]] = None
    _suffix_tokens: Optional[List[int]] = None
    _max_length: int = 8192

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "auto",
        attn_implementation: str = "flash_attention_2",
        max_length: int = 8192,
        token: str = "",
    ) -> None:
        """Initialize the reranker with a HuggingFace transformer model.

        Args:
            model_name_or_path: Path to a local model or HF model name.
            torch_dtype: The torch dtype to use for the model.
            device: The device to run the model on ('auto', 'cuda', 'cpu').
            attn_implementation: Attention implementation for the model.
            max_length: Maximum sequence length.
            token: HuggingFace token for accessing gated models.
        """
        logger.info(f"Initializing TransformersReranker with model: {model_name_or_path}")
        start_time = time.time()
        with _LOCK:
            if TransformersReranker._model is None:
                logger.info("No cached model found. Starting model initialization...")
                self._initialize_model(
                    model_name_or_path,
                    torch_dtype,
                    device,
                    attn_implementation,
                    max_length,
                    token,
                )
            else:
                logger.info("Using cached model instance")

        # Alias for readability and type-safety
        self.model = TransformersReranker._model
        self.tokenizer = TransformersReranker._tokenizer
        self.device = TransformersReranker._device
        self.token_true_id = TransformersReranker._token_true_id
        self.token_false_id = TransformersReranker._token_false_id
        self.prefix_tokens = TransformersReranker._prefix_tokens
        self.suffix_tokens = TransformersReranker._suffix_tokens
        self.max_length = TransformersReranker._max_length
        logger.info(f"TransformersReranker initialization completed in {time.time() - start_time:.2f}s")
        logger.info(f"Model loaded on device: {self.device}")
        logger.info(f"Model max_length: {self.max_length}")

    def _initialize_model(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype,
        device_str: str,
        attn_implementation: str,
        max_length: int,
        token: str = "",
    ) -> None:
        """Lazily initialize the shared model and tokenizer."""
        logger.info(f"Starting model initialization from: {model_name_or_path}")
        init_start = time.time()
        if device_str == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_str
        logger.info(f"Using device: {device}")

        # Check if model_name_or_path is a local directory
        is_local = os.path.isdir(model_name_or_path)
        if is_local:
            logger.info(f"Loading from local directory: {model_name_or_path}")
        else:
            logger.info(f"Loading from HuggingFace Hub: {model_name_or_path}")
        
        # Add token if provided and not using local path
        tokenizer_kwargs = {"padding_side": "left", "trust_remote_code": True}
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
        }
        
        if token and not is_local:
            logger.info("Using HF token for authentication")
            tokenizer_kwargs["token"] = token
            model_kwargs["token"] = token
        
        logger.info("Loading tokenizer...")
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        logger.info(f"Tokenizer loaded in {time.time() - tokenizer_start:.2f}s")
        
        logger.info("Loading model...")
        model_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs).to(device).eval()
        logger.info(f"Model loaded in {time.time() - model_start:.2f}s")

        TransformersReranker._model = model
        TransformersReranker._tokenizer = tokenizer
        TransformersReranker._device = device
        TransformersReranker._max_length = max_length

        TransformersReranker._token_false_id = tokenizer.convert_tokens_to_ids("no")
        TransformersReranker._token_true_id = tokenizer.convert_tokens_to_ids("yes")
        logger.info(f"Token IDs - yes: {TransformersReranker._token_true_id}, no: {TransformersReranker._token_false_id}")

        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the "
            'requirements based on the Query and the Instruct provided. Note that '
            'the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        TransformersReranker._prefix_tokens = tokenizer.encode(
            prefix, add_special_tokens=False
        )
        TransformersReranker._suffix_tokens = tokenizer.encode(
            suffix, add_special_tokens=False
        )
        logger.info(f"Prefix tokens length: {len(TransformersReranker._prefix_tokens)}")
        logger.info(f"Suffix tokens length: {len(TransformersReranker._suffix_tokens)}")
        logger.info(f"Model initialization completed in {time.time() - init_start:.2f}s")

    def rerank(
        self, query: str, docs: List[Dict[str, Any]], *, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return *top_k* docs ranked by relevance to *query*."""
        start_time = time.time()
        logger.info(f"Reranking {len(docs)} documents for query: {query[:50]}...")
        if not docs:
            logger.info("No documents to rerank, returning empty list")
            return []

        logger.info("Preparing document-query pairs")
        pairs = [
            self._format_prompt(query, doc.get("text", "")) for doc in docs
        ]

        logger.info("Tokenizing inputs")
        tokenize_start = time.time()
        inputs = self._prepare_inputs(pairs)
        logger.info(f"Tokenization completed in {time.time() - tokenize_start:.2f}s")
        
        logger.info("Computing relevance scores")
        inference_start = time.time()
        scores = self._compute_scores(inputs)
        logger.info(f"Inference completed in {time.time() - inference_start:.2f}s")
        logger.info(f"Score range: min={min(scores):.4f}, max={max(scores):.4f}")

        for d, s in zip(docs, scores):
            d["score"] = s
            d["rerank_score"] = s

        sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)[:top_k]
        logger.info(f"Reranking completed in {time.time() - start_time:.2f}s")
        logger.info(f"Top document score: {sorted_docs[0]['score']:.4f}")
        return sorted_docs

    def _prepare_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize and prepare a batch of query-document pairs."""
        logger.info(f"Preparing inputs for {len(pairs)} pairs")
        start_time = time.time()
        # Calculate effective max length for the content (excluding prefix/suffix)
        content_max_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        logger.info(f"Content max length: {content_max_length}")
        
        # Prepare inputs with prefix and suffix
        prepared_pairs = []
        for pair in pairs:
            # Tokenize without prefix/suffix first to apply truncation to content only
            temp_encoding = self.tokenizer(
                pair, 
                add_special_tokens=False,
                truncation=True,
                max_length=content_max_length,
                return_attention_mask=False
            )
            
            # Get the tokenized content and add prefix/suffix
            token_ids = self.prefix_tokens + temp_encoding["input_ids"] + self.suffix_tokens
            prepared_pairs.append(token_ids)
            if len(prepared_pairs) == 1:
                logger.info(f"Sample tokenized length: {len(token_ids)} tokens")
        
        # Use the tokenizer's pad method with max_length and return as tensors
        pad_start = time.time()
        inputs = self.tokenizer.pad(
            {"input_ids": prepared_pairs},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        logger.info(f"Padding completed in {time.time() - pad_start:.2f}s")

        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        logger.info(f"Input preparation completed in {time.time() - start_time:.2f}s")
        logger.info(f"Input tensor shape: {inputs['input_ids'].shape}")
        return inputs

    @torch.inference_mode()
    def _compute_scores(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """Compute relevance scores for a batch of inputs."""
        logger.info("Running model inference")
        start_time = time.time()
        logits = self.model(**inputs).logits[:, -1, :]
        logger.info(f"Model forward pass completed in {time.time() - start_time:.2f}s")
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]

        scores = torch.stack([false_vector, true_vector], dim=1)
        log_softmax_scores = torch.nn.functional.log_softmax(scores, dim=1)
        
        result = log_softmax_scores[:, 1].exp().tolist()
        logger.info(f"Score computation completed in {time.time() - start_time:.2f}s")
        return result

    @staticmethod
    def _format_prompt(query: str, doc: str) -> str:
        """Format the prompt for the Qwen3 reranker model."""
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )


class TransformersRerankerPlugin(SentioPlugin):
    """Plugin wrapper for HF Transformers reranker."""

    name = "transformers_reranker"
    plugin_type = "reranker"

    def __init__(self, **kwargs: Any) -> None:
        self.reranker = TransformersReranker(**kwargs)

    def register(self, pipeline: Any) -> None:
        pipeline.reranker = self.reranker


def get_plugin() -> SentioPlugin:
    """Return plugin instance."""
    return TransformersRerankerPlugin()
