"""Beam Cloud chat provider for Sentio.

This module provides a chat provider that runs models on Beam Cloud.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from root.src.utils.settings import settings
from root.src.integrations.beam.ai_model import BeamModel

logger = logging.getLogger(__name__)


class BeamChatProvider:
    """Chat provider that runs models on Beam Cloud."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_token: Optional[str] = None,
    ) -> None:
        """Initialize BeamChatProvider.

        Args:
            model_id: Model identifier, defaults to settings.beam_model_id.
            api_token: Beam API token, defaults to settings.beam_api_token.
        """
        self.model_id = model_id or settings.beam_model_id
        self.api_token = api_token or settings.beam_api_token
        
        if not self.api_token:
            raise RuntimeError("BEAM_API_TOKEN is required")
            
        self._beam_model = None

    async def _ensure_model(self) -> BeamModel:
        """Ensure model is initialized.

        Returns:
            Initialized BeamModel instance.
        """
        if self._beam_model is None:
            self._beam_model = BeamModel.get_instance(self.model_id)
            await self._beam_model.initialize()
        return self._beam_model

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate chat completion using Beam model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stream: Whether to stream the output.
            **kwargs: Additional model-specific parameters.

        Returns:
            Generated text or async generator for streaming.
        """
        model = await self._ensure_model()
        
        # Convert messages to a prompt
        # This is a simple implementation - can be enhanced based on model requirements
        prompt = self._messages_to_prompt(messages)
        
        return await model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs,
        )

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
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

    async def chat_completion(
        self,
        payload: Dict[str, Any],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """OpenAI-compatible chat completion interface.

        Args:
            payload: OpenAI-compatible payload.
            **kwargs: Additional parameters.

        Returns:
            OpenAI-compatible response or streaming generator.
        """
        messages = payload.get("messages", [])
        temperature = payload.get("temperature", 0.7)
        max_tokens = payload.get("max_tokens", 1024)
        stream = payload.get("stream", False)
        
        result = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )
        
        if stream:
            return self._stream_to_openai_format(result)
        
        # Convert plain text to OpenAI format
        return {
            "id": f"beam-{self.model_id}-{asyncio.get_event_loop().time()}",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    async def _stream_to_openai_format(
        self, stream_gen: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """Convert streaming text to OpenAI format.

        Args:
            stream_gen: Text stream generator.

        Yields:
            OpenAI-compatible streaming format.
        """
        chunk_id = f"beam-{self.model_id}-{asyncio.get_event_loop().time()}"
        i = 0
        
        async for chunk in stream_gen:
            data = {
                "id": f"{chunk_id}-chunk-{i}",
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            i += 1
            yield f"data: {data}\n\n"
            
        # Final chunk with finish_reason
        data = {
            "id": f"{chunk_id}-chunk-{i}",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
