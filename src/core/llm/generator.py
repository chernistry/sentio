"""LLM generator for LangGraph RAG pipeline.

This module provides a generator node for the LangGraph RAG pipeline,
which uses an LLM to generate responses based on retrieved context.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from src.core.graph.state import RAGState, set_response, add_metadata
from src.core.llm.chat_adapter import ChatAdapter
from src.core.llm.prompt_builder import PromptBuilder
from src.core.models.document import Document

logger = logging.getLogger(__name__)


class LLMGenerator:
    """LLM generator for RAG pipeline.
    
    This class handles generating responses using an LLM based on
    retrieved context.
    """

    def __init__(
        self,
        chat_adapter: ChatAdapter | None = None,
        prompt_builder: PromptBuilder | None = None,
        mode: str = "balanced",
        max_tokens: int = 1024,
    ) -> None:
        """Initialize the LLM generator.
        
        Args:
            chat_adapter: Chat adapter to use
            prompt_builder: Prompt builder to use
            mode: Generation mode (fast, balanced, quality, creative)
            max_tokens: Maximum tokens to generate
        """
        self.chat_adapter = chat_adapter or ChatAdapter()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.mode = mode
        self.max_tokens = max_tokens

        logger.debug(
            "Initialized LLMGenerator with mode=%s, max_tokens=%d",
            self.mode,
            self.max_tokens,
        )

    async def generate(
        self,
        query: str,
        documents: list[Document],
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Generate a response based on the query and documents.
        
        Args:
            query: User query
            documents: Retrieved documents
            stream: Whether to stream the response
            
        Returns:
            Generated response text or streaming generator
        """
        # Prepare context from documents
        context = self._prepare_context(documents)

        # Build prompt
        system_message = self.prompt_builder.build_system_message()
        user_prompt = self.prompt_builder.build_generation_prompt(
            query=query,
            context=context,
            mode=self.mode,  # type: ignore
        )

        # Prepare payload
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._get_temperature_for_mode(),
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

        # Generate response
        logger.info("Generating response for query: %s", query)
        response = await self.chat_adapter.chat_completion(payload)

        if not stream:
            # Extract content from response
            content = self._extract_content(response)
            return content
        # Return streaming generator
        return self._stream_content(response)  # type: ignore

    async def generate_for_state(self, state: RAGState) -> RAGState:
        """Generate a response for the RAG state.
        
        This method is used as a LangGraph node.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with generated response
        """
        if not state["selected_documents"]:
            logger.warning("No documents selected for generation")
            set_response(state, "I don't have enough information to answer that question.")
            return state

        try:
            # Generate response
            response = await self.generate(
                query=state["query"],
                documents=state["selected_documents"],
                stream=False,
            )

            # Update state
            set_response(state, response)
            add_metadata(state, "generator_mode", self.mode)

            logger.info("Generated response of length %d", len(response))
        except Exception as e:
            logger.error("Error generating response: %s", e)
            add_metadata(state, "generator_error", str(e))
            set_response(state, "I encountered an error while generating a response.")

        return state

    async def astream_for_state(self, state: RAGState) -> AsyncGenerator[RAGState, None]:
        """Stream a response for the RAG state.
        
        This method is used for streaming responses.
        
        Args:
            state: Current RAG state
            
        Yields:
            Updated RAG states with partial responses
        """
        if not state["selected_documents"]:
            logger.warning("No documents selected for generation")
            set_response(
                state,
                "I don't have enough information to answer that question.",
            )
            yield state
            return

        try:
            # Generate streaming response
            stream_gen = await self.generate(
                query=state["query"],
                documents=state["selected_documents"],
                stream=True,
            )

            # Process streaming response
            buffer = ""
            async for chunk in stream_gen:  # type: ignore
                # Extract content from chunk
                content = self._extract_streaming_content(chunk)
                if content:
                    buffer += content

                    # Update state with current buffer
                    new_state = dict(state)  # shallow copy of TypedDict
                    set_response(new_state, buffer)
                    yield new_state  # type: ignore[arg-type]

            # Final state update
            set_response(state, buffer)
            add_metadata(state, "generator_mode", self.mode)

            logger.info("Generated streaming response of length %d", len(buffer))
        except Exception as e:
            logger.error("Error generating streaming response: %s", e)
            add_metadata(state, "generator_error", str(e))
            set_response(state, "I encountered an error while generating a response.")
            yield state

    async def close(self) -> None:
        """Close the chat adapter."""
        await self.chat_adapter.close()

    def _prepare_context(self, documents: list[Document]) -> str:
        """Prepare context from documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ''
        
        # Add logging to verify document structure
        for i, doc in enumerate(documents):
            logger.info(f"Document {i}: text='{doc.text[:100] if doc.text else ''}...', metadata_content='{doc.metadata.get('content', '')[:100] if doc.metadata.get('content') else ''}...'")
        
        context_parts = []
        for i, doc in enumerate(documents):
            # Get content from either text field or metadata.content
            content = doc.text or doc.metadata.get('content', '')
            if not content:
                logger.warning(f"Document {i} has no content in either text or metadata.content")
                continue
                
            source = doc.metadata.get('source', f'Document {i+1}')
            context_parts.append(f'Source: {source}\nContent: {content}')
        
        if not context_parts:
            logger.warning("No content available in any retrieved documents")
            return 'No content available in retrieved documents.'
            
        return '\n\n'.join(context_parts) + '\n\nUse this context to answer accurately, focusing on key facts.'

    def _get_temperature_for_mode(self) -> float:
        """Get temperature based on generation mode.
        
        Returns:
            Temperature value
        """
        mode_temps = {
            "fast": 0.0,
            "balanced": 0.3,
            "quality": 0.2,
            "creative": 0.7,
        }
        return mode_temps.get(self.mode, 0.3)

    def _extract_content(self, response: dict[str, Any]) -> str:
        """Extract content from response.
        
        Args:
            response: Response from chat completion
            
        Returns:
            Content string
        """
        # Try common response shapes from OpenAI-compatible providers
        try:
            content = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if content:
                return content
        except (IndexError, AttributeError):
            pass

        # Fallbacks for other providers
        for key in ("content", "text", "output_text", "response"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _extract_streaming_content(self, chunk: str) -> str:
        """Extract content from streaming chunk.
        
        Args:
            chunk: Streaming chunk
            
        Returns:
            Content string
        """
        try:
            # Parse SSE format
            if chunk.startswith("data: "):
                data = chunk[6:].strip()
                if data == "[DONE]":
                    return ""

                import json
                parsed = json.loads(data)
                return parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
            return ""
        except Exception:
            return ""

    async def _stream_content(self, stream_gen: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """Process streaming content.
        
        Args:
            stream_gen: Streaming generator
            
        Yields:
            Content chunks
        """
        async for chunk in stream_gen:
            content = self._extract_streaming_content(chunk)
            if content:
                yield content
