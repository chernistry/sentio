"""Example usage of Beam Cloud integration.

This module provides examples of how to use the Beam Cloud integration
for running models and tasks.
"""

import asyncio
from typing import Dict, Any

from root.src.integrations.beam.runtime import local_task
from root.src.integrations.beam.ai_model import BeamModel, run_inference
from root.src.core.llm.providers.beam_chat import BeamChatProvider


async def example_direct_model_usage() -> None:
    """Example of direct model usage."""
    # Create and initialize model
    model = BeamModel.get_instance()
    await model.initialize()
    
    # Generate text
    result = await model.generate(
        prompt="Explain quantum computing in simple terms.",
        max_tokens=1024,
        temperature=0.7,
    )
    
    print(f"Model result: {result}")
    
    # Streaming example
    print("Streaming results:")
    async for chunk in await model.generate(
        prompt="Write a short poem about AI.",
        stream=True,
    ):
        print(chunk, end="", flush=True)
    print()


async def example_chat_provider() -> None:
    """Example of chat provider usage."""
    provider = BeamChatProvider()
    
    # Simple chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Beam Cloud."},
    ]
    
    result = await provider.chat(messages=messages)
    print(f"Chat result: {result}")
    
    # OpenAI-compatible interface
    payload = {
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 500,
    }
    
    response = await provider.chat_completion(payload)
    print(f"OpenAI-compatible response: {response}")


@local_task(name="example_beam_task")
async def example_task(query: str) -> Dict[str, Any]:
    """Example task that can run locally or on Beam.

    Args:
        query: Input query.

    Returns:
        Dictionary with task results.
    """
    # This function can be called locally or deployed to Beam
    model = BeamModel.get_instance()
    await model.initialize()
    
    result = await model.generate(prompt=query)
    
    return {
        "query": query,
        "result": result,
    }


async def main() -> None:
    """Run all examples."""
    print("Running direct model usage example...")
    await example_direct_model_usage()
    
    print("\nRunning chat provider example...")
    await example_chat_provider()
    
    print("\nRunning task example locally...")
    result = await example_task("What is Beam Cloud?")
    print(f"Task result: {result}")
    
    print("\nRunning inference task...")
    inference_result = await run_inference(
        prompt="Explain the benefits of cloud computing.",
        max_tokens=500,
    )
    print(f"Inference result: {inference_result}")


if __name__ == "__main__":
    asyncio.run(main()) 