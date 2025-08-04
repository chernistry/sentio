"""Utility for extracting structured JSON from LLM responses.

This module provides utilities for extracting JSON blocks from LLM responses,
with various fallback strategies to handle malformed JSON.
"""

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# Try importing json5, but don't make it a hard requirement
try:
    import json5
    JSON5_AVAILABLE = True
except ImportError:
    json5 = None
    JSON5_AVAILABLE = False


class JsonBlock(BaseModel):
    """Model for validating and representing JSON blocks."""
    text: str = Field(..., description="Text content containing potential JSON")

    @model_validator(mode="before")
    @classmethod
    def check_text_is_str(cls, values):
        if not isinstance(values.get("text", ""), str):
            raise ValueError("Text must be a string")
        return values


class JsonExtractResult(BaseModel):
    """Model for JSON extraction results."""
    data: dict[str, Any] | None = Field(None, description="Extracted JSON data")
    success: bool = Field(False, description="Whether extraction was successful")
    error: str | None = Field(None, description="Error message if extraction failed")


async def _find_json_block(text: str) -> str | None:
    """Asynchronously finds a JSON block in the text, preferring blocks in markdown.
    
    Args:
        text: The input text to search for a JSON block
        
    Returns:
        The found JSON block or None if no block is found
    """
    # Pattern for markdown code blocks (json, javascript, or none)
    code_block_match = re.search(r"```(?:json|javascript)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # If no code block, find the largest top-level JSON object
    brace_level = 0
    max_len = 0
    best_match = None
    start_index = -1

    for i, char in enumerate(text):
        if char == "{":
            if brace_level == 0:
                start_index = i
            brace_level += 1
        elif char == "}":
            if brace_level > 0:
                brace_level -= 1
                if brace_level == 0 and start_index != -1:
                    length = i - start_index + 1
                    if length > max_len:
                        max_len = length
                        best_match = text[start_index:i+1]

    return best_match


async def _repair_json_string(s: str) -> str:
    """Asynchronously fixes common errors in JSON strings from LLM.
    
    Args:
        s: The JSON string to fix
        
    Returns:
        The repaired JSON string
    """
    # Remove trailing commas
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    # Fix unquoted keys - simplified pattern
    s = re.sub(r"([{,]\s*)([a-zA-Z_]\w*)(\s*:)", r'\1"\2"\3', s)
    # Replace single quotes with double quotes (basic)
    s = re.sub(r"':\s*'([^']*)'", r'": "\1"', s)  # For values
    s = re.sub(r"'([\w_]+)':", r'"\1":', s)  # For keys

    # Handle python constants
    s = s.replace("True", "true").replace("False", "false").replace("None", "null")

    return s


async def extract_json_dict(raw_text: str | None) -> JsonExtractResult:
    """Robustly extracts a JSON dictionary from a raw LLM response string.
    This function implements a chain of strategies for finding and parsing JSON.
    
    Args:
        raw_text: The raw LLM response text
        
    Returns:
        A JsonExtractResult model with extracted data and operation status
    """
    result = JsonExtractResult()

    if not raw_text or not isinstance(raw_text, str):
        result.error = "Invalid input: text is empty or not a string"
        return result

    try:
        # Validate input through Pydantic
        validated_input = JsonBlock(text=raw_text)

        # Find the JSON block
        json_block = await _find_json_block(validated_input.text)
        if not json_block:
            result.error = "No potential JSON block found in the text"
            logger.debug(result.error)
            return result

        # Strategy 1: Try to parse directly
        try:
            result.data = json.loads(json_block)
            result.success = True
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"Initial json.loads failed: {e}. Trying repairs.")

        # Strategy 2: Repair the string and try again
        repaired_block = await _repair_json_string(json_block)
        try:
            result.data = json.loads(repaired_block)
            result.success = True
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"json.loads on repaired string failed: {e}. Trying json5.")

        # Strategy 3: Use json5 for more lenient parsing
        if JSON5_AVAILABLE:
            try:
                result.data = json5.loads(repaired_block)
                result.success = True
                return result
            except Exception as e:
                error_msg = f"json5 parsing failed: {e}. All parsing strategies exhausted."
                logger.debug(error_msg)
                result.error = error_msg

        # If all else fails, log the failure
        result.error = "All JSON parsing attempts failed for the text block"
        logger.error(result.error)
        return result

    except Exception as e:
        result.error = f"Unexpected error during JSON extraction: {e!s}"
        logger.exception("Error in extract_json_dict")
        return result


# Backward compatibility for synchronous use
def extract_json_dict_sync(raw_text: str | None) -> dict[str, Any] | None:
    """Synchronous wrapper for the asynchronous extract_json_dict function.
    
    Args:
        raw_text: The raw LLM response text
        
    Returns:
        The extracted JSON dictionary or None in case of an error
    """
    try:
        # Check if an event loop is running
        try:
            loop = asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        if has_running_loop:
            # If an event loop is already running, create a task
            future = asyncio.ensure_future(extract_json_dict(raw_text))

            # If we are in an async context but want a synchronous result
            if not loop.is_running():
                result = loop.run_until_complete(future)
                return result.data if result.success else None
            # In this case, we cannot block execution
            # Use synchronous check as a fallback
            logger.warning("Cannot wait for async extract_json_dict in running event loop. Using fallback extraction.")

            # Simple synchronous implementation for fallback
            if not raw_text or not isinstance(raw_text, str):
                return None

            # Find the JSON block
            code_block_match = re.search(r"```(?:json|javascript)?\s*(\{[\s\S]*?\})\s*```", raw_text, re.DOTALL)
            json_block = code_block_match.group(1).strip() if code_block_match else None

            if not json_block:
                # Find the largest top-level JSON object
                brace_level = 0
                max_len = 0
                best_match = None
                start_index = -1

                for i, char in enumerate(raw_text):
                    if char == "{":
                        if brace_level == 0:
                            start_index = i
                        brace_level += 1
                    elif char == "}":
                        if brace_level > 0:
                            brace_level -= 1
                            if brace_level == 0 and start_index != -1:
                                length = i - start_index + 1
                                if length > max_len:
                                    max_len = length
                                    best_match = raw_text[start_index:i+1]

                json_block = best_match

            if not json_block:
                return None

            # Try to parse directly
            try:
                return json.loads(json_block)
            except json.JSONDecodeError:
                # Use json5 if available
                if JSON5_AVAILABLE:
                    try:
                        return json5.loads(json_block)
                    except Exception:
                        pass
                return None
        else:
            # If no event loop is running, use asyncio.run
            result = asyncio.run(extract_json_dict(raw_text))
            return result.data if result.success else None
    except Exception as e:
        logger.error(f"Error in synchronous extraction wrapper: {e}")
        return None
