import json
import re
import logging
from typing import Optional, Dict, Any, Union
import asyncio
from pydantic import BaseModel, Field, root_validator

# Try importing json5, but don't make it a hard requirement if unavailable
try:
    import json5
    JSON5_AVAILABLE = True
except ImportError:
    json5 = None
    JSON5_AVAILABLE = False

# Optional pretty-printing dependencies
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:  # pragma: no cover
    RICH_AVAILABLE = False
    _console = None

logger = logging.getLogger(__name__)

class JsonBlock(BaseModel):
    """Модель для валидации и представления JSON блоков."""
    text: str = Field(..., description="Text content containing potential JSON")
    
    @root_validator(pre=True)
    def check_text_is_str(cls, values):
        if not isinstance(values.get('text', ''), str):
            raise ValueError("Text must be a string")
        return values

class JsonExtractResult(BaseModel):
    """Модель результата извлечения JSON."""
    data: Optional[Dict[str, Any]] = Field(None, description="Extracted JSON data")
    success: bool = Field(False, description="Whether extraction was successful")
    error: Optional[str] = Field(None, description="Error message if extraction failed")

async def _find_json_block(text: str) -> Optional[str]:
    """
    Асинхронно находит JSON блок в тексте, предпочитая блоки в markdown-разметке.
    
    Args:
        text: Исходный текст для поиска JSON блока
        
    Returns:
        Найденный JSON блок или None, если блок не найден
    """
    # Pattern for markdown code blocks (json, javascript, or none)
    code_block_match = re.search(r'```(?:json|javascript)?\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # If no code block, find the largest top-level JSON object
    # This is more robust than just finding the first and last brace.
    brace_level = 0
    max_len = 0
    best_match = None
    start_index = -1

    for i, char in enumerate(text):
        if char == '{':
            if brace_level == 0:
                start_index = i
            brace_level += 1
        elif char == '}':
            if brace_level > 0:
                brace_level -= 1
                if brace_level == 0 and start_index != -1:
                    length = i - start_index + 1
                    if length > max_len:
                        max_len = length
                        best_match = text[start_index:i+1]
    
    return best_match

async def _repair_json_string(s: str) -> str:
    """
    Асинхронно исправляет распространенные ошибки в JSON строках от LLM.
    
    Args:
        s: JSON строка для исправления
        
    Returns:
        Исправленная JSON строка
    """
    # Remove trailing commas
    s = re.sub(r',\s*([\}\]])', r'\1', s)
    # Fix unquoted keys - simplified pattern
    s = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', s)
    # Replace single quotes with double quotes (basic)
    # A bit risky, but often necessary. Let's make it safer.
    s = re.sub(r"':\s*'([^']*)'", r'": "\1"', s) # For values
    s = re.sub(r"'([\w_]+)':", r'"\1":', s) # For keys

    # Handle python constants
    s = s.replace('True', 'true').replace('False', 'false').replace('None', 'null')
    
    # Attempt to escape newlines within strings
    def escape_newlines(match):
        return match.group(1) + match.group(2).replace('\n', '\\n')
    
    return s

async def extract_json_dict(raw_text: Optional[str]) -> JsonExtractResult:
    """
    Робастно извлекает JSON словарь из сырой строки ответа LLM.
    Эта функция реализует цепочку стратегий для поиска и парсинга JSON.
    
    Args:
        raw_text: Сырой текст ответа LLM
        
    Returns:
        JsonExtractResult модель с извлеченными данными и статусом операции
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
        logger.error(result.error, extra={"json_block": json_block})
        return result
    
    except Exception as e:
        result.error = f"Unexpected error during JSON extraction: {str(e)}"
        logger.exception("Error in extract_json_dict", extra={"error": str(e)})
        return result

# Обратная совместимость с синхронным использованием
def extract_json_dict_sync(raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Синхронная обертка для асинхронной функции extract_json_dict.
    Для обратной совместимости.
    
    Args:
        raw_text: Сырой текст ответа LLM
        
    Returns:
        Извлеченный JSON словарь или None в случае ошибки
    """
    try:
        # Проверяем, запущен ли event loop
        try:
            loop = asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False
        
        if has_running_loop:
            # Если event loop уже запущен, создаем задачу
            future = asyncio.ensure_future(extract_json_dict(raw_text))
            
            # Если мы в асинхронном контексте, но хотим синхронный результат
            if not loop.is_running():
                result = loop.run_until_complete(future)
                return result.data if result.success else None
            else:
                # В этом случае мы не можем блокировать выполнение
                # Используем синхронную проверку как резервный вариант
                logger.warning("Cannot wait for async extract_json_dict in running event loop. Using fallback extraction.")
                
                # Простая синхронная реализация для резервного варианта
                if not raw_text or not isinstance(raw_text, str):
                    return None
                
                # Ищем JSON блок
                code_block_match = re.search(r'```(?:json|javascript)?\s*(\{[\s\S]*?\})\s*```', raw_text, re.DOTALL)
                json_block = code_block_match.group(1).strip() if code_block_match else None
                
                if not json_block:
                    # Ищем самый большой JSON объект
                    brace_level = 0
                    max_len = 0
                    best_match = None
                    start_index = -1

                    for i, char in enumerate(raw_text):
                        if char == '{':
                            if brace_level == 0:
                                start_index = i
                            brace_level += 1
                        elif char == '}':
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
                
                # Пытаемся парсить напрямую
                try:
                    return json.loads(json_block)
                except json.JSONDecodeError:
                    # Используем json5 если доступен
                    if JSON5_AVAILABLE:
                        try:
                            return json5.loads(json_block)
                        except Exception:
                            pass
                    return None
        else:
            # Если event loop не запущен, используем asyncio.run
            result = asyncio.run(extract_json_dict(raw_text))
            return result.data if result.success else None
    except Exception as e:
        logger.error(f"Error in synchronous extraction wrapper: {e}")
        return None

# ---------------------------------------------------------------------------
# Human-friendly rendering helpers
# ---------------------------------------------------------------------------

def _render_plain(data: Dict[str, Any], max_sources: int = 5) -> str:
    """Fallback plain-text formatter when Rich is unavailable."""
    answer = data.get("answer", "").strip()
    sources = data.get("sources", []) or []

    lines = ["Answer:\n" + answer, "\nSources:"]
    for i, src in enumerate(sources[:max_sources], start=1):
        title = src.get("title") or src.get("metadata", {}).get("title") or f"Source {i}"
        score = src.get("score", "?")
        lines.append(f"  {i}. {title}  (score={score})")
    if len(sources) > max_sources:
        lines.append(f"  … and {len(sources) - max_sources} more")
    return "\n".join(lines)


def _render_rich(data: Dict[str, Any], max_sources: int = 5) -> str:
    """Render reply with Rich, captured as an ANSI string."""
    assert _console is not None  # should be guarded by caller

    answer = data.get("answer", "").strip() or "—"
    sources = data.get("sources", []) or []

    # Clean up the answer text by removing potential AI self-instructions
    # (often found after "--" or "----" markers)
    answer_parts = answer.split('-' * 4)
    if len(answer_parts) > 1:
        # Take only the first part before the divider
        answer = answer_parts[0].strip()

    # Format answer panel
    table = Table(title="Sources", box=box.SIMPLE, show_lines=False, expand=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Source", style="blue")
    table.add_column("Score", justify="right", style="green")

    for i, src in enumerate(sources[:max_sources], start=1):
        title = src.get("title") or src.get("metadata", {}).get("title") or "—"
        source = src.get("source", "Unknown")
        score_val = src.get("score")
        score = f"{score_val:.2f}" if isinstance(score_val, (int, float)) else str(score_val or "—")
        table.add_row(str(i), title, source, score)

    if len(sources) > max_sources:
        table.caption = f"… and {len(sources) - max_sources} more"

    with _console.capture() as capture:
        _console.print(Panel(answer, title="Answer", border_style="magenta"))
        _console.print(table)
    return capture.get()


def format_llm_response(
    raw_response: Union[str, Dict[str, Any]],
    *,
    max_sources: int = 5,
    use_rich: Optional[bool] = None,
) -> str:
    """Parse and prettify an LLM JSON response.

    Parameters
    ----------
    raw_response: str | dict
        Either a raw string potentially containing a JSON block, or a parsed
        dictionary with at least the keys ``answer`` & ``sources``.
    max_sources: int, default 5
        Maximum number of sources to show before truncating.
    use_rich: bool | None
        Force using Rich (``True``) or plain text (``False``). If *None*,
        autodetects based on availability.

    Returns
    -------
    str
        A formatted ANSI (if Rich) or plain-text representation ready for
        printing to the terminal.
    """
    # Decide whether to use Rich
    if use_rich is None:
        use_rich = RICH_AVAILABLE

    # Parse input if needed
    if isinstance(raw_response, dict):
        data = raw_response
    else:
        data = extract_json_dict_sync(raw_response) or {}

    if not data:
        # Fallback to raw string if parsing failed
        return str(raw_response)

    # Clean up text in sources if present
    if "sources" in data and isinstance(data["sources"], list):
        for src in data["sources"]:
            if "text" in src and isinstance(src["text"], str):
                # Replace tabs with spaces and normalize whitespace
                text = src["text"].replace("\t", " ")
                # Handle line breaks more elegantly
                text = text.replace("\n", " ").strip()
                # Normalize multiple spaces to single space
                text = " ".join(text.split())
                src["text"] = text

    renderer = _render_rich if use_rich and RICH_AVAILABLE else _render_plain
    return renderer(data, max_sources=max_sources)
