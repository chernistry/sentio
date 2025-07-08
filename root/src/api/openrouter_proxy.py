import os, random, httpx, asyncio, time
from typing import Dict, Any, AsyncGenerator, Union

OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

# Prefer CSV file (one key per line, header optional)
_CSV_PATH = os.getenv("OPENROUTER_KEYS_CSV", "/app/keys.csv")

def _load_csv(path: str):
    if not os.path.isfile(path):
        return []
    keys: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.lower() == "key":
                continue
            if line.startswith("sk-"):
                keys.append(line)
    return keys

OPENROUTER_KEYS = _load_csv(_CSV_PATH)

# Fallback to env variable list
if not OPENROUTER_KEYS:
    _KEYS_ENV = os.getenv("OPENROUTER_KEYS", "")
    OPENROUTER_KEYS = [k.strip() for k in _KEYS_ENV.split(";") if k.strip()]
    if not OPENROUTER_KEYS:
        single = os.getenv("OPENROUTER_KEY", "").strip()
        if single:
            OPENROUTER_KEYS.append(single)
        # Additional fallback: support legacy singular variable name
        if not OPENROUTER_KEYS:
            legacy = os.getenv("OPENROUTER_API_KEY", "").strip()
            if legacy:
                OPENROUTER_KEYS.append(legacy)

class _KeyPool:
    """Round-robin pool with simple disable/timeout logic (in-memory)."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise RuntimeError("No OPENROUTER_KEYS provided")
        # store dicts: {key:str, disabled_until:float}
        self._keys = [{"key": k, "disabled_until": 0.0} for k in keys]
        self._idx = 0
        self._lock = asyncio.Lock()

    async def next(self) -> str:
        async with self._lock:
            start = self._idx
            n = len(self._keys)
            while True:
                item = self._keys[self._idx]
                self._idx = (self._idx + 1) % n
                if time.time() >= item["disabled_until"]:
                    return item["key"]
                if self._idx == start:
                    # all keys disabled; wait 1s
                    await asyncio.sleep(1)

    async def disable(self, key: str, timeout: int = 600):
        async with self._lock:
            for item in self._keys:
                if item["key"] == key:
                    item["disabled_until"] = max(item["disabled_until"], time.time() + timeout)
                    break

auth_pool = _KeyPool(OPENROUTER_KEYS) if OPENROUTER_KEYS else None

# --- helpers -----------------------------------------------------------

def _stealth_headers() -> Dict[str, str]:
    """Mimic headers OpenRouter recommends (prevents quota pooling detection)."""
    ref = os.getenv("OPENROUTER_REFERER", "https://openwebui.com/")
    title = os.getenv("OPENROUTER_TITLE", "SentioGateway")
    return {
        "HTTP-Referer": ref,
        "X-Title": title,
    }

# --- public API --------------------------------------------------------

async def list_models() -> Dict[str, Any]:
    if not auth_pool:
        return {"object": "list", "data": []}
    key = await auth_pool.next()
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{OPENROUTER_API_BASE}/models",
            headers={
                "Authorization": f"Bearer {key}",
                **_stealth_headers(),
            },
        )
        resp.raise_for_status()
        return resp.json()

async def chat_completion(payload: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
    """Forward /chat/completions to OpenRouter with key rotation.
    If stream=True in payload – returns an async generator yielding raw event lines.
    Otherwise returns parsed JSON dict.
    """
    if not auth_pool:
        raise RuntimeError("OPENROUTER_KEYS not configured")
    key = await auth_pool.next()
    stream = bool(payload.get("stream", False))
    
    # Replace deprecated model with the current one
    if payload.get("model") == "deepseek-chat-v3-0324:free":
        payload["model"] = "mistralai/mistral-small-3.2-24b-instruct:free"
    
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            resp = await client.post(
                f"{OPENROUTER_API_BASE}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    **_stealth_headers(),
                }
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # 401/403 – key invalid, disable long
            if status in (401, 403):
                await auth_pool.disable(key, 24 * 3600)
            # 429/5xx – temporary
            elif status in (429, 500, 502, 503, 504):
                await auth_pool.disable(key, 600)
            raise
        if not stream:
            return resp.json()

        async def _gen():
            # Process event stream from OpenRouter
            try:
                async for line in resp.aiter_lines():
                    if line.strip():
                        # Remove "OPENROUTER PROCESSING" from the beginning of the line, if present
                        if line.startswith(": OPENROUTER PROCESSING"):
                            line = line[len(": OPENROUTER PROCESSING"):]
                        yield line
            except Exception as e:
                yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"
                return
        return _gen() 