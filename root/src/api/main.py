from fastapi import FastAPI, HTTPException, Body, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import AsyncGenerator, List
from prometheus_fastapi_instrumentator import Instrumentator
import json, uuid, time, logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Use relative path for import that will work with the new structure
from src.core.pipeline import SentioRAGPipeline
from src.api.openrouter_proxy import list_models as or_list_models, chat_completion as or_chat_completion

app = FastAPI(title="Sentio RAG API", version="0.1.0")

# Initialize metrics instrumentator (Prometheus)
instrumentator = (
    Instrumentator()
    .instrument(app)
    .expose(app, include_in_schema=False, should_gzip=True)
)

# CORS for local dev UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: SentioRAGPipeline | None = None


# ---------------------------- OpenAI compatibility Models ----------------------------

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "Sentio"

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]

# --------------------------------------------------------------------------------

def get_pipeline() -> SentioRAGPipeline:
    global pipeline
    if pipeline is None:
        # Lazy init to avoid heavy deps during import (esp. tests)
        try:
            pipeline = SentioRAGPipeline()
        except Exception as e:  # capture environment issues
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    return pipeline


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    status: str
    files: int
    chunks_indexed: int


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok"}


async def stream_generator(agen: AsyncGenerator[str, None]):
    """Wrap tokens as Server-Sent Events (SSE)."""
    async for token in agen:
        # Each SSE event must end with a double newline
        yield f"data: {token}\n\n"


@app.post("/chat/stream", tags=["chat"])
async def chat_stream(req: ChatRequest):
    """Endpoint to handle chat requests with a streaming response (SSE)."""
    try:
        response_generator = get_pipeline().query_stream(req.question, top_k=req.top_k)
        # Return SSE stream
        return StreamingResponse(
            stream_generator(response_generator),
            media_type="text/event-stream",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest):
    try:
        answer = await get_pipeline().query(req.question, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    return ChatResponse(answer=answer)


# ---------------------------- OpenAI compatibility ----------------------------

@app.get("/v1/models", response_model=ModelList, tags=["openai"])
async def list_models():
    """Return union of local RAG model plus OpenRouter models."""
    cards: List[ModelCard] = [ModelCard(id="Sentio-rag-model")]
    try:
        or_resp = await or_list_models()
        for m in or_resp.get("data", []):
            mid = m.get("id") or m.get("name")
            if mid:
                cards.append(ModelCard(id=mid, owned_by="openrouter"))
    except Exception as e:
        logger.error(f"Failed to fetch OpenRouter models: {str(e)}")
        pass  # graceful degradation
    return ModelList(data=cards)

@app.post("/v1/chat/completions", tags=["openai"])
async def openai_chat_completions(request: Request, payload: dict = Body(...)):
    """Minimal OpenAI-compatible endpoint for Open WebUI integration.

    Only a subset of the full specification is implemented:
    - `messages`: list with at least one user message (the last user message is used).
    - `stream`: if true, SSE-style streaming chunks are returned following OpenAI format.
    Additional fields are accepted but ignored for now.
    """
    model = payload.get("model", "Sentio-rag-model")
    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="'messages' field is required")

    # Extract user content from the last message
    user_content = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    if not user_content:
        raise HTTPException(status_code=400, detail="No user message found in 'messages'")

    top_k = payload.get("top_k", 5)
    stream = bool(payload.get("stream", False))
    
    logger.info(f"Chat request: model={model}, stream={stream}, content={user_content[:50]}...")

    # Add headers for Open WebUI compatibility
    headers = {
        "Content-Type": "text/event-stream" if stream else "application/json",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }

    # Route to OpenRouter if not local RAG
    if model != "Sentio-rag-model":
        # Ensure we respect stream flag
        try:
            if stream:
                # OpenRouter generator
                logger.info(f"Streaming from OpenRouter: {model}")
                gen = await or_chat_completion(payload)
                
                async def debug_stream():
                    async for chunk in gen:
                        logger.debug(f"Stream chunk: {chunk[:100]}")
                        yield chunk
                
                return StreamingResponse(debug_stream(), media_type="text/event-stream", headers=headers)
            else:
                logger.info(f"Non-streaming from OpenRouter: {model}")
                response = await or_chat_completion(payload)
                logger.debug(f"OpenRouter response: {json.dumps(response)[:200]}")
                return JSONResponse(content=response, headers=headers)
        except Exception as e:
            logger.error(f"OpenRouter error: {str(e)}")
            raise HTTPException(status_code=502, detail=str(e))

    try:
        if stream:
            logger.info("Streaming from local RAG model")
            async def _event_stream():
                async for token in get_pipeline().query_stream(user_content, top_k=top_k):
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "delta": {"role": "assistant", "content": token},
                                "index": 0,
                                "finish_reason": None,
                            }
                        ],
                    }
                    chunk_str = json.dumps(chunk, ensure_ascii=False)
                    logger.debug(f"RAG stream chunk: {chunk_str[:100]}")
                    yield f"data: {chunk_str}\n\n"
                
                # Send final chunk with finish_reason: "stop"
                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "delta": {"role": "assistant", "content": ""},
                            "index": 0,
                            "finish_reason": "stop",
                        }
                    ],
                }
                final_str = json.dumps(final_chunk, ensure_ascii=False)
                logger.debug(f"RAG final chunk: {final_str}")
                yield f"data: {final_str}\n\n"
                
                # Signal completion per OpenAI spec
                logger.debug("Sending [DONE]")
                yield "data: [DONE]\n\n"

            return StreamingResponse(_event_stream(), media_type="text/event-stream", headers=headers)

        # Non-streaming path
        logger.info("Non-streaming from local RAG model")
        answer = await get_pipeline().query(user_content, top_k=top_k)
        response_body = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            # Token usage accounting is skipped (set to zero for now)
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        logger.debug(f"RAG response: {json.dumps(response_body)[:200]}")
        return JSONResponse(content=response_body, headers=headers)
    except Exception as e:
        # If the local model does not work, report an error
        logger.error(f"RAG model error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG model error: {str(e)}")


# ---------------------------- Docs Ingestion ----------------------------

@app.post("/v1/docs/upload", response_model=UploadResponse, tags=["docs"])
async def upload_docs(files: List[UploadFile] = File(...)):
    """Upload and ingest documents into the vector store.

    Accepts one or multiple files (text, markdown, etc.). Each file is split into chunks,
    embedded and indexed in Qdrant.
    """
    try:
        contents: List[str] = []
        sources: List[str] = []
        for file in files:
            data = await file.read()
            # Attempt to decode as UTF-8; fallback to Latin-1
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1", errors="ignore")
            contents.append(text)
            sources.append(file.filename or f"file_{len(sources)}")

        chunks = await get_pipeline().ingest_texts(contents, sources)

        return UploadResponse(status="ok", files=len(files), chunks_indexed=chunks)
    except Exception as e:
        logger.error(f"Doc upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Doc upload failed: {str(e)}") 