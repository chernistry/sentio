from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import json
import time
import os

from ..core.pipeline import SentioRAGPipeline, PipelineConfig

logger = logging.getLogger(__name__)

# Initialize the pipeline
pipeline_config = PipelineConfig(
    collection_name=os.getenv("QDRANT_COLLECTION", "Sentio_docs"),
)
pipeline = SentioRAGPipeline(pipeline_config)

# Initialize router
router = APIRouter()

# Models
class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    top_k: int = 3
    temperature: float = 0.7

class EmbedRequest(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = {}

# Routes
@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Process a chat request and return an answer with sources."""
    try:
        # Initialize pipeline if not already done
        if not pipeline.initialized:
            await pipeline.initialize()
        
        # Process the query
        result = await pipeline.query(
            request.question,
            top_k=request.top_k
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed")
async def embed_endpoint(request: EmbedRequest):
    """Embed a document and store it in the vector database."""
    try:
        # Initialize pipeline if not already done
        if not pipeline.initialized:
            await pipeline.initialize()
        
        # Process the document
        chunks_added = await pipeline.ingest_texts(
            [request.content],
            [request.metadata.get("source", request.id)]
        )
        
        return {
            "status": "success",
            "document_id": request.id,
            "chunks_added": chunks_added
        }
    except Exception as e:
        logger.error(f"Error in embed endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_endpoint():
    """Clear the vector database."""
    try:
        # Reset stats
        pipeline.reset_stats()
        return {"status": "success", "message": "Collection cleared"}
    except Exception as e:
        logger.error(f"Error in clear endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_endpoint():
    """Check the health of the system."""
    try:
        # Initialize pipeline if not already done
        if not pipeline.initialized:
            await pipeline.initialize()
            
        health_status = await pipeline.health_check()
        return health_status
    except Exception as e:
        logger.error(f"Error in health endpoint: {e}")
        return {"status": "unhealthy", "error": str(e)}

@router.get("/evaluation/history")
async def evaluation_history_endpoint():
    """Get the evaluation history."""
    try:
        # Initialize pipeline if not already done
        if not pipeline.initialized:
            await pipeline.initialize()
        
        # Check if evaluation history method exists
        if hasattr(pipeline, "get_evaluation_history"):
            history = pipeline.get_evaluation_history()
            return history
        else:
            raise HTTPException(status_code=404, detail="RAGAS evaluation not enabled")
    except Exception as e:
        logger.error(f"Error in evaluation history endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluation/metrics")
async def evaluation_metrics_endpoint():
    """Get the average evaluation metrics."""
    try:
        # Initialize pipeline if not already done
        if not pipeline.initialized:
            await pipeline.initialize()
        
        # Check if average metrics method exists
        if hasattr(pipeline, "get_average_metrics"):
            metrics = pipeline.get_average_metrics()
            return metrics
        else:
            raise HTTPException(status_code=404, detail="RAGAS evaluation not enabled")
    except Exception as e:
        logger.error(f"Error in evaluation metrics endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 