from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import json
import time
import os
import asyncio

from ..core.pipeline import SentioRAGPipeline, PipelineConfig
from ..core.graph import build_basic_graph, build_streaming_graph
from ..utils.settings import settings

logger = logging.getLogger(__name__)

# Initialize the pipeline
pipeline_config = PipelineConfig(
    collection_name=os.getenv("QDRANT_COLLECTION", "Sentio_docs"),
)
pipeline = SentioRAGPipeline(pipeline_config)

# Initialize LangGraph (lazily)
graph = None
streaming_graph = None

# Initialize router
router = APIRouter()

# Models
class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    top_k: int = 3
    temperature: float = 0.7
    stream: bool = False

class EmbedRequest(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = {}

# Helper function to get or initialize the LangGraph
async def get_graph():
    global graph
    if graph is None:
        # Initialize pipeline if not already done
        if not pipeline.initialized:
            await pipeline.initialize()
        # Build graph using the pipeline components
        graph = build_basic_graph(pipeline_config, pipeline)
        logger.info("LangGraph initialized successfully")
    return graph

# Helper function to get or initialize the Streaming LangGraph
async def get_streaming_graph():
    global streaming_graph
    if streaming_graph is None:
        # Initialize pipeline if not already done
        if not pipeline.initialized:
            await pipeline.initialize()
        # Build streaming graph using the pipeline components
        streaming_graph = build_streaming_graph(pipeline_config, pipeline)
        logger.info("Streaming LangGraph initialized successfully")
    return streaming_graph

# Routes
@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Process a chat request and return an answer with sources."""
    try:
        # Initialize pipeline if not already done
        if not pipeline.initialized:
            await pipeline.initialize()
        
        # If streaming is requested, return a StreamingResponse
        if request.stream:
            return await chat_stream_endpoint(request)
        
        # Process the query using either LangGraph or classic pipeline
        if settings.use_langgraph:
            logger.info(f"Using LangGraph for query: {request.question}")
            graph_instance = await get_graph()
            result = await graph_instance.ainvoke({"query": request.question})
            
            # Format response to match classic pipeline output
            response = {
                "answer": result.answer,
                "sources": result.sources,
                "metadata": result.metadata
            }
            return response
        else:
            logger.info(f"Using classic pipeline for query: {request.question}")
            result = await pipeline.query(
                request.question,
                top_k=request.top_k
            )
            return result
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def chat_stream_endpoint(request: ChatRequest):
    """Process a streaming chat request and return a StreamingResponse."""
    try:
        if not settings.use_langgraph:
            # If LangGraph is disabled, fall back to non-streaming response
            logger.warning("Streaming requested but LangGraph is disabled. Falling back to non-streaming response.")
            result = await pipeline.query(request.question, top_k=request.top_k)
            
            # Return a streaming response that only emits one item
            async def fake_stream():
                yield json.dumps({"answer": result["answer"], "done": True}) + "\n"
            
            return StreamingResponse(
                fake_stream(),
                media_type="text/event-stream"
            )
        
        # Get or initialize streaming graph
        streaming_graph_instance = await get_streaming_graph()
        
        # Create a streaming response
        async def stream_response():
            try:
                # Start the streaming process
                async for chunk in streaming_graph_instance.astream({"query": request.question}):
                    if "answer" in chunk:
                        # Format each chunk as a JSON string followed by a newline
                        yield json.dumps({"answer": chunk["answer"], "done": False}) + "\n"
                
                # Send a final message indicating completion
                yield json.dumps({"done": True}) + "\n"
            except Exception as e:
                logger.error(f"Error in streaming response: {e}", exc_info=True)
                yield json.dumps({"error": str(e), "done": True}) + "\n"
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error setting up streaming response: {e}", exc_info=True)
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
        
        # Add LangGraph status if enabled
        if settings.use_langgraph:
            try:
                graph_instance = await get_graph()
                health_status["langgraph"] = "healthy" if graph_instance else "not initialized"
                
                # Check streaming graph too
                streaming_graph_instance = await get_streaming_graph() 
                health_status["langgraph_streaming"] = "healthy" if streaming_graph_instance else "not initialized"
            except Exception as e:
                health_status["langgraph"] = f"error: {str(e)}"
                
        return health_status
    except Exception as e:
        logger.error(f"Error in health endpoint: {e}")
        return {"status": "unhealthy", "error": str(e)}

@router.get("/evaluation/history")
async def evaluation_history_endpoint():
    """Get the evaluation history."""
    try:
        logger.debug("Handling /evaluation/history request")
        # Initialize pipeline if not already done
        if not pipeline.initialized:
            logger.debug("Pipeline not initialized, initializing...")
            await pipeline.initialize()
            logger.debug("Pipeline initialized")
        
        # Check if evaluation history method exists
        logger.debug(f"Pipeline has get_evaluation_history: {hasattr(pipeline, 'get_evaluation_history')}")
        logger.debug(f"Pipeline has evaluator: {hasattr(pipeline, 'evaluator')}")
        if hasattr(pipeline, 'evaluator'):
            logger.debug(f"Pipeline.evaluator has get_evaluation_history: {hasattr(pipeline.evaluator, 'get_evaluation_history')}")
        
        if hasattr(pipeline, "get_evaluation_history"):
            logger.debug("Calling pipeline.get_evaluation_history()")
            history = pipeline.get_evaluation_history()
            logger.debug(f"Got history with {len(history)} entries")
            return history
        elif hasattr(pipeline, "evaluator") and hasattr(pipeline.evaluator, "get_evaluation_history"):
            logger.debug("Calling pipeline.evaluator.get_evaluation_history()")
            history = pipeline.evaluator.get_evaluation_history()
            logger.debug(f"Got history with {len(history)} entries")
            return history
        else:
            logger.warning("RAGAS evaluation not enabled, neither pipeline nor evaluator has get_evaluation_history method")
            raise HTTPException(status_code=404, detail="RAGAS evaluation not enabled")
    except Exception as e:
        logger.error(f"Error in evaluation history endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluation/metrics")
async def evaluation_metrics_endpoint():
    """Get the average evaluation metrics."""
    try:
        logger.debug("Handling /evaluation/metrics request")
        # Initialize pipeline if not already done
        if not pipeline.initialized:
            logger.debug("Pipeline not initialized, initializing...")
            await pipeline.initialize()
            logger.debug("Pipeline initialized")
        
        # Check if average metrics method exists
        logger.debug(f"Pipeline has get_average_metrics: {hasattr(pipeline, 'get_average_metrics')}")
        logger.debug(f"Pipeline has evaluator: {hasattr(pipeline, 'evaluator')}")
        if hasattr(pipeline, 'evaluator'):
            logger.debug(f"Pipeline.evaluator has get_average_metrics: {hasattr(pipeline.evaluator, 'get_average_metrics')}")
        
        if hasattr(pipeline, "get_average_metrics"):
            logger.debug("Calling pipeline.get_average_metrics()")
            metrics = pipeline.get_average_metrics()
            logger.debug(f"Got metrics: {metrics}")
            return metrics
        elif hasattr(pipeline, "evaluator") and hasattr(pipeline.evaluator, "get_average_metrics"):
            logger.debug("Calling pipeline.evaluator.get_average_metrics()")
            metrics = pipeline.evaluator.get_average_metrics()
            logger.debug(f"Got metrics: {metrics}")
            return metrics
        else:
            logger.warning("RAGAS evaluation not enabled, neither pipeline nor evaluator has get_average_metrics method")
            raise HTTPException(status_code=404, detail="RAGAS evaluation not enabled")
    except Exception as e:
        logger.error(f"Error in evaluation metrics endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 