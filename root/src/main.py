import os
import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router
from .core.plugin_manager import PluginManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Sentio RAG API",
    description="API for Sentio Retrieval-Augmented Generation",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(router)

# Initialize plugin manager
plugin_manager = PluginManager()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        logger.info("Loading plugins...")
        # Load plugins from environment variables
        plugin_manager.load_from_env()
        
        # Load RAGAS evaluator plugin
        try:
            # Import from the new core module location
            from root.src.core.llm.ragas import get_plugin
            ragas_plugin = get_plugin()
            plugin_manager.register_plugin(ragas_plugin)
            logger.info("✅ RAGAS evaluation plugin loaded successfully")
        except ImportError as e:
            logger.warning(f"RAGAS plugin not available: {e}")
        except Exception as e:
            logger.error(f"Error loading RAGAS plugin: {e}")
            
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    try:
        from .api.routes import pipeline
        if hasattr(pipeline, "aclose"):
            await pipeline.aclose()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=True,
    ) 