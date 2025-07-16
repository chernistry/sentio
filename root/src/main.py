import os
import logging
import asyncio
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router, pipeline
from .core.plugin_manager import PluginManager
import importlib

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

plugin_manager = PluginManager()

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline and plugins on startup."""
    try:
        logger.info("Server starting up...")
        
        # Initialize the main RAG pipeline first
        if not pipeline.initialized:
            await pipeline.initialize()
        
        # Discover and load plugins from the environment
        plugin_manager.load_from_env()
        
        # Register all loaded plugins with the pipeline instance
        plugin_manager.register_all(pipeline)
        
        # Explicitly load and register RAGAS plugin from plugins directory
        try:
            # Import the plugin module
            ragas_module = importlib.import_module("plugins.ragas_eval")
            ragas_plugin = ragas_module.get_plugin()
            
            logger.info(f"Explicitly loading RAGAS plugin: {ragas_plugin.name}")
            plugin_manager.register_plugin(ragas_plugin, pipeline)
            
            # Проверка успешности регистрации
            if hasattr(pipeline, "evaluator"):
                logger.info("✅ RAGAS plugin registered successfully, evaluator is available")
                logger.info(f"Evaluator methods: get_evaluation_history={hasattr(pipeline.evaluator, 'get_evaluation_history')}, get_average_metrics={hasattr(pipeline.evaluator, 'get_average_metrics')}")
                logger.info(f"Pipeline methods: get_evaluation_history={hasattr(pipeline, 'get_evaluation_history')}, get_average_metrics={hasattr(pipeline, 'get_average_metrics')}")
                
                # Доступ к методам через pipeline.evaluator
                if hasattr(pipeline.evaluator, "get_evaluation_history"):
                    history = pipeline.evaluator.get_evaluation_history()
                    logger.info(f"Evaluation history entries: {len(history)}")
            else:
                logger.warning("⚠️ RAGAS plugin registered but evaluator attribute not found on pipeline")
        except Exception as e:
            logger.error(f"Failed to load RAGAS plugin: {e}", exc_info=True)

    except Exception as e:
        logger.critical(f"FATAL: Server startup failed - {e}", exc_info=True)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add a custom header to the response to measure processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include the API router
app.include_router(router)

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