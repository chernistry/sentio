#!/usr/bin/env python3
"""
LangGraph Server Launcher

This script provides a convenient way to start the LangGraph Server for visualizing
and debugging the RAG pipeline graph in the browser.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("langgraph_server")

def main():
    """Start the LangGraph Server with the appropriate configuration."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    # Check if langgraph.json exists
    config_path = project_root / "langgraph.json"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Please create a langgraph.json file in the project root.")
        sys.exit(1)
    
    # Check if .env file exists
    env_path = project_root / ".env"
    if not env_path.exists():
        logger.warning(f".env file not found: {env_path}")
        logger.warning("LangSmith API key may not be configured.")
    
    # Change to the project root directory
    os.chdir(project_root)
    logger.info(f"Changed working directory to: {project_root}")
    
    # Start the LangGraph Server
    logger.info("Starting LangGraph Server...")
    try:
        # Use subprocess to run the langgraph dev command
        process = subprocess.run(
            ["langgraph", "dev"],
            check=True,
        )
        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start LangGraph Server: {e}")
        return e.returncode
    except KeyboardInterrupt:
        logger.info("LangGraph Server stopped by user.")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 