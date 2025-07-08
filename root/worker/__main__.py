"""
Sentio Worker - Main entry point when running as a module
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/data/worker.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point that runs the worker.py script functionality
    """
    try:
        # Add parent directory to path so we can import worker.py
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        # Import worker functions without exiting
        from worker import main as worker_main
        
        # Run the worker main function but don't exit
        result = worker_main()
        logger.info(f"Worker process completed with status: {result}")
        return result
    except Exception as e:
        logger.critical(f"Fatal error in worker module: {str(e)}")
        return 1

if __name__ == "__main__":
    # Call main but don't exit
    main() 