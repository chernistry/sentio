"""
Sentio Worker - Main entry point when running as a module
"""

import os
import sys
import logging
from pathlib import Path

# Get log file path from environment or use default
log_file = os.environ.get('LOG_FILE', '/app/data/worker.log')

# Make sure the directory exists
log_dir = os.path.dirname(log_file)
try:
    os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create log directory {log_dir}: {e}")

# Configure logging
logging.basicConfig(
    level=logging.info,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
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