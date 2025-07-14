#!/usr/bin/env python3
"""
Sentio Worker - Azure Queue Message Processor

Polls Azure Storage Queue for messages and processes them according to type.
- Document ingestion
- Chat completions tracking
- Other asynchronous tasks
"""

import os
import sys
import json
import time
import logging
import signal
from typing import Dict, Any, Optional

# Import Azure integration modules
from src.integrations.azure.queue import AzureQueueClient

# Configure logging
logging.basicConfig(
    level=logging.info,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/worker.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Signal handling for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received, completing current task before exiting...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process_message(message: Dict[str, Any]) -> bool:
    """
    Process a message based on its type.
    
    Args:
        message: Message dictionary from queue
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        event_type = message.get("event_type")
        logger.info(f"Processing message with event_type: {event_type}")
        
        if event_type == "document_ingestion":
            # Process document ingestion request
            import subprocess
            file_path = message.get("file_path")
            if file_path:
                logger.info(f"Starting ingestion for file: {file_path}")
                result = subprocess.run(
                    ["python", "-m", "src.etl.ingest", "--data-dir", os.path.dirname(file_path)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    logger.info(f"Ingestion successful: {file_path}")
                else:
                    logger.error(f"Ingestion failed: {result.stderr}")
                    return False
            else:
                logger.warning("Missing file_path in document_ingestion message")
                return False
                
        elif event_type == "chat_completion":
            # Process chat completion (e.g., save to database, analytics)
            logger.info(f"Processing chat completion: {message.get('request_id')}")
            # Implementation specific to your needs
            time.sleep(1)  # Simulate processing time
            
        else:
            logger.warning(f"Unknown event type: {event_type}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return False

def main():
    """
    Main worker function that polls queue and processes messages.
    """
    logger.info("Starting Sentio Worker - Azure Queue Processor")
    
    try:
        # Initialize Azure Queue client
        queue_client = AzureQueueClient()
        logger.info("Connected to Azure Queue")
        
        poll_interval = int(os.getenv("QUEUE_POLL_INTERVAL_SECONDS", "10"))
        visibility_timeout = int(os.getenv("MESSAGE_VISIBILITY_TIMEOUT", "300"))  # 5 minutes
        
        # Main processing loop
        while running:
            try:
                # Receive message from queue
                messages = queue_client.receive_messages(
                    max_messages=1,
                    visibility_timeout=visibility_timeout
                )
                
                if messages:
                    message = messages[0]
                    logger.info(f"Received message: {message.get('event_type', 'unknown')}")
                    
                    # Extract queue metadata
                    queue_metadata = message.pop("_queue_metadata", {})
                    message_id = queue_metadata.get("id")
                    pop_receipt = queue_metadata.get("pop_receipt")
                    
                    if not message_id or not pop_receipt:
                        logger.warning("Missing message ID or pop receipt, skipping")
                        continue
                    
                    # Process the message
                    success = process_message(message)
                    
                    # Delete message from queue if processed successfully
                    if success:
                        queue_client.delete_message(message_id, pop_receipt)
                        logger.info(f"Message {message_id} processed and deleted")
                    else:
                        logger.warning(f"Failed to process message {message_id}, leaving in queue")
                else:
                    logger.debug("No messages in queue, waiting...")
                    time.sleep(poll_interval)
                    
            except Exception as e:
                logger.error(f"Error in main processing loop: {str(e)}")
                time.sleep(poll_interval)  # Back off on error
                
        logger.info("Worker shutting down gracefully")
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    # Don't exit after running - let startup.sh handle the looping
    main() 