#!/usr/bin/env python3
"""
Azure Storage Queue Integration Module

Provides functionality for interacting with Azure Storage Queues
for asynchronous message processing.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

# Azure Storage Queue SDK
from azure.storage.queue import QueueClient, BinaryBase64EncodePolicy
from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)

class AzureQueueClient:
    """Client for interacting with Azure Storage Queue."""
    
    def __init__(self):
        """Initialize Azure Queue client using environment variables."""
        conn_string = os.getenv("AZURE_QUEUE_CONNECTION_STRING")
        queue_name = os.getenv("AZURE_QUEUE_NAME", "submissions")
        
        if not conn_string:
            logger.error("AZURE_QUEUE_CONNECTION_STRING environment variable not set")
            raise ValueError("AZURE_QUEUE_CONNECTION_STRING environment variable not set")
        
        self.queue_client = QueueClient.from_connection_string(
            conn_string=conn_string, 
            queue_name=queue_name,
            message_encode_policy=BinaryBase64EncodePolicy()
        )
        logger.info(f"Initialized Azure Queue client for queue '{queue_name}'")
    
    def send_message(self, message: Dict[str, Any], visibility_timeout: int = None) -> str:
        """
        Send a message to the Azure Storage Queue.
        
        Args:
            message: Dictionary to be sent as JSON message
            visibility_timeout: Optional visibility timeout in seconds
            
        Returns:
            Message ID if successful
            
        Raises:
            AzureError: If message sending fails
        """
        try:
            message_json = json.dumps(message)
            response = self.queue_client.send_message(
                message_json,
                visibility_timeout=visibility_timeout
            )
            logger.debug(f"Message sent to queue: {response.id}")
            return response.id
        except AzureError as e:
            logger.error(f"Failed to send message to Azure Queue: {str(e)}")
            raise
    
    def receive_messages(self, max_messages: int = 1, visibility_timeout: int = 30) -> List[Dict]:
        """
        Receive messages from the Azure Storage Queue.
        
        Args:
            max_messages: Maximum number of messages to receive (1-32)
            visibility_timeout: Visibility timeout in seconds
            
        Returns:
            List of messages as Python dictionaries
            
        Raises:
            AzureError: If message retrieval fails
        """
        try:
            messages = []
            for msg in self.queue_client.receive_messages(
                max_messages=max_messages,
                visibility_timeout=visibility_timeout
            ):
                try:
                    message_content = json.loads(msg.content)
                    message_content["_queue_metadata"] = {
                        "id": msg.id,
                        "pop_receipt": msg.pop_receipt,
                        "inserted_on": msg.inserted_on.isoformat() if msg.inserted_on else None
                    }
                    messages.append(message_content)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping non-JSON message: {msg.id}")
                    
            return messages
        except AzureError as e:
            logger.error(f"Failed to receive messages from Azure Queue: {str(e)}")
            raise
    
    def delete_message(self, message_id: str, pop_receipt: str) -> None:
        """
        Delete a message from the queue.
        
        Args:
            message_id: Message ID
            pop_receipt: Pop receipt from the receive_messages call
            
        Raises:
            AzureError: If deletion fails
        """
        try:
            self.queue_client.delete_message(message_id, pop_receipt)
            logger.debug(f"Message {message_id} deleted from queue")
        except AzureError as e:
            logger.error(f"Failed to delete message {message_id}: {str(e)}")
            raise 