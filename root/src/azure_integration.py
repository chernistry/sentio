#!/usr/bin/env python3
"""
Sentio RAG Azure Integration Module

This module provides integration classes for Azure services used in the Sentio RAG application.
- Azure Storage Queue for message passing
- Azure Cosmos DB for document metadata storage
- Azure Application Insights for telemetry and logging

These components replace their non-Azure counterparts in the migration process.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional

# Azure Storage Queue SDK
from azure.storage.queue import QueueClient, BinaryBase64EncodePolicy
from azure.core.exceptions import AzureError

# Azure Cosmos DB SDK
from azure.cosmos import CosmosClient, PartitionKey, exceptions as cosmos_exceptions

# Azure Application Insights
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure import metrics_exporter
from opencensus.stats import stats as stats_module
from opencensus.trace import config_integration

# Configure logger
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


class AzureCosmosDBClient:
    """Client for interacting with Azure Cosmos DB."""
    
    def __init__(self):
        """Initialize Azure Cosmos DB client using environment variables."""
        endpoint = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")
        database_name = os.getenv("COSMOS_DATABASE_NAME", "sentio-db")
        container_name = os.getenv("COSMOS_CONTAINER_NAME", "metadata")
        
        if not endpoint or not key:
            # Try to construct endpoint from account name if direct endpoint not provided
            account_name = os.getenv("COSMOS_ACCOUNT_NAME")
            if account_name:
                endpoint = f"https://{account_name}.documents.azure.com:443/"
            else:
                logger.error("Cosmos DB endpoint/key not configured")
                raise ValueError("COSMOS_ENDPOINT and COSMOS_KEY or COSMOS_ACCOUNT_NAME required")
        
        # Initialize Cosmos client
        self.cosmos_client = CosmosClient(endpoint, credential=key)
        self.database = self.cosmos_client.get_database_client(database_name)
        self.container = self.database.get_container_client(container_name)
        
        logger.info(f"Initialized Cosmos DB client for database '{database_name}', container '{container_name}'")
    
    def create_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in Cosmos DB.
        
        Args:
            item: Dictionary representing the item to create
            
        Returns:
            Created item with additional metadata
            
        Raises:
            CosmosHttpResponseError: If creation fails
        """
        try:
            # Ensure the item has an id
            if "id" not in item:
                item["id"] = str(hash(frozenset(item.items())))
            
            created_item = self.container.create_item(body=item)
            logger.debug(f"Created item in Cosmos DB with id {item.get('id')}")
            return created_item
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to create item in Cosmos DB: {str(e)}")
            raise
    
    def query_items(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query items from Cosmos DB.
        
        Args:
            query: SQL query string
            parameters: Optional query parameters
            
        Returns:
            List of matching items
            
        Raises:
            CosmosHttpResponseError: If query fails
        """
        try:
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            logger.debug(f"Query returned {len(items)} items")
            return items
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to query items from Cosmos DB: {str(e)}")
            raise
    
    def get_item(self, item_id: str, partition_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a specific item by ID.
        
        Args:
            item_id: Item ID
            partition_key: Optional partition key value
            
        Returns:
            Item if found
            
        Raises:
            CosmosHttpResponseError: If retrieval fails or item not found
        """
        try:
            item = self.container.read_item(item=item_id, partition_key=partition_key or item_id)
            return item
        except cosmos_exceptions.CosmosHttpResponseError as e:
            if e.status_code == 404:
                logger.warning(f"Item with id {item_id} not found")
                return None
            logger.error(f"Failed to get item {item_id}: {str(e)}")
            raise
    
    def delete_item(self, item_id: str, partition_key: Optional[str] = None) -> None:
        """
        Delete an item by ID.
        
        Args:
            item_id: Item ID
            partition_key: Optional partition key value
            
        Raises:
            CosmosHttpResponseError: If deletion fails
        """
        try:
            self.container.delete_item(item=item_id, partition_key=partition_key or item_id)
            logger.debug(f"Deleted item {item_id} from Cosmos DB")
        except cosmos_exceptions.CosmosHttpResponseError as e:
            if e.status_code == 404:
                logger.warning(f"Item with id {item_id} not found for deletion")
                return
            logger.error(f"Failed to delete item {item_id}: {str(e)}")
            raise


def configure_azure_app_insights():
    """Configure Azure Application Insights for telemetry and logging."""
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        logger.warning("APPLICATIONINSIGHTS_CONNECTION_STRING not set, App Insights integration disabled")
        return False
    
    # Add Azure Log Handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(AzureLogHandler(connection_string=connection_string))
    
    # Configure metrics exporter
    exporter = metrics_exporter.new_metrics_exporter(
        connection_string=connection_string
    )
    stats = stats_module.stats
    view_manager = stats.view_manager
    view_manager.register_exporter(exporter)
    
    # Configure trace integration
    config_integration.trace_integrations(["logging", "requests", "sqlalchemy", "fastapi"])
    
    logger.info("Azure Application Insights integration configured")
    return True 