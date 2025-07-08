#!/usr/bin/env python3
"""
Azure Cosmos DB Integration Module

Provides functionality for interacting with Azure Cosmos DB
for document metadata storage.
"""

import os
import logging
from typing import Dict, Any, List, Optional

# Azure Cosmos DB SDK
from azure.cosmos import CosmosClient, PartitionKey, exceptions as cosmos_exceptions

logger = logging.getLogger(__name__)

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