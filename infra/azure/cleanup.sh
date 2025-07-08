#!/bin/bash
# Cleanup Azure resources for Sentio RAG project

set -e

# Variables
RESOURCE_GROUP="rg-sentio-free"
KEY_VAULT_NAME="kv-sentio-free"
QDRANT_CONTAINER="aci-qdrant-free"
API_CONTAINER="aci-sentio-api"
WORKER_CONTAINER="aci-sentio-worker"

# Check if logged in to Azure CLI
echo "Checking Azure CLI login..."
az account show > /dev/null || { echo "Please login to Azure CLI using 'az login'"; exit 1; }

# Check if resource group exists
if ! az group show --name $RESOURCE_GROUP &>/dev/null; then
    echo "Resource group '$RESOURCE_GROUP' does not exist. Nothing to clean up."
    exit 0
fi

echo "WARNING: This will delete all resources in the resource group '$RESOURCE_GROUP'!"
echo "This includes:"
echo "- Qdrant Container Instance"
echo "- API Container Instance"
echo "- Worker Container Instance"
echo "- Key Vault"
echo "- Storage Account"
echo "- Log Analytics Workspace"
echo "- All other resources in the resource group"

read -p "Are you sure you want to continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Stop containers first if they exist
echo "Stopping container instances if they exist..."

# Check and stop Qdrant container
if az container show --resource-group $RESOURCE_GROUP --name $QDRANT_CONTAINER &>/dev/null; then
    echo "Stopping Qdrant container..."
    az container stop --resource-group $RESOURCE_GROUP --name $QDRANT_CONTAINER --no-wait
fi

# Check and stop API container
if az container show --resource-group $RESOURCE_GROUP --name $API_CONTAINER &>/dev/null; then
    echo "Stopping API container..."
    az container stop --resource-group $RESOURCE_GROUP --name $API_CONTAINER --no-wait
fi

# Check and stop Worker container
if az container show --resource-group $RESOURCE_GROUP --name $WORKER_CONTAINER &>/dev/null; then
    echo "Stopping Worker container..."
    az container stop --resource-group $RESOURCE_GROUP --name $WORKER_CONTAINER --no-wait
fi

# Wait a moment for containers to stop
echo "Waiting for containers to stop..."
sleep 10

# Get storage account name if available
STORAGE_ACCOUNT_NAME=$(az storage account list --resource-group $RESOURCE_GROUP --query "[0].name" -o tsv 2>/dev/null || echo "")

if [ ! -z "$STORAGE_ACCOUNT_NAME" ]; then
    # Get storage account key
    STORAGE_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT_NAME --query "[0].value" -o tsv 2>/dev/null || echo "")
    
    if [ ! -z "$STORAGE_KEY" ]; then
        echo "Cleaning up file shares in storage account $STORAGE_ACCOUNT_NAME..."
        
        # List all file shares
        SHARES=$(az storage share list --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY --query "[].name" -o tsv 2>/dev/null || echo "")
        
        # Delete each share
        for SHARE in $SHARES; do
            echo "Deleting file share $SHARE..."
            az storage share delete --name $SHARE --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY --output none || true
        done
    fi
fi

echo "Deleting resource group '$RESOURCE_GROUP'..."
az group delete --name $RESOURCE_GROUP --yes --no-wait

echo "Resource group deletion initiated. It may take a few minutes to complete."
echo "You can check the status with: az group show --name $RESOURCE_GROUP"

# Note about Key Vault soft delete
echo "Note: Key Vault resources are subject to soft-delete protection."
echo "If you want to recreate the Key Vault with the same name immediately,"
echo "you may need to purge the deleted Key Vault using:"
echo "az keyvault list-deleted"
echo "az keyvault purge --name $KEY_VAULT_NAME"

# Check if there are other deleted key vaults
OTHER_DELETED_KVS=$(az keyvault list-deleted --query "[?name!='$KEY_VAULT_NAME'].name" -o tsv)
if [ ! -z "$OTHER_DELETED_KVS" ]; then
    echo "Other deleted Key Vaults found that may need purging:"
    echo "$OTHER_DELETED_KVS"
fi 