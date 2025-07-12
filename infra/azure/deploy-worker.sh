#!/bin/bash
# Deploy only Sentio Worker to Azure Container Instances

set -e

# Variables
RESOURCE_GROUP="rg-sentio-free"
LOCATION="westeurope"
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
GITHUB_USERNAME="chernistry"  # Fixed username for image access
WORKER_CONTAINER_NAME="aci-sentio-worker"
WORKER_SHARE_NAME="worker-share"

# Load GitHub Container Registry credentials from .env
if [ -f "../../root/.env" ]; then
  source ../../root/.env
  GHCR_USERNAME=$GITHUB_USERNAME
  GHCR_PASSWORD=$GHCR_PAT_WRITE
else
  echo "Warning: ../../root/.env file not found, using manually set credentials"
  GHCR_USERNAME="chernistry"
  GHCR_PASSWORD="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"
fi

# Check if logged in to Azure CLI
echo "Checking Azure CLI login..."
az account show > /dev/null || { echo "Please login to Azure CLI using 'az login'"; exit 1; }

# Get Key Vault name from deployment
KEY_VAULT_NAME=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query properties.outputs.keyVaultName.value \
  -o tsv)

# Check and fetch secrets from Key Vault
echo "Fetching secrets from Key Vault..."
JINA_API_KEY=$(az keyvault secret show --vault-name $KEY_VAULT_NAME --name "jina-api-key" --query "value" -o tsv)
QDRANT_API_KEY=$(az keyvault secret show --vault-name $KEY_VAULT_NAME --name "qdrant-api-key" --query "value" -o tsv)
OPENROUTER_API_KEY=$(az keyvault secret show --vault-name $KEY_VAULT_NAME --name "openrouter-api-key" --query "value" -o tsv)
QUEUE_CONNECTION_STRING=$(az keyvault secret show --vault-name $KEY_VAULT_NAME --name "queue-connection-string" --query "value" -o tsv)
QDRANT_URL=$(az keyvault secret show --vault-name $KEY_VAULT_NAME --name "qdrant-url" --query "value" -o tsv)

echo "Using Qdrant URL: $QDRANT_URL"

# Get storage account name from deployment
STORAGE_ACCOUNT_NAME=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query properties.outputs.storageAccountName.value \
  -o tsv)

# Get storage key
STORAGE_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT_NAME --query "[0].value" -o tsv)

# Check Worker share
WORKER_SHARE_EXISTS=$(az storage share exists --name $WORKER_SHARE_NAME --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY --query exists -o tsv)
if [ "$WORKER_SHARE_EXISTS" != "true" ]; then
  echo "Creating Worker file share..."
  az storage share create --name $WORKER_SHARE_NAME --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY --output none
fi

# Deploy Worker container instance
echo "Deploying Sentio Worker container instance..."
az container create \
  --resource-group $RESOURCE_GROUP \
  --name $WORKER_CONTAINER_NAME \
  --image "ghcr.io/$GITHUB_USERNAME/sentio-worker:latest" \
  --registry-login-server "ghcr.io" \
  --registry-username "$GHCR_USERNAME" \
  --registry-password "$GHCR_PASSWORD" \
  --cpu 1 \
  --memory 2 \
  --os-type Linux \
  --restart-policy Always \
  --environment-variables \
    USE_AZURE=true \
    QDRANT_URL=$QDRANT_URL \
    COLLECTION_NAME=Sentio_docs_v2 \
    AZURE_QUEUE_NAME=submissions \
    LOG_LEVEL=INFO \
    QUEUE_POLL_INTERVAL_SECONDS=10 \
    MESSAGE_VISIBILITY_TIMEOUT=300 \
    OPENROUTER_URL=https://openrouter.ai/api/v1 \
    OPENROUTER_MODEL=deepseek/deepseek-r1-0528-qwen3-8b:free \
  --secure-environment-variables \
    JINA_API_KEY="$JINA_API_KEY" \
    QDRANT_API_KEY="$QDRANT_API_KEY" \
    OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
    AZURE_QUEUE_CONNECTION_STRING="$QUEUE_CONNECTION_STRING" \
  --azure-file-volume-account-name $STORAGE_ACCOUNT_NAME \
  --azure-file-volume-account-key $STORAGE_KEY \
  --azure-file-volume-share-name $WORKER_SHARE_NAME \
  --azure-file-volume-mount-path "/app/data"

# Wait for Worker container to be provisioned
echo "Waiting for Worker container to be provisioned..."
# az container wait --resource-group $RESOURCE_GROUP --name $WORKER_CONTAINER_NAME --timeout 300
echo "Waiting 30 seconds for Worker container to start..."
sleep 30

WORKER_STATE=$(az container show \
  --resource-group $RESOURCE_GROUP \
  --name $WORKER_CONTAINER_NAME \
  --query "properties.provisioningState" \
  --output tsv)
echo "Worker container state: $WORKER_STATE"

echo "✅ Worker deployment completed!"
echo "You can check the status with:"
echo "az container list --resource-group $RESOURCE_GROUP -o table"
echo "To view container logs:"
echo "az container logs --resource-group $RESOURCE_GROUP --name $WORKER_CONTAINER_NAME" 