#!/bin/bash
# Deploy Qdrant to Azure Container Instance

set -e

# Variables
RESOURCE_GROUP="rg-sentio-free"
LOCATION="westeurope"
CONTAINER_NAME="aci-qdrant-free"
DNS_NAME_LABEL="qdrant-sentio-free"
VOLUME_NAME="qdrantvolume"
SHARE_NAME="qdrantshare"

# Check if logged in to Azure CLI
echo "Checking Azure CLI login..."
az account show > /dev/null || { echo "Please login to Azure CLI using 'az login'"; exit 1; }

# Get storage account name from deployment
STORAGE_ACCOUNT_NAME=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query properties.outputs.storageAccountName.value \
  -o tsv)

# Get storage account key
STORAGE_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT_NAME --query "[0].value" -o tsv)

# Check if file share exists
echo "Checking if file share exists..."
SHARE_EXISTS=$(az storage share exists --name $SHARE_NAME --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY --query exists -o tsv)

if [ "$SHARE_EXISTS" != "true" ]; then
  echo "Creating file share $SHARE_NAME..."
  az storage share create --name $SHARE_NAME --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY --output none
fi

echo "Deploying Qdrant container instance..."
az container create \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --image "qdrant/qdrant:latest" \
  --ports 6333 6334 \
  --dns-name-label $DNS_NAME_LABEL \
  --cpu 1 \
  --memory 2 \
  --os-type Linux \
  --restart-policy OnFailure \
  --azure-file-volume-account-name $STORAGE_ACCOUNT_NAME \
  --azure-file-volume-account-key $STORAGE_KEY \
  --azure-file-volume-share-name $SHARE_NAME \
  --azure-file-volume-mount-path "/qdrant/storage" \
  --environment-variables \
    QDRANT_ALLOW_RECOVERY_MODE=true \
    QDRANT_SERVICE_PORT=6333 \
    QDRANT_TELEMETRY_DISABLED=true

# Wait for container to be provisioned
echo "Waiting for Qdrant container to be provisioned..."
az container show \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --query "properties.provisioningState" \
  --output tsv

# Get the Qdrant URL
QDRANT_FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.fqdn -o tsv)
QDRANT_URL="http://$QDRANT_FQDN:6333"

# Save Qdrant URL to .env.azure
echo "Saving Qdrant URL to .env.azure..."
echo "QDRANT_URL=$QDRANT_URL" > .env.azure

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to be ready..."
for i in {1..30}; do
  if curl -s --max-time 2 "$QDRANT_URL/collections" | grep -q 'collections'; then
    echo "Qdrant is ready."
    break
  fi
  echo "Waiting for Qdrant to start... ($i/30)"
  sleep 5
  if [ $i -eq 30 ]; then
    echo "Qdrant did not become ready in time. Check container logs."
    echo "az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
  fi
done

echo "✅ Qdrant deployment completed successfully!"
echo "Qdrant URL: $QDRANT_URL"
echo "Qdrant Dashboard: $QDRANT_URL/dashboard"
echo "Next steps:"
echo "1. Set up secrets in Key Vault with './setup-secrets.sh'"
echo "2. Deploy API and Worker containers with './deploy-container-apps.sh'" 