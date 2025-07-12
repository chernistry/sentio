#!/bin/bash
# Deploy infrastructure for Sentio RAG project

set -e

# Variables
RESOURCE_GROUP="rg-sentio-free"
LOCATION="westeurope"
KEY_VAULT_NAME="kv-sentio-free"

# Check if logged in to Azure CLI
echo "Checking Azure CLI login..."
az account show > /dev/null || { echo "Please login to Azure CLI using 'az login'"; exit 1; }

# Check for deleted Key Vault with the same name
echo "Checking for deleted Key Vault with the same name..."
DELETED_KV=$(az keyvault list-deleted --query "[?name=='$KEY_VAULT_NAME'].name" -o tsv)
if [ ! -z "$DELETED_KV" ]; then
  echo "Found deleted Key Vault with name '$KEY_VAULT_NAME'. Purging it..."
  az keyvault purge --name $KEY_VAULT_NAME --no-wait
  echo "Waiting for purge to complete..."
  sleep 30
fi

# Create resource group if it doesn't exist
echo "Creating resource group if it doesn't exist..."
az group create --name $RESOURCE_GROUP --location $LOCATION --output none

# Deploy main Bicep template
echo "Deploying main Bicep template..."
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --template-file main.bicep \
  --parameters environmentName=free \
  --output none

# Get current user objectId
USER_ID=$(az ad signed-in-user show --query id -o tsv)

# Set access policy for current user
echo "Setting access policy for current user..."
az keyvault set-policy \
  --name $KEY_VAULT_NAME \
  --resource-group $RESOURCE_GROUP \
  --object-id $USER_ID \
  --secret-permissions get set list delete backup restore recover purge \
  --output none

echo "✅ Infrastructure deployment completed successfully!"
echo "Next steps:"
echo "1. Deploy Qdrant with './deploy-qdrant.sh'"
echo "2. Set up secrets in Key Vault with './setup-secrets.sh'"
echo "3. Deploy API and Worker containers with './deploy-container-apps.sh'" 