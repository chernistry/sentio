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

# Function to check and register resource provider
check_resource_provider() {
  local provider=$1
  local state=$(az provider show --namespace $provider --query "registrationState" -o tsv 2>/dev/null || echo "NotRegistered")
  
  if [[ "$state" != "Registered" ]]; then
    echo "Resource provider $provider is not registered. Registering now..."
    az provider register --namespace $provider --wait
    echo "✅ Resource provider $provider registered"
  else
    echo "✅ Resource provider $provider already registered"
  fi
}

# Register necessary resource providers
echo "Checking and registering required resource providers..."
check_resource_provider "Microsoft.KeyVault"
check_resource_provider "Microsoft.Storage"
check_resource_provider "Microsoft.App"
check_resource_provider "Microsoft.OperationalInsights"

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
  --template-file $(dirname "$0")/../bicep/main.bicep \
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

# Get deployment outputs and save to .env.azure
echo "Saving deployment outputs to .env.azure..."
KEY_VAULT_NAME=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query properties.outputs.keyVaultName.value \
  -o tsv)

STORAGE_ACCOUNT_NAME=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query properties.outputs.storageAccountName.value \
  -o tsv)

QUEUE_CONNECTION_STRING=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query properties.outputs.queueConnectionString.value \
  -o tsv)

CONTAINER_APP_ENV_ID=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query properties.outputs.containerAppEnvId.value \
  -o tsv)

CONTAINER_APP_ENV_NAME=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query properties.outputs.containerAppEnvName.value \
  -o tsv)

# Create or update .env.azure file
cat > ../.env.azure << EOF
# Azure Resource IDs
CONTAINER_APP_ENV_ID="${CONTAINER_APP_ENV_ID}"
KEY_VAULT_ID="/subscriptions/$(az account show --query id -o tsv)/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.KeyVault/vaults/${KEY_VAULT_NAME}"
STORAGE_ACCOUNT_ID="/subscriptions/$(az account show --query id -o tsv)/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.Storage/storageAccounts/${STORAGE_ACCOUNT_NAME}"

# Connection Strings
QUEUE_CONNECTION_STRING="${QUEUE_CONNECTION_STRING}"

# Service URLs
# QDRANT_URL will be set in setup-secrets.sh

# Resource Names
STORAGE_ACCOUNT_NAME="${STORAGE_ACCOUNT_NAME}"
KEY_VAULT_NAME="${KEY_VAULT_NAME}"
CONTAINER_APP_ENV_NAME="${CONTAINER_APP_ENV_NAME}"
EOF

echo "✅ Infrastructure deployment completed successfully!"
echo "Next steps:"
echo "1. Set up secrets in Key Vault with './setup-secrets.sh'"
echo "2. Deploy API, Worker, and UI containers with './deploy-apps.sh'" 