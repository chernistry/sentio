#!/bin/bash
# Setup secrets in Key Vault for Sentio RAG project

set -e

# Variables
RESOURCE_GROUP="rg-sentio-free"
LOCATION="westeurope"

# Load environment variables from .env.azure if exists
if [ -f ".env.azure" ]; then
  source .env.azure
  echo "Loaded environment variables from .env.azure"
else
  echo "Warning: .env.azure file not found. Qdrant URL may not be set correctly."
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

if [ -z "$KEY_VAULT_NAME" ]; then
  echo "Error: Could not retrieve Key Vault name from deployment. Make sure you've run './deploy-infra.sh' first."
  exit 1
fi

echo "Using Key Vault: $KEY_VAULT_NAME"

# Check if user has access to Key Vault
echo "Checking access to Key Vault..."
if ! az keyvault show --name $KEY_VAULT_NAME --resource-group $RESOURCE_GROUP &>/dev/null; then
  echo "Error: Could not access Key Vault $KEY_VAULT_NAME. Make sure it exists and you have access."
  exit 1
fi

# Get current user objectId
USER_ID=$(az ad signed-in-user show --query id -o tsv)

# Check if user has set permissions
echo "Checking if current user has set permissions on Key Vault..."
HAS_PERMISSIONS=$(az keyvault show --name $KEY_VAULT_NAME --query "properties.accessPolicies[?objectId=='$USER_ID'].permissions.secrets" -o tsv)

if [ -z "$HAS_PERMISSIONS" ] || [[ ! "$HAS_PERMISSIONS" =~ "set" ]]; then
  echo "Setting access policy for current user..."
  az keyvault set-policy --name $KEY_VAULT_NAME --object-id $USER_ID --secret-permissions get set list delete backup restore recover purge --output none
fi

# Get queue connection string from deployment
echo "Getting queue connection string from deployment..."
QUEUE_CONNECTION_STRING=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query properties.outputs.queueConnectionString.value \
  -o tsv)

if [ -z "$QUEUE_CONNECTION_STRING" ]; then
  echo "Warning: Could not retrieve queue connection string from deployment."
fi

echo "Setting up secrets in Key Vault $KEY_VAULT_NAME..."

# Set up secrets with error handling
set_secret() {
  local name=$1
  local value=$2
  
  echo "Setting secret: $name"
  if az keyvault secret set --vault-name $KEY_VAULT_NAME --name "$name" --value "$value" --output none; then
    echo "✓ Secret '$name' set successfully"
  else
    echo "✗ Failed to set secret '$name'"
  fi
}

# Set up dummy API keys (replace with real values in production)
set_secret "jina-api-key" "dummy-jina-api-key"
set_secret "qdrant-api-key" ""
set_secret "openrouter-api-key" "dummy-openrouter-api-key"
set_secret "queue-connection-string" "$QUEUE_CONNECTION_STRING"

# Set Qdrant URL if available
if [ -n "$QDRANT_URL" ]; then
  set_secret "qdrant-url" "$QDRANT_URL"
  echo "Set Qdrant URL: $QDRANT_URL"
else
  echo "Warning: QDRANT_URL not found. Please deploy Qdrant with './deploy-qdrant.sh' first."
  # Set a default localhost URL for now
  set_secret "qdrant-url" "http://localhost:6333"
  echo "Set default Qdrant URL: http://localhost:6333"
fi

echo "✅ Secrets setup completed successfully!"
echo "Next step: Deploy container apps with './deploy-container-apps.sh'" 