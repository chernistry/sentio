#!/bin/bash
# Setup secrets in Key Vault for Sentio RAG project

set -e

# Variables
RESOURCE_GROUP="rg-sentio-free"
LOCATION="westeurope"

# Load environment variables from .env.azure if exists
if [ -f "../.env.azure" ]; then
  source ../.env.azure
  echo "Loaded environment variables from .env.azure"
else
  echo "Warning: .env.azure file not found. Creating one from deployment outputs."
fi

# Load main .env file for API keys if exists
if [ -f "../../.env" ]; then
  source ../../.env
  echo "Loaded environment variables from .env"
else
  echo "Warning: .env file not found. Using default values for API keys."
fi

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

# Check if resource group exists
echo "Checking if resource group exists..."
if ! az group show --name $RESOURCE_GROUP &>/dev/null; then
  echo "Error: Resource group $RESOURCE_GROUP does not exist. Please run deploy-infra.sh first."
  exit 1
fi

# Get Key Vault name from deployment if not set in .env.azure
if [ -z "$KEY_VAULT_NAME" ]; then
  echo "Getting Key Vault name from deployment..."
  KEY_VAULT_NAME=$(az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name main \
    --query properties.outputs.keyVaultName.value \
    -o tsv)

  if [ -z "$KEY_VAULT_NAME" ]; then
    echo "Error: Could not retrieve Key Vault name from deployment. Make sure you've run './deploy-infra.sh' first."
    exit 1
  fi
  
  # Save to .env.azure for future use
  echo "KEY_VAULT_NAME=\"$KEY_VAULT_NAME\"" >> ../.env.azure
  echo "Saved KEY_VAULT_NAME to .env.azure"
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
  echo "✅ Access policy set for current user"
fi

# Get queue connection string from deployment if not set in .env.azure
if [ -z "$QUEUE_CONNECTION_STRING" ]; then
  echo "Getting queue connection string from deployment..."
  QUEUE_CONNECTION_STRING=$(az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name main \
    --query properties.outputs.queueConnectionString.value \
    -o tsv)

  if [ -z "$QUEUE_CONNECTION_STRING" ]; then
    echo "Warning: Could not retrieve queue connection string from deployment."
  else
    # Save to .env.azure for future use
    echo "QUEUE_CONNECTION_STRING=\"$QUEUE_CONNECTION_STRING\"" >> ../.env.azure
    echo "Saved QUEUE_CONNECTION_STRING to .env.azure"
  fi
fi

echo "Setting up secrets in Key Vault $KEY_VAULT_NAME..."

# Function to get existing secret from Key Vault
get_existing_secret() {
  local name=$1
  local value=$(az keyvault secret show --vault-name $KEY_VAULT_NAME --name "$name" --query "value" -o tsv 2>/dev/null || echo "")
  echo "$value"
}

# Set up secrets with error handling
set_secret() {
  local name=$1
  local value=$2
  local default_value=$3
  local existing_value=$(get_existing_secret "$name")
  
  echo "Setting secret: $name"
  
  # Check if there is already a value in Key Vault
  if [ -n "$existing_value" ] && [ -z "$value" ]; then
    echo "Using existing value from Key Vault for $name"
    return 0
  fi
  
  if [ -z "$value" ]; then
    if [ -n "$default_value" ]; then
      value="$default_value"
      echo "Using provided default value for $name"
    else
      value="dummy-${name}-value"
      echo "Warning: Empty value for $name, using dummy value"
    fi
  fi
  
  if az keyvault secret set --vault-name $KEY_VAULT_NAME --name "$name" --value "$value" --output none; then
    echo "✓ Secret '$name' set successfully"
  else
    echo "✗ Failed to set secret '$name'"
  fi
}

# Check and normalize API keys
# Priority: value from .env > existing in Key Vault > default value
normalize_api_keys() {
  # Get existing values from Key Vault
  local existing_jina=$(get_existing_secret "jina-api-key")
  local existing_qdrant=$(get_existing_secret "qdrant-api-key")
  local existing_openrouter=$(get_existing_secret "openrouter-api-key")
  local existing_embedding=$(get_existing_secret "embedding-model-api-key")
  local existing_chat_llm=$(get_existing_secret "chat-llm-api-key")
  
  # Normalize values considering priorities
  JINA_API_KEY=${JINA_API_KEY:-$existing_jina}
  QDRANT_API_KEY=${QDRANT_API_KEY:-$existing_qdrant}
  OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-$existing_openrouter}
  
  # For embedding-model-api-key use: value from .env > existing > jina-api-key
  EMBEDDING_MODEL_API_KEY=${EMBEDDING_MODEL_API_KEY:-$existing_embedding}
  if [ -z "$EMBEDDING_MODEL_API_KEY" ]; then
    EMBEDDING_MODEL_API_KEY=$JINA_API_KEY
    echo "Using JINA_API_KEY as fallback for EMBEDDING_MODEL_API_KEY"
  fi
  
  # For chat-llm-api-key use: value from .env > existing > openrouter-api-key
  CHAT_LLM_API_KEY=${CHAT_LLM_API_KEY:-$existing_chat_llm}
  if [ -z "$CHAT_LLM_API_KEY" ]; then
    CHAT_LLM_API_KEY=$OPENROUTER_API_KEY
    echo "Using OPENROUTER_API_KEY as fallback for CHAT_LLM_API_KEY"
  fi
}

# Normalize API keys before setting
normalize_api_keys

# Set up API keys with better error handling
set_secret "jina-api-key" "$JINA_API_KEY" ""
set_secret "qdrant-api-key" "$QDRANT_API_KEY" ""
set_secret "openrouter-api-key" "$OPENROUTER_API_KEY" ""
set_secret "queue-connection-string" "$QUEUE_CONNECTION_STRING" ""
set_secret "embedding-model-api-key" "$EMBEDDING_MODEL_API_KEY" "$JINA_API_KEY"
set_secret "chat-llm-api-key" "$CHAT_LLM_API_KEY" "$OPENROUTER_API_KEY"

# Set Qdrant URL if available
if [ -n "$QDRANT_URL" ]; then
  # Remove port 6333 if it's a cloud URL
  if [[ "$QDRANT_URL" == *"cloud.qdrant.io"* ]]; then
    QDRANT_URL=$(echo "$QDRANT_URL" | sed 's/:6333//')
  fi
  set_secret "qdrant-url" "$QDRANT_URL" ""
  echo "Set Qdrant URL: $QDRANT_URL"
else
  # Check existing URL in Key Vault
  existing_qdrant_url=$(get_existing_secret "qdrant-url")
  if [ -n "$existing_qdrant_url" ]; then
    echo "Using existing Qdrant URL from Key Vault: $existing_qdrant_url"
  else
    # Use default URL for Qdrant
    QDRANT_URL_FROM_ENV="https://1b8ab421-90a1-47d6-bd52-ac8eab597146.eu-central-1-0.aws.cloud.qdrant.io"
    set_secret "qdrant-url" "$QDRANT_URL_FROM_ENV" ""
    echo "Set default Qdrant URL: $QDRANT_URL_FROM_ENV"
  fi
fi

# Check for all required secrets
echo "Verifying all required secrets are set..."
required_secrets=("jina-api-key" "qdrant-api-key" "openrouter-api-key" "embedding-model-api-key" "chat-llm-api-key" "qdrant-url")
missing_secrets=0

for secret in "${required_secrets[@]}"; do
  value=$(get_existing_secret "$secret")
  if [ -z "$value" ]; then
    echo "⚠️ Warning: Secret '$secret' is still empty or not set"
    missing_secrets=$((missing_secrets + 1))
  else
    echo "✓ Secret '$secret' is properly set"
  fi
done

if [ $missing_secrets -gt 0 ]; then
  echo "⚠️ Warning: $missing_secrets required secrets are missing or empty"
else
  echo "✅ All required secrets are properly set"
fi

echo "✅ Secrets setup completed successfully!"
echo "Next step: Deploy container apps with './deploy-apps.sh'" 