#!/bin/bash
# Deploy Sentio RAG applications to Azure Container Apps

set -e

# Variables
RESOURCE_GROUP="rg-sentio-free"
LOCATION="westeurope"
GITHUB_USERNAME=${GITHUB_USERNAME:-"$GITHUB_REPOSITORY_OWNER"}  # Use env var or repository owner

# Load GitHub Container Registry credentials from .env
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
ENV_FILE="$ROOT_DIR/.env"
AZURE_ENV_FILE="$ROOT_DIR/infra/azure/.env.azure"

if [ -f "$ENV_FILE" ]; then
  source "$ENV_FILE"
  # Use GITHUB_USERNAME from env or fall back to repository owner
  GHCR_USERNAME=${GITHUB_USERNAME:-$GITHUB_REPOSITORY_OWNER}
  GHCR_PASSWORD=${GHCR_PAT_WRITE:-$GITHUB_TOKEN}
  echo "Loaded GitHub credentials from $ENV_FILE"
else
  echo "Warning: .env file not found at $ENV_FILE, using environment variables"
  GHCR_USERNAME=${GITHUB_USERNAME:-$GITHUB_REPOSITORY_OWNER}
  GHCR_PASSWORD=${GHCR_PAT_WRITE:-$GITHUB_TOKEN}
fi

# Check if GitHub token is available
if [ -z "$GHCR_PASSWORD" ]; then
  echo "Error: GitHub Container Registry token (GHCR_PAT_WRITE or GITHUB_TOKEN) is not set"
  echo "Please set it in .env file or as an environment variable"
  exit 1
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
check_resource_provider "Microsoft.App"

# Load environment variables from .env.azure if exists
if [ -f "$AZURE_ENV_FILE" ]; then
  source "$AZURE_ENV_FILE"
  echo "Loaded environment variables from $AZURE_ENV_FILE"
else
  echo "Warning: .env.azure file not found. Creating from main deployment..."
fi

# Check if resource group exists
echo "Checking if resource group exists..."
if ! az group show --name $RESOURCE_GROUP &>/dev/null; then
  echo "Error: Resource group $RESOURCE_GROUP does not exist. Please run deploy-infra.sh first."
  exit 1
fi

# Get Container App Environment ID from deployment
if [ -z "$CONTAINER_APP_ENV_ID" ]; then
  echo "Getting Container App Environment ID from deployment..."
  CONTAINER_APP_ENV_ID=$(az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name main \
    --query properties.outputs.containerAppEnvId.value \
    -o tsv)
  
  if [ -z "$CONTAINER_APP_ENV_ID" ]; then
    echo "Error: Could not retrieve Container App Environment ID. Make sure you've run deploy-infra.sh first."
    exit 1
  fi
fi

# Get Key Vault name from deployment
if [ -z "$KEY_VAULT_NAME" ]; then
  echo "Getting Key Vault name from deployment..."
  KEY_VAULT_NAME=$(az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name main \
    --query properties.outputs.keyVaultName.value \
    -o tsv)
  
  if [ -z "$KEY_VAULT_NAME" ]; then
    echo "Error: Could not retrieve Key Vault name. Make sure you've run deploy-infra.sh first."
    exit 1
  fi
fi

# Get Queue connection string from deployment
if [ -z "$QUEUE_CONNECTION_STRING" ]; then
  echo "Getting Queue connection string from deployment..."
  QUEUE_CONNECTION_STRING=$(az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name main \
    --query properties.outputs.queueConnectionString.value \
    -o tsv)
  
  if [ -z "$QUEUE_CONNECTION_STRING" ]; then
    echo "Warning: Could not retrieve queue connection string from deployment."
  fi
fi

# Check and fetch secrets from Key Vault
echo "Fetching secrets from Key Vault..."
if ! az keyvault show --name $KEY_VAULT_NAME --resource-group $RESOURCE_GROUP &>/dev/null; then
  echo "Error: Key Vault $KEY_VAULT_NAME not found. Make sure you've run deploy-infra.sh first."
  exit 1
fi

# Function to get secret safely
get_secret() {
  local name=$1
  local default=$2
  
  local value=$(az keyvault secret show --vault-name $KEY_VAULT_NAME --name "$name" --query "value" -o tsv 2>/dev/null || echo "")
  
  if [ -z "$value" ]; then
    echo "Warning: Secret $name not found in Key Vault. Using default value."
    echo "$default"
  else
    echo "$value"
  fi
}

JINA_API_KEY=$(get_secret "jina-api-key" "dummy-jina-api-key")
QDRANT_API_KEY=$(get_secret "qdrant-api-key" "")
OPENROUTER_API_KEY=$(get_secret "openrouter-api-key" "dummy-openrouter-api-key")
QDRANT_URL=$(get_secret "qdrant-url" "http://localhost:6333")
EMBEDDING_MODEL_API_KEY=$(get_secret "embedding-model-api-key" "$JINA_API_KEY")
CHAT_LLM_API_KEY=$(get_secret "chat-llm-api-key" "$OPENROUTER_API_KEY")

# Check for all required secrets
if [ -z "$EMBEDDING_MODEL_API_KEY" ]; then
  echo "Warning: embedding-model-api-key is empty, setting it to jina-api-key"
  EMBEDDING_MODEL_API_KEY=$JINA_API_KEY
  # Add to Key Vault if missing
  az keyvault secret set --vault-name $KEY_VAULT_NAME --name "embedding-model-api-key" --value "$EMBEDDING_MODEL_API_KEY" --output none
fi

if [ -z "$CHAT_LLM_API_KEY" ]; then
  echo "Warning: chat-llm-api-key is empty, setting it to openrouter-api-key"
  CHAT_LLM_API_KEY=$OPENROUTER_API_KEY
  # Add to Key Vault if missing
  az keyvault secret set --vault-name $KEY_VAULT_NAME --name "chat-llm-api-key" --value "$CHAT_LLM_API_KEY" --output none
fi

echo "Using Qdrant URL: $QDRANT_URL"

# Verify GitHub Container Registry credentials
echo "Verifying GitHub Container Registry credentials..."
echo $GHCR_PASSWORD | docker login ghcr.io -u $GHCR_USERNAME --password-stdin || { echo "Error: Failed to login to GitHub Container Registry"; exit 1; }
echo "✅ GitHub Container Registry login successful"

# Deploy API Container App
echo "Deploying API Container App..."
az containerapp create \
  --resource-group $RESOURCE_GROUP \
  --name "ca-sentio-api" \
  --environment $CONTAINER_APP_ENV_ID \
  --image "ghcr.io/$GHCR_USERNAME/sentio-api:latest" \
  --registry-server "ghcr.io" \
  --registry-username "$GHCR_USERNAME" \
  --registry-password "$GHCR_PASSWORD" \
  --target-port 8000 \
  --ingress "external" \
  --min-replicas 1 \
  --max-replicas 1 \
  --cpu 0.5 \
  --memory 1.0Gi \
  --secrets "jina-api-key=$JINA_API_KEY" "qdrant-api-key=$QDRANT_API_KEY" "openrouter-api-key=$OPENROUTER_API_KEY" "queue-connection-string=$QUEUE_CONNECTION_STRING" "embedding-model-api-key=$EMBEDDING_MODEL_API_KEY" "chat-llm-api-key=$CHAT_LLM_API_KEY" \
  --env-vars "USE_AZURE=true" "QDRANT_URL=$QDRANT_URL" "COLLECTION_NAME=Sentio_docs" "AZURE_QUEUE_NAME=submissions" "LOG_LEVEL=info" "ENABLE_METRICS=true" "ENABLE_CORS=true" "OPENROUTER_URL=https://openrouter.ai/api/v1" "OPENROUTER_MODEL=deepseek/deepseek-r1-0528-qwen3-8b:free" "JINA_API_KEY=secretref:jina-api-key" "QDRANT_API_KEY=secretref:qdrant-api-key" "OPENROUTER_API_KEY=secretref:openrouter-api-key" "AZURE_QUEUE_CONNECTION_STRING=secretref:queue-connection-string" "EMBEDDING_MODEL_API_KEY=secretref:embedding-model-api-key" "EMBEDDING_MODEL=jina-embeddings-v3" "EMBEDDING_PROVIDER=jina" "QDRANT_API_KEY_HEADER=api-key" "CHAT_LLM_API_KEY=secretref:chat-llm-api-key" "CHAT_LLM_BASE_URL=https://openrouter.ai/api/v1" "CHAT_LLM_MODEL=deepseek/deepseek-chat-v3-0324:free"

# Deploy Worker Container App
echo "Deploying Worker Container App..."
az containerapp create \
  --resource-group $RESOURCE_GROUP \
  --name "ca-sentio-worker" \
  --environment $CONTAINER_APP_ENV_ID \
  --image "ghcr.io/$GHCR_USERNAME/sentio-worker:latest" \
  --registry-server "ghcr.io" \
  --registry-username "$GHCR_USERNAME" \
  --registry-password "$GHCR_PASSWORD" \
  --target-port 8000 \
  --ingress "internal" \
  --min-replicas 0 \
  --max-replicas 1 \
  --cpu 0.5 \
  --memory 1.0Gi \
  --secrets "jina-api-key=$JINA_API_KEY" "qdrant-api-key=$QDRANT_API_KEY" "openrouter-api-key=$OPENROUTER_API_KEY" "queue-connection-string=$QUEUE_CONNECTION_STRING" "embedding-model-api-key=$EMBEDDING_MODEL_API_KEY" "chat-llm-api-key=$CHAT_LLM_API_KEY" \
  --env-vars "USE_AZURE=true" "QDRANT_URL=$QDRANT_URL" "COLLECTION_NAME=Sentio_docs" "AZURE_QUEUE_NAME=submissions" "LOG_LEVEL=info" "QUEUE_POLL_INTERVAL_SECONDS=10" "MESSAGE_VISIBILITY_TIMEOUT=300" "OPENROUTER_URL=https://openrouter.ai/api/v1" "OPENROUTER_MODEL=deepseek/deepseek-r1-0528-qwen3-8b:free" "JINA_API_KEY=secretref:jina-api-key" "QDRANT_API_KEY=secretref:qdrant-api-key" "OPENROUTER_API_KEY=secretref:openrouter-api-key" "AZURE_QUEUE_CONNECTION_STRING=secretref:queue-connection-string" "EMBEDDING_MODEL_API_KEY=secretref:embedding-model-api-key" "EMBEDDING_MODEL=jina-embeddings-v3" "EMBEDDING_PROVIDER=jina" "QDRANT_API_KEY_HEADER=api-key" "LOG_FILE=/tmp/worker.log" "CHAT_LLM_API_KEY=secretref:chat-llm-api-key" \
  --command "/app/startup.sh"

# Get API FQDN for UI configuration
API_FQDN=$(az containerapp show --resource-group $RESOURCE_GROUP --name "ca-sentio-api" --query properties.configuration.ingress.fqdn -o tsv)
if [ -z "$API_FQDN" ]; then
  echo "Warning: Could not get API FQDN. Using default Container App domain pattern."
  API_FQDN="ca-sentio-api.${LOCATION}.azurecontainerapps.io"
fi

# Deploy UI Container App
echo "Deploying UI Container App..."
az containerapp create \
  --resource-group $RESOURCE_GROUP \
  --name "ca-sentio-ui" \
  --environment $CONTAINER_APP_ENV_ID \
  --image "ghcr.io/$GHCR_USERNAME/sentio-ui:latest" \
  --registry-server "ghcr.io" \
  --registry-username "$GHCR_USERNAME" \
  --registry-password "$GHCR_PASSWORD" \
  --target-port 8501 \
  --ingress "external" \
  --min-replicas 0 \
  --max-replicas 1 \
  --cpu 0.5 \
  --memory 1.0Gi \
  --secrets "jina-api-key=$JINA_API_KEY" "qdrant-api-key=$QDRANT_API_KEY" "openrouter-api-key=$OPENROUTER_API_KEY" "queue-connection-string=$QUEUE_CONNECTION_STRING" \
  --env-vars "SENTIO_BACKEND_URL=https://$API_FQDN" "LOG_LEVEL=info"

# Get endpoints
API_FQDN=$(az containerapp show --resource-group $RESOURCE_GROUP --name "ca-sentio-api" --query properties.configuration.ingress.fqdn -o tsv)
UI_FQDN=$(az containerapp show --resource-group $RESOURCE_GROUP --name "ca-sentio-ui" --query properties.configuration.ingress.fqdn -o tsv)

echo "✅ Sentio applications deployment completed!"
echo "API URL: https://$API_FQDN"
echo "API Docs: https://$API_FQDN/docs"
echo "UI URL: https://$UI_FQDN"

# Deactivation of old revisions with issues
echo "Checking for old revisions to deactivate..."
# Get list of all revisions for API
API_REVISIONS=$(az containerapp revision list --name "ca-sentio-api" --resource-group $RESOURCE_GROUP --query "[].name" -o tsv)
WORKER_REVISIONS=$(az containerapp revision list --name "ca-sentio-worker" --resource-group $RESOURCE_GROUP --query "[].name" -o tsv)
UI_REVISIONS=$(az containerapp revision list --name "ca-sentio-ui" --resource-group $RESOURCE_GROUP --query "[].name" -o tsv)

# Get the latest revisions
LATEST_API_REVISION=$(az containerapp revision list --name "ca-sentio-api" --resource-group $RESOURCE_GROUP --query "sort_by([],&createdTime)[-1].name" -o tsv)
LATEST_WORKER_REVISION=$(az containerapp revision list --name "ca-sentio-worker" --resource-group $RESOURCE_GROUP --query "sort_by([],&createdTime)[-1].name" -o tsv)
LATEST_UI_REVISION=$(az containerapp revision list --name "ca-sentio-ui" --resource-group $RESOURCE_GROUP --query "sort_by([],&createdTime)[-1].name" -o tsv)

# Deactivate all old revisions
for rev in $API_REVISIONS; do
  if [ "$rev" != "$LATEST_API_REVISION" ]; then
    echo "Deactivating old API revision: $rev"
    az containerapp revision deactivate --name "ca-sentio-api" --resource-group $RESOURCE_GROUP --revision $rev || true
  fi
done

for rev in $WORKER_REVISIONS; do
  if [ "$rev" != "$LATEST_WORKER_REVISION" ]; then
    echo "Deactivating old Worker revision: $rev"
    az containerapp revision deactivate --name "ca-sentio-worker" --resource-group $RESOURCE_GROUP --revision $rev || true
  fi
done

for rev in $UI_REVISIONS; do
  if [ "$rev" != "$LATEST_UI_REVISION" ]; then
    echo "Deactivating old UI revision: $rev"
    az containerapp revision deactivate --name "ca-sentio-ui" --resource-group $RESOURCE_GROUP --revision $rev || true
  fi
done

# Create a basic smoke test script
SMOKE_TEST_FILE="$ROOT_DIR/tests/smoke.sh"
cat > "$SMOKE_TEST_FILE" << 'EOF'
#!/bin/bash
# Basic smoke test for Sentio RAG deployment

set -e

# Get API URL from command line argument or use default
API_URL=${1:-""}

if [ -z "$API_URL" ]; then
  echo "Error: API URL not provided."
  echo "Usage: ./smoke.sh <api_url>"
  exit 1
fi

echo "Running smoke tests against API at $API_URL..."

# Test API health
echo "Testing API health..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
if [ "$HEALTH_RESPONSE" -eq 200 ]; then
  echo "✅ API health endpoint: OK"
else
  echo "❌ API health endpoint failed with status: $HEALTH_RESPONSE"
  exit 1
fi

# Test API docs
echo "Testing API docs..."
DOCS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs")
if [ "$DOCS_RESPONSE" -eq 200 ]; then
  echo "✅ API docs endpoint: OK"
else
  echo "❌ API docs endpoint failed with status: $DOCS_RESPONSE"
  exit 1
fi

# Test a simple query (adjust as needed)
echo "Testing search query..."
QUERY_RESPONSE=$(curl -s -X POST "$API_URL/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 3}' \
  -o /dev/null -w "%{http_code}")

if [ "$QUERY_RESPONSE" -eq 200 ]; then
  echo "✅ Search query: OK"
else
  echo "❌ Search query failed with status: $QUERY_RESPONSE"
  exit 1
fi

echo "✅ All smoke tests passed!"
EOF

chmod +x "$SMOKE_TEST_FILE"

echo "Smoke test script created at '$SMOKE_TEST_FILE'"
echo "To run smoke tests: ./tests/smoke.sh https://$API_FQDN" 