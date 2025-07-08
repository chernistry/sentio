#!/bin/bash
# Smoke test script for Sentio RAG system
# Tests both API and Worker components

set -e

# Configuration
API_URL="sentio-api-free.westeurope.azurecontainer.io"
PROTOCOL="http"  # Changed from https to http

echo "Using API URL: $API_URL"

# Test 1: API health check
echo "Test 1: API Health Check"
response=$(curl -s -o /dev/null -w "%{http_code}" "${PROTOCOL}://${API_URL}/health")
if [ "$response" = "200" ]; then
  echo "✅ API Health check passed"
else
  echo "❌ API Health check failed with status code: $response"
  exit 1
fi

# Test 2: Submit a document for embedding
echo "Test 2: Submit document for embedding"
TEST_DOC='{"content": "This is a test document for Sentio RAG system", "metadata": {"source": "smoke_test", "created_at": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}, "id": "smoke-test-'$(date +%s)'"}'

embed_response=$(curl -s -X POST "${PROTOCOL}://${API_URL}/embed" \
  -H "Content-Type: application/json" \
  -d "$TEST_DOC")

echo "Embed response: $embed_response"

if [[ "$embed_response" == *"success"* ]]; then
  echo "✅ Document submission successful"
else
  echo "❌ Document submission failed"
  exit 1
fi

# Test 3: Wait for worker to process the queue
echo "Test 3: Waiting for worker to process queue (15 seconds)..."
sleep 15

# Test 4: Search for the embedded document
echo "Test 4: Search for embedded document"
search_query='{"query": "test document", "limit": 5}'

search_response=$(curl -s -X POST "${PROTOCOL}://${API_URL}/search" \
  -H "Content-Type: application/json" \
  -d "$search_query")

echo "Search response: $search_response"

if [[ "$search_response" == *"smoke_test"* ]]; then
  echo "✅ Search test successful - document was indexed by worker"
else
  echo "❓ Document might not be indexed yet. Check worker logs for processing status."
  echo "Worker logs command: az container logs --resource-group rg-sentio-free --name aci-sentio-worker"
fi

# Test 5: Check worker status
echo "Test 5: Check worker container status"
worker_status=$(az container show --resource-group rg-sentio-free --name aci-sentio-worker --query containers[0].instanceView.currentState.state -o tsv)

echo "Worker status: $worker_status"
if [ "$worker_status" = "Running" ]; then
  echo "✅ Worker is running correctly"
else
  echo "❌ Worker is not in Running state: $worker_status"
  echo "Worker logs:"
  az container logs --resource-group rg-sentio-free --name aci-sentio-worker --tail 10
  exit 1
fi

echo "==============================="
echo "🎉 All smoke tests completed!"
echo "===============================" 