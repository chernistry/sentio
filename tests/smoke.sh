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
