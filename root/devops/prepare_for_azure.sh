#!/bin/bash
# Prepare Docker images for Azure deployment
# This script builds and pushes images to GitHub Container Registry

set -e

# Load environment variables
source .env

echo "===== Building and pushing Docker images for Azure deployment ====="

# Check GitHub credentials
if [ -z "${GHCR_PAT_WRITE}" ]; then
  echo "Error: GHCR_PAT_WRITE environment variable is not set in .env file"
  echo "Please set it to a valid GitHub Personal Access Token with write:packages scope"
  exit 1
fi

# Login to GitHub Container Registry
echo "Logging in to GitHub Container Registry..."
echo "${GHCR_PAT_WRITE}" | docker login ghcr.io -u "chernistry" --password-stdin

# Enable Docker BuildKit for better performance
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
export COMPOSE_BAKE=true

# Build base image first to ensure it's available for other services
echo "Building base image..."
docker compose build --build-arg BUILDKIT_INLINE_CACHE=1 base

# Build images in parallel using bake
echo "Building Docker images in parallel using Bake..."
docker compose bake

# Tag images
echo "Tagging images..."
docker tag sentio-api:latest ghcr.io/chernistry/sentio-api:latest
docker tag sentio-worker:latest ghcr.io/chernistry/sentio-worker:latest
docker tag sentio-ui:latest ghcr.io/chernistry/sentio-ui:latest

# Push images
echo "Pushing images to GitHub Container Registry..."
docker push ghcr.io/chernistry/sentio-api:latest
docker push ghcr.io/chernistry/sentio-worker:latest
docker push ghcr.io/chernistry/sentio-ui:latest

echo "===== Docker images successfully built and pushed ====="
echo "Next step: Deploy infrastructure with './infra/azure/scripts/deploy-infra.sh'"

echo "Note: For multi-architecture builds, use './infra/azure/scripts/build-multi-arch.sh' instead" 