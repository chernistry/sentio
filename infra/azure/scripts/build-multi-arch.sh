#!/bin/bash
# Build multi-architecture Docker images for Sentio RAG project

set -e

# Variables
GITHUB_USERNAME=${GITHUB_USERNAME:-"chernistry"}  # Use env var or default to chernistry
CACHE_DIR=${DOCKER_CACHE_DIR:-"${HOME}/.docker/sentio-cache"}
REGISTRY=${REGISTRY:-"ghcr.io"}
PLATFORMS=${PLATFORMS:-"linux/amd64,linux/arm64"}

# Load GitHub Container Registry credentials from .env
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
ENV_FILE="$ROOT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
  source "$ENV_FILE"
  GITHUB_USERNAME=${GITHUB_USERNAME:-"chernistry"}
  GHCR_USERNAME=${GHCR_USERNAME:-$GITHUB_USERNAME}
  GHCR_PASSWORD=${GHCR_PAT_WRITE:-$GITHUB_TOKEN}
  echo "Loaded GitHub credentials from $ENV_FILE"
else
  echo "Warning: .env file not found at $ENV_FILE, using environment variables"
  GITHUB_USERNAME=${GITHUB_USERNAME:-"chernistry"}
  GHCR_USERNAME=${GHCR_USERNAME:-$GITHUB_USERNAME}
  GHCR_PASSWORD=${GHCR_PAT_WRITE:-$GITHUB_TOKEN}
fi

# Check if GitHub token is available
if [ -z "$GHCR_PASSWORD" ]; then
  echo "Error: GitHub Container Registry token (GHCR_PAT_WRITE or GITHUB_TOKEN) is not set"
  echo "Please set it in .env file or as an environment variable"
  exit 1
fi

# Verify required files exist before proceeding
echo "Verifying project structure..."
if [ ! -f "root/app.py" ]; then
  echo "Error: root/app.py not found. Make sure you're running this script from the project root directory."
  exit 1
fi

if [ ! -d "root/worker" ] || [ ! -f "root/worker/__main__.py" ]; then
  echo "Error: root/worker/__main__.py not found. Check project structure."
  exit 1
fi

if [ ! -f "root/streamlit_app.py" ]; then
  echo "Error: root/streamlit_app.py not found. Check project structure."
  exit 1
fi

echo "✅ Project structure verification passed"

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR"

# Login to GitHub Container Registry
echo "Logging in to GitHub Container Registry..."
if [ -z "$CI" ]; then
  # Only login if not in CI environment (GitHub Actions handles login)
  # Check if already logged in
  if ! docker info | grep -q "Username: $GHCR_USERNAME"; then
    # Use password-stdin for more reliable auth
    echo "$GHCR_PASSWORD" | docker login $REGISTRY -u $GHCR_USERNAME --password-stdin || { echo "Error: Failed to login to GitHub Container Registry"; exit 1; }
  fi
  echo "✅ GitHub Container Registry login successful"
else
  echo "Skipping login in CI environment"
fi

# Setup buildx for multi-architecture builds
echo "Setting up Docker Buildx..."
BUILDER_NAME="multiarch"
if ! docker buildx inspect "$BUILDER_NAME" &>/dev/null; then
  docker buildx create --name "$BUILDER_NAME" --driver docker-container --bootstrap
fi
docker buildx use "$BUILDER_NAME"
echo "✅ Docker Buildx setup complete"

# Version metadata
VERSION=$(grep -oP 'version\s*=\s*"\K[^"]+' pyproject.toml 2>/dev/null || echo "0.1.0")
DATE_TAG=$(date +'%Y%m%d')
SHORT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "local")

# Function to build and push multi-architecture image with caching
build_and_push() {
  local component=$1
  local dockerfile=$2
  local context=$3
  local push_flag=${4:-"--push"}  # Use --push by default, --load for local testing
  
  # Create tag list
  local tag_base="$REGISTRY/$GHCR_USERNAME/sentio-$component"
  local tags=(
    "$tag_base:latest"
    "$tag_base:$VERSION"
    "$tag_base:$SHORT_SHA"
    "$tag_base:$VERSION-$DATE_TAG"
  )
  
  # Prepare tag args for buildx
  local tag_args=""
  for tag in "${tags[@]}"; do
    tag_args="$tag_args --tag $tag"
  done
  
  echo "Building multi-architecture image for $component..."
  
  # Set cache locations
  local cache_to="type=local,dest=$CACHE_DIR/$component,mode=max"
  local cache_from="type=local,src=$CACHE_DIR/$component"
  
  # Check if in CI environment, use GitHub Actions cache if available
  if [ -n "$CI" ]; then
    # Use registry-based cache for CI builds
    cache_to="type=registry,ref=$REGISTRY/$GHCR_USERNAME/sentio-cache-$component,mode=max"
    cache_from="type=registry,ref=$REGISTRY/$GHCR_USERNAME/sentio-cache-$component"
    echo "Using registry-based cache for CI builds"
  elif [ -n "$USE_REGISTRY_CACHE" ]; then
    # Use registry-based cache if explicitly requested
    cache_to="type=registry,ref=$REGISTRY/$GHCR_USERNAME/sentio-cache-$component,mode=max"
    cache_from="type=registry,ref=$REGISTRY/$GHCR_USERNAME/sentio-cache-$component"
    echo "Using registry-based cache"
  elif [ ! -d "$CACHE_DIR/$component" ]; then
    echo "No local cache found at $CACHE_DIR/$component, will create it"
  else
    echo "Using local cache from $CACHE_DIR/$component"
  fi
  
  # Build with optimized settings
  docker buildx build \
    --platform "$PLATFORMS" \
    $tag_args \
    -f "$dockerfile" \
    --cache-from="$cache_from" \
    --cache-to="$cache_to" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --build-arg VERSION="$VERSION" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$SHORT_SHA" \
    --build-arg PYTHON_GPG=off \
    --build-arg APT_OPTIONS="Acquire::Retries=3" \
    $push_flag \
    "$context"
  
  echo "✅ Built $component with tags: ${tags[*]}"
}

# Parse component argument
COMPONENT="$1"

# Navigate to project root (we should already be there but ensure it)
cd "$ROOT_DIR"
echo "Working directory: $(pwd)"

# Use optimized Dockerfiles from devops directory
if [[ "$COMPONENT" == "api" || -z "$COMPONENT" ]]; then
  echo "Building API image..."
  build_and_push "api" "root/devops/Dockerfile.api" "."
fi

if [[ "$COMPONENT" == "worker" || -z "$COMPONENT" ]]; then
  echo "Building Worker image..."
  build_and_push "worker" "root/devops/Dockerfile.worker" "."
fi

if [[ "$COMPONENT" == "ui" || -z "$COMPONENT" ]]; then
  echo "Building UI image..."
  build_and_push "ui" "root/devops/Dockerfile.ui" "."
fi

echo "✅ All requested images built and pushed successfully!"
echo "Next step: Deploy container apps with './deploy-apps.sh'"