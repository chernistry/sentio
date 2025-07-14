#!/bin/bash
# Script to set environment variables based on BEAM_MODE

# Load variables from the .env file
set -a
source .env
set +a

# Determine BEAM_EMBEDDING_BASE_URL based on BEAM_MODE
if [ "$BEAM_MODE" = "local" ]; then
  export BEAM_EMBEDDING_BASE_URL="$BEAM_EMBEDDING_BASE_LOCAL_URL"
  echo "BEAM_MODE=local, using BEAM_EMBEDDING_BASE_LOCAL_URL: $BEAM_EMBEDDING_BASE_LOCAL_URL"
else
  export BEAM_EMBEDDING_BASE_URL="$BEAM_EMBEDDING_BASE_CLOUD_URL"
  echo "BEAM_MODE=cloud, using BEAM_EMBEDDING_BASE_CLOUD_URL: $BEAM_EMBEDDING_BASE_CLOUD_URL"
fi

# Run docker-compose with the provided arguments
exec docker-compose "$@" 