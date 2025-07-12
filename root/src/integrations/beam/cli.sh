#!/bin/bash

# CLI script for managing Beam Cloud deployments

set -e

# -------------------- Configuration -------------------- #

# Project root autodetection (fallback to current dir)
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
ENV_FILE="${PROJECT_ROOT}/.env"
APP_PATH="root/src/integrations/beam/app.py"

# Colours for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No colour

# -------------------------------------------------------- #
# Helper – load variables from .env if present
load_env() {
    if [[ -f "$ENV_FILE" ]]; then
        # shellcheck disable=SC1090
        set -a && source "$ENV_FILE" && set +a
    fi
}

# Helper – print error and exit
abort() {
    echo -e "${RED}Error:${NC} $1" >&2
    exit 1
}

# Helper – Beam CLI presence check
require_beam_cli() {
    command -v beam &>/dev/null || abort "Beam CLI not installed. Run: pip install beam-client"
}

# Helper – ensure BEAM_API_TOKEN present
require_token() {
    if [[ -z "$BEAM_API_TOKEN" ]]; then
        abort "BEAM_API_TOKEN not set. Add it to .env or pass --token <TOKEN>."
    fi
}

# Helper – configure Beam profile with token if not yet configured
configure_beam() {
    require_beam_cli
    require_token
    echo -e "${GREEN}Configuring Beam CLI...${NC}"
    beam configure default --token "$BEAM_API_TOKEN"
    echo -e "${GREEN}Beam CLI configured.${NC}"
}

# -------------------- Commands -------------------- #

show_help() {
    cat <<EOF
${GREEN}Beam Cloud CLI for Sentio${NC}
Usage: ./cli.sh <command> [options]
Commands:
  setup                Configure Beam CLI with API token
  deploy [--chat|--inference|--all]
  destroy [--chat|--inference|--all]
  list                 List deployed resources
  invoke <queue> <json>
  setup-env            Add Beam variables to .env (idempotent)
  help                 Show this help message
Options:
  --token <TOKEN>      Override BEAM_API_TOKEN for this invocation
EOF
}

# Load env variables before parsing args (allows override via CLI)
load_env

# Default option values
DEPLOY_CHAT=false DEPLOY_INF=false DEPLOY_ALL=false
DESTROY_CHAT=false DESTROY_INF=false DESTROY_ALL=false

COMMAND=$1; shift || true

# Option parse loop
while [[ $# -gt 0 ]]; do
    case $1 in
        --chat)
            DEPLOY_CHAT=true DESTROY_CHAT=true ; shift ;;
        --inference)
            DEPLOY_INF=true DESTROY_INF=true ; shift ;;
        --all)
            DEPLOY_ALL=true DESTROY_ALL=true ; shift ;;
        --token)
            BEAM_API_TOKEN=$2 ; shift 2 ;;
        --help)
            show_help ; exit 0 ;;
        *)
            abort "Unknown option $1" ;;
    esac
done

# Defaults if no specific deploy/destroy flags set
$DEPLOY_CHAT || $DEPLOY_INF || $DEPLOY_ALL || DEPLOY_ALL=true
$DESTROY_CHAT || $DESTROY_INF || $DESTROY_ALL || DESTROY_ALL=true

# -------------------- Actions -------------------- #

setup_cmd() {
    configure_beam
}

deploy_cmd() {
    configure_beam
    require_beam_cli

    if $DEPLOY_ALL || $DEPLOY_CHAT; then
        echo -e "${GREEN}Deploying chat endpoint...${NC}"
        beam deploy "${APP_PATH}:chat_endpoint"
    fi
    if $DEPLOY_ALL || $DEPLOY_INF; then
        echo -e "${GREEN}Deploying inference task queue...${NC}"
        beam deploy "${APP_PATH}:inference_task"
    fi
}

destroy_cmd() {
    configure_beam
    require_beam_cli

    if $DESTROY_ALL || $DESTROY_CHAT; then
        echo -e "${YELLOW}Removing chat endpoint...${NC}"
        beam stop endpoint chat || echo -e "${YELLOW}Chat endpoint not found.${NC}"
    fi
    if $DESTROY_ALL || $DESTROY_INF; then
        echo -e "${YELLOW}Removing inference task queue...${NC}"
        beam stop task-queue inference || echo -e "${YELLOW}Inference queue not found.${NC}"
    fi
}

list_cmd() {
    configure_beam
    echo -e "${GREEN}Endpoints:${NC}" && beam list endpoints || true
    echo -e "${GREEN}Task Queues:${NC}" && beam list task-queues || true
}

invoke_cmd() {
    configure_beam
    QUEUE=$1 JSON=$2
    [[ -z $QUEUE || -z $JSON ]] && abort "invoke requires queue and JSON payload"
    echo -e "${GREEN}Invoking $QUEUE...${NC}"
    beam invoke "$QUEUE" "$JSON"
}

setup_env_cmd() {
    # Ensure .env exists
    [[ -f "$ENV_FILE" ]] || touch "$ENV_FILE"

    grep -q "CHAT_PROVIDER=" "$ENV_FILE" || echo "CHAT_PROVIDER=beam" >> "$ENV_FILE"
    grep -q "BEAM_API_TOKEN=" "$ENV_FILE" || echo "BEAM_API_TOKEN=" >> "$ENV_FILE"
    grep -q "BEAM_VOLUME=" "$ENV_FILE" || echo "BEAM_VOLUME=comfy-weights" >> "$ENV_FILE"
    grep -q "BEAM_MODEL_ID=" "$ENV_FILE" || echo "BEAM_MODEL_ID=mistral-7b" >> "$ENV_FILE"
    grep -q "BEAM_GPU=" "$ENV_FILE" || echo "BEAM_GPU=A10G" >> "$ENV_FILE"
    grep -q "BEAM_MEMORY=" "$ENV_FILE" || echo "BEAM_MEMORY=32Gi" >> "$ENV_FILE"
    grep -q "BEAM_CPU=" "$ENV_FILE" || echo "BEAM_CPU=4" >> "$ENV_FILE"

    echo -e "${GREEN}Beam variables ensured in .env (edit as needed).${NC}"
}

case $COMMAND in
    setup) setup_cmd ;;
    deploy) deploy_cmd ;;
    destroy) destroy_cmd ;;
    list) list_cmd ;;
    invoke) invoke_cmd "$@" ;;
    setup-env) setup_env_cmd ;;
    help|*) show_help ;;
esac
