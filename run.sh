#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Sentio RAG System Controller
# --------------------------------------------------------------
# Purpose: Unified, unambiguous CLI for local/dev/prod workflows.
# Usage  : ./run.sh <command> [options]
# Commands:
#   start [--mode dev|prod] - Start Sentio stack in dev or prod mode
#   stop                    - Stop all running services
#   restart [--mode dev|prod] - Restart stack in specified mode
#   status                  - Show running service statuses & health checks
#   logs [svc] [-n lines]   - Tail logs (all services or specific one)
#   ingest [path]           - Ingest documents (runs ingest container)
#   test                    - Execute full test suite
#   chat-test [-p preset] [-v] - Run sample chat queries against API
#   env                     - Environment diagnostics
#   index ...               - Build/rebuild indexes
#   locust                  - Launch Locust load test UI
#   azure-push              - Build & push images to GHCR
#   infra start             - Start Azure infrastructure (Container Apps)
#   infra stop              - Stop Azure infrastructure (scale to zero)
#   infra status            - Show status of Azure resources
#   infra destroy           - Destroy all Azure resources in specified resource group
#   infra destroy-all       - Destroy ALL Azure resources in ALL resource groups
#   infra full-deploy       - Complete Azure deployment (infra, secrets, apps)
#   docker up [options]     - Start Docker containers with proper env setup
#   docker down [options]   - Stop Docker containers
#   docker build [service]  - Build Docker images (sequential)
#   docker bake [service]   - Build Docker images (parallel, optimized)
#   help                    - Display this help
# --------------------------------------------------------------
# Author : Sentio Team
# License: MIT
# ══════════════════════════════════════════════════════════════
set -euo pipefail
IFS=$'\n\t'

# ------------------------- Styling --------------------------- #
RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"; BLUE="\033[0;34m"; NC="\033[0m"
print_header() { echo -e "${BLUE}════ Sentio RAG Controller ═══════════════════════${NC}"; }

# >>> ADD: Friendly colour-coded help printer
show_help() {
  echo -e "${BLUE}Usage${NC}: ./run.sh <command> [options]\n"

  echo -e "${YELLOW}STACK:${NC}"
  echo -e "  ${GREEN}start${NC} [-m dev|prod]        Start Sentio stack"
  echo -e "  ${GREEN}stop${NC}                      Stop all services"
  echo -e "  ${GREEN}restart${NC} [-m dev|prod]      Restart stack"
  echo -e "  ${GREEN}status${NC}                     Show service status & health checks"
  echo -e "  ${GREEN}logs${NC} [svc] [-n N]          Tail logs (all or specific service)"
  echo -e "  ${GREEN}build${NC} [service]           Fast build of all services (parallel + cache)"
  echo -e "  ${GREEN}flush${NC} [-y]                 Delete all Qdrant collections\n"

  echo -e "${YELLOW}DEV TOOLS:${NC}"
  echo -e "  ${GREEN}ingest${NC} [path]              Ingest documents via helper container"
  echo -e "  ${GREEN}index${NC} ...                  Build / rebuild indexes (see --help)"
  echo -e "  ${GREEN}test${NC} [-s scope]            Run pytest suite (scope: local|cloud)"
  echo -e "  ${GREEN}chat-test${NC} [-p preset]       Run sample chat queries"
  echo -e "  ${GREEN}locust${NC} [-p port]           Launch Locust load-test UI"
  echo -e "  ${GREEN}env${NC}                        Environment diagnostics\n"

  echo -e "${YELLOW}DOCKER:${NC}"
  echo -e "  ${GREEN}docker up${NC} [service]        Start docker-compose stack"
  echo -e "  ${GREEN}docker down${NC}               Stop docker-compose stack"
  echo -e "  ${GREEN}docker build${NC} [service]     Build images sequentially"
  echo -e "  ${GREEN}docker bake${NC} [service]     Build images in parallel with caching\n"

  echo -e "${YELLOW}INFRA (Azure):${NC}"
  echo -e "  ${GREEN}infra deploy${NC}              Deploy core infrastructure (Bicep)"
  echo -e "  ${GREEN}infra apps${NC}                Deploy API / Worker / UI Container Apps"
  echo -e "  ${GREEN}infra secrets${NC}             Setup Key-Vault secrets"
  echo -e "  ${GREEN}infra build-images${NC}        Build & push multi-arch images to GHCR"
  echo -e "  ${GREEN}infra full-deploy${NC}         Complete deployment (infra+secrets+apps)"
  echo -e "  ${GREEN}infra start|stop${NC}          Scale Container Apps (on/off)"
  echo -e "  ${GREEN}infra update${NC}              Update Container Apps with latest images"
  echo -e "  ${GREEN}infra status${NC}              Show resource status"
  echo -e "  ${GREEN}infra destroy${NC}             Destroy resources in one RG"
  echo -e "  ${GREEN}infra destroy-all${NC}         Destroy ALL resources in ALL RGs\n"

  echo -e "${YELLOW}MISC:${NC}"
  echo -e "  ${GREEN}azure-push${NC}                Build & push images using docker-compose"
  echo -e "  ${GREEN}help${NC}                       Show this help\n"
}
# <<< END add show_help()

# -------------------- Environment loading ------------------- #
load_env() {
  local env_path=".env"
  if [[ -f "$env_path" ]]; then
    # Strip comments / trailing whitespace then export
    set -a
    # shellcheck disable=SC2046
    export $(grep -v '^#' "$env_path" | sed -e 's/#.*$//' -e 's/[[:space:]]*$//' | xargs || true)
    set +a
  fi
}

# Export environment early so child Python CLI inherits variables
load_env

# ---------------------- CLI delegation --------------------- #
# Preserve command components irrespective of IFS (space removed above)
PY=(python -m root.cli.sentio_cli)

# -------------------- Docker Compose Wrapper ----------------- #
run_docker_compose() {
  # Enable Docker Compose Bake for better build performance
  export COMPOSE_BAKE=true
  
  # Check for the presence of the setup-env.sh script
  if [[ -f "root/devops/setup-env.sh" ]]; then
    # Make the script executable
    chmod +x root/devops/setup-env.sh
    # Run docker-compose through setup-env.sh
    root/devops/setup-env.sh "$@"
  else
    # Run docker-compose directly if the script is not found
    docker compose "$@"
  fi
}

# Docker Bake wrapper for parallel builds
run_docker_bake() {
  # Enable Docker BuildKit for better performance
  export DOCKER_BUILDKIT=1
  export COMPOSE_DOCKER_CLI_BUILD=1
  export BUILDKIT_INLINE_CACHE=1
  
  # Default to all services if no argument provided
  local services=${1:-"all"}
  # Check if base service exists and build it first if needed
  if grep -q "service.*base:" docker-compose.yml 2>/dev/null; then
    echo -e "${GREEN}• Base image found in docker-compose.yml${NC}"
    
    # Build base image first if needed
    if [[ "$services" == "all" || "$services" == "base" ]]; then
      echo -e "${GREEN}• Building base image first...${NC}"
      run_docker_compose build base
    fi
  else
    echo -e "${YELLOW}• Base image not found in docker-compose.yml, skipping its build${NC}"
  fi
  
  # Parse services to build
  if [[ "$services" == "all" ]]; then
    echo -e "${GREEN}• Building all services in parallel...${NC}"
    # Build all services with buildx in parallel
    run_docker_compose build --parallel
  else
    echo -e "${GREEN}• Building services: $services${NC}"
    # Build specific services
    run_docker_compose build $services
  fi
}

# --------------------- Local embed server ----------------- #
EMBED_PID_FILE=".local_embed_server.pid"
start_embed_server() {
  if [[ "${BEAM_MODE:-cloud}" != "local" ]]; then
    return 0 # only start in local mode
  fi
  if [[ -f "$EMBED_PID_FILE" ]] && kill -0 "$(cat "$EMBED_PID_FILE")" 2>/dev/null; then
    echo -e "${YELLOW}• Embed server already running (PID $(cat "$EMBED_PID_FILE"))${NC}"
    return 0
  fi
  echo -e "${GREEN}• Starting local embedding server...${NC}"
  nohup python -m root.src.integrations.beam.local_model_server > embed_server.log 2>&1 &
  echo $! > "$EMBED_PID_FILE"
  echo -e "${GREEN}  ↳ PID $! | logs → embed_server.log${NC}"
}

stop_embed_server() {
  if [[ -f "$EMBED_PID_FILE" ]]; then
    PID=$(cat "$EMBED_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo -e "${GREEN}• Stopping local embedding server (PID $PID)...${NC}"
      kill "$PID" && rm -f "$EMBED_PID_FILE"
    fi
  fi
}

# --------------------- Azure Infrastructure Control ----------------- #
run_azure_infra_control() {
  local cmd=$1
  local script_path="infra/azure/scripts/infra-control.sh"
  
  if [[ ! -f "$script_path" ]]; then
    echo -e "${RED}Error: Azure infrastructure control script not found at $script_path${NC}"
    return 1
  fi
  
  chmod +x "$script_path"
  (cd "$(dirname "$script_path")" && "./$(basename "$script_path")" "$cmd")
}

# Function to update Azure Container Apps
update_azure_apps() {
  echo -e "${GREEN}• Updating Azure Container Apps...${NC}"
  
  local RESOURCE_GROUP="rg-sentio-free"
  local APPS=("ca-sentio-worker" "ca-sentio-api" "ca-sentio-ui")
  
  # Ensure logged in
  if ! az account show &>/dev/null; then
    echo -e "${RED}Not logged in to Azure CLI. Run 'az login' first.${NC}"
    return 1
  fi

  # Iterate over each Container App and restart its latest revision
  for APP in "${APPS[@]}"; do
    echo -e "${GREEN}• Processing $APP...${NC}"

    # Fetch latest revision name (by creation time)
    LATEST_REV=$(az containerapp revision list --resource-group "$RESOURCE_GROUP" --name "$APP" \
      --query "sort_by([],&createdTime)[-1].name" -o tsv 2>/dev/null || echo "")

    if [[ -z "$LATEST_REV" ]]; then
      echo -e "${YELLOW}  ↳ No revisions found for $APP, skipping restart${NC}"
      continue
    fi

    echo -e "  ↳ Restarting revision $LATEST_REV"
    if az containerapp revision restart --resource-group "$RESOURCE_GROUP" --name "$APP" --revision "$LATEST_REV"; then
      echo -e "  ✓ Restarted $APP revision $LATEST_REV"
    else
      echo -e "${YELLOW}  ⚠️  Failed to restart $APP revision $LATEST_REV${NC}"
    fi
  done

  # After restart ensure apps are set to min replicas >=1 for API/UI
  echo -e "${GREEN}• Ensuring API and UI have at least 1 replica...${NC}"
  az containerapp update --resource-group "$RESOURCE_GROUP" --name "ca-sentio-api" --min-replicas 1 --max-replicas 1 >/dev/null || true
  az containerapp update --resource-group "$RESOURCE_GROUP" --name "ca-sentio-ui" --min-replicas 1 --max-replicas 1 >/dev/null || true

  echo -e "${GREEN}• Starting all container apps...${NC}"
  run_azure_infra_control "start"

  echo -e "${GREEN}✅ Azure Container Apps updated!${NC}"
}

# Function for full automatic deployment to Azure
full_azure_deploy() {
  echo -e "${GREEN}• Starting full automatic deployment to Azure...${NC}"
  
  # Step 1: Infrastructure preparation
  echo -e "${GREEN}• Step 1/5: Deploying base infrastructure...${NC}"
  chmod +x infra/azure/scripts/deploy-infra.sh
  infra/azure/scripts/deploy-infra.sh
  
  # Step 2: Secrets setup
  echo -e "${GREEN}• Step 2/5: Setting up secrets in Key Vault...${NC}"
  chmod +x infra/azure/scripts/setup-secrets.sh
  infra/azure/scripts/setup-secrets.sh
  
  # Step 3: Build and publish images
  echo -e "${GREEN}• Step 3/5: Building and publishing Docker images...${NC}"
  chmod +x infra/azure/scripts/build-multi-arch.sh
  infra/azure/scripts/build-multi-arch.sh
  
  # Step 4: Application deployment
  echo -e "${GREEN}• Step 4/5: Deploying Container Apps...${NC}"
  chmod +x infra/azure/scripts/deploy-apps.sh
  infra/azure/scripts/deploy-apps.sh
  
  # Step 5: Application startup
  echo -e "${GREEN}• Step 5/5: Starting all services...${NC}"
  run_azure_infra_control "start"
  
  echo -e "${GREEN}✅ Full deployment to Azure completed successfully!${NC}"
  echo -e "${GREEN}• Check resource status: ./run.sh infra status${NC}"
}

# -------------------------- Main ---------------------------- #
print_header
case "${1:-help}" in
  start)      shift; start_embed_server; "${PY[@]}" start "$@" ;;
  stop)       stop_embed_server; "${PY[@]}" stop ;;
  restart)    shift; stop_embed_server; start_embed_server; "${PY[@]}" restart "$@" ;;
  status)     "${PY[@]}" status ;;
  logs)       shift; "${PY[@]}" logs "$@" ;;
  ingest)     shift; "${PY[@]}" ingest "$@" ;;
  test)       shift; "${PY[@]}" tests --scope local "$@" ;;
  test-local) shift; "${PY[@]}" tests --scope local "$@" ;;
  test-cloud) shift; "${PY[@]}" tests --scope cloud "$@" ;;
  chat-test)  shift; "${PY[@]}" chat-test "$@" ;;
  env)        "${PY[@]}" env ;;
  index)      shift; "${PY[@]}" build-index "$@" ;;
  locust)     shift; "${PY[@]}" locust "$@" ;;
  flush)      shift; "${PY[@]}" flush "$@" ;;
  build)      shift; run_docker_bake "$@" ;;
  docker)
    if [ $# -lt 2 ]; then
      echo -e "${RED}Error: Missing docker sub-command. See './run.sh help'.${NC}"
      exit 1
    fi
    subcmd=$2; shift 2
    case "$subcmd" in
      up)     run_docker_compose up "$@" ;;
      down)   run_docker_compose down "$@" ;;
      build)  run_docker_compose build "$@" ;;
      bake)   run_docker_bake "$@" ;;
      *)      echo -e "${RED}Unknown docker sub-command: $subcmd${NC}"; exit 1 ;;
    esac
    ;;
  azure-push)
    if [[ -f "root/devops/prepare_for_azure.sh" ]]; then
      chmod +x root/devops/prepare_for_azure.sh
      ./root/devops/prepare_for_azure.sh
    else
      echo -e "${RED}prepare_for_azure.sh not found${NC}"
      exit 1
    fi
    ;;
  infra)
    if [ $# -lt 2 ]; then
      echo -e "${RED}Error: Missing infra sub-command. See 'infra help' or run './run.sh help'.${NC}"
      exit 1
    fi
    subcmd=$2; shift 2
    case "$subcmd" in
      start|stop|status|destroy|destroy-all)
        run_azure_infra_control "$subcmd" "$@" ;;
      deploy)
        chmod +x infra/azure/scripts/deploy-infra.sh
        infra/azure/scripts/deploy-infra.sh "$@" ;;
      apps)
        chmod +x infra/azure/scripts/deploy-apps.sh
        infra/azure/scripts/deploy-apps.sh "$@" ;;
      secrets)
        chmod +x infra/azure/scripts/setup-secrets.sh
        infra/azure/scripts/setup-secrets.sh "$@" ;;
      build-images|build)
        chmod +x infra/azure/scripts/build-multi-arch.sh
        infra/azure/scripts/build-multi-arch.sh "$@" ;;
      update)
        update_azure_apps ;;
      full-deploy)
        full_azure_deploy ;;
      *)
        echo -e "${RED}Unknown infra sub-command: $subcmd${NC}"; exit 1 ;;
    esac
    ;;
  help|*)     show_help ;;
esac 