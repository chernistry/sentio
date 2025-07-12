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
#   docker-up [options]     - Start Docker containers with proper env setup
#   docker-down [options]   - Stop Docker containers
#   docker-build [options]  - Build Docker images
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
  # Check for the presence of the setup-env.sh script
  if [[ -f "root/devops/setup-env.sh" ]]; then
    # Make the script executable
    chmod +x root/devops/setup-env.sh
    # Run docker-compose through setup-env.sh
    root/devops/setup-env.sh "$@"
  else
    # Run docker-compose directly if the script is not found
    docker-compose "$@"
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
  rebuild)    shift; "${PY[@]}" rebuild "$@" ;;
  azure-push) shift; "${PY[@]}" azure-prepare "$@" ;;
  docker-up)  shift; run_docker_compose up "$@" ;;
  docker-down)  shift; run_docker_compose down "$@" ;;
  docker-build)  shift; run_docker_compose build "$@" ;;
  help|*)     grep -E "^#   [a-z]" "$0" | sed 's/#   /  /' ;;
esac 