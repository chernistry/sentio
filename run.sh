#!/bin/bash
# Sentio RAG System Controller
# Usage: ./run.sh [command]
# Commands:
#   start       - Start the Sentio stack
#   stop        - Stop the Sentio stack
#   restart     - Restart the Sentio stack
#   status      - Show status of Sentio services
#   logs        - Show logs from all or specific service
#   test        - Run tests
#   testchat    - Test RAG API with sample questions
#   ingest      - Ingest documents
#   app         - Start ollama serve + all stack (qdrant, api, webui)
#   deploy      - Deploy to Azure Container Instances
#   destroy     - Destroy Azure resources
#   deployinfra - Deploy Azure infrastructure only

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'

# Functions
print_header() {
    echo -e "${BLUE}════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Sentio RAG System Controller${NC}"
    echo -e "${BLUE}════════════════════════════════════════════${NC}"
}

check_ollama() {
    echo -e "${YELLOW}Checking Ollama service...${NC}"
    if ! pgrep -x "ollama" > /dev/null; then
        echo -e "${RED}Ollama is not running! Starting Ollama...${NC}"
        ollama serve > /dev/null 2>&1 &
        sleep 2
    fi

    echo -e "${GREEN}Checking available models...${NC}"
    ollama list
}

wait_for_qdrant() {
    local url="http://localhost:6333/collections"
    echo -e "${YELLOW}Waiting for Qdrant to be ready...${NC}"
    for i in {1..30}; do
        if curl -s --max-time 2 "$url" | grep -q 'collections'; then
            echo -e "${GREEN}Qdrant is ready.${NC}"
            return 0
        fi
        sleep 1
    done
    echo -e "${RED}Qdrant did not become ready in time.${NC}"
    exit 1
}

is_qdrant_running() {
    docker compose ps | grep -q 'qdrant' && docker compose ps | grep qdrant | grep -q 'Up'
}

ingest_docs() {
    echo -e "${YELLOW}Ingesting documents...${NC}"
    local started_qdrant=0
    if ! is_qdrant_running; then
        echo -e "${YELLOW}Qdrant not running, starting...${NC}"
        docker compose up -d qdrant
        started_qdrant=1
    fi
    wait_for_qdrant
    if [ -f root/.env ]; then
        export $(grep -v '^#' root/.env | xargs)
    fi
    docker compose run --no-deps --rm ingest
    if [ "$started_qdrant" = "1" ]; then
        echo -e "${YELLOW}Stopping Qdrant...${NC}"
        docker compose stop qdrant
    fi
}

app_stack() {
    echo -e "${YELLOW}Starting Ollama serve (if not running)...${NC}"
    if ! pgrep -x "ollama" > /dev/null; then
        ollama serve > /dev/null 2>&1 &
        sleep 2
    fi
    # Load environment variables (e.g., JINA_API_KEY) if not already set
    if [ -f root/.env ]; then
        export $(grep -v '^#' root/.env | xargs)
    fi
    echo -e "${YELLOW}Starting Sentio stack (qdrant, api, webui)...${NC}"
    docker compose up -d qdrant api webui
    wait_for_qdrant
    echo -e "${GREEN}Stack started. WebUI: http://localhost:3001${NC}"
}

test_stack() {
    app_stack
    echo -e "${YELLOW}Running E2E tests...${NC}"
    cd root && pytest tests/e2e/test_chat.py -v
}

start_stack() {
    # Load environment variables (e.g., JINA_API_KEY) if not already set
    if [ -f root/.env ]; then
        export $(grep -v '^#' root/.env | xargs)
    fi
    docker compose up -d
    echo -e "${GREEN}Sentio stack started!${NC}"
    echo -e "Web UI: ${BLUE}http://localhost:3001${NC}"
    echo -e "API: ${BLUE}http://localhost:8000${NC}"
    echo -e "Qdrant UI: ${BLUE}http://localhost:6333/dashboard${NC}"
}

stop_stack() {
    echo -e "${YELLOW}Stopping Sentio stack...${NC}"
    docker compose down
    echo -e "${GREEN}Sentio stack stopped${NC}"
}

restart_stack() {
    stop_stack
    start_stack
}

show_status() {
    echo -e "${YELLOW}Sentio services status:${NC}"
    docker compose ps
    
    echo -e "\n${YELLOW}Ollama status:${NC}"
    if pgrep -x "ollama" > /dev/null; then
        echo -e "${GREEN}Ollama is running${NC}"
        echo -e "${YELLOW}Available models:${NC}"
        ollama list
    else
        echo -e "${RED}Ollama is not running${NC}"
    fi
    
    echo -e "\n${YELLOW}Qdrant collection status:${NC}"
    python debug/stack_debug.py --checks collection_info --pretty || echo -e "${RED}Failed to get Qdrant collection info${NC}"
}

show_logs() {
    if [ "$1" == "" ]; then
        echo -e "${YELLOW}Showing logs for all services...${NC}"
        docker compose logs --tail=50 -f
    else
        echo -e "${YELLOW}Showing logs for $1...${NC}"
        docker compose logs --tail=50 -f $1
    fi
}

run_tests() {
    test_stack
}

test_chat() {
    # Make sure stack is running
    if ! curl -s http://localhost:8000/docs > /dev/null; then
        echo -e "${YELLOW}API not running, starting stack...${NC}"
        app_stack
        sleep 3
    fi

    # Load environment variables (e.g., JINA_API_KEY) if not already set
    if [ -f root/.env ]; then
        export $(grep -v '^#' root/.env | xargs)
    fi

    # Check if verbose flag (-v) was passed
    local verbose_flag=""
    if [ "$1" == "-v" ]; then
        verbose_flag="--verbose"
        echo -e "${YELLOW}Running chat tests with sample questions (verbose mode)...${NC}"
    else
        echo -e "${YELLOW}Running chat tests with sample questions...${NC}"
    fi

    cd root && python -m tests.e2e.test_chat --preset all $verbose_flag
}

deploy_to_azure() {
    echo -e "${YELLOW}Deploying to Azure Container Instances...${NC}"
    cd infra/azure
    
    if [ "$1" == "infra" ]; then
        echo -e "${YELLOW}Deploying infrastructure only...${NC}"
        ./deploy-infra.sh
        echo -e "${GREEN}Infrastructure deployed. You can now run './run.sh deploy' to deploy the apps.${NC}"
        return
    fi
    
    # Check if infrastructure exists
    if ! az group show --name rg-sentio-free &>/dev/null; then
        echo -e "${YELLOW}Infrastructure not found. Deploying infrastructure first...${NC}"
        ./deploy-infra.sh
    fi
    
    # Setup secrets
    echo -e "${YELLOW}Setting up secrets in Key Vault...${NC}"
    ./setup-secrets.sh
    
    # Deploy container instances
    echo -e "${YELLOW}Deploying container instances...${NC}"
    ./deploy-container-apps.sh
    
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    cd ../..
}

destroy_azure_resources() {
    echo -e "${RED}WARNING: This will destroy all Azure resources in the resource group!${NC}"
    read -p "Are you sure you want to continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Destroying Azure resources...${NC}"
        cd infra/azure
        ./cleanup.sh
        cd ../..
    else
        echo -e "${GREEN}Operation cancelled.${NC}"
    fi
}

# Main
print_header

case "$1" in
    start)
        start_stack
        ;;
    stop)
        stop_stack
        ;;
    restart)
        restart_stack
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs $2
        ;;
    test)
        run_tests
        ;;
    testchat)
        test_chat $2
        ;;
    ingest)
        ingest_docs
        ;;
    app)
        app_stack
        ;;
    deploy)
        deploy_to_azure
        ;;
    deployinfra)
        deploy_to_azure infra
        ;;
    destroy)
        destroy_azure_resources
        ;;
    *)
        print_header
        echo -e "${YELLOW}Usage: $0 {start|stop|restart|status|logs|test|ingest|app|deploy|deployinfra|destroy}${NC}"
        echo -e "  start       - Start the Sentio stack"
        echo -e "  stop        - Stop the Sentio stack"
        echo -e "  restart     - Restart the Sentio stack"
        echo -e "  status      - Show status of Sentio services"
        echo -e "  logs [svc]  - Show logs for all or specific service"
        echo -e "  test        - Run E2E tests (auto stack up)"
        echo -e "  testchat    - Test RAG API with sample questions"
        echo -e "  ingest      - Ingest documents (auto qdrant up)"
        echo -e "  app         - Start ollama serve + all stack"
        echo -e "  deploy      - Deploy to Azure Container Instances"
        echo -e "  deployinfra - Deploy Azure infrastructure only"
        echo -e "  destroy     - Destroy Azure resources"
        exit 1
esac

exit 0 