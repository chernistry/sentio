# Sentio RAG Deployment on Azure

## Overview

This document describes the process of deploying Sentio RAG on Microsoft Azure using free-tier resources. The deployment includes setting up the base infrastructure, building and publishing Docker images, and configuring container applications.

## Deployment Architecture

The Sentio RAG deployment on Azure consists of the following components:

1. **API container** - main API for request processing
2. **Worker container** - background processor for document indexing
3. **UI container** - user interface based on Streamlit
4. **Azure Key Vault** - storage for secrets and credentials
5. **Azure Storage Account** - storage for data and application state
6. **Azure Container Apps** - runtime environment for containers
7. **Qdrant Cloud** - external vector storage for embeddings

## Prerequisites

- Azure account with active subscription
- Azure CLI installed
- Docker and Docker Buildx for multi-arch image builds
- GitHub account and token for GitHub Container Registry
- Qdrant Cloud account for vector storage

## Deployment Steps

### 1. Environment Preparation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentio.git
   cd sentio
   ```

2. Create a `.env` file with the required environment variables:
   ```
   GHCR_USERNAME=your_github_username
   GHCR_PAT_WRITE=your_github_token
   AZURE_SUBSCRIPTION_ID=your_subscription_id
   AZURE_RESOURCE_GROUP=sentio-rg
   AZURE_LOCATION=eastus
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

### 2. Build and Publish Docker Images

1. Run the multi-arch image build script:
   ```bash
   chmod +x infra/azure/scripts/build-multi-arch.sh
   infra/azure/scripts/build-multi-arch.sh
   ```

   The script will create three images for amd64 and arm64 architectures:
   - `ghcr.io/yourusername/sentio-api:latest`
   - `ghcr.io/yourusername/sentio-worker:latest`
   - `ghcr.io/yourusername/sentio-ui:latest`

2. Check the created images:
   ```bash
   docker images | grep sentio
   ```

### 3. Create Base Azure Infrastructure

1. Run the infrastructure creation script:
   ```bash
   chmod +x infra/azure/scripts/create-infra.sh
   infra/azure/scripts/create-infra.sh
   ```

   The script will create:
   - Resource group
   - Storage account
   - Key Vault
   - Container App environment

2. Check the created resources:
   ```bash
   az group show --name sentio-rg
   ```

### 4. Configure Secrets in Azure Key Vault

1. Run the secrets setup script:
   ```bash
   chmod +x infra/azure/scripts/setup-secrets.sh
   infra/azure/scripts/setup-secrets.sh
   ```

   The script will add all required secrets to Azure Key Vault, including Qdrant Cloud credentials.

2. Check the added secrets:
   ```bash
   az keyvault secret list --vault-name <your_vault_name>
   ```

### 5. Deploy Container Applications

1. Run the container app deployment script:
   ```bash
   chmod +x infra/azure/scripts/deploy-apps.sh
   infra/azure/scripts/deploy-apps.sh
   ```

   The script will create three container apps:
   - API container
   - Worker container
   - UI container

2. Check the deployed apps:
   ```bash
   az containerapp list --resource-group sentio-rg --output table
   ```

### 6. Deployment Testing

1. Run the smoke test script:
   ```bash
   chmod +x tests/smoke.sh
   ./tests/smoke.sh https://<api_fqdn>
   ```

2. Check UI availability:
   ```bash
   curl -I https://<ui_fqdn>
   ```

## Troubleshooting

### Issue: Containers fail to start due to incorrect paths

**Symptoms:**
- API container: error "No module named root.src.app"
- UI container: error "File does not exist: root/src/ui/app.py"
- Worker container: similar path issues

**Solution:**
1. Fix paths in Dockerfile:
   - API: `CMD ["python", "root/app.py"]`
   - Worker: `CMD ["python", "-m", "root.worker"]`
   - UI: `CMD ["streamlit", "run", "root/streamlit_app.py"]`

2. Add file existence check in the container before startup:
   ```Dockerfile
   RUN ls -la && \
       echo "Checking entrypoint..." && \
       ls -la root/ && \
       test -f root/app.py || (echo "ERROR: file not found" && exit 1)
   ```

3. Rebuild and republish Docker images:
   ```bash
   infra/azure/scripts/build-multi-arch.sh
   ```

4. Restart containers:
   ```bash
   az containerapp restart --name ca-sentio-api --resource-group sentio-rg
   az containerapp restart --name ca-sentio-worker --resource-group sentio-rg
   az containerapp restart --name ca-sentio-ui --resource-group sentio-rg
   ```

### Issue: Authentication problems with GitHub Container Registry

**Symptoms:**
- Error "unauthorized: authentication required" when pushing images

**Solution:**
1. Check GitHub token presence and correctness:
   ```bash
   echo $GHCR_PAT_WRITE | docker login ghcr.io -u $GHCR_USERNAME --password-stdin
   ```

2. Update the token in the .env file and in Azure Key Vault:
   ```bash
   az keyvault secret set --vault-name <vault_name> --name "github-token" --value "<new_token>"
   ```

### Issue: Containers cannot connect to Qdrant Cloud

**Symptoms:**
- Errors in container logs about inability to connect to Qdrant

**Solution:**
1. Check Qdrant URL and API key in Azure Key Vault:
   ```bash
   az keyvault secret show --vault-name <vault_name> --name "qdrant-url"
   az keyvault secret show --vault-name <vault_name> --name "qdrant-api-key"
   ```

2. Update secrets if needed:
   ```bash
   az keyvault secret set --vault-name <vault_name> --name "qdrant-url" --value "<correct_url>"
   az keyvault secret set --vault-name <vault_name> --name "qdrant-api-key" --value "<correct_key>"
   ```

3. Restart containers

## Monitoring and Optimization

### Monitoring

1. Set up Azure Monitor for container apps:
   ```bash
   az monitor log-analytics workspace create --resource-group sentio-rg --workspace-name sentio-logs
   az containerapp env update --name sentio-env --resource-group sentio-rg --logs-workspace-id <workspace-id>
   ```

2. Set up alerts for high resource usage:
   ```bash
   az monitor alert create --resource-group sentio-rg --name high-cpu-alert --condition "CPU > 80%" --action-group <action-group-id>
   ```

3. View container logs:
   ```bash
   az containerapp logs show --name ca-sentio-api --resource-group sentio-rg
   ```

### Optimization for Free Tier

1. Configure autoscaling to save resources:
   ```bash
   az containerapp update --name ca-sentio-api --resource-group sentio-rg --min-replicas 0 --max-replicas 1
   ```

2. Use minimal sizes for containers:
   ```bash
   az containerapp update --name ca-sentio-api --resource-group sentio-rg --cpu 0.5 --memory 1.0Gi
   ```

3. Set scaling rules based on load:
   ```bash
   az containerapp scale rule add --name ca-sentio-api --resource-group sentio-rg --type http --http-concurrency 10
   ```

## Application Update

1. Make code changes

2. Rebuild and publish Docker images:
   ```bash
   infra/azure/scripts/build-multi-arch.sh
   ```

3. Update container apps:
   ```bash
   az containerapp update --name ca-sentio-api --resource-group sentio-rg --image ghcr.io/yourusername/sentio-api:latest
   az containerapp update --name ca-sentio-worker --resource-group sentio-rg --image ghcr.io/yourusername/sentio-worker:latest
   az containerapp update --name ca-sentio-ui --resource-group sentio-rg --image ghcr.io/yourusername/sentio-ui:latest
   ```

## Conclusion

Deploying Sentio RAG on Azure using free-tier resources enables you to create a fully functional application for document processing and analysis using the RAG approach. Proper Docker image path configuration and resource optimization allow you to efficiently use the Azure free tier. 