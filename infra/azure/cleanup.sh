#!/bin/bash
# Cleanup Azure resources for Sentio RAG project

set -e

# Check if logged in to Azure CLI
echo "Checking Azure CLI login..."
az account show > /dev/null || { echo "Please login to Azure CLI using 'az login'"; exit 1; }

# Ask user if they want to delete ALL resources across all subscriptions or just specific resources
echo "WARNING: This script can delete ALL Azure resources across ALL resource groups and subscriptions."
read -p "Do you want to delete ALL resources (a) or just specific resources (s)? (a/s): " -n 1 -r DELETE_ALL
echo

if [[ $DELETE_ALL =~ ^[Aa]$ ]]; then
    echo "You've chosen to delete ALL Azure resources across ALL subscriptions."
    
    # List all subscriptions with Enabled state
    echo "Listing all enabled subscriptions..."
    SUBSCRIPTIONS=$(az account list --query "[?state=='Enabled'].id" -o tsv)
    
    if [ -z "$SUBSCRIPTIONS" ]; then
        echo "No enabled subscriptions found."
    fi
    
    for SUB_ID in $SUBSCRIPTIONS; do
        echo "Switching to subscription $SUB_ID..."
        az account set --subscription $SUB_ID || {
            echo "Failed to switch to subscription $SUB_ID, skipping..."
            continue
        }
        
        # List all resource groups in current subscription
        echo "Listing resource groups in subscription $SUB_ID..."
        RESOURCE_GROUPS=$(az group list --query "[].name" -o tsv 2>/dev/null || echo "")
        
        if [ -z "$RESOURCE_GROUPS" ]; then
            echo "No resource groups found in subscription $SUB_ID."
            continue
        fi
        
        for RG in $RESOURCE_GROUPS; do
            echo "Processing resource group: $RG"
            
            # Stop all container instances first
            CONTAINERS=$(az container list --resource-group $RG --query "[].name" -o tsv 2>/dev/null || echo "")
            for CONTAINER in $CONTAINERS; do
                echo "Stopping container $CONTAINER in resource group $RG..."
                az container stop --resource-group $RG --name $CONTAINER || true
            done
            
            # Get storage accounts for file share cleanup
            STORAGE_ACCOUNTS=$(az storage account list --resource-group $RG --query "[].name" -o tsv 2>/dev/null || echo "")
            for SA in $STORAGE_ACCOUNTS; do
                echo "Cleaning up storage account $SA..."
                # Get storage key
                STORAGE_KEY=$(az storage account keys list --resource-group $RG --account-name $SA --query "[0].value" -o tsv 2>/dev/null || echo "")
                
                if [ ! -z "$STORAGE_KEY" ]; then
                    # List and delete all file shares
                    SHARES=$(az storage share list --account-name $SA --account-key $STORAGE_KEY --query "[].name" -o tsv 2>/dev/null || echo "")
                    for SHARE in $SHARES; do
                        echo "Deleting file share $SHARE..."
                        az storage share delete --name $SHARE --account-name $SA --account-key $STORAGE_KEY --output none || true
                    done
                    
                    # Delete all blobs in all containers
                    BLOB_CONTAINERS=$(az storage container list --account-name $SA --account-key $STORAGE_KEY --query "[].name" -o tsv 2>/dev/null || echo "")
                    for BLOB_CONTAINER in $BLOB_CONTAINERS; do
                        echo "Deleting blobs in container $BLOB_CONTAINER..."
                        az storage blob delete-batch --source $BLOB_CONTAINER --account-name $SA --account-key $STORAGE_KEY || true
                    done
                fi
            done
            
            # Stop all VMs
            VMS=$(az vm list --resource-group $RG --query "[].name" -o tsv 2>/dev/null || echo "")
            for VM in $VMS; do
                echo "Stopping VM $VM..."
                az vm deallocate --resource-group $RG --name $VM || true
            done
            
            # Stop all App Services
            WEBAPPS=$(az webapp list --resource-group $RG --query "[].name" -o tsv 2>/dev/null || echo "")
            for WEBAPP in $WEBAPPS; do
                echo "Stopping Web App $WEBAPP..."
                az webapp stop --resource-group $RG --name $WEBAPP || true
            done
            
            # Stop AKS clusters
            AKS_CLUSTERS=$(az aks list --resource-group $RG --query "[].name" -o tsv 2>/dev/null || echo "")
            for AKS in $AKS_CLUSTERS; do
                echo "Stopping AKS cluster $AKS..."
                az aks stop --resource-group $RG --name $AKS || true
            done
            
            # Delete the resource group
            echo "Deleting resource group $RG..."
            az group delete --name $RG --yes
        done
    done
    
    # Clean up deleted Key Vaults that are in soft-delete state
    echo "Checking for soft-deleted Key Vaults to purge..."
    DELETED_KVNAMES=$(az keyvault list-deleted --query "[].name" -o tsv 2>/dev/null || echo "")
    for KV in $DELETED_KVNAMES; do
        echo "Purging deleted Key Vault: $KV"
        az keyvault purge --name $KV || true
    done

    echo "All Azure resources have been cleaned up across all subscriptions."
else
    # Variables for specific resources
    RESOURCE_GROUP="rg-sentio-free"
    KEY_VAULT_NAME="kv-sentio-free"
    QDRANT_CONTAINER="aci-qdrant-free"
    API_CONTAINER="aci-sentio-api"
    WORKER_CONTAINER="aci-sentio-worker"

    # Check if resource group exists
    if ! az group show --name $RESOURCE_GROUP &>/dev/null; then
        echo "Resource group '$RESOURCE_GROUP' does not exist. Nothing to clean up."
        
        # Check if there are any other resource groups
        echo "Checking for any other resource groups..."
        OTHER_RGS=$(az group list --query "[].name" -o tsv 2>/dev/null || echo "")
        
        if [ ! -z "$OTHER_RGS" ]; then
            echo "Found other resource groups:"
            echo "$OTHER_RGS"
            
            read -p "Do you want to delete these resource groups? (y/n): " -n 1 -r DELETE_OTHER
            echo
            
            if [[ $DELETE_OTHER =~ ^[Yy]$ ]]; then
                for RG in $OTHER_RGS; do
                    echo "Deleting resource group $RG..."
                    az group delete --name $RG --yes
                done
            fi
        else
            echo "No resource groups found."
        fi
        
        # Check for soft-deleted Key Vaults
        echo "Checking for soft-deleted Key Vaults..."
        DELETED_KVS=$(az keyvault list-deleted --query "[].name" -o tsv 2>/dev/null || echo "")
        
        if [ ! -z "$DELETED_KVS" ]; then
            echo "Found soft-deleted Key Vaults:"
            echo "$DELETED_KVS"
            
            read -p "Do you want to purge these Key Vaults? (y/n): " -n 1 -r PURGE_KV
            echo
            
            if [[ $PURGE_KV =~ ^[Yy]$ ]]; then
                for KV in $DELETED_KVS; do
                    echo "Purging Key Vault $KV..."
                    az keyvault purge --name $KV || true
                done
            fi
        else
            echo "No soft-deleted Key Vaults found."
        fi
        
        exit 0
    fi

    echo "WARNING: This will delete all resources in the resource group '$RESOURCE_GROUP'!"
    echo "This includes:"
    echo "- Qdrant Container Instance"
    echo "- API Container Instance"
    echo "- Worker Container Instance"
    echo "- Key Vault"
    echo "- Storage Account"
    echo "- Log Analytics Workspace"
    echo "- All other resources in the resource group"

    read -p "Are you sure you want to continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 0
    fi

    # Stop containers first if they exist
    echo "Stopping container instances if they exist..."

    # Check and stop Qdrant container
    if az container show --resource-group $RESOURCE_GROUP --name $QDRANT_CONTAINER &>/dev/null; then
        echo "Stopping Qdrant container..."
        az container stop --resource-group $RESOURCE_GROUP --name $QDRANT_CONTAINER
    fi

    # Check and stop API container
    if az container show --resource-group $RESOURCE_GROUP --name $API_CONTAINER &>/dev/null; then
        echo "Stopping API container..."
        az container stop --resource-group $RESOURCE_GROUP --name $API_CONTAINER
    fi

    # Check and stop Worker container
    if az container show --resource-group $RESOURCE_GROUP --name $WORKER_CONTAINER &>/dev/null; then
        echo "Stopping Worker container..."
        az container stop --resource-group $RESOURCE_GROUP --name $WORKER_CONTAINER
    fi

    # Wait a moment for containers to stop
    echo "Waiting for containers to stop..."
    sleep 10

    # Get storage account name if available
    STORAGE_ACCOUNT_NAME=$(az storage account list --resource-group $RESOURCE_GROUP --query "[0].name" -o tsv 2>/dev/null || echo "")

    if [ ! -z "$STORAGE_ACCOUNT_NAME" ]; then
        # Get storage account key
        STORAGE_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT_NAME --query "[0].value" -o tsv 2>/dev/null || echo "")
        
        if [ ! -z "$STORAGE_KEY" ]; then
            echo "Cleaning up file shares in storage account $STORAGE_ACCOUNT_NAME..."
            
            # List all file shares
            SHARES=$(az storage share list --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY --query "[].name" -o tsv 2>/dev/null || echo "")
            
            # Delete each share
            for SHARE in $SHARES; do
                echo "Deleting file share $SHARE..."
                az storage share delete --name $SHARE --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY --output none || true
            done
        fi
    fi

    echo "Deleting resource group '$RESOURCE_GROUP'..."
    az group delete --name $RESOURCE_GROUP --yes

    echo "Resource group deletion initiated. It may take a few minutes to complete."
    echo "You can check the status with: az group show --name $RESOURCE_GROUP"

    # Note about Key Vault soft delete
    echo "Note: Key Vault resources are subject to soft-delete protection."
    echo "If you want to recreate the Key Vault with the same name immediately,"
    echo "you may need to purge the deleted Key Vault using:"
    echo "az keyvault list-deleted"
    echo "az keyvault purge --name $KEY_VAULT_NAME"

    # Check if there are other deleted key vaults
    OTHER_DELETED_KVS=$(az keyvault list-deleted --query "[?name!='$KEY_VAULT_NAME'].name" -o tsv)
    if [ ! -z "$OTHER_DELETED_KVS" ]; then
        echo "Other deleted Key Vaults found that may need purging:"
        echo "$OTHER_DELETED_KVS"
    fi 
fi 