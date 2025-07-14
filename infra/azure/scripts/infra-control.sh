#!/bin/bash
# Control Azure infrastructure for Sentio RAG project
# Allows stopping, starting, and destroying resources

set -e

# Variables
RESOURCE_GROUP="rg-sentio-free"
LOCATION="westeurope"

# Load environment variables from .env.azure if exists
if [ -f "../.env.azure" ]; then
  source ../.env.azure
  echo "Loaded environment variables from .env.azure"
else
  echo "Warning: .env.azure file not found. Using default values."
fi

# Check if logged in to Azure CLI
echo "Checking Azure CLI login..."
az account show > /dev/null || { echo "Please login to Azure CLI using 'az login'"; exit 1; }

# Function to show usage
show_usage() {
  echo "Usage: $0 [start|stop|status|destroy]"
  echo ""
  echo "Commands:"
  echo "  start   - Start all Container Apps"
  echo "  stop    - Stop all Container Apps (scale to zero)"
  echo "  status  - Show status of all resources"
  echo "  destroy - Destroy all resources in the resource group"
  echo ""
  exit 1
}

# Function to check if resource group exists
check_resource_group() {
  az group show --name $RESOURCE_GROUP &>/dev/null
  return $?
}

# Function to check if resource provider is registered
check_resource_provider() {
  local provider=$1
  local state=$(az provider show --namespace $provider --query "registrationState" -o tsv 2>/dev/null || echo "NotRegistered")
  
  if [[ "$state" != "Registered" ]]; then
    echo "Resource provider $provider is not registered. Registering now..."
    az provider register --namespace $provider --wait
    echo "✅ Resource provider $provider registered"
  fi
}

# Function to get Container App names
get_container_apps() {
  if ! check_resource_group; then
    echo "Resource group $RESOURCE_GROUP does not exist."
    return 1
  fi
  
  # Check if Microsoft.App provider is registered
  check_resource_provider "Microsoft.App"
  
  # Get container apps, handle case when none exist
  local apps=$(az containerapp list --resource-group $RESOURCE_GROUP --query "[].name" -o tsv 2>/dev/null || echo "")
  if [ -z "$apps" ]; then
    echo "No Container Apps found in resource group $RESOURCE_GROUP."
    return 1
  fi
  
  echo "$apps"
}

# Function to deactivate old revisions
deactivate_old_revisions() {
  local app_name=$1
  echo "Checking for old revisions of $app_name to deactivate..."
  
  # Get list of all revisions
  local revisions=$(az containerapp revision list --name $app_name --resource-group $RESOURCE_GROUP --query "[].name" -o tsv 2>/dev/null || echo "")
  if [ -z "$revisions" ]; then
    echo "No revisions found for $app_name, skipping..."
    return
  fi
  
  # Get the latest revision
  local latest_revision=$(az containerapp revision list --name $app_name --resource-group $RESOURCE_GROUP --query "sort_by([],&createdTime)[-1].name" -o tsv 2>/dev/null || echo "")
  if [ -z "$latest_revision" ]; then
    echo "Could not determine latest revision for $app_name, skipping..."
    return
  fi
  
  echo "Latest revision for $app_name: $latest_revision"
  
  # Deactivate all old revisions
  for rev in $revisions; do
    if [ "$rev" != "$latest_revision" ]; then
      echo "Deactivating old revision: $rev"
      az containerapp revision deactivate --name $app_name --resource-group $RESOURCE_GROUP --revision $rev || true
    fi
  done
}

# Function to start Container Apps
start_container_apps() {
  echo "Starting Container Apps..."
  
  if ! check_resource_group; then
    echo "Resource group $RESOURCE_GROUP does not exist. Please deploy infrastructure first."
    exit 1
  fi
  
  # Check if Microsoft.App provider is registered
  check_resource_provider "Microsoft.App"
  
  local apps=$(get_container_apps)
  if [ $? -ne 0 ] || [ -z "$apps" ]; then
    echo "No Container Apps to start."
    return
  fi
  
  for app in $apps; do
    echo "Starting $app..."
    
    # Check if app has any revisions
    local revisions=$(az containerapp revision list --name $app --resource-group $RESOURCE_GROUP --query "[].name" -o tsv 2>/dev/null || echo "")
    if [ -z "$revisions" ]; then
      echo "No revisions found for $app, skipping..."
      continue
    fi
    
    # Deactivate old revisions before starting
    deactivate_old_revisions $app
    
    # Get the latest active revision instead of using 'latest' literal string
    local latest_revision=$(az containerapp revision list --name $app --resource-group $RESOURCE_GROUP --query "sort_by([],&createdTime)[-1].name" -o tsv 2>/dev/null || echo "")
    if [ ! -z "$latest_revision" ]; then
      echo "Activating revision $latest_revision for $app..."
      # Use || true to continue even if activation fails (e.g. already active)
      az containerapp revision activate --name $app --resource-group $RESOURCE_GROUP --revision $latest_revision || true
      
      # Check that the revision is activated and running
      local revision_status=$(az containerapp revision show --name $app --resource-group $RESOURCE_GROUP --revision $latest_revision --query "properties.status" -o tsv 2>/dev/null || echo "Unknown")
      echo "Revision status: $revision_status"
    else
      echo "No active revision found for $app, skipping activation..."
    fi
    
    # Set min replicas to 1 for API and UI to ensure they're running
    if [[ "$app" == *"api"* ]] || [[ "$app" == *"ui"* ]]; then
      echo "Setting min replicas to 1 for $app..."
      az containerapp update --name $app --resource-group $RESOURCE_GROUP --min-replicas 1
    fi
  done
  
  echo "✅ Container Apps started"
  
  # Check status of started applications
  echo "Checking status of started apps..."
  for app in $apps; do
    local status=$(az containerapp show --name $app --resource-group $RESOURCE_GROUP --query "properties.provisioningState" -o tsv 2>/dev/null || echo "Unknown")
    local replicas=$(az containerapp show --name $app --resource-group $RESOURCE_GROUP --query "properties.template.scale.minReplicas" -o tsv 2>/dev/null || echo "Unknown")
    echo "$app: Status=$status, MinReplicas=$replicas"
  done
}

# Function to stop Container Apps
stop_container_apps() {
  echo "Stopping Container Apps..."
  
  if ! check_resource_group; then
    echo "Resource group $RESOURCE_GROUP does not exist. Nothing to stop."
    return
  fi
  
  # Check if Microsoft.App provider is registered
  check_resource_provider "Microsoft.App"
  
  local apps=$(get_container_apps)
  if [ $? -ne 0 ] || [ -z "$apps" ]; then
    echo "No Container Apps to stop."
    return
  fi
  
  for app in $apps; do
    echo "Setting min replicas to 0 for $app..."
    az containerapp update --name $app --resource-group $RESOURCE_GROUP --min-replicas 0
    
    # Deactivate latest revision to ensure it's not running
    echo "Deactivating $app..."
    local latest_revision=$(az containerapp revision list --name $app --resource-group $RESOURCE_GROUP --query "[0].name" -o tsv 2>/dev/null || echo "")
    if [ ! -z "$latest_revision" ]; then
      az containerapp revision deactivate --name $app --resource-group $RESOURCE_GROUP --revision $latest_revision
    fi
  done
  
  echo "✅ All Container Apps stopped"
}

# Function to show status
show_status() {
  echo "Resource Group: $RESOURCE_GROUP"
  
  if ! check_resource_group; then
    echo "Resource group $RESOURCE_GROUP does not exist."
    return
  fi
  
  # Check if Microsoft.App provider is registered
  check_resource_provider "Microsoft.App"
  
  echo -e "\nContainer Apps:"
  az containerapp list --resource-group $RESOURCE_GROUP --query "[].{Name:name, Status:properties.provisioningState, Replicas:properties.template.scale.minReplicas, Endpoint:properties.configuration.ingress.fqdn}" -o table 2>/dev/null || echo "No Container Apps found."
  
  # Show active revisions for each application
  local apps=$(get_container_apps)
  if [ $? -eq 0 ] && [ ! -z "$apps" ]; then
    echo -e "\nActive Revisions:"
    for app in $apps; do
      local active_revisions=$(az containerapp revision list --name $app --resource-group $RESOURCE_GROUP --query "[?properties.active==\`true\`].{Name:name, Created:properties.createdTime, Replicas:properties.replicas}" -o table 2>/dev/null || echo "None")
      echo -e "$app:"
      echo "$active_revisions"
    done
  fi
  
  if [ ! -z "$KEY_VAULT_NAME" ]; then
    echo -e "\nKey Vault:"
    az keyvault show --name $KEY_VAULT_NAME --resource-group $RESOURCE_GROUP --query "{Name:name, Status:properties.provisioningState}" -o table 2>/dev/null || echo "Key Vault $KEY_VAULT_NAME not found."
  fi
  
  if [ ! -z "$STORAGE_ACCOUNT_NAME" ]; then
    echo -e "\nStorage Account:"
    az storage account show --name $STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP --query "{Name:name, Status:provisioningState}" -o table 2>/dev/null || echo "Storage Account $STORAGE_ACCOUNT_NAME not found."
  fi
}

# Function to destroy all resources
destroy_resources() {
  echo "WARNING: This will destroy all resources in the resource group $RESOURCE_GROUP!"
  echo "This action cannot be undone."
  read -p "Are you sure you want to continue? (y/n): " -n 1 -r
  echo
  
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Destroying all resources in $RESOURCE_GROUP..."
    
    # Check if resource group exists
    if check_resource_group; then
      # First try to stop all container apps if they exist
      if [ ! -z "$KEY_VAULT_NAME" ]; then
        # Check for soft-deleted Key Vault
        DELETED_KV=$(az keyvault list-deleted --query "[?name=='$KEY_VAULT_NAME'].name" -o tsv 2>/dev/null || echo "")
      fi
      
      # Try to stop container apps, but don't fail if it doesn't work
      stop_container_apps || true
      
      # Delete the resource group
      echo "Deleting resource group $RESOURCE_GROUP..."
      az group delete --name $RESOURCE_GROUP --yes
      
      # Purge soft-deleted Key Vault if exists
      if [ ! -z "$DELETED_KV" ]; then
        echo "Purging soft-deleted Key Vault $KEY_VAULT_NAME..."
        az keyvault purge --name $KEY_VAULT_NAME --no-wait
      fi
      
      echo "✅ All resources destroyed successfully"
    else
      echo "Resource group $RESOURCE_GROUP does not exist. Nothing to destroy."
    fi
  else
    echo "Operation cancelled"
  fi
}

# Function to destroy all resources in all resource groups
destroy_all_resources() {
  echo "Checking for Azure resources..."
  
  # Get all resource groups
  echo "Fetching all resource groups..."
  RESOURCE_GROUPS=$(az group list --query "[].name" -o tsv)
  
  if [ -z "$RESOURCE_GROUPS" ]; then
    echo "No resource groups found in your Azure subscription. Nothing to destroy."
    return
  fi
  
  echo "Found the following resource groups:"
  echo "$RESOURCE_GROUPS" | tr '\t' '\n' | sed 's/^/- /'
  
  echo "⚠️ WARNING: This will destroy ALL resources in ALL resource groups listed above!"
  echo "This action cannot be undone and will delete EVERYTHING in your Azure subscription."
  read -p "Are you absolutely sure you want to continue? (yes/no): " -r
  echo
  
  if [[ $REPLY == "yes" ]]; then
    echo "Destroying ALL resources in ALL resource groups..."
    
    read -p "Last chance! Type 'DELETE ALL' to confirm deletion: " -r
    echo
    
    if [[ $REPLY == "DELETE ALL" ]]; then
      # Loop through each resource group and delete
      for rg in $RESOURCE_GROUPS; do
        echo "Deleting resource group $rg..."
        az group delete --name "$rg" --yes
        echo "✅ Resource group $rg deleted"
      done
      
      echo "✅ All Azure resources in all resource groups have been destroyed"
    else
      echo "Operation cancelled"
    fi
  else
    echo "Operation cancelled"
  fi
}

# Check command argument
if [ $# -ne 1 ]; then
  show_usage
fi

# Execute command
case "$1" in
  start)
    start_container_apps
    ;;
  stop)
    stop_container_apps
    ;;
  status)
    show_status
    ;;
  destroy)
    destroy_resources
    ;;
  destroy-all)
    destroy_all_resources
    ;;
  *)
    show_usage
    ;;
esac