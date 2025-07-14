@description('The location for all resources.')
param location string = resourceGroup().location

@description('Environment name (dev, test, prod)')
param environmentName string = 'free'

@description('Prefix for all resources')
param prefix string = 'sentio'

// Variables
var keyVaultName = 'kv-${prefix}-${environmentName}'
var storageAccountName = 'stg${prefix}${uniqueString(resourceGroup().id)}'
var queueName = 'submissions'
var qdrantShareName = 'qdrantshare'
var containerAppEnvName = 'cae-${prefix}-${environmentName}'

// Common tags for all resources
var tags = {
  project: prefix
  environment: environmentName
  deploymentType: 'bicep'
}

// Log Analytics Workspace for monitoring
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'log-${prefix}-${environmentName}'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

// Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2022-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    enabledForDeployment: true
    enabledForTemplateDeployment: true
    enabledForDiskEncryption: true
    tenantId: subscription().tenantId
    accessPolicies: []
    sku: {
      name: 'standard'
      family: 'A'
    }
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// Storage Account
resource storageAccount 'Microsoft.Storage/storageAccounts@2022-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Allow'
    }
  }
}

// Queue Service
resource queueService 'Microsoft.Storage/storageAccounts/queueServices@2022-05-01' = {
  name: 'default'
  parent: storageAccount
  properties: {
    cors: {
      corsRules: []
    }
  }
}

// Queue
resource queue 'Microsoft.Storage/storageAccounts/queueServices/queues@2022-05-01' = {
  name: queueName
  parent: queueService
  properties: {
    metadata: {}
  }
}

// File Service
resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2022-05-01' = {
  name: 'default'
  parent: storageAccount
  properties: {
    cors: {
      corsRules: []
    }
    shareDeleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

// File Share for Qdrant
resource fileShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2022-05-01' = {
  name: qdrantShareName
  parent: fileService
  properties: {
    shareQuota: 5
    enabledProtocols: 'SMB'
  }
}

// Container App Environment
resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2022-03-01' = {
  name: containerAppEnvName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
    zoneRedundant: false
  }
}

// Outputs
output keyVaultName string = keyVault.name
output storageAccountName string = storageAccount.name
output queueConnectionString string = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${listKeys(storageAccount.id, storageAccount.apiVersion).keys[0].value};EndpointSuffix=${environment().suffixes.storage}' 
output containerAppEnvId string = containerAppEnvironment.id
output containerAppEnvName string = containerAppEnvironment.name 