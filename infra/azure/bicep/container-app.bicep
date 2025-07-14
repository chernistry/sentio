@description('The location for all resources.')
param location string = resourceGroup().location

@description('Container App Environment ID')
param containerAppEnvId string

@description('Container image to deploy')
param containerImage string = 'ghcr.io/username/sentio-api:latest'

@description('Container app name')
param containerAppName string = 'sentio-api'

@description('Container app CPU cores')
param containerCpuCores string = '0.5'

@description('Container app memory')
param containerMemory string = '1.0Gi'

@description('Container registry server')
param containerRegistryServer string = 'ghcr.io'

@description('Container registry username')
param containerRegistryUsername string = ''

@description('Container registry password')
@secure()
param containerRegistryPassword string = ''

@description('Environment variables for the container')
param environmentVariables array = []

@description('Secret environment variables for the container')
param secretEnvironmentVariables array = []

// Container App
resource containerApp 'Microsoft.App/containerApps@2022-03-01' = {
  name: containerAppName
  location: location
  properties: {
    managedEnvironmentId: containerAppEnvId
    configuration: {
      ingress: {
        external: true
        targetPort: 80
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: !empty(containerRegistryUsername) && !empty(containerRegistryPassword) ? [
        {
          server: containerRegistryServer
          username: containerRegistryUsername
          passwordSecretRef: 'registry-password'
        }
      ] : []
      secrets: !empty(containerRegistryPassword) ? [
        {
          name: 'registry-password'
          value: containerRegistryPassword
        }
        ,{
          name: 'openrouter-api-key'
          value: 'your-openrouter-api-key'
        }
      ] : []
    }
    template: {
      containers: [
        {
          name: containerAppName
          image: containerImage
          resources: {
            cpu: json(containerCpuCores)
            memory: containerMemory
          }
          env: concat(
            environmentVariables,
            secretEnvironmentVariables,
            [
              {
                name: 'OPENROUTER_API_KEY'
                secretRef: 'openrouter-api-key'
              }
            ]
          )
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
        rules: [
          {
            name: 'http-scale'
            http: {
              metadata: {
                concurrentRequests: '1'
              }
            }
          }
        ]
      }
    }
  }
}

// Outputs
output containerAppFqdn string = containerApp.properties.configuration.ingress.fqdn 