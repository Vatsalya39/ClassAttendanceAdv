// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/storage.bicep
//
// Blob Storage for enrollment / session / review-crop images, with lifecycle
// policies matching docs/data-retention.md:
//   session-images  → delete blobs 90 days after last modification
//   review-crops    → delete blobs 30 days after last modification
//   enrollment-images → never auto-deleted (removed on erasure request via
//                       POST /admin/students/{id}/purge-biometrics)
//
// SECURITY: containers are created PRIVATE. The current AzureBlobStorageBackend
// returns plain blob URLs — before production, switch to SAS-signed URLs
// (tracked in the deployment README pre-production checklist).
// ─────────────────────────────────────────────────────────────────────────────

@description('Storage account name (globally unique, lowercase alphanumeric).')
param storageAccountName string

param location string = resourceGroup().location

param tags object

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: { name: 'Standard_GRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false // private containers; SAS support is a pre-prod checklist item
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    accessTier: 'Hot'
    networkAcls: {
      defaultAction: 'Allow' // tighten with private endpoints once SAS flow lands
      bypass: 'AzureServices'
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

resource enrollmentContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'enrollment-images'
  properties: { publicAccess: 'None' }
}

resource sessionsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'session-images'
  properties: { publicAccess: 'None' }
}

resource reviewContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'review-crops'
  properties: { publicAccess: 'None' }
}

resource lifecycle 'Microsoft.Storage/storageAccounts/managementPolicies@2023-05-01' = {
  parent: storage
  name: 'default'
  properties: {
    policy: {
      rules: [
        {
          enabled: true
          name: 'expire-session-images-90d'
          type: 'Lifecycle'
          definition: {
            actions: {
              baseBlob: { delete: { daysAfterModificationGreaterThan: 90 } }
            }
            filters: {
              blobTypes: ['blockBlob']
              prefixMatch: ['session-images/']
            }
          }
        }
        {
          enabled: true
          name: 'expire-review-crops-30d'
          type: 'Lifecycle'
          definition: {
            actions: {
              baseBlob: { delete: { daysAfterModificationGreaterThan: 30 } }
            }
            filters: {
              blobTypes: ['blockBlob']
              prefixMatch: ['review-crops/']
            }
          }
        }
      ]
    }
  }
}

output storageAccountName string = storage.name
output primaryBlobEndpoint string = storage.properties.primaryEndpoints.blob
output connectionString string = 'DefaultEndpointsProtocol=https;AccountName=${storage.name};AccountKey=${storage.listKeys().keys[0].value};EndpointSuffix=${environment().suffixes.storage}'
