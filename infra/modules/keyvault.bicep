// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/keyvault.bicep
//
// Key Vault (RBAC-authorized) holding every runtime secret. The Container
// Apps user-assigned managed identity gets "Key Vault Secrets User" so apps
// pull secrets at startup without any connection string in Bicep state or CI.
//
// Secret values are passed in from main.bicep, which sources them either
// from module outputs (connection strings) or parameters (JWT secret).
// ─────────────────────────────────────────────────────────────────────────────

@description('Key Vault name (globally unique).')
param keyVaultName string

@description('User-assigned managed identity principal id that may read secrets.')
param identityPrincipalId string

@secure()
param jwtSecret string

@secure()
param postgresConnectionString string

@secure()
param postgresSyncConnectionString string

@secure()
param redisConnectionString string

@secure()
param serviceBusConnectionString string

@secure()
param storageConnectionString string

param location string = resourceGroup().location

param tags object

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
  }
}

var secrets = [
  { name: 'Database--ConnectionString', value: postgresConnectionString }
  { name: 'Database--SyncConnectionString', value: postgresSyncConnectionString }
  { name: 'Redis--Url', value: redisConnectionString }
  { name: 'ServiceBus--ConnectionString', value: serviceBusConnectionString }
  { name: 'Storage--ConnectionString', value: storageConnectionString }
  { name: 'JwtSecret', value: jwtSecret }
]

resource secretResources 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = [for s in secrets: {
  parent: vault
  name: s.name
  properties: {
    value: s.value
    contentType: 'text/plain'
  }
}]

resource secretsUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, identityPrincipalId, 'Key Vault Secrets User')
  scope: vault
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '4633458b-17de-408a-b874-0445c86b69e6' // Key Vault Secrets User
    )
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

output vaultUri string = vault.properties.vaultUri
output vaultName string = vault.name
