// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/acr.bicep
//
// Azure Container Registry (Basic) for api / worker / ml images.
// Admin user stays DISABLED — Phase 7 CI pushes via OIDC/service principal,
// Container Apps pull via the managed identity (AcrPull assigned in main).
// ─────────────────────────────────────────────────────────────────────────────

@description('Registry name (globally unique, alphanumeric).')
param acrName string

@allowed(['Basic', 'Standard'])
param sku string = 'Standard'

param location string = resourceGroup().location

param tags object

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  tags: tags
  sku: { name: sku }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled' // GitHub Actions push; lock down with runners in VNet if required
  }
}

output loginServer string = registry.properties.loginServer
output registryId string = registry.id
