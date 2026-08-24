// ─────────────────────────────────────────────────────────────────────────────
// infra/main.bicep
//
// Root template — composes every module for the attendance system:
// networking → database(+replica) → redis / servicebus / storage / acr →
// key vault (secrets composed here) → container apps env + tiers + jobs →
// static web app.
//
// Deploy (see infra/README.md for the full walkthrough):
//   az deployment group create -f infra/main.bicep -p infra/parameters/staging.bicepparam
// ─────────────────────────────────────────────────────────────────────────────

@description('Short environment name, e.g. stg / prd — used in every resource name.')
param environmentName string = 'stg'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Postgres administrator login.')
param postgresAdminLogin string = 'attendance_admin'

@secure()
@description('Postgres administrator password (generate per environment; never commit real values).')
param postgresAdminPassword string

@secure()
@description('JWT signing secret — generate with: python -c "import secrets; print(secrets.token_hex(64))"')
param jwtSecret string

@description('Postgres SKU — Standard_B1ms is free-tier eligible for 12 months on student subscriptions.')
param postgresSkuName string = 'Standard_B1ms'

@allowed(['ZoneRedundant', 'SameZone', 'Disabled'])
param postgresHighAvailability string = 'Disabled'

@description('GPU workload profile requires supported region + quota; disable for staging.')
param enableGpuWorkloadProfile bool = false

@description('Image tag pulled by apps/jobs (Phase 7 CI pushes real tags).')
param imageTag string = 'latest'

param workerMaxReplicas int = 20

param mlMaxReplicas int = 8

param apiMinReplicas int = 1

param apiMaxReplicas int = 10

@description('Tags applied to all resources.')
param tags object = {
  project: 'attendance'
  environment: environmentName
}

var nameSuffix = uniqueString(resourceGroup().id)
var resourceNames = {
  vnet: 'vnet-attendance-${environmentName}'
  postgresPrimary: 'pg-att-${environmentName}-${nameSuffix}'
  servicebus: 'sb-att-${environmentName}-${nameSuffix}'
  storage: 'stattn${environmentName}${nameSuffix}' // lowercase alphanumeric only
  keyvault: 'kv-att-${environmentName}-${nameSuffix}'
  acr: 'acratt${environmentName}${nameSuffix}'
  logAnalytics: 'log-att-${environmentName}-${nameSuffix}'
  identity: 'id-att-${environmentName}-${nameSuffix}'
  containerEnv: 'cae-att-${environmentName}'
}
var databaseName = 'attendance_db'

// ── Identity ─────────────────────────────────────────────────────────────────

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: resourceNames.identity
  location: location
  tags: tags
}

// ── Networking ───────────────────────────────────────────────────────────────

module networking 'modules/networking.bicep' = {
  name: 'networking'
  params: {
    location: location
    tags: tags
  }
}

// ── Observability ────────────────────────────────────────────────────────────

module logAnalytics 'modules/log_analytics.bicep' = {
  name: 'logAnalytics'
  params: {
    workspaceName: resourceNames.logAnalytics
    location: location
    tags: tags
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────

module acr 'modules/acr.bicep' = {
  name: 'acr'
  params: {
    acrName: resourceNames.acr
    location: location
    tags: tags
  }
}

resource acrExisting 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: resourceNames.acr
}

resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acrExisting
  name: guid(acrExisting.id, 'acrpull')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d' // AcrPull
    )
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ── Database ─────────────────────────────────────────────────────────────────

module database 'modules/database.bicep' = {
  name: 'database'
  params: {
    primaryServerName: resourceNames.postgresPrimary
    administratorLogin: postgresAdminLogin
    administratorPassword: postgresAdminPassword
    skuName: postgresSkuName
    highAvailabilityMode: postgresHighAvailability
    publicNetworkAccess: 'Disabled' // never public; migrations run inside the VNet
    delegatedSubnetResourceId: networking.outputs.postgresSubnetId
    privateDnsZoneId: networking.outputs.postgresPrivateDnsZoneId
    databaseName: databaseName
    location: location
    tags: tags
  }
}

// ── Cache / queue / storage ──────────────────────────────────────────────────

module servicebus 'modules/servicebus.bicep' = {
  name: 'servicebus-ref'
  params: {
    namespaceName: resourceNames.servicebus
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    storageAccountName: resourceNames.storage
    location: location
    tags: tags
  }
}

// ── Key Vault (secrets composed from module outputs) ─────────────────────────

var pgFqdn = database.outputs.primaryServerFqdn
// Staging: internal Container App redis (see container_apps.bicep).
// Production: swap for Azure Managed Redis URL here.
var redisUrl = 'redis://attendance-redis:6379/0'

module keyvault 'modules/keyvault.bicep' = {
  name: 'keyvault'
  params: {
    keyVaultName: resourceNames.keyvault
    identityPrincipalId: identity.properties.principalId
    jwtSecret: jwtSecret
    postgresConnectionString: 'postgresql+asyncpg://${postgresAdminLogin}:${postgresAdminPassword}@${pgFqdn}:5432/${databaseName}?sslmode=require'
    postgresSyncConnectionString: 'postgresql+psycopg2://${postgresAdminLogin}:${postgresAdminPassword}@${pgFqdn}:5432/${databaseName}?sslmode=require'
    redisConnectionString: redisUrl
    serviceBusConnectionString: servicebus.outputs.connectionString
    storageConnectionString: storage.outputs.connectionString
    location: location
    tags: tags
  }
}

// ── Container Apps ───────────────────────────────────────────────────────────

module containerApps 'modules/container_apps.bicep' = {
  name: 'containerApps'
  params: {
    environmentName: resourceNames.containerEnv
    location: location
    tags: tags
    logAnalyticsWorkspaceId: logAnalytics.outputs.workspaceId
    infrastructureSubnetId: networking.outputs.appsSubnetId
    enableGpuWorkloadProfile: enableGpuWorkloadProfile
    acrLoginServer: acr.outputs.loginServer
    imageTag: imageTag
    identityResourceId: identity.id
    serviceBusNamespaceName: resourceNames.servicebus
    vaultUri: keyvault.outputs.vaultUri
    apiMinReplicas: apiMinReplicas
    apiMaxReplicas: apiMaxReplicas
    workerMaxReplicas: workerMaxReplicas
    mlMaxReplicas: mlMaxReplicas
    mlInferenceWorkloadProfile: enableGpuWorkloadProfile ? 'gpu' : 'consumption'
  }
}

// ── Frontend ─────────────────────────────────────────────────────────────────
// NOTE: Static Web Apps cannot be created via ARM without a linked git repo,
// so infra/README.md §7 provisions it as a "disconnected" SWA via
// `az staticwebapp create` (no --source); the Phase 7 workflow then deploys
// builds with the SWA deploy token.

// ── Outputs ──────────────────────────────────────────────────────────────────

output apiFqdn string = containerApps.outputs.apiFqdn
output acrLoginServer string = acr.outputs.loginServer
output keyVaultName string = keyvault.outputs.vaultName
output migrationJobName string = containerApps.outputs.migrationJobName
output postgresPrimaryFqdn string = pgFqdn
