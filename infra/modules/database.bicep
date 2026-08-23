// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/database.bicep
//
// Azure Database for PostgreSQL Flexible Server + read replica.
//
// SIZING RATIONALE (initial estimate — refine after Phase 8 burst tests):
//   SKU GeneralPurpose D4ds_v5 (4 vCore / 16 GiB). The roster-scoping rule
//   means every match query touches ~60-70 students × 3-5 embeddings, so the
//   HNSW index for 50k students (~250k vectors ≈ 0.5-1 GB) sits entirely in
//   RAM and per-query latency is independent of total enrollment. 4 vCores
//   absorb the "start of period" write burst (hundreds of sections capturing
//   within a few minutes) without overprovisioning.
//   The replica serves reporting/analytics so those scans never contend with
//   live attendance writes on the primary.
//
// pgvector note: 'vector' is in Azure's extension allowlist; the Alembic
// migration runs CREATE EXTENSION IF NOT EXISTS vector — no Bicep flag needed.
// ─────────────────────────────────────────────────────────────────────────────

@description('Primary server name.')
param primaryServerName string

@description('Read-replica server name.')
param replicaServerName string

@description('Admin login for Postgres (NOT used by apps — they use this too today; rotate via Key Vault).')
param administratorLogin string

@secure()
@description('Admin password — provide via parameter file / Key Vault at deploy time.')
param administratorPassword string

@allowed(['14', '15', '16'])
param postgresVersion string = '16'

@description('SKU name, e.g. Standard_D4ds_v5 (4 vCore / 16 GiB).')
param skuName string = 'Standard_D4ds_v5'

@description('Storage size in GB.')
param storageSizeGB int = 64

@description('Whether the primary accepts public network traffic. Keep Disabled in production.')
@allowed(['Enabled', 'Disabled'])
param publicNetworkAccess string = 'Disabled'

@description('Zone-redundant HA for production; None for staging cost savings.')
@allowed(['ZoneRedundant', 'SameZone', 'None'])
param highAvailabilityMode string = 'None'

param delegatedSubnetResourceId string

param privateDnsZoneId string

param databaseName string = 'attendance_db'

param location string = resourceGroup().location

param tags object

var storageAutoGrow = 'Enabled'
var maintenanceWindow = {
  customWindow: 'Enabled'
  dayOfWeek: 6 // Saturday
  startHour: 2
  startMinute: 0
}

resource primary 'Microsoft.DBforPostgreSQL/flexibleServers@2024-03-01' = {
  name: primaryServerName
  location: location
  tags: tags
  sku: {
    name: skuName
    tier: 'GeneralPurpose'
  }
  properties: {
    version: postgresVersion
    administratorLogin: administratorLogin
    administratorLoginPassword: administratorPassword
    network: {
      delegatedSubnetResourceId: delegatedSubnetResourceId
      privateDnsZoneArmResourceId: privateDnsZoneId
      publicNetworkAccess: publicNetworkAccess
    }
    storage: {
      storageSizeGB: storageSizeGB
      autoGrow: storageAutoGrow
    }
    backup: {
      backupRetentionDays: 14
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: highAvailabilityMode
    }
    maintenanceWindow: maintenanceWindow
  }
}

resource attendanceDb 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-03-01' = {
  parent: primary
  name: databaseName
  properties: {
    charset: 'utf8'
    collation: 'en_US.utf8'
  }
}

// ── Read replica: same SKU, for reporting/analytics isolation. ───────────────
resource replica 'Microsoft.DBforPostgreSQL/flexibleServers@2024-03-01' = {
  name: replicaServerName
  location: location
  tags: tags
  sku: {
    name: skuName
    tier: 'GeneralPurpose'
  }
  properties: {
    createMode: 'Replica'
    sourceServerResourceId: primary.id
    version: postgresVersion
    network: {
      delegatedSubnetResourceId: delegatedSubnetResourceId
      privateDnsZoneArmResourceId: privateDnsZoneId
      publicNetworkAccess: publicNetworkAccess
    }
    storage: {
      autoGrow: storageAutoGrow
    }
  }
}

output primaryServerFqdn string = primary.properties.fullyQualifiedDomainName
output replicaServerFqdn string = replica.properties.fullyQualifiedDomainName
output primaryServerName string = primary.name
