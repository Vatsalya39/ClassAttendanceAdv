// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/database.bicep
//
// Azure Database for PostgreSQL Flexible Server + read replica.
//
// SIZING (staging): Burstable_B1ms (2 vCore burst / 1-2 GiB) — free-tier
// eligible for 12 months on Azure for Students. The roster-scoping rule keeps
// per-query load tiny, so burstable CPU is fine at this scale. The read
// replica was removed for staging; re-add it for production reporting load
// (see git history of this file).
//
// pgvector note: 'vector' is in Azure's extension allowlist; the Alembic
// migration runs CREATE EXTENSION IF NOT EXISTS vector — no Bicep flag needed.
// ─────────────────────────────────────────────────────────────────────────────

@description('Primary server name.')
param primaryServerName string

@description('Admin login for Postgres (NOT used by apps — they use this too today; rotate via Key Vault).')
param administratorLogin string

@secure()
@description('Admin password — provide via parameter file / Key Vault at deploy time.')
param administratorPassword string

@allowed(['14', '15', '16'])
param postgresVersion string = '16'

@description('SKU name, e.g. Standard_B1ms (burstable, free-tier eligible) or Standard_D4ds_v5 (production).')
param skuName string = 'Standard_B1ms'

@description('Storage size in GB (32 is the Burstable minimum).')
param storageSizeGB int = 32

@description('Whether the primary accepts public network traffic. Keep Disabled in production.')
@allowed(['Enabled', 'Disabled'])
param publicNetworkAccess string = 'Disabled'

@description('Zone-redundant HA for production; None for staging cost savings.')
@allowed(['ZoneRedundant', 'SameZone', 'Disabled'])
param highAvailabilityMode string = 'Disabled'

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

resource primary 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: primaryServerName
  location: location
  tags: tags
  sku: {
    name: skuName
    tier: startsWith(skuName, 'Standard_B') ? 'Burstable' : 'GeneralPurpose'
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
    // Omitting the block entirely disables HA — some 2024+ API versions
    // reject an explicit 'Disabled'.
    highAvailability: highAvailabilityMode == 'Disabled' ? null : {
      mode: highAvailabilityMode
    }
    maintenanceWindow: maintenanceWindow
  }
}

// NOTE (pgvector allow-listing): the configurations child resource rejects
// ARM PUTs ("Source is invalid"), so allow-listing is a MANUAL post-provision
// step — see infra/README.md §5b:
//   az postgres flexible-server parameter set \
//     -g <rg> -s <server> -n azure.extensions --value vector
resource attendanceDb 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: primary
  name: databaseName
  properties: {
    charset: 'utf8'
    collation: 'en_US.utf8'
  }
}

output primaryServerFqdn string = primary.properties.fullyQualifiedDomainName
output primaryServerName string = primary.name
