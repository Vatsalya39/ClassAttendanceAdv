// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/redis.bicep
//
// Azure Cache for Redis — roster/session caching, capture-job status,
// rate limiting.
//
//   production: Premium_P1, VNet-injected (not publicly reachable)
//   staging:    Standard_C1, public endpoint (acceptable for staging only)
// ─────────────────────────────────────────────────────────────────────────────

@description('Redis cache name (globally unique).')
param redisName string

@allowed(['Standard_C1', 'Premium_P1'])
param skuName string = 'Premium_P1'

@description('VNet subnet for Premium injection. Required when skuName is Premium; empty for Standard.')
param subnetId string = ''

param location string = resourceGroup().location

param tags object

var isPremium = startsWith(skuName, 'Premium')

resource redis 'Microsoft.Cache/redis@2024-03-01' = {
  name: redisName
  location: location
  tags: tags
  properties: {
    sku: {
      name: isPremium ? 'Premium' : 'Standard'
      family: isPremium ? 'P' : 'C'
      capacity: isPremium ? 1 : 1
    }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
    redisConfiguration: {
      'maxmemory-policy': 'volatile-ttl' // job status keys carry TTLs — evict those first
    }
    subnetId: isPremium ? subnetId : null
    publicNetworkAccess: isPremium ? 'Disabled' : 'Enabled'
  }
}

output redisHostName string = redis.properties.hostName
#disable-next-line outputs-should-not-contain-secrets
output redisPrimaryKey string = redis.listKeys().primaryKey // consumed only by main.bicep → Key Vault
