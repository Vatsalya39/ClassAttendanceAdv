// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/networking.bicep
//
// VNet layout (production: Postgres and Redis are NOT publicly reachable):
//
//   vnet /16
//   ├── snet-postgres   /24  delegated to Microsoft.DBforPostgreSQL/flexibleServers
//   ├── snet-redis      /24  injected Azure Cache for Redis (Premium)
//   ├── snet-apps       /23  Container Apps environment infrastructure subnet
//   └── snet-endpoints  /26  private endpoints (staging / future services)
//
// A private DNS zone links Postgres privatelink names into the VNet so the
// apps resolve the database privately.
// ─────────────────────────────────────────────────────────────────────────────

@description('Virtual network address prefix (CIDR).')
param vnetAddressPrefix string = '10.10.0.0/16'

@description('Location for all networking resources.')
param location string = resourceGroup().location

@description('Tags applied to every resource.')
param tags object

resource vnet 'Microsoft.Network/virtualNetworks@2023-11-01' = {
  name: 'vnet-${location}-attendance'
  location: location
  tags: tags
  properties: {
    addressSpace: { addressPrefixes: [vnetAddressPrefix] }
  }
}

resource postgresSubnet 'Microsoft.Network/virtualNetworks/subnets@2023-11-01' = {
  parent: vnet
  name: 'snet-postgres'
  properties: {
    addressPrefix: '10.10.1.0/24'
    delegations: [
      {
        name: 'postgres-delegation'
        properties: { serviceName: 'Microsoft.DBforPostgreSQL/flexibleServers' }
      }
    ]
    privateEndpointNetworkPolicies: 'Disabled'
  }
}

resource redisSubnet 'Microsoft.Network/virtualNetworks/subnets@2023-11-01' = {
  parent: vnet
  name: 'snet-redis'
  properties: {
    addressPrefix: '10.10.2.0/24'
    privateEndpointNetworkPolicies: 'Disabled'
  }
}

resource appsSubnet 'Microsoft.Network/virtualNetworks/subnets@2023-11-01' = {
  parent: vnet
  name: 'snet-apps'
  properties: {
    addressPrefix: '10.10.4.0/23'
    privateEndpointNetworkPolicies: 'Disabled'
  }
}

resource endpointsSubnet 'Microsoft.Network/virtualNetworks/subnets@2023-11-01' = {
  parent: vnet
  name: 'snet-endpoints'
  properties: {
    addressPrefix: '10.10.6.0/26'
    privateEndpointNetworkPolicies: 'Disabled'
  }
}

resource postgresPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink.postgres.database.azure.com'
  location: 'global'
  tags: tags
}

resource dnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: postgresPrivateDnsZone
  name: 'link-${vnet.name}'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: { id: vnet.id }
  }
}

output vnetName string = vnet.name
output vnetId string = vnet.id
output postgresSubnetId string = postgresSubnet.id
output redisSubnetId string = redisSubnet.id
output appsSubnetId string = appsSubnet.id
output endpointsSubnetId string = endpointsSubnet.id
output postgresPrivateDnsZoneId string = postgresPrivateDnsZone.id
