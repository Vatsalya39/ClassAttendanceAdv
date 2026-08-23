// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/servicebus.bicep
//
// Azure Service Bus namespace + the capture-jobs queue.
//
//   LockDuration 5 min  — a classroom photo takes seconds-to-minutes on GPU;
//                         the lock must outlive slow processing.
//   MaxDeliveryCount 5  — after 5 failed deliveries (nack) the message lands
//                         in the dead-letter sub-queue automatically; the
//                         worker's nack() dead-letters immediately with a
//                         reason, matching backend/core/queue.py.
//   MessageTTL 1 day    — stale jobs expire into the DLQ instead of looping.
//
// KEDA reads ActiveMessageCount on this queue to scale both worker tiers.
// ─────────────────────────────────────────────────────────────────────────────

@description('Service Bus namespace name (globally unique).')
param namespaceName string

@description('Queue name — keep in sync with QUEUE_NAME app setting.')
param queueName string = 'capture-jobs'

param location string = resourceGroup().location

param tags object

resource sbns 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: namespaceName
  location: location
  tags: tags
  sku: {
    name: 'Standard'
    tier: 'Standard'
  }
  properties: {
    minimumTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled' // reachable by GitHub runners for smoke tests; namespace keys live in Key Vault
    disableLocalAuth: false // connection-string auth used by the apps via Key Vault
  }
}

resource queue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: sbns
  name: queueName
  properties: {
    lockDuration: 'PT5M'
    maxDeliveryCount: 5
    defaultMessageTimeToLive: 'P1D'
    deadLetteringOnMessageExpiration: true
    requiresDuplicateDetection: true
    duplicateDetectionHistoryTimeWindow: 'PT10M'
    maxSizeInMegabytes: 2048
  }
}

output serviceBusNamespaceName string = sbns.name
output queueName string = queue.name
#disable-next-line outputs-should-not-contain-secrets
output connectionString string = sbns.listKeys().primaryConnectionString // consumed only by main.bicep → Key Vault
