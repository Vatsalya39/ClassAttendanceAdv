// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/servicebus.bicep
//
// The Service Bus RP rejects template-driven PUTs on some subscriptions
// (empty BadRequest) while CLI creation works — so the namespace + queue are
// provisioned ONCE via az cli (see infra/README.md §0b) and referenced here
// as EXISTING resources purely to emit the connection string into Key Vault.
//
// Queue settings applied by the CLI (Basic tier):
//   lockDuration PT5M · maxDeliveryCount 5 (→ DLQ) · message TTL P1D
//   dead-lettering on expiration enabled
// Idempotency does NOT rely on SB duplicate detection (unavailable on Basic):
// workers claim jobs via Redis SET NX; attendance writes are ON CONFLICT.
// ─────────────────────────────────────────────────────────────────────────────

@description('Service Bus namespace name (must already exist — created by az cli).')
param namespaceName string

param namespaceResourceGroupName string = resourceGroup().name

resource ns 'Microsoft.ServiceBus/namespaces@2024-01-01' existing = {
  name: namespaceName
  scope: resourceGroup(namespaceResourceGroupName)
}

// Connection strings live on authorization RULES — namespace-level listKeys
// does not exist and fails with an opaque BadRequest.
resource rootRule 'Microsoft.ServiceBus/namespaces/authorizationRules@2024-01-01' existing = {
  parent: ns
  name: 'RootManageSharedAccessKey'
}

#disable-next-line outputs-should-not-contain-secrets
output connectionString string = rootRule.listKeys().primaryConnectionString // consumed only by main.bicep → Key Vault
output serviceBusNamespaceName string = ns.name
