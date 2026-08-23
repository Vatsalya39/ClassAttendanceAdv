// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/log_analytics.bicep
//
// Log Analytics workspace backing the Container Apps environment (console
// logs, KEDA scaler events). Retention kept short to control cost; ship to
// longer-term storage later if compliance requires it.
// ─────────────────────────────────────────────────────────────────────────────

@description('Workspace name.')
param workspaceName string

param location string = resourceGroup().location

param tags object

resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: workspaceName
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    features: { searchVersion: 1 }
  }
}

output workspaceId string = workspace.id
output workspaceNameOutput string = workspace.name
