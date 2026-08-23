// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/static_web_app.bicep
//
// Azure Static Web Apps hosting the Next.js frontend.
// The GitHub Actions workflow (Phase 7) deploys the built app; here we only
// provision the resource and its API token placeholder.
// ─────────────────────────────────────────────────────────────────────────────

@description('Static Web App name.')
param staticSiteName string

param location string = resourceGroup().location

param tags object

resource swa 'Microsoft.Web/staticSites@2023-12-01' = {
  name: staticSiteName
  location: location // SWA apps are served globally; this is the metadata region
  tags: tags
  properties: {
    // Repository link is configured by the Phase 7 workflow instead of Bicep,
    // so provisioning works from a clean subscription without a git remote.
    repositoryUrl: ''
    branch: 'main'
    buildProperties: {
      appLocation: 'frontend'
      outputLocation: '.next'
      appArtifactLocation: '.next'
    }
  }
  sku: {
    name: 'Free'
    tier: 'Free'
  }
}

@secure()
output apiKey string = swa.listSecrets().properties.apiKey

output defaultHostname string = swa.properties.defaultHostname
