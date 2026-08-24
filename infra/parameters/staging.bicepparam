// ─────────────────────────────────────────────────────────────────────────────
// staging.bicepparam — cheap, GPU-off, no HA.
// Replace the two secure placeholders before deploying (see infra/README.md).
// ─────────────────────────────────────────────────────────────────────────────
using '../main.bicep'

param environmentName = 'stg'
param postgresAdminPassword = 'REPLACE_ME_generate_and_store_securely'
param jwtSecret = 'REPLACE_ME_run_python_c_import_secrets_token_hex_64'

param postgresSkuName = 'Standard_B1ms' // free-tier eligible 12 months
param postgresHighAvailability = 'Disabled'
param enableGpuWorkloadProfile = false // no GPU quota needed for staging
param imageTag = 'latest'
param apiMinReplicas = 0 // scale-to-zero staging
param apiMaxReplicas = 5
param workerMaxReplicas = 3
param mlMaxReplicas = 2
