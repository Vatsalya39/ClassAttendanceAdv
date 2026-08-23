// ─────────────────────────────────────────────────────────────────────────────
// staging.bicepparam — cheap, GPU-off, no HA.
// Replace the two secure placeholders before deploying (see infra/README.md).
// ─────────────────────────────────────────────────────────────────────────────
using '../main.bicep'

param environmentName = 'stg'
param postgresAdminPassword = 'REPLACE_ME_generate_and_store_securely'
param jwtSecret = 'REPLACE_ME_run_python_c_import_secrets_token_hex_64'

param postgresSkuName = 'Standard_D2ds_v5'
param postgresHighAvailability = 'None'
param redisSkuName = 'Standard_C1'
param enableGpuWorkloadProfile = false // no GPU quota needed for staging
param imageTag = 'latest'
param apiMinReplicas = 1
param apiMaxReplicas = 5
param workerMaxReplicas = 10
param mlMaxReplicas = 4
