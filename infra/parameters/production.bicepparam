// ─────────────────────────────────────────────────────────────────────────────
// production.bicepparam — Premium Redis (VNet-injected), zone-redundant
// Postgres HA, GPU workload profile enabled.
// Replace the two secure placeholders before deploying.
// ─────────────────────────────────────────────────────────────────────────────
using '../main.bicep'

param environmentName = 'prd'

@secure()
param postgresAdminPassword = 'REPLACE_ME_generate_and_store_securely'

@secure()
param jwtSecret = 'REPLACE_ME_run_python_c_import_secrets_token_hex_64'

param postgresSkuName = 'Standard_D4ds_v5' // 4 vCore / 16 GiB — see modules/database.bicep rationale
param postgresHighAvailability = 'ZoneRedundant'
param redisSkuName = 'Premium_P1'
param enableGpuWorkloadProfile = true // requires region + quota; see README
param imageTag = 'latest'
param apiMinReplicas = 2
param apiMaxReplicas = 15
param workerMaxReplicas = 20
param mlMaxReplicas = 8
