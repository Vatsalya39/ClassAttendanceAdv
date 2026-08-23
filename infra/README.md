# Infra — Azure IaC (Bicep, Phase 6)

Infrastructure as Code for the production Azure deployment. **Bicep** was
chosen over Terraform because it is first-party for Azure (day-one support
for new Container Apps / GPU workload-profile features), needs no state file
to drift or secure, and supports `az what-if` previews natively.

## Layout

```
infra/
├── main.bicep                  # Orchestrator — composes every module
├── modules/
│   ├── networking.bicep        # VNet, delegated subnets, private DNS zone
│   ├── database.bicep          # PostgreSQL Flexible Server + read replica
│   ├── redis.bicep             # Azure Cache for Redis (Premium = VNet-injected)
│   ├── servicebus.bicep        # Namespace + capture-jobs queue (DLQ built in)
│   ├── storage.bicep           # Blob containers + 90d/30d lifecycle rules
│   ├── keyvault.bicep          # RBAC vault + all runtime secrets
│   ├── acr.bicep               # Container registry (identity-based pull)
│   ├── log_analytics.bicep     # Workspace behind Container Apps logs
│   ├── container_apps.bicep    # Environment + api/worker/ml tiers + 2 jobs
│   └── static_web_app.bicep    # Frontend hosting
├── parameters/
│   ├── staging.bicepparam      # cheap: no GPU profile, Standard Redis
│   └── production.bicepparam   # Premium Redis, ZoneRedundant HA, GPU tier
└── scripts/
    └── create_partitions.sql   # idempotent monthly-partition maintenance
```

## What gets deployed

| Resource | Sizing (initial) | Why |
|---|---|---|
| PostgreSQL Flexible Server | `Standard_D4ds_v5` 4 vCore/16 GiB, private VNet, + read replica | Roster-scoping keeps the HNSW index tiny (~250k vectors ≈ <1 GB → fits in RAM); 4 vCores absorb the start-of-period write burst. Replica isolates reporting scans. **Re-size after Phase 8 burst tests.** |
| Azure Cache for Redis | Prod `Premium_P1` (VNet), staging `Standard_C1` | Premium is required for VNet injection. `volatile-ttl` eviction matches our TTL'd job-status keys. |
| Service Bus namespace | Standard, queue `capture-jobs`, lock 5 min, MaxDeliveryCount 5 → DLQ | Matches `backend/core/queue.py`: nack() dead-letters with reason; redelivery ≤5 then DLQ. Duplicate detection on (job_id message id). |
| Container Apps env | consumption profile (+ optional GPU profile `NC24ads-A100-v4`) | api scales on HTTP concurrency; worker & ml-inference scale on **queue depth via KEDA** (`azure-servicebus` scaler). |
| api app | min 1-2, max 10-15, public ingress :8000 | `/health` liveness probe. |
| worker (CPU) | **min 0**, max 20; 1 replica / 5 queued msgs | Scales to zero when idle. |
| ml-inference (GPU) | **min 0**, max 8; 1 replica / 3 queued msgs | GPU nodes drain when idle — cost control. Falls back to the consumption profile when the region/quota lacks GPU support. |
| Key Vault | RBAC-only; apps pull via managed identity | No connection strings live in Bicep state, env files or CI. |
| Static Web App | Free SKU | Phase 7 workflow deploys the Next.js build. |

**Tier note:** today one worker image performs ML inference *and* matching,
so both worker tiers consume the same queue. Production shifts bulk capture
processing to the GPU tier by keeping the CPU worker ceiling small; splitting
matching out of `backend/worker/main.py` into a separate inference service is
a deliberate future refactor.

## Deploy from a clean subscription

```bash
# 0. Variables
RG=rg-attendance-stg
LOC=eastus2            # pick a region supporting Container Apps (+GPU if enabled)
az group create -n $RG -l $LOC

# 1. Secrets (never commit these values)
export PG_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
export JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(64))')"

# 2. Provision everything (~15-25 min; Postgres replica is the slow part)
az deployment group create \
  -g $RG \
  -f infra/main.bicep \
  -p infra/parameters/staging.bicepparam \
  -p postgresAdminPassword="$PG_PASSWORD" \
  -p jwtSecret="$JWT_SECRET"

# 3. Capture outputs
API_FQDN=$(az deployment group show -g $RG -n main \
  --query properties.outputs.apiFqdn.value -o tsv)
ACR=$(az deployment group show -g $RG -n main \
  --query properties.outputs.acrLoginServer.value -o tsv)
MIGRATION_JOB=$(az deployment group show -g $RG -n main \
  --query properties.outputs.migrationJobName.value -o tsv)

# 4. Build & push the backend image (api + worker share one image;
#    ml-inference uses the same image on the GPU workload profile)
az acr login -n $ACR
docker build --target worker -t $ACR/backend:latest .
docker push $ACR/backend:latest

# 5. Run migrations INSIDE the VNet (GitHub runners cannot reach private Postgres)
az containerapp job start -g $RG -n $MIGRATION_JOB
az containerapp job execution list -g $RG -n $MIGRATION_JOB -o table

# 6. Bootstrap the first super admin (runs inside the api container)
az containerapp exec -g $RG --name attendance-api \
  --command "python scripts/create_superadmin.py --username admin --password $(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')"

# 7. Point the frontend at the API and deploy it (Phase 7 wires CI to do this)
FRONTEND_HOST=$(az deployment group show -g $RG -n main \
  --query properties.outputs.frontendHostname.value -o tsv)
az staticwebapp app settings set -n swa-att-stg-* \
  --setting-names NEXT_PUBLIC_API_URL="https://$API_FQDN"
cd frontend && npx @azure/static-web-apps-cli deploy   # or the GitHub workflow
```

Preview changes before any redeploy:

```bash
az deployment group what-if -g $RG -f infra/main.bicep \
  -p infra/parameters/staging.bicepparam
```

## Partition maintenance

`job-partition-maintenance` runs daily at 03:00 UTC (`psql` executing
`scripts/create_partitions.sql`) and pre-creates the current month + next 4
months of `attendance` partitions plus the DEFAULT catch-all — idempotent,
mirrors `backend/db/partitions.py`. Nothing else schedules partitions.

## Raising ceilings beyond ~50k students

The KEDA rules convert queue depth into replicas:

```
replicas_needed ≈ peak_jobs_per_minute × avg_seconds_per_job / 60 / jobs_per_replica_concurrency
```

1. Run `scripts/bench_throughput.py` on one GPU replica to measure
   `faces/sec` → derive `avg_seconds_per_job` for your typical classroom photo.
2. Raise ceilings (they are just parameters — no code change):
   ```bash
   az containerapp update -g $RG -n attendance-worker \
     --min-replicas 0 --max-replicas 40
   az containerapp update -g $RG -n attendance-ml-inference \
     --min-replicas 0 --max-replicas 20
   ```
3. If Postgres CPU saturates first (watch `txid`/CPU metrics on the primary),
   move reporting fully onto the replica connection string, then scale the
   primary SKU up one tier (`D4ds_v5 → D8ds_v5`). Because matching is always
   roster-scoped, DB size grows linearly with enrollment but per-query load
   does not — CPU pressure comes from write bursts only.

## Pre-production checklist

- [ ] **Blob SAS**: containers are created private; switch
      `AzureBlobStorageBackend` to SAS-signed URLs before go-live so review
      crops render without public access.
- [ ] GPU quota + region support for `NC24ads-A100-v4` (preview feature) —
      otherwise deploy with `enableGpuWorkloadProfile=false`.
- [ ] Set `CORS_ORIGINS` to the deployed SWA hostname (Key Vault secret or
      container env override).
- [ ] Rotate `postgresAdminPassword` post-deploy; consider Entra-ID (passwordless)
      auth for Flexible Server.
- [ ] Enable `disableLocalAuth` on Service Bus once apps use managed identity
      fully.
- [ ] Alerts: Service Bus DLQ depth > 0, queue age, Postgres storage %,
      Redis evictions.
