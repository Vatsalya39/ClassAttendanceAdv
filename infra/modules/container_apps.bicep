// ─────────────────────────────────────────────────────────────────────────────
// infra/modules/container_apps.bicep
//
// Azure Container Apps environment hosting three compute tiers plus two jobs:
//
//   api           CPU consumption profile · public ingress :8000
//                 scales on HTTP concurrency (built-in rule).
//   worker        CPU consumption profile · queue consumer, NO ingress.
//                 KEDA azure-servicebus scaler: one new replica per 5 queued
//                 messages. min=0 → scales to zero when idle.
//   ml-inference  GPU workload profile (fallback: consumption when the
//                 subscription/region lacks GPU quota) · same queue, scaled
//                 more aggressively (one replica per 3 messages) because each
//                 job is detection+embedding heavy. min=0 controls idle cost.
//
//   job-migrations  Manual-trigger job running `alembic upgrade head` INSIDE
//                   the VNet (GitHub runners cannot reach private Postgres).
//   job-partitions  Scheduled daily job pre-creating next months of
//                   attendance partitions via scripts/create_partitions.sql.
//
// NOTE on tier responsibilities (documented decision): today one worker image
// performs ML inference AND matching. Staging without GPU quota lets CPU
// workers process everything; production shifts bulk capture processing to
// the GPU tier by keeping cpu-worker maxReplicas small. Splitting matching
// out of the worker is a deliberate future refactor, tracked in README.
// ─────────────────────────────────────────────────────────────────────────────

@description('Container Apps environment name.')
param environmentName string

param location string = resourceGroup().location

param tags object

param logAnalyticsWorkspaceId string

param infrastructureSubnetId string

@description('Deploy a GPU workload profile? Requires supported region + subscription quota.')
param enableGpuWorkloadProfile bool = true

@description('GPU workload profile type (region-dependent, preview feature).')
param gpuWorkloadProfileType string = 'NC24ads-A100-v4'

@description('ACR login server, e.g. myregistry.azurecr.io.')
param acrLoginServer string

@description('Image tag pulled by every app/job.')
param imageTag string = 'latest'

param identityResourceId string

@description('Service Bus namespace short name — required by the KEDA azure-servicebus scaler.')
param serviceBusNamespaceName string

@description('Comma-separated CORS origins for the API (frontend URL).')
param corsOrigins string = 'http://localhost:3000'

param vaultUri string

@description('Minimum API replicas — 0 in staging: cold start on first request, zero idle cost.')
param apiMinReplicas int = 0
param apiMaxReplicas int = 10

@description('CPU worker ceiling — initial estimate until Phase 8 burst tests.')
param workerMaxReplicas int = 20

@description('GPU ml-inference ceiling — sized for worst-case concurrent sections; refine after Phase 8.')
param mlMaxReplicas int = 8

@allowed(['consumption', 'gpu'])
param mlInferenceWorkloadProfile string = 'gpu'

// ── Environment ──────────────────────────────────────────────────────────────

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: reference(logAnalyticsWorkspaceId, '2023-09-01').customerId
        sharedKey: listKeys(logAnalyticsWorkspaceId, '2023-09-01').primarySharedKey
      }
    }
    vnetConfiguration: {
      internal: false // api exposes public ingress through the environment
      infrastructureSubnetId: infrastructureSubnetId
    }
    workloadProfiles: enableGpuWorkloadProfile
      ? [
          { name: 'consumption', workloadProfileType: 'Consumption' }
          {
            name: 'gpu'
            workloadProfileType: gpuWorkloadProfileType
            minimumCount: 0 // GPU nodes scale away when idle — cost control
            maximumCount: 3
          }
        ]
      : [{ name: 'consumption', workloadProfileType: 'Consumption' }]
  }
}

var kvSecrets = [
  { ref: 'database-url', secretName: 'Database--ConnectionString' }
  { ref: 'database-url-sync', secretName: 'Database--SyncConnectionString' }
  { ref: 'jwt-secret', secretName: 'JwtSecret' }
  { ref: 'redis-url', secretName: 'Redis--Url' }
  { ref: 'servicebus-connection-string', secretName: 'ServiceBus--ConnectionString' }
  { ref: 'storage-connection-string', secretName: 'Storage--ConnectionString' }
]

var trimmedVaultUri = endsWith(vaultUri, '/') ? substring(vaultUri, 0, length(vaultUri) - 1) : vaultUri

var commonSecrets = [for s in kvSecrets: {
  name: s.ref
  keyVaultUrl: '${trimmedVaultUri}/secrets/${s.secretName}'
  identity: identityResourceId
}]

// Settings.Environment expects development|staging|production; resource
// naming uses short codes (stg/prd) — translate here.
var appEnvironmentMap = {
  dev: 'development'
  stg: 'staging'
  prd: 'production'
  development: 'development'
  staging: 'staging'
  production: 'production'
}
var appEnvironment = appEnvironmentMap[tags.environment]

var commonEnv = [
  { name: 'QUEUE_BACKEND', value: 'servicebus' }
  { name: 'STORAGE_BACKEND', value: 'azure_blob' }
  { name: 'ENVIRONMENT', value: appEnvironment }
  { name: 'DB_SSL', value: 'true' }
  { name: 'LOG_LEVEL', value: 'INFO' }
  { name: 'FACE_MATCH_THRESHOLD', value: '0.45' } // recalibrate in Phase 8
  { name: 'AZURE_KEYVAULT_URI', value: vaultUri }
]

var registryBlock = [
  {
    server: acrLoginServer
    identity: identityResourceId // AcrPull assigned in main.bicep
  }
]

// ── API tier ─────────────────────────────────────────────────────────────────

resource apiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'attendance-api'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identityResourceId}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    workloadProfileName: 'consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: commonSecrets
      registries: registryBlock
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: '${acrLoginServer}/attendance-api:${imageTag}'
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: concat(
            commonEnv,
            [
              { name: 'DATABASE_URL', secretRef: 'database-url' }
              { name: 'DATABASE_URL_SYNC', secretRef: 'database-url-sync' }
              { name: 'AZURE_STORAGE_CONNECTION_STRING', secretRef: 'storage-connection-string' }
              { name: 'JWT_SECRET_KEY', secretRef: 'jwt-secret' }
              { name: 'REDIS_URL', secretRef: 'redis-url' }
              { name: 'SERVICE_BUS_CONNECTION_STRING', secretRef: 'servicebus-connection-string' }
              { name: 'CORS_ORIGINS', value: corsOrigins }
            ]
          )
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000, httpHeaders: [] }
              initialDelaySeconds: 10
              periodSeconds: 15
            }
          ]
        }
      ]
      scale: {
        minReplicas: apiMinReplicas
        maxReplicas: apiMaxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: { concurrentRequests: '50' }
            }
          }
        ]
      }
    }
  }
}


// ── Internal Redis (staging) ─────────────────────────────────────────────────
// Classic Azure Cache for Redis is retired for new creations and its
// replacement (Azure Managed Redis) is not available in centralindia.
// For staging we run Redis as an always-on internal container — reachable
// from every tier as `attendance-redis` via the environment's internal DNS.
// Production should deploy Azure Managed Redis in a supported region.
resource redisApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'attendance-redis'
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: env.id
    workloadProfileName: 'consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: false
        targetPort: 6379
        transport: 'tcp'
      }
    }
    template: {
      containers: [
        {
          name: 'redis'
          image: 'redis:7.2-alpine'
          command: ['redis-server', '--save', '60', '1', '--loglevel', 'warning']
          resources: { cpu: json('0.5'), memory: '1Gi' }
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 1 } // always-on cache
    }
  }
}

// ── CPU worker tier ──────────────────────────────────────────────────────────

var servicebusScaleMetadata = {
  queueName: 'capture-jobs'
  namespace: serviceBusNamespaceName
  messageCount: '5' // one replica per 5 pending jobs
  cloud: 'AzurePublicCloud'
}

var servicebusScaleAuth = [
  {
    triggerParameter: 'connection'
    secretRef: 'servicebus-connection-string'
  }
]

resource workerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'attendance-worker'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identityResourceId}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    workloadProfileName: 'consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: commonSecrets
      registries: registryBlock
      // No ingress — pure queue consumer.
    }
    template: {
      containers: [
        {
          name: 'worker'
          image: '${acrLoginServer}/attendance-worker-ml:${imageTag}'
          command: ['python', '-m', 'backend.worker.main']
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: concat(
            commonEnv,
            [
              { name: 'DATABASE_URL', secretRef: 'database-url' }
              { name: 'DATABASE_URL_SYNC', secretRef: 'database-url-sync' }
              { name: 'AZURE_STORAGE_CONNECTION_STRING', secretRef: 'storage-connection-string' }
              { name: 'JWT_SECRET_KEY', secretRef: 'jwt-secret' }
              { name: 'REDIS_URL', secretRef: 'redis-url' }
              { name: 'SERVICE_BUS_CONNECTION_STRING', secretRef: 'servicebus-connection-string' }
            ]
          )
        }
      ]
      scale: {
        minReplicas: 0 // scale to zero when the queue is empty
        maxReplicas: workerMaxReplicas
        rules: [
          {
            name: 'queue-depth'
            custom: {
              type: 'azure-servicebus'
              metadata: servicebusScaleMetadata
              auth: servicebusScaleAuth
            }
          }
        ]
      }
    }
  }
}

// ── GPU ml-inference tier ────────────────────────────────────────────────────

// GPU tier scales faster: one replica per 3 pending jobs.
var mlScaleMetadata = union(servicebusScaleMetadata, { messageCount: '3' })

resource mlInferenceApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'attendance-ml-inference'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identityResourceId}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    workloadProfileName: mlInferenceWorkloadProfile // 'gpu' or fallback 'consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: commonSecrets
      registries: registryBlock
      ingress: {
        external: false // internal-only; health probe reachable inside the env
        targetPort: 8000
        transport: 'auto'
      }
    }
    template: {
      containers: [
        {
          name: 'ml-inference'
          image: enableGpuWorkloadProfile ? '${acrLoginServer}/attendance-worker-gpu:${imageTag}' : '${acrLoginServer}/attendance-worker-ml:${imageTag}'
          command: ['python', '-m', 'backend.worker.main']
          resources: {
            cpu: json('2.0')
            memory: '4Gi' // buffalo_l models ≈ 600 MB. 4Gi matches Azure's 1:2 ratio and protects student credits.
          }
          env: concat(
            commonEnv,
            [
              { name: 'DATABASE_URL', secretRef: 'database-url' }
              { name: 'DATABASE_URL_SYNC', secretRef: 'database-url-sync' }
              { name: 'AZURE_STORAGE_CONNECTION_STRING', secretRef: 'storage-connection-string' }
              { name: 'JWT_SECRET_KEY', secretRef: 'jwt-secret' }
              { name: 'REDIS_URL', secretRef: 'redis-url' }
              { name: 'SERVICE_BUS_CONNECTION_STRING', secretRef: 'servicebus-connection-string' }
            ]
          )
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000, httpHeaders: [] }
              initialDelaySeconds: 30 // model warm-up
              periodSeconds: 20
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0 // GPU replicas scale away when idle — cost control
        maxReplicas: mlMaxReplicas
        rules: [
          {
            name: 'queue-depth-gpu'
            custom: {
              type: 'azure-servicebus'
              metadata: mlScaleMetadata
              auth: servicebusScaleAuth
            }
          }
        ]
      }
    }
  }
}


// ── Frontend (Next.js standalone) ────────────────────────────────────────────
resource frontendApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'attendance-frontend'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identityResourceId}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    workloadProfileName: 'consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      registries: registryBlock
      ingress: {
        external: true
        targetPort: 3000
        transport: 'auto'
        allowInsecure: false
      }
    }
    template: {
      containers: [
        {
          name: 'frontend'
          image: '${acrLoginServer}/attendance-frontend:${imageTag}'
          resources: { cpu: json('0.5'), memory: '1Gi' }
        }
      ]
      scale: {
        minReplicas: 0 // cold start on first visit in staging
        maxReplicas: 3
        rules: [
          { name: 'http', http: { metadata: { concurrentRequests: '50' } } }
        ]
      }
    }
  }
}

// ── Migration job (manual trigger — Phase 7 pipeline starts it pre-deploy) ───


resource migrationJob 'Microsoft.App/jobs@2024-03-01' = {
  name: 'job-db-migrations'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identityResourceId}': {} }
  }
  properties: {
    environmentId: env.id
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 1800
      replicaRetryLimit: 0
      manualTriggerConfig: {
        parallelism: 1
        replicaCompletionCount: 1
      }
      registries: registryBlock
      secrets: commonSecrets
    }
    template: {
      containers: [
        {
          name: 'migrator'
          image: '${acrLoginServer}/attendance-api:${imageTag}'
          command: [
            'sh'
            '-c'
            'DATABASE_URL_SYNC=$DATABASE_URL_SYNC alembic upgrade head'
          ]
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: concat(
            commonEnv,
            [
              { name: 'DATABASE_URL_SYNC', secretRef: 'database-url-sync' }
              { name: 'AZURE_STORAGE_CONNECTION_STRING', secretRef: 'storage-connection-string' }
              { name: 'JWT_SECRET_KEY', secretRef: 'jwt-secret' }
              { name: 'REDIS_URL', secretRef: 'redis-url' }
            ]
          )
        }
      ]
    }
  }
}

// ── Partition maintenance job (daily 03:00 UTC) ──────────────────────────────

resource partitionJob 'Microsoft.App/jobs@2024-03-01' = {
  name: 'job-partition-maintenance'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identityResourceId}': {} }
  }
  properties: {
    environmentId: env.id
    configuration: {
      triggerType: 'Schedule'
      replicaTimeout: 600
      replicaRetryLimit: 2
      scheduleTriggerConfig: {
        cronExpression: '0 3 * * *' // seconds minutes hours day month weekday
        parallelism: 1
        replicaCompletionCount: 1
      }
      secrets: concat(
        commonSecrets,
        [
          {
            name: 'partition-sql'
            value: loadTextContent('../scripts/create_partitions.sql')
          }
        ]
      )
    }
    template: {
      containers: [
        {
          name: 'partition-maintainer'
          image: 'postgres:16-alpine'
          command: [
            'sh'
            '-c'
            'psql "$DATABASE_URL_SYNC" -v ON_ERROR_STOP=1 -c "$PARTITION_SQL"'
          ]
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
          env: [
            { name: 'DATABASE_URL_SYNC', secretRef: 'database-url-sync' }
            { name: 'PGOPTIONS', value: '-c lock_timeout=30000 -c statement_timeout=600000' }
            { name: 'PARTITION_SQL', secretRef: 'partition-sql' }
          ]
        }
      ]
    }
  }
}

output environmentId string = env.id
output apiFqdn string = apiApp.properties.configuration.ingress.fqdn
output frontendFqdn string = frontendApp.properties.configuration.ingress.fqdn
output migrationJobName string = migrationJob.name
