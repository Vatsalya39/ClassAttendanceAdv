# GitHub Actions ↔ Azure setup (OIDC — no stored cloud secrets)

One-time setup after creating the repo and deploying `infra/`. Uses
**OpenID Connect federation**: workflows authenticate to Azure with short-lived
tokens — no client secrets to rotate or leak.

## 1. Create the repo variables/secrets

After running the commands below, set these in **GitHub → Settings → Secrets
and variables → Actions**:

| Type | Name | Value |
|---|---|---|
| Secret | `AZURE_CLIENT_ID` | `$APP_ID` from step 2 |
| Secret | `AZURE_TENANT_ID` | `$TENANT_ID` |
| Secret | `AZURE_SUBSCRIPTION_ID` | `$SUBSCRIPTION` |
| Variable | `AZURE_RESOURCE_GROUP_STAGING` | e.g. `rg-attendance-stg` |
| Variable | `AZURE_RESOURCE_GROUP_PRODUCTION` | e.g. `rg-attendance-prd` |
| Variable | `ACR_LOGIN_SERVER` | output `acrLoginServer` (staging deploy) |
| Variable | `PRODUCTION_ACR_LOGIN_SERVER` | prod ACR login server |
| Variable | `STAGING_API_URL` | `https://<apiFqdn>` (staging) |
| Secret | `AZURE_STATIC_WEB_APPS_API_TOKEN` | staging SWA deploy token |
| Secret | `AZURE_STATIC_WEB_APPS_API_TOKEN_PROD` | production SWA deploy token |

SWA tokens: `az staticwebapp secrets list -g <rg> -n <swa-name> --query properties.apiKey -o tsv`

## 2. Create the Entra app + federated credentials

```bash
SUBSCRIPTION=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
REPO="<your-github-user>/classAttendenceAdvanced"   # adjust

APP=$(az ad app create --display-name "gh-attendance-actions" --query appId -o tsv)
OBJECT_ID=$(az ad app show --id "$APP" --query id -o tsv)

# Main branch pushes (backend.yml / frontend.yml deploy-staging jobs)
az ad app federated-credential create --id "$OBJECT_ID" --parameters "{
  \"name\": \"gh-main\",
  \"issuer\": \"https://token.actions.githubusercontent.com\",
  \"subject\": \"repo:$REPO:ref:refs/heads/main\",
  \"audiences\": [\"api://AzureADTokenExchange\"]
}"

# Production environment (deploy-production.yml runs inside env 'production')
az ad app federated-credential create --id "$OBJECT_ID" --parameters "{
  \"name\": \"gh-env-production\",
  \"issuer\": \"https://token.actions.githubusercontent.com\",
  \"subject\": \"repo:$REPO:environment:production\",
  \"audiences\": [\"api://AzureADTokenExchange\"]
}"

echo "APP_ID=$APP TENANT_ID=$TENANT_ID SUBSCRIPTION=$SUBSCRIPTION"
```

## 3. Grant permissions on each resource group

The workflows need Contributor over the staging/production resource groups
(update container apps, start jobs, pull ACR).

```bash
for RG in rg-attendance-stg rg-attendance-prd; do
  az role assignment create \
    --assignee "$APP" \
    --role Contributor \
    --scope "$(az group show -n $RG --query id -o tsv)"
done
```

## 4. Enable the production approval gate

GitHub → **Settings → Environments → production → Required reviewers**:
add yourself (or the release team). `deploy-production.yml` declares
`environment: production`, so every run pauses for approval before touching
production.

Optionally do the same for `staging`.

## 5. First deploy order of operations

1. Provision infra: see [infra/README.md](../infra/README.md)
2. Push `main` → `backend.yml` builds/pushes images and deploys staging
   (the migration job creates the schema on first run, including
   `CREATE EXTENSION vector`)
3. Bootstrap the first super admin (command in infra/README.md §6)
4. Create institutions/sections via `/admin/console`
