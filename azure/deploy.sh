#!/bin/bash
# Deploy Waterfall XIRR to Azure
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Docker installed
#
# Usage:
#   chmod +x azure/deploy.sh
#   ./azure/deploy.sh
#
# First-time setup creates all resources. Subsequent runs just push a new image.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
APP_NAME="${APP_NAME:-waterfall-xirr}"
RESOURCE_GROUP="${RESOURCE_GROUP:-${APP_NAME}-rg}"
LOCATION="${LOCATION:-eastus}"
PG_ADMIN_USER="${PG_ADMIN_USER:-pgadmin}"

# Prompt for PostgreSQL password if not set
if [ -z "${PG_ADMIN_PASSWORD:-}" ]; then
    read -sp "PostgreSQL admin password: " PG_ADMIN_PASSWORD
    echo
fi

echo "=== Waterfall XIRR Azure Deployment ==="
echo "App Name:       $APP_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location:       $LOCATION"
echo

# ── Step 1: Create Resource Group ────────────────────────────────────
echo "1/6 Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

# ── Step 2: Deploy Infrastructure (Bicep) ────────────────────────────
echo "2/6 Deploying infrastructure (PostgreSQL, App Service, Container Registry)..."
DEPLOY_OUTPUT=$(az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --template-file azure/main.bicep \
    --parameters \
        appName="$APP_NAME" \
        pgAdminUser="$PG_ADMIN_USER" \
        pgAdminPassword="$PG_ADMIN_PASSWORD" \
    --query "properties.outputs" \
    --output json)

ACR_SERVER=$(echo "$DEPLOY_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['acrLoginServer']['value'])")
APP_URL=$(echo "$DEPLOY_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['appUrl']['value'])")
PG_HOST=$(echo "$DEPLOY_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['pgHost']['value'])")

echo "  ACR:    $ACR_SERVER"
echo "  App:    $APP_URL"
echo "  PG:     $PG_HOST"

# ── Step 3: Build Docker Image ───────────────────────────────────────
echo "3/6 Building Docker image..."
docker build -t "${ACR_SERVER}/${APP_NAME}:latest" .

# ── Step 4: Push to Azure Container Registry ─────────────────────────
echo "4/6 Pushing image to ACR..."
az acr login --name "${APP_NAME}acr" 2>/dev/null || \
    az acr login --name "$(echo ${APP_NAME}acr | tr -d '-')"
docker push "${ACR_SERVER}/${APP_NAME}:latest"

# ── Step 5: Set App Settings ─────────────────────────────────────────
echo "5/6 Configuring app settings..."

# Generate secrets
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")

az webapp config appsettings set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --settings \
        SECRET_KEY="$SECRET_KEY" \
        JWT_SECRET="$JWT_SECRET" \
        CORS_ORIGINS="$APP_URL" \
    --output none

# ── Step 6: Restart App ──────────────────────────────────────────────
echo "6/6 Restarting app..."
az webapp restart --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --output none

echo
echo "=== Deployment Complete ==="
echo "App URL:      $APP_URL"
echo "PostgreSQL:   $PG_HOST"
echo
echo "Next steps:"
echo "  1. Migrate data:  DATABASE_URL=postgresql://${PG_ADMIN_USER}:***@${PG_HOST}:5432/waterfall_xirr?sslmode=require python -m flask_app.migrate_to_postgres"
echo "  2. Open app:      $APP_URL"
echo "  3. Login:         admin / admin (change password immediately)"
echo
echo "To configure SSO, set these in Azure Portal > App Service > Configuration:"
echo "  SSO_PROVIDER, SSO_CLIENT_ID, SSO_CLIENT_SECRET, SSO_TENANT_ID"
