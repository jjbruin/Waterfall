#!/bin/bash
# azure-complete-setup.sh
# Reference script documenting the Azure infrastructure that was provisioned.
# This is NOT meant to be re-run — it documents what was done.

set -euo pipefail

# ── Resources Provisioned (2026-04-09) ─────────────────────────────────
# Resource Group:     rg-waterfall-dev (eastus)
# Container Registry: acrwaterfalldev.azurecr.io (Basic SKU, admin enabled)
# PostgreSQL:         psql-waterfall-dev.postgres.database.azure.com (B1ms, v16, eastus2)
# Database:           waterfall_xirr
# Storage Account:    stwaterfalldev (Standard LRS, csv-data container)
# Container App Env:  cae-waterfall-dev (Consumption plan, eastus)
# Container App:      app-waterfall-dev
# App URL:            https://app-waterfall-dev.victoriousforest-f83586cf.eastus.azurecontainerapps.io

# ── GitHub Actions Secrets Needed ──────────────────────────────────────
# ACR_LOGIN_SERVER     = acrwaterfalldev.azurecr.io
# ACR_USERNAME         = acrwaterfalldev
# ACR_PASSWORD         = (az acr credential show --name acrwaterfalldev --query "passwords[0].value" -o tsv)
# AZURE_RG             = rg-waterfall-dev
# AZURE_CONTAINERAPP   = app-waterfall-dev
# AZURE_CREDENTIALS    = (service principal JSON — requires CSP partner to create)

# ── Useful Commands ────────────────────────────────────────────────────
# View app logs:
#   az containerapp logs show -g rg-waterfall-dev -n app-waterfall-dev --type console --tail 50
#
# Update image after manual ACR build:
#   az containerapp update -g rg-waterfall-dev -n app-waterfall-dev --image acrwaterfalldev.azurecr.io/waterfall-xirr:latest
#
# Build image in ACR (no local Docker needed):
#   az acr build --registry acrwaterfalldev -g rg-waterfall-dev --image waterfall-xirr:latest --no-logs .
#
# Check build status:
#   az acr task list-runs --registry acrwaterfalldev --top 5 -o table
#
# PostgreSQL connection:
#   DATABASE_URL=postgresql://wfadmin:<password>@psql-waterfall-dev.postgres.database.azure.com:5432/waterfall_xirr?sslmode=require
