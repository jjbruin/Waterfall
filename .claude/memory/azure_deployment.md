---
name: Azure Deployment
description: Azure infrastructure details and deployment workflow — target for all future development
type: project
---

## Decision (2026-04-10)
All future development targets the Azure deployment. No more local-only features. Single codebase, single deployment target.

## Azure Resources

### Container App (VNet-integrated, May 2026)
- **App URL**: https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io
- **Resource Group**: rg-waterfall-dev (eastus)
- **Container Registry**: acrwaterfalldev.azurecr.io (Basic SKU, admin enabled)
- **PostgreSQL**: psql-waterfall-dev.postgres.database.azure.com (B1ms, v16, eastus2)
  - Database: waterfall_xirr, User: wfadmin
- **Container App Env**: cae-waterfall-vnet (VNet-integrated, Consumption plan, eastus)
- **Container App**: app-waterfall-dev-v2 (1 CPU, 2GB RAM, 1 Gunicorn worker)

### Networking (May 2026)
- **VNet**: vnet-waterfall-dev (10.0.0.0/16)
- **Container Apps Subnet**: snet-containerapp (10.0.0.0/23, delegated to Microsoft.App)
- **NAT Gateway**: nat-waterfall-dev → static IP **20.127.96.240** (pip-nat-waterfall)
- **VPN Gateway**: vpngw-waterfall-dev (VpnGw1AZ SKU) → public IP **48.194.101.189** (pip-vpngw-waterfall)
- **Gateway Subnet**: GatewaySubnet (10.0.2.0/27)
- **Local Network Gateway**: lgw-mri (peer: 172.191.157.134, subnet: 10.219.226.0/24)
- **VPN Connection**: vpn-to-mri (Site-to-Site IPsec, pre-shared key, status: NotConnected)

### Legacy (still running, can be deleted)
- **Old Container App Env**: cae-waterfall-dev (no VNet)
- **Old Container App**: app-waterfall-dev (revision v99)
- **Old URL**: https://app-waterfall-dev.victoriousforest-f83586cf.eastus.azurecontainerapps.io

## Architecture
- Docker multi-stage build: Vue frontend → Python 3.12-slim + Gunicorn
- Includes ODBC Driver 18 for SQL Server + pyodbc (for MRI access)
- SQL query files copied into image at `/app/queries/`
- SQLAlchemy abstraction (`flask_app/db.py`): DATABASE_URL env var switches SQLite/PostgreSQL
- Data adapters (`data_adapters.py`): pluggable per-table loading (DB or MRI API)

## Deployment — Azure CLI (not GitHub Actions)
GitHub Actions secrets (`AZURE_CREDENTIALS`) are not configured, so all deploys use Azure CLI directly.
After committing changes, run these two commands in sequence:

1. **Build image in ACR**: `az acr build --registry acrwaterfalldev -g rg-waterfall-dev --image waterfall-xirr:latest --no-logs .`
2. **Deploy to Container Apps**: `az containerapp update -g rg-waterfall-dev -n app-waterfall-dev-v2 --image acrwaterfalldev.azurecr.io/waterfall-xirr:latest --revision-suffix vNN`

**Note**: Find the latest revision suffix first: `az containerapp revision list -g rg-waterfall-dev -n app-waterfall-dev-v2 --query "[].name" -o tsv | sort | tail -1`

## Desktop Shortcut
- `launch_app.bat` opens the Azure app URL in browser
- Points to: https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io/dashboard

## Useful Commands
- View logs: `az containerapp logs show -g rg-waterfall-dev -n app-waterfall-dev-v2 --type console --tail 50`
- Check VPN status: `az network vpn-connection show -g rg-waterfall-dev -n vpn-to-mri --query "{connectionStatus: connectionStatus}" -o json`
- Check resources: `az containerapp show -g rg-waterfall-dev -n app-waterfall-dev-v2 --query "properties.template.containers[0].resources" -o json`

## MRI Connectivity from Azure — Status (May 6, 2026)
- **Problem**: MRI SQL Servers are on private IPs (10.219.226.x) behind MRI's VPN. Azure container can't reach them directly.
- **VPN Gateway provisioned** (48.194.101.189) — ready for dedicated tunnel.
- **Decision**: Purchase 2nd VPN license from MRI for a dedicated Azure↔MRI tunnel. MRI DBA confirmed this is supported for internal apps in Azure. Contact MRI account exec for pricing.
- **Workaround (until tunnel live)**: Run Flask locally with VPN connected, pointing at Azure DATABASE_URL. Admin clicks "Refresh All Data from MRI" locally; data flows: local machine → MRI (via FortiClient VPN) → local machine → Azure PostgreSQL.
- **When tunnel is purchased**: MRI provides new VPN peer IP + IKE params. Update Local Network Gateway + VPN Connection in Azure. Container traffic routes: Azure Container App → VNet → VPN Gateway → MRI tunnel → SQL Servers.

## Migration Scripts (in scripts/)
- `migrate_to_postgres.py` — bulk SQLite → PostgreSQL migration
- `fix_tables.py` — fix tables with type mismatches (occupancy, tenants, prospective_loans)
- `azure-complete-setup.sh` — reference doc of provisioned infrastructure
