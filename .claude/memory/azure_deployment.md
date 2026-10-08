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
  - `GUNICORN_WORKERS=1` is set on the Container App and overrides the Dockerfile's 2;
    one replica (min = max = 1). Verified Oct 7 2026.
  - **Before raising either, fix cache invalidation.** `data_service.load_all` caches in
    a module-level dict per PROCESS, and every refresh / reload / CSV import clears only
    the process that served the request (`mri_service._clear_all_caches`,
    `/api/data/reload`). With two workers or two replicas the others keep serving the
    pre-refresh data until restarted -- e.g. a deleted MRI loan still counted on half of
    the requests. The same goes for the Dashboard caps cache and the compute cache.

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

## Refreshing local data from production (moved from CLAUDE.md, Oct 5 2026)

Added to CLAUDE.md on origin/main (`40c6796`) while the compaction branch was open, so
it was never in the compaction baseline. The COMMAND stays in CLAUDE.md under Local
Development, because that is the thing you need in front of you; the cautions are here,
verbatim as they were written.

### Refreshing local data from production
`waterfall.db` is NOT in git, so every clone holds its own copy, current only as of
its owner's last load -- Jim's and Charlene's are separate and drift independently.
`scripts/pull_production_db.py` copies production's PostgreSQL into it (Oct 5 2026).

Stop the local Flask server first (Windows will not replace an open file), then in
ONE PowerShell terminal -- each tab has its own variables, which is how the first
attempt failed -- paste this single line:

```powershell
$env:DATABASE_URL = (az containerapp secret show -g rg-waterfall-dev -n app-waterfall-dev-v2 --secret-name db-url --query value -o tsv); .venv\Scripts\python scripts\pull_production_db.py; Remove-Item Env:DATABASE_URL
```

- **The password stays in your terminal.** Read from `DATABASE_URL`, never printed.
  Needs an `az login` with rights to read the container app's secrets, and this
  machine's IP allowed on `psql-waterfall-dev` (Networking) if the connection times out.
- **Rows are replaced, the local schema is kept** -- recreating tables from
  PostgreSQL would lose SQLite's `INTEGER PRIMARY KEY AUTOINCREMENT`, and the next
  locally inserted row would get a NULL id. New columns are added, new tables created.
- **Local logins are kept**: `users`, `password_reset_tokens` and
  `user_section_access` are not copied.
- **Nothing is lost**: the previous file stays as `waterfall.db.bak-<timestamp>`, and
  the new one is swapped in only when every table's row count matches production.
- **Only the last two backups are kept** (Oct 8 2026): after a SUCCESSFUL swap, older
  `<dest>.bak-*` files beyond `--keep-backups` (default 2, by modified time) are
  deleted and named; a failed run deletes nothing; `--keep-all-backups` disables it.
  Guardrail: `scripts/pull_production_db_backups_check.py`.
- **Dates are written the way SQLite holds them here** -- `2026-06-30`, or a time
  after a SPACE. SQLite compares dates as TEXT; the first version wrote
  `2026-06-30T00:00:00`, which sorts after `2026-06-30 23:59:59`, and 879 IA rows
  dated 6/30 silently fell outside a "through 6/30" filter. `--repair-dates` fixes a
  file written that way, in place.
- `--no-files` leaves stored PDFs and images empty (the full copy is ~1.4 GB).
- **The Database Tools "Export Database" button is NOT a substitute**: it reads a
  SQLite file, so on Azure it exports the container's empty local file, not
  PostgreSQL. See `open_items.md`.

## Never set an env var that starts with "/" from Git Bash (Oct 8 2026)
Git Bash's MSYS path conversion rewrites any argument that looks like a Unix path BEFORE `az` sees it:
`--set-env-vars SSO_REDIRECT_URL=/login` stored `C:/Program Files/Git/login`, and Microsoft sign-in was
broken from `v580` to `v609` (Chrome: ERR_UNSAFE_REDIRECT). Set such values from PowerShell, or prefix the
Bash command with `MSYS_NO_PATHCONV=1`, and read the value back from the new revision afterwards:
`az containerapp revision show ... --query "properties.template.containers[0].env[?name=='X'].value | [0]"`.
