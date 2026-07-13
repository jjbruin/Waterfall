# MRI Database Sources

## Two MRI Databases (both SQL Server, VPN required)

### 1. PMX (Property Management)
- **URL**: pmx7b.cloud.mrisoftware.com
- **MRI tables**: `IA_Contribution`, `IA_Distribution`, `IA_Subtype`, `IA_Relationship`, `Entity`, `IA_Commitment`
- **Queries**:
  - `accounting_feed.sql` — contributions + distributions with running capital balance, Capital/ROE_Income flags
  - `MRI_IA_Relationship.sql` — entity relationships with ownership % (joins Entity for Name)
  - `MRI_Commitments.sql` — active investor commitments (EndDate IS NULL)

### 2. Investment Management (IM)
- **URL**: psc.investment.mrisoftware.com (Aurelia Query Tool)
- **MRI tables**: `vstaging_journal_entry`, `coa`, `Loan`, `Loan_Date`, `Valuation`, `Occupancy`, `Occupancy_Tenants`, `Vendor`, `property`, `txprop`, `Investors`, `fund`, `propint`, `Event_Dates`
- **Queries**:
  - `ISBS_Download.sql` — all ISBS data from vstaging_journal_entry (no COA join). Val_IS_2025 merged into Valuation IS.
  - `coa.sql` — chart of accounts from IM `vCOA` view (COA table permission-denied; GACC on PMX has incompatible M-prefix numbering)
  - `MRI_Loans.sql` — loan details + dates from Loan_Date, joined to property for name
  - `MRI_VAL.sql` — valuations with cap rates, NOI, concluded values
  - `MRI_Occupancy_Download.sql` — residential + commercial occupancy with computed Occ% (CTE-based)
  - `Tenant_Report.sql` — commercial tenant detail; parameterized with @filterdt for as-of date
  - `ROE_Download.sql` — ROE analysis (download-only, not imported to app DB)
  - `Prop_Info_Core.sql` — core property metadata for deals table upsert (one row per vCode)
  - `Prop_Info_DealTerms.sql` — PE deal terms pivoted from txfinancial_IC (coupon, IRR lookback, PE split, econ occ at close)
  - `Prop_Info_AtClose.sql` — at-close underwriting NOI with dynamic dates (earliest Dec 31 per deal)

### Remaining Tables (no MRI query needed)
- `capital_calls` — app-managed (CRUD in Waterfall app)
- `planned_loans` — app-managed (Prospective Loans in Waterfall app)
- `forecasts` — remains CSV upload (workflow constraints)
- `deals` (investment_map) — upsert from `Prop_Info_Core.sql`; preserves manual columns (InvestmentID, Portfolio_Name, Sale_Status)

## Query Files
- Stored in two locations:
  - Local: `queries/` folder (10 .sql files, committed to git, copied into Docker image)
  - Network: `C:\Users\jbruin\peaceablestreet.com\...\Asset Mgmt\7. Azure App\queries` (SharePoint-synced, team accessible)
- `mri_service.py` checks network folder first, falls back to local `queries/`
- Plain SQL, executed via `pyodbc` + `pd.read_sql()`
- See `queries/README.md` for full mapping table

## Connection Details (confirmed May 5, 2026)

### Two Separate SQL Server Instances
| Server | IP | Database | Content |
|--------|----|----------|---------|
| **PMX** | 10.219.226.17,1433 | BV6899900001 | Accounting, relationships, commitments, contributions, distributions, GL |
| **IM** | 10.219.226.18,1433 | PSC | ISBS, valuations, loans, occupancy, tenants, investments |

### Credentials
- **Username**: PSCVPN
- **Password**: NVc8MkB^PlRuv*
- **ODBC Driver**: `ODBC Driver 18 for SQL Server` (Driver 17 not installed locally)
- **Connection string**: `DRIVER={ODBC Driver 18 for SQL Server};SERVER=<ip>,1433;DATABASE=<db>;UID=PSCVPN;PWD=NVc8MkB^PlRuv*;TrustServerCertificate=yes;`

### VPN
- **Client**: FortiClient (local machine only)
- **VPN source IP**: 10.212.134.1
- **MRI VPN peer**: 172.191.157.134 (MRI-controlled Fortinet)
- **Public IP**: 50.251.58.254 (not used for MRI access — traffic goes over VPN tunnel)
- **Bandwidth**: 5 Mbps licensed throughput
- **Latency**: ~40ms RTT to both servers

### Table Mapping (verified)
| Our CSV | PMX (.17) Table | IM (.18) Table |
|---------|----------------|----------------|
| accounting_feed | `IA_Contribution` + `IA_Distribution` | — |
| relationships | `IA_Relationship` | `MRI_IA_Relationship` |
| commitments | `IA_Commitment` | `MRI_IA_Commitment` |
| investment_map (deals) | `IA_Investment` | `MRI_IA_Investment` |
| ISBS | — | `Statement` |
| MRI_Val | — | `Valuation` |
| MRI_Loans | — | `Loan` |
| MRI_Occupancy | — | `Occupancy` |
| tenants | — | `CommercialLease` |
| COA | `GACC` (wrong format) | `vCOA` view ✓ / `COA` table (denied) |

### Notes
- `IA_InvestmentLedger` exists on both servers but returns 0 rows
- **COA resolution (Jun 2026)**: `COA` table on IM is permission-denied. `GACC` on PMX has M-prefix numbering (all rows start with "MR..."), incompatible with our 4xxx/5xxx scheme. Solution: use `vCOA` **view** on IM with `ISNUMERIC(vaccount)=1` filter → 176 distinct account/type rows. Network `coa.sql` updated accordingly.
- Both servers have duplicate copies of IA tables — PMX is authoritative

## MRI Query Service (built May 5, 2026)
- **Service**: `flask_app/services/mri_service.py` — connects to MRI, runs queries, imports to Azure PostgreSQL
- **Docker**: Image includes ODBC Driver 18 + pyodbc. Query .sql files at `/app/queries/`.
- **API endpoints** in `flask_app/api/data.py`:
  - `GET /api/data/mri/status` — test VPN connectivity (any user)
  - `GET /api/data/mri/queries` — list available queries (any user)
  - `POST /api/data/mri/queries/<name>/run` — run query, save CSV to network downloads folder (any user)
  - `GET /api/data/mri/queries/<name>/download` — run query, return CSV as browser download (any user)
  - `POST /api/data/mri/refresh` — refresh ALL importable tables from MRI (admin only)
  - `POST /api/data/mri/refresh/<name>` — refresh single table from MRI (admin only)
- **Vue UI**: MRI Data section in sidebar (AppSidebar.vue) — server status, query list, download/run/import buttons
- **Network folders** (SharePoint-synced, team accessible):
  - Queries: `C:\Users\jbruin\peaceablestreet.com\Peaceable Street Capital - Documents\Asset Mgmt\7. Azure App\queries`
  - Downloads: `...\data-downloads` — timestamped CSVs: `{query}_{YYYYMMDD_HHMMSS}.csv`
- **Full refresh**: ~4 min (ISBS is 137s for 900K rows; others are <6s each)
- **ISBS special handling**: Auto-splits by vSource into 5 tables, merges Val_IS_2025 into Valuation IS
- **BOM handling**: `utf-8-sig` encoding strips BOM from .sql files
- **Error logging**: Full traceback logged on query/refresh failures
- **Query timeout**: 10 minutes (for large queries like ISBS)
- **CSV save**: Gracefully skips when network folder unavailable (Azure container)

## Azure MRI Connectivity — Status (Jun 22, 2026)
- **MRI queries work locally** (FortiClient VPN) — all 8 importable queries verified, ~900K ISBS rows
- **MRI queries do NOT work from Azure yet** — 2nd VPN tunnel being configured

### New Tunnel (2nd VPN license — Jun 2026)
- **MRI encryption domain**: `10.219.226.8/29`
- **New server IPs** (Azure tunnel only — different from local VPN IPs):
  | Server | New IP (Azure tunnel) | Old IP (local VPN) | Database |
  |--------|----------------------|-------------------|----------|
  | **PMX** | 10.219.226.9 | 10.219.226.17 | BV6899900001 |
  | **IM** (Investment Central) | 10.219.225.10 | 10.219.226.18 | PSC |
- **Azure VPN Gateway**: vpngw-waterfall-dev (VpnGw1AZ, 48.194.101.189)
- **Pre-shared key**: `QVYGlQSlBrn2zwv$A*4F6$1W#CverF38`
- **Next steps**:
  1. Get MRI's peer IP (their VPN gateway public IP for this tunnel) — still pending
  2. Create Azure Local Network Gateway with MRI peer IP + address spaces (10.219.226.8/29, 10.219.225.0/24)
  3. Create VPN Connection on vpngw-waterfall-dev with pre-shared key + IKE settings
  4. Update `mri_service.py` connection strings to use new IPs when running from Azure (detect via env var)
  5. Test connectivity from container

### Local VPN (existing — unchanged)
- **Client**: FortiClient (local machine only)
- **Server IPs**: PMX=10.219.226.17, IM=10.219.226.18
- **Workaround (until Azure tunnel is live)**: Run Flask locally with FortiClient VPN → Refresh All → imports to local SQLite; sync to Azure PG manually
