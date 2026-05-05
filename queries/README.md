# SQL Queries

MRI source queries for direct database access via VPN. Each `.sql` file maps to one CSV/table in the application.

## Databases

| Database | Server | Auth | Tables |
|----------|--------|------|--------|
| **PMX** (Property Management) | pmx7b.cloud.mrisoftware.com | VPN + credentials (TBD) | IA_Contribution, IA_Distribution, IA_Subtype, IA_Relationship, Entity, IA_Commitment |
| **IM** (Investment Management) | psc.investment.mrisoftware.com | VPN + credentials (TBD) | vstaging_journal_entry, coa, Loan, Valuation, Occupancy, Occupancy_Tenants, property, Vendor, etc. |

## Query → CSV → Table Mapping

### PMX Database
| Query File | CSV | App Table | Notes |
|-----------|-----|-----------|-------|
| `accounting_feed.sql` | `accounting_feed.csv` | `accounting` | Contributions + distributions with running capital balance |
| `MRI_IA_Relationship.sql` | `MRI_IA_Relationship.csv` | `relationships` | Entity relationships with ownership % |
| `MRI_Commitments.sql` | `MRI_Commitments.csv` | `commitments` | Active investor commitments (EndDate IS NULL) |

### IM Database
| Query File | CSV | App Table | Notes |
|-----------|-----|-----------|-------|
| `ISBS_Download.sql` | `ISBS_*.csv` (6 splits) | `isbs_*` | All ISBS data; filter by vSource for splits |
| `coa.sql` | `coa.csv` | `coa` | Chart of accounts (excludes M-prefixed) |
| `MRI_Loans.sql` | `MRI_Loans.csv` | `loans` | Loan details with dates from Loan_Date |
| `MRI_VAL.sql` | `MRI_Val.csv` | `valuations` | Cap rates, NOI, concluded values |
| `MRI_Occupancy_Download.sql` | `MRI_Occupancy_Download.csv` | `occupancy` | Residential + commercial occupancy with computed Occ% |
| `Tenant_Report.sql` | `Tenant_Report.csv` | `tenants` | Commercial tenant detail; `@filterdt` parameter for as-of date |

### Not Yet Queried
| CSV | App Table | Status | Notes |
|-----|-----------|--------|-------|
| `investment_map.csv` | `deals` | TBD | Data spans multiple MRI tables; will optimize retrieval once VPN is live |
| `forecast_feed.csv` | `forecasts` | CSV only | Remains CSV upload due to workflow constraints |
| `MRI_Supp.csv` | `planned_loans` | App-managed | Created/maintained in Waterfall app (Prospective Loans) |
| `MRI_Capital_Calls.csv` | `capital_calls` | App-managed | Created/maintained in Waterfall app (Capital Calls CRUD) |

### App-Only Tables (no MRI source)
| CSV | App Table | Notes |
|-----|-----------|-------|
| `waterfalls.csv` | `waterfalls` | PROTECTED — managed in Waterfall Setup |
| `fund_deals.csv` | `fund_deals` | Fund-to-deal mappings |
| `investor_waterfalls.csv` | `investor_waterfalls` | LP/GP waterfall definitions |
| `investor_accounting.csv` | `investor_accounting` | LP/GP historical transactions |
| `OnePager_Comments.csv` | `one_pager_comments` | PROTECTED — managed in One Pager |

## Usage
- Plain SQL files, loadable by `data_adapters.py` via `pd.read_sql()`
- Parameterized queries use SQL Server `DECLARE @param` syntax
- ISBS query returns all vSource types — filter in Python or add WHERE clause per split
- `Tenant_Report.sql` accepts `@filterdt` parameter for report as-of date

## VPN Testing Checklist
1. [ ] VPN client installed and connected
2. [ ] PMX connection string verified (`pyodbc` or `pymssql`)
3. [ ] IM connection string verified
4. [ ] Test small query on each database (e.g., `SELECT TOP 10 * FROM IA_Subtype`)
5. [ ] Run `accounting_feed.sql` — compare row count to current CSV
6. [ ] Run `ISBS_Download.sql` — verify no row limits (was 800K+ in CSV)
7. [ ] Run remaining queries — compare columns to existing CSVs
8. [ ] Add `MRI_PMX_URL` and `MRI_IM_URL` env vars to Flask config
9. [ ] Build adapter in `data_adapters.py` to load from MRI via SQL files
10. [ ] Test full app with live MRI data (Dashboard → Deal Analysis → One Pager)
