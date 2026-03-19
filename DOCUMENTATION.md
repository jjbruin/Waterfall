# Waterfall Model - Complete Documentation

**Last Updated:** March 2026

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation & Setup](#installation--setup)
3. [Data Files](#data-files)
4. [Database Management](#database-management)
5. [Key Concepts](#key-concepts)
6. [Capital Pools & Routing](#capital-pools--routing)
7. [Ownership Relationships](#ownership-relationships)
8. [Capital Calls & Commitments](#capital-calls--commitments)
9. [Performance & Caching](#performance--caching)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

A Flask + Vue financial modeling application for calculating investment waterfalls, XIRR, and related performance metrics for real estate investments. Supports multi-layer distribution waterfalls with preferred returns, capital accounts, and investor-level tracking.

### Tech Stack

- **Python 3.x** with virtual environment (`.venv/`)
- **Flask** - REST API backend (`flask_app/`)
- **Vue 3 + Vite** - Modern frontend (`vue_app/`)
- **pandas/numpy** - Data manipulation
- **scipy** - XIRR/NPV calculations (Brent's method)
- **ECharts** - Interactive charts (Vue)
- **SQLite** - Local database (`waterfall.db`), PostgreSQL ready
- **JWT** - Authentication (Flask + Vue)

### Project Structure

```
waterfall-xirr/
├── config.py                 # Constants, account classifications, dynamic defaults
├── compute.py                # Deal computation engine (core logic)
├── one_pager.py              # One Pager data logic (performance calcs, cap stack, PE metrics)
├── models.py                 # Data classes (InvestorState, Loan)
├── waterfall.py              # Waterfall calculation engine
├── metrics.py                # XIRR, XNPV, ROE, MOIC calculations
├── loaders.py                # Data loading from database/CSV
├── database.py               # SQLite management, migrations, table definitions, CSV import/export
├── loans.py                  # Debt service modeling
├── planned_loans.py          # Future loan projections
├── capital_calls.py          # Capital call handling
├── cash_management.py        # Cash flow management
├── consolidation.py          # Sub-portfolio deal/property aggregation
├── portfolio.py              # Fund/portfolio aggregation
├── reporting.py              # Annual aggregation tables, formatting utilities
├── ownership_tree.py         # Investor ownership structures
├── utils.py                  # Helper utilities
├── waterfall.db              # SQLite database (not in git, >100MB)
│
├── flask_app/                # Flask REST API backend (105 routes)
│   ├── __init__.py           # App factory, SPA serving
│   ├── run.py                # Dev server entry point
│   ├── config.py             # Flask configuration (dynamic defaults, ACTUALS_THROUGH)
│   ├── db.py                 # SQLAlchemy engine (SQLite or PostgreSQL)
│   ├── auth/                 # JWT authentication, SSO (Azure AD + Okta)
│   ├── api/                  # 10 API blueprints (dashboard, data, deals, reviews, financials, reports, etc.)
│   └── services/             # Business logic (compute_service, data_service, review_service, etc.)
│
└── vue_app/                  # Vue 3 + Vite frontend
    ├── src/
    │   ├── api/client.ts     # Axios instance with JWT interceptors
    │   ├── stores/           # Pinia stores (auth, data, dashboard, deals)
    │   ├── views/            # Page components (DashboardView, DealAnalysisView, OnePagerView, etc.)
    │   └── components/       # Shared components (KpiCard, DataTable, ReviewPanel, AppSidebar)
    ├── vite.config.ts        # Vite config (proxies /api to Flask)
    └── package.json
```

### Module Reference

| Module | Purpose |
|--------|---------|
| `compute.py` | Deal computation engine — `compute_deal_analysis()`, `build_partner_results()`, `run_interleaved_waterfalls()` |
| `one_pager.py` | One Pager data: property performance, cap stack, PE metrics, comments CRUD |
| `waterfall.py` | Pref accrual, waterfall step processing, investor state management |
| `loans.py` | Loan amortization schedules |
| `consolidation.py` | Sub-portfolio aggregation (deals with properties) |
| `ownership_tree.py` | Multi-tier ownership tracing |
| `metrics.py` | XIRR, ROE, MOIC calculations |
| `capital_calls.py` | Capital calls processing |
| `cash_management.py` | Cash reserves and CapEx management |
| `database.py` | Database operations, table definitions, migrations, indexes, CSV import/export |
| `reporting.py` | Annual aggregation tables and formatting utilities |
| `config.py` | BS_ACCOUNTS, IS_ACCOUNTS, capital pool routing, dynamic default settings |
| `flask_app/services/compute_service.py` | Deal cache, ROE/MOIC audit builders, 8 Excel generators |
| `flask_app/services/financials_service.py` | Performance chart, IS, BS, tenants, one-pager data |
| `flask_app/services/review_service.py` | Review workflow (approval pipeline, tracking, roles) |

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
cd C:\Users\jbruin\Documents\GitHub\waterfall-xirr
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Initialize Database

```bash
python migrate_to_database.py --folder "C:\Users\jbruin\OneDrive - peaceablestreet.com\Documents\WaterfallApp\data"
```

### Step 3: Run Application

```bash
# Flask API backend
python -m flask_app.run          # API on http://localhost:5000

# Vue frontend (separate terminal)
cd vue_app && npm run dev        # Frontend on http://localhost:5173
# Default login: admin / admin
```

---

## Data Files

### Required Files

Located in: `C:\Users\jbruin\OneDrive - peaceablestreet.com\Documents\WaterfallApp\data`

| File | Description |
|------|-------------|
| `investment_map.csv` | Deal master data (vcode, Investment_Name, Portfolio_Name) |
| `forecast_feed.csv` | Financial projections |
| `waterfalls.csv` | Waterfall step definitions |
| `accounting_feed.csv` | Historical accounting transactions |
| `coa.csv` | Chart of accounts |

### Optional Files

| File | Description |
|------|-------------|
| `MRI_Loans.csv` | Loan details |
| `MRI_Supp.csv` | Supplemental/planned loan data |
| `MRI_Val.csv` | Valuation data |
| `MRI_Capital_Calls.csv` | Capital call schedule |
| `ISBS_Download.csv` | Income statement and balance sheet data (Interim IS, Budget IS, Projected IS, Interim BS) |
| `MRI_IA_Relationship.csv` | Ownership relationships |
| `MRI_Commitments.csv` | Investor commitments |
| `MRI_Occupancy_Download.csv` | Monthly occupancy data by property (dtReported, Qtr, Occ%) |
| `Tenant_Report.csv` | Commercial tenant roster by property (leases, SF, rent) |

### Sub-Portfolio Structure

Deals can have child properties linked via the `Portfolio_Name` field:

- **Parent Deal**: `Portfolio_Name` is NULL or self-referencing, `Investment_Name` is the portfolio identifier
- **Child Property**: `Portfolio_Name` = parent deal's `Investment_Name`

Example:
```
vcode      Investment_Name    Portfolio_Name
P0000033   OREI Portfolio     (null)
P0000061   Whitney Manor      OREI Portfolio
P0000062   Westchase Apts     OREI Portfolio
```

**Consolidation Rules for Sub-Portfolios:**
- **Forecasts**: ONLY property-level forecasts are used (summed across all properties). Parent-level forecasts are **ignored** even if they exist.
- **Loans**: Aggregate from both parent AND property levels
- **Debt Service**: Sum of amortization schedules for all loans (parent + properties)
- **Loan Payoff**: Sum of outstanding balances at sale date for all loans

---

## Database Management

### Refresh Workflow

1. Update CSV files in the data folder
2. Close Excel (releases file locks)
3. Run migration:

```bash
cd C:\Users\jbruin\Documents\GitHub\waterfall-xirr
python migrate_to_database.py --folder "C:\Users\jbruin\OneDrive - peaceablestreet.com\Documents\WaterfallApp\data"
```

4. Verify output shows green checkmarks for each file

### Backup

Before major changes:
```bash
copy waterfall.db waterfall_backup.db
```

### Restore

If needed, delete `waterfall.db` and either:
- Re-run migration from CSV files, or
- Rename backup file back to `waterfall.db`

---

## Key Concepts

### Waterfall Types

| Type | Purpose | Capital Impact |
|------|---------|----------------|
| **CF_WF** | Operating cash distributions | Does NOT reduce capital outstanding |
| **Cap_WF** | Refi/sale proceeds | DOES reduce capital outstanding |

### Preferred Returns

- **Accrual**: Daily using Act/365 Fixed day count
- **Compounding**: Annually on 12/31 (with 45-day grace period)
- **Grace Period**: Distributions within 45 days after year-end can reduce pref before compounding
- **Tracking**: Three buckets per tier:
  - `pref_unpaid_compounded` - Prior years' unpaid pref (already compounded)
  - `pref_accrued_prior_year` - Last year's unpaid (compounds after Feb 14)
  - `pref_accrued_current_year` - Current year accrual

### Payment Priority

When paying pref, the order is:
1. Compounded unpaid pref (oldest)
2. Prior year accrued (if past grace period)
3. Current year accrued (newest)

### Account Classifications

- Revenue accounts: 4xxx series
- Expense accounts: 5xxx series
- Interest: 7030
- Principal: 7060
- CapEx tracked separately

### Actuals Through Cutoff

Global setting that allows mid-year model updates using actual results through a specified date.

| Component | Behavior with `actuals_through` set |
|-----------|-------------------------------------|
| **Partner Cash Flows** | Actual distributions from `accounting_feed` through cutoff (via `seed_states_from_accounting`); waterfall-computed distributions only AFTER cutoff |
| **Operating Forecast** | Forecast Revenue + Expense rows for months <= cutoff removed from `fc_deal_full` |
| **Waterfall** | `cf_period_cash` and `cap_period_cash` filtered to `event_date > cutoff` only |
| **Cache Key** | Includes `actuals_through` so toggling triggers full recomputation |
| **Default** | `None` (full forecast, current behavior preserved) |

**Dynamic Defaults** (as of March 2026):
- `DEFAULT_START_YEAR = date.today().year` (was hardcoded 2026)
- `PRO_YR_BASE_DEFAULT = date.today().year - 1` (was hardcoded 2025)

**UI Controls**:
- Vue: Report Settings section in `AppSidebar.vue` with checkbox + month-end date selector

**Flask**: `ACTUALS_THROUGH` in `flask_app/config.py`, passed via query params and request body, included in `GET /api/data/config` and `PUT /api/data/config`.

### Conventions

- Cashflow signs: negative = contribution, positive = distribution
- Rates as decimals (0.08 = 8%)
- Use Python `date` objects for dates

---

## Capital Pools & Routing

### Pool Types

| Pool | Description | Pref Accrual |
|------|-------------|--------------|
| `initial` | Primary investment capital | Yes |
| `additional` | Additional capital (OP/dev) | Yes |
| `special` | Special capital (PPI/investor) | Yes |
| `operating` | Operating capital contributions | Typically 0% (capped) |
| `cost_overrun` | Cost overrun capital | Yes |

### Typename Routing

Contributions are routed to pools based on `Typename` field:

| Typename contains | Routes to Pool |
|-------------------|----------------|
| "operating capital" | operating |
| "cost overrun" | cost_overrun |
| "special capital" | special |
| "additional capital" | additional |
| (anything else) | initial |

### Operating Capital Caps

Operating capital pools have a cumulative return cap equal to total contributions. The cap is enforced automatically.

### vState Values

| vState | Description |
|--------|-------------|
| `Pref` | Pay accrued preferred return |
| `Initial` | Return initial capital (Cap_WF only) |
| `Add` | Route to specific capital pool |
| `Tag` | Proportional follower (no pool routing) |
| `Share` | Residual distribution |
| `IRR` | IRR hurdle gate |
| `Amt` | Fixed-amount distribution |
| `AMFee` | Post-distribution fee (pool-neutral) — deducts from source investor (vNotes), pays to recipient (PropCode). `nPercent` = annual rate, `mAmount` = periods/yr |
| `Promote` | Cumulative catch-up — `FXRate` = carry share, `nPercent` = target carry %. `vNotes` = comma-separated capital investors. Math: `E >= target/(1-target) * P` |

**Critical Rule**: Operating capital MUST use `Add`, never `Tag`. See `waterfall_setup_rules.txt` for details.

---

## Ownership Relationships

### Multi-Tier Structure

The system traces ownership through multiple layers:

1. **Deal Level**: Property-owning entity (e.g., PPIBEL)
2. **Fund Level**: Investor funds (e.g., TGA24, I1BA)
3. **Ultimate Investors**: End beneficial owners

### Passthrough Entities

Entities 100% owned by a single parent are marked as passthroughs. Distributions flow through automatically.

### Waterfall Requirements

Entities with multiple investors need their own waterfall definitions. The system identifies these automatically from `MRI_IA_Relationship.csv`.

---

## Capital Calls & Commitments

### Data Sources

| File | Purpose |
|------|---------|
| `MRI_Commitments.csv` | Who committed what to whom |
| `MRI_Capital_Calls.csv` | Planned future calls |

### Capital Calls Data Pipeline

1. **CSV Import**: `MRI_Capital_Calls.csv` imported to `capital_calls` SQLite table
2. **Table Auto-Creation**: `database.py` → `create_additional_tables()` creates the table if it doesn't exist:
   ```sql
   CREATE TABLE IF NOT EXISTS capital_calls (
       Vcode TEXT NOT NULL, PropCode TEXT, CallDate TEXT,
       Amount REAL, CallType TEXT, FundingSource TEXT,
       Notes TEXT, Typename TEXT DEFAULT 'Contribution: Investments'
   )
   ```
3. **Loading**: `load_capital_calls()` normalizes column names (`entityid` → `deal_name`, `propcode` → `investor_id`, `calldate` → `call_date`), drops null rows, parses dates with `format='mixed'`
4. **Schedule**: `build_capital_call_schedule()` filters by deal vcode and builds event list
5. **Application**: `apply_capital_calls_to_states()` routes contributions to capital pools via `typename_to_pool()` and records XIRR cashflows

### Capital Call CRUD (Deal Analysis)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/deals/<vcode>/capital-calls` | POST | Create new capital call |
| `/api/deals/<vcode>/capital-calls` | PUT | Update existing capital call |
| `/api/deals/<vcode>/capital-calls` | DELETE | Delete capital call |

All CRUD operations call `data_service.reload()` for full cache invalidation (not `refresh_table()`) to ensure consistent recomputation.

### Refi Shortfall Auto-Clear

When a deal has a refinancing shortfall (`refi_capital_call_required = True`), capital calls are checked after processing. If total capital calls cover the shortfall within $1 tolerance, the warning flag is cleared:
```python
remaining = refi_capital_call_amount - total_cc
if remaining <= 1.0:
    refi_capital_call_required = False
```

### Date Handling

Capital calls use `pd.to_datetime(format='mixed', dayfirst=False)` to handle both:
- US dates from CSV import: `"6/30/2026"`
- ISO dates from HTML form inputs: `"2026-06-01"`

### Key Calculations

**Unfunded Commitment**:
```
Unfunded = Committed Amount - Funded to Date
```

**Outstanding Equity**:
```
Outstanding = Contributions - Return of Capital
```

**Pro-Rata Calls**:
```
Investor's Share = Total Call Amount × Investor's CapitalPercent
```
(Capped at unfunded balance)

### Cash Flow Integration

```
Beginning Cash
+ Capital Calls (if any)
- CapEx (paid from cash first)
+ Operating CF (if positive)
- Operating Deficits (covered from cash if available)
= Distributable Amount → Goes to Waterfall
= Ending Cash (carries forward)

At Sale:
Sale Proceeds + Remaining Cash Reserves → Capital Waterfall
```

---

## Balloon Loan Payoff & Prospective Loans

### Balloon Detection

When a loan's amortization schedule shows the balance being paid down to near-zero at maturity (a balloon payment), the system detects this automatically:

| Check | Threshold |
|-------|-----------|
| `ending_balance` on final row | < $1.00 (float tolerance) |
| `principal` on final row | > 0 |
| `ending_balance` on prior row | > 0 |

### Forecast vs Sale Treatment

- **Forecast debt service**: Balloon principal is **excluded** — it is not an operating expense. Only interest is included for the balloon period.
- **Sale proceeds**: Net sale deducts the **pre-balloon** loan balance. Formula: `total_loan_balance_at(sale_date) + balloon_total`
- **Loan schedule groupby**: Uses `["vcode", "LoanID", "event_date"]` to correctly distinguish multiple loans with overlapping dates

### Prospective Loan Refinancing

`size_prospective_loan()` in `planned_loans.py` models future refinancing:

- Returns a sizing dict including `refi_date` for Vue form pre-fill
- When `Sale_ME` < new loan maturity, sale date is extended to the new maturity
- Net sale proceeds: `sale_price - loan_balances - balloon_total + cash_reserves`
- The refi shortfall amount triggers the "Capital Call Required" warning in Deal Analysis

---

## Deal Analysis — Excel Downloads & Audit Expanders

The Deal Analysis tab includes per-section Excel download buttons on every section header, plus a "Download Full Deal Analysis (Excel)" button at the top that produces a comprehensive 7-sheet workbook. Three collapsible audit sections below the waterfall results provide full calculation transparency for every metric.

### Excel Downloads

**Per-Section Downloads** (small "Excel" button on each section header):
- Partner Returns — Partner metrics + deal total row
- Annual Forecast — Pivoted line items × years
- Debt Service — Loan Summary + Amortization Schedule (2 sheets)
- Cash Management — Cash flow schedule
- Capital Calls — Capital call data
- XIRR Cash Flows — Merged side-by-side partner cashflows
- ROE Audit / MOIC Audit — Individual audit workbooks

**Full Workbook** — 7 sheets: Partner Returns, Annual Forecast, Debt Service, Cash Schedule, XIRR Cash Flows, ROE Audit, MOIC Audit. All formatted with blue `#4472C4` headers, currency/percentage/multiple number formats, bold total rows with top borders, auto-sized columns.

**API Endpoints** (`GET /api/deals/<vcode>/excel/`):
| Endpoint | Description |
|----------|-------------|
| `partner-returns` | Partner Returns sheet |
| `forecast` | Annual Forecast sheet |
| `debt-service` | Loan Summary + Amortization |
| `cash-schedule` | Cash Schedule sheet |
| `capital-calls` | Capital Calls sheet |
| `xirr-cashflows` | XIRR Cash Flows sheet |
| `full` | 7-sheet comprehensive workbook |

**Generator Functions** (`flask_app/services/compute_service.py`): Shared helpers `_excel_styles()`, `_write_header_row()`, `_autosize_columns()`. Per-section generators and `generate_full_deal_excel()` for the comprehensive workbook. ROE/MOIC audit sheets are generated by existing `generate_roe_audit_excel()` / `generate_moic_audit_excel()` and copied via `load_workbook`.

### XIRR Cash Flows
- Pivot table showing Date, Description, Pref Equity, Ptr Equity, and Deal columns
- Download as CSV for independent verification

### ROE Audit — Return on Equity Breakdown

Formula: `ROE = (Total CF Distributions ÷ Weighted Avg Capital) ÷ Years`

Per-partner sections, then deal-level total:

**Capital Balance Timeline** — Step-by-step replay of the `calculate_roe()` logic showing:
- Each capital event (contributions increase balance, capital returns decrease balance)
- Holding periods between events with days held and weighted capital (Balance × Days)
- Running capital balance after each event (floored at 0)
- Totals row: sum of days and weighted average capital (WAC = total weighted ÷ total days)

**CF Distributions** — Itemized list of CF waterfall distributions (ROE numerator). Only operating income distributions count — return of capital and gain on sale are excluded.

**Summary Metric Cards**: Inception → End Date, Total Days / Years, CF Distributions, Weighted Avg Capital, ROE

**Deal-Level ROE**: Same timeline using aggregated cashflows across all partners. CF distributions collected from all partners' `cf_only_distributions`.

**Excel Download** (`roe_audit.xlsx`): Per-partner sections (timeline + CF distributions + summary), then deal-level section. Blue `#4472C4` headers, currency/percentage formats, bold summary rows with top border.

### MOIC Audit — Multiple on Invested Capital

Formula: `MOIC = (Total Distributions + Unrealized NAV) ÷ Total Contributions`

Per-partner sections, then deal-level total:

**Cashflow Breakdown** — Every cashflow classified by type:
| Type | Criteria |
|------|----------|
| Contribution | Amount < 0 |
| CF Distribution | Amount > 0, operating income |
| Cap Distribution | Amount > 0, capital/refi/sale event |
| Terminal Value | Unrealized NAV entry |

**Summary Metric Cards**: Contributions, CF Distributions, Cap Distributions, Total Distributions, Unrealized NAV, MOIC

**Deal-Level MOIC**: Uses **realized distributions only** (no unrealized NAV), per `compute.py`. Displayed with an info callout: `Deal MOIC = Total Distributions ÷ Total Contributions`.

**Excel Download** (`moic_audit.xlsx`): Per-partner sections (breakdown + summary), then deal-level section. Same formatting conventions as ROE audit.

### Data Sources

All audit data comes from `partner_results` (computed by `build_partner_results()` in `compute.py`):
- `combined_cashflows` — All capital events for ROE timeline
- `cf_only_distributions` — CF waterfall distributions for ROE numerator
- `cashflow_details` — Labeled cashflow rows for MOIC breakdown

### Helper Functions (app.py, module-level)

| Function | Returns |
|----------|---------|
| `_build_roe_timeline(combined_cashflows, cf_only_distributions, end_date)` | `(timeline_df, cf_dist_df, summary_dict)` |
| `_build_moic_breakdown(cashflow_details, pr_dict)` | `(breakdown_df, summary_dict)` |
| `_generate_roe_audit_excel(partner_results, deal_summary, sale_me)` | Excel bytes |
| `_generate_moic_audit_excel(partner_results, deal_summary, sale_me)` | Excel bytes |
| `generate_partner_returns_excel(result)` | Partner Returns Excel bytes (compute_service.py) |
| `generate_forecast_excel(result, start_year, horizon_years)` | Annual Forecast Excel bytes (compute_service.py) |
| `generate_debt_service_excel(result)` | Debt Service Excel bytes (compute_service.py) |
| `generate_full_deal_excel(result, start_year, horizon_years)` | 7-sheet workbook bytes (compute_service.py) |

---

## Property Financials Tab

The Property Financials tab is rendered by `property_financials_ui.py` and displays financial data for the selected property. Sections appear in this order:

### 1. Performance Chart
- **Altair dual-axis chart**: Occupancy bars (left axis, 0-100%) + Actual NOI and Underwritten NOI lines (right axis, dollar scale)
- **Controls**: Frequency (Monthly / Quarterly / Annually), Period End selector
- **Data pipeline**: Cumulative YTD balances from ISBS → periodic monthly NOI → aggregated to selected frequency
- **NOI calculation**: Revenue (credit accounts, negated) minus Expenses (debit accounts), using `IS_ACCOUNTS` from `config.py`
- **Occupancy source**: `occupancy` table (`Occ%` column preferred over `OccupancyPercent` which has nulls). `dtReported` parsed as Excel serial date. Quarterly values averaged from monthly data.
- Shows trailing 12 periods by default

### 2. Income Statement Comparison
- Two-column comparison with configurable period type (TTM, YTD, Full Year, Current Year Estimate, Custom Month)
- Four source options: Actual, Budget, Underwriting, Valuation
- Displays Revenue, Expenses, NOI, Debt Service, DSCR, Other Below the Line
- Apply Settings button to refresh comparison

### 3. Balance Sheet Comparison
- Two-period comparison (Prior Period vs Current Period)
- Assets, Liabilities, Equity with variance columns
- Balance check validation (Assets + L&E = 0)

### 4. Tenant Roster & Lease Rollover
- Commercial properties only (skipped if no tenant data)
- Sortable tenant table with SF, rent, RPSF, % GLA, % ABR
- Expiring leases highlighted (within 2 years)
- Lease Maturity Profile chart (Altair) with 10-year forecast
- Lease Maturity by Year table
- Printable HTML report

### 5. One Pager Investor Report
- Rendered by `one_pager_ui.py` calling `one_pager.py` for data
- Quarter selector, then sections: General Info, Cap Stack, Property Performance, PE Performance, Comments, NOI/Occupancy Chart
- Chart uses same data pipeline as Performance Chart but fixed to quarterly, trailing 12 quarters
- Print report generates single-page HTML; CSV export available

---

## Reports Tab

The Reports tab is rendered by `reports_ui.py` and provides exportable cross-deal reports.

### Projected Returns Summary

Columns: Deal Name, Partner, Contributions, CF Distributions, Capital Distributions, IRR, ROE, MOIC.

Each deal's partner rows are followed by a **Deal Total** row (bold, solid top border) showing deal-level aggregated metrics:
- **IRR**: Combined XIRR across all partner cashflows
- **ROE**: Weighted-average-capital method using CF distributions only
- **MOIC**: Total distributions / total contributions

### Population Selectors

| Option | Description |
|--------|-------------|
| **Current Deal** | Uses the deal selected in the Deal Analysis tab (instant — reads cached result) |
| **Select Deals** | Multi-select from eligible deals |
| **By Partner** | Select a partner (PropCode from waterfalls); resolves to all deals where that partner appears |
| **By Upstream Investor** | Select an investor from the ownership tree; resolves to all deals where that investor has exposure (direct or through intermediate entities) |
| **All Deals** | All deals that have waterfall definitions (child properties excluded) |

### Excel Export

- Generated via `openpyxl` with formatting:
  - Currency columns (`$#,##0`): Contributions, CF Distributions, Capital Distributions
  - Percentage columns (`0.00%`): IRR, ROE
  - Multiple column (`0.00"x"`): MOIC
  - Deal-total rows: bold font + medium top border
  - Auto-column-width based on content
- Download button appears below the preview table

### Performance Notes

- Child properties are filtered out using a vectorized `Portfolio_Name` check (no per-row function calls)
- The "Current Deal" option reuses the `_deal_cache` from session_state — no recomputation
- Multi-deal reports show a progress bar during computation

---

## PSCKOC Entity Analysis Tab

The PSCKOC tab is rendered by `psckoc_ui.py` and provides upstream waterfall analysis for the PSCKOC holding entity, showing how deal-level distributions flow through PPI entities to PSCKOC members.

### Members

| Member | Role | Unit Type |
|--------|------|-----------|
| PSC1 | GP co-invest (Peaceable) | Capital Units |
| KCREIT | LP investor (KOC Member) | Capital Units |
| PCBLE | GP promote + AM fee recipient (Peaceable) | Carry Units |

### Waterfall Structure (Section 6.02)

1. **Preferred Return** (8%) — PSC1 + KCREIT pro rata
2. **GP Catch-up** — 80% PCBLE / 20% Capital until PCBLE has 20% of aggregate (Pref + Catch-up)
3. **Residual Split** — 20% PCBLE / 80% Capital (pro rata PSC1 + KCREIT)
4. **AM Fee** — 1.5% annual (quarterly) on KCREIT capital balance, deducted from KCREIT and paid to PCBLE

### New Waterfall vStates

| vState | Purpose | Key Fields |
|--------|---------|------------|
| `AMFee` | Post-distribution fee (pool-neutral) | `PropCode` = recipient, `nPercent` = annual rate, `mAmount` = periods/yr, `vNotes` = source investor |
| `Promote` | Cumulative catch-up | `PropCode` = carry holder, `FXRate` = carry share, `nPercent` = target carry %, `vNotes` = capital investors (comma-separated) |

**AMFee** is pool-neutral (`allocated = 0`) — it does NOT reduce the remaining cash pool. The waterfall loop has a special check to continue processing AMFee steps even when remaining cash is zero.

**Promote** tracks cumulative catch-up across periods via `promote_base` (pref paid) and `promote_carry` (carry earned) on `InvestorState`. The catch-up formula: `carry_needed = (target_pct / (1 - target_pct)) * promote_base - promote_carry`.

**promote_base tracking**: Pref steps with `vNotes` containing "promote_base" increment the investor's `promote_base` after paying pref.

### Deal Discovery

`_find_psckoc_deals()` traces the entity chain:
1. Find entities where PSCKOC is an investor (via relationships table)
2. Find deals whose waterfalls reference those entities as PropCode
3. Returns deal vcodes with PPI entity and ownership percentage

### Tab Sections

1. **Deal Portfolio** — Table of underlying deals with asset type, PPI entity, ownership %
2. **Compute Button** — Triggers `get_cached_deal_result()` per deal + `run_recursive_upstream_waterfalls()` for CF and Cap
3. **Partner Returns** — KPI cards (IRR, ROE, MOIC) per member + styled summary table
4. **Income Schedule** — PSCKOC's projected income by period and source deal
5. **Waterfall Allocations** — CF_WF and Cap_WF allocation tables per member
6. **AM Fee Schedule** — Quarterly AM fee amounts (date, KCREIT balance, fee)
7. **XIRR Cash Flows** — Combined cashflow table per member
8. **Excel Export** — 4-sheet workbook (Partner Returns, Income Schedule, AM Fee Schedule, XIRR Cash Flows)

Results cached in module-level dict in `psckoc_service.py`.

---

## Performance & Caching

### Shared Multi-Deal Computation Cache

All deal computations flow through `get_cached_deal_result()` in `flask_app/services/compute_service.py`. This function wraps `compute_deal_analysis()` with a module-level `_deal_cache` dict, keyed by `{vcode}|{start_year}|{horizon_years}|{pro_yr_base}|{actuals_through}`.

Deals computed by any consumer accumulate in the shared cache and are reused everywhere:

| Consumer | Location | Behavior |
|----------|----------|----------|
| Deal Analysis | `deals.py` → `compute_service.py` | Computes on first request, instant on return |
| Dashboard Computed Returns | `dashboard.py` → `compute_service.py` | Loops all eligible deals, cached for later |
| Reports | `reports.py` → `compute_service.py` | Loops selected deals, reuses any cached results |
| PSCKOC | `psckoc.py` → `psckoc_service.py` | Computes underlying deals + upstream waterfalls |

Cache invalidation happens via `compute_service.clear_cache()` when "Reload Data" or "Import CSVs" is triggered.

### Module-Level Caches

| Service | Cache | Cleared by |
|---------|-------|------------|
| `compute_service.py` | `_deal_cache` — deal computation results | `clear_cache()` on reload/import |
| `data_service.py` | `_data_cache` — loaded DataFrames | `reload()` on reload/import |
| `dashboard_service.py` | Per-request computation | No persistent cache |

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| NaT comparison error | Null dates in loan data | Fixed in loans.py; re-load MRI_Loans.csv if needed |
| Investor capital never decreases | Step uses Tag instead of Add | Change vState to Add |
| Operating capital exceeds cap | mAmount set incorrectly | Update mAmount to match contributions |
| Wrong pool receives distributions | Incorrect vtranstype | Update vtranstype text |
| Properties not aggregating to parent | Portfolio_Name mismatch | Ensure property's Portfolio_Name = parent's Investment_Name |
| Excessive pref accrual | Wrong last_accrual_date seed | Check accounting data has correct historical dates |
| Capital calls not computing | Mixed date formats crash `pd.to_datetime` silently | Ensure `format='mixed'` in `load_capital_calls()` |
| Capital calls empty after save | `refresh_table()` insufficient for initial load | Use `data_service.reload()` for full cache clear |
| Refi shortfall persists after capital calls | Floating point rounding ($0.39 remainder) | $1 tolerance applied: `remaining <= 1.0` |
| Balloon payment double-counted | Balloon in forecast AND sale deduction | Balloon excluded from forecast, added to sale loan balance |
| Loan ending_balance not exactly 0 | Float rounding (e.g., 8.2e-08) | Detection uses `< 1.0` threshold, not `== 0` |

### Validation Checks

1. **Commitment percentages sum to 100%** for each entity
2. **Funded <= Committed** for each investor
3. **Capital calls <= Unfunded balance**
4. **Outstanding equity >= 0**
5. **FXRates at each iOrder sum to 1.0** (typically)

### Database Issues

- **"File not found"**: Check CSV path and filename spelling
- **"Database is locked"**: Close Flask app and Excel
- **"Invalid CSV format"**: Check for blank rows, special characters, merged cells

---

## Additional Reference Documents

- `waterfall_setup_rules.txt` - Detailed guide for deal modeling team on waterfall step configuration
- `typename_rules.txt` - Complete Typename routing rules for capital pool assignment
- `CLAUDE.md` - Project instructions for Claude Code assistant

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 3.1 | Mar 18 2026 | Capital call CRUD with full cache invalidation, mixed date format handling (`format='mixed'`), null row filtering, `capital_calls` table auto-creation, refi shortfall auto-clear after capital calls cover gap. Balloon loan payoff: detection (ending_balance < $1), exclusion from forecast debt service, pre-balloon balance deduction from net sale proceeds. Prospective loan `refi_date` in sizing dict for Vue form pre-fill. |
| 3.0 | Mar 16 2026 | Global "Actuals Through" cutoff setting (sidebar control, forecast trimming, post-cutoff waterfall), dynamic year defaults, Deal Analysis per-section Excel downloads (7 section buttons + full 7-sheet workbook), capitalization table bug fix (TypeName normalization), annual forecast formatting (black border lines), Flask/Vue actuals_through threading |
| 2.6 | Feb 11 2026 | PSCKOC entity analysis tab: AMFee + Promote vStates in waterfall engine, promote_base/promote_carry tracking on InvestorState, upstream waterfall aggregation for PSCKOC members (PSC1/KCREIT/PCBLE), income schedule, AM fee schedule, partner returns, Excel export |
| 2.5 | Feb 11 2026 | ROE & MOIC audit expanders on Deal Analysis tab: full calculation breakdowns, capital balance timelines, metric summary cards, Excel downloads |
| 2.4 | Feb 7 2026 | Reports tab with Projected Returns Summary, Excel export, population selectors (Current Deal, Select Deals, By Partner, By Upstream Investor, All Deals), deal-level total rows |
| 2.3 | Feb 7 2026 | Performance chart (NOI + occupancy), occupancy table, One Pager chart upgrade, section reorder (IS before BS) |
| 2.2 | Feb 2026 | Property Financials tab extraction (IS, BS, Tenants, One Pager into property_financials_ui.py), compute.py extraction |
| 2.1 | Feb 2026 | Loan aggregation for sub-portfolios, 45-day grace period compounding, NaT handling |
| 2.0 | Jan 2026 | Capital calls, cash management, database migration |
| 1.0 | Initial | Core waterfall engine, ownership tree, metrics |
