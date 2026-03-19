# Waterfall XIRR - Multi-Layer Waterfall Model

## Project Overview

A Flask + Vue financial modeling application for calculating investment waterfalls, XIRR, and related performance metrics for real estate investments. The application supports multi-layer distribution waterfalls with preferred returns, capital accounts, and investor-level tracking.

## Tech Stack

- **Python 3.x** with virtual environment (`.venv/`)
- **Flask** - REST API backend (`flask_app/`)
- **Vue 3 + Vite** - Modern frontend (`vue_app/`)
- **pandas/numpy** - Data manipulation
- **scipy** - XIRR/NPV calculations (Brent's method)
- **ECharts** - Interactive charts (Vue)
- **SQLite** - Local database (`waterfall.db`)
- **JWT** - Authentication (Flask + Vue)

## Project Structure

```
waterfall-xirr/
├── config.py                 # Constants, account classifications, rates, dynamic defaults
├── compute.py                # Deal computation logic (core engine)
├── one_pager.py              # One Pager data logic (general info, cap stack, property perf, PE metrics, comments)
├── models.py                 # Data classes (InvestorState, Loan)
├── waterfall.py              # Waterfall calculation engine
├── metrics.py                # XIRR, XNPV, ROE, MOIC calculations
├── loaders.py                # Data loading from database/CSV
├── database.py               # SQLite management, migrations, table definitions, CSV import/export
├── loans.py                  # Debt service modeling
├── planned_loans.py          # Future loan projections
├── capital_calls.py          # Capital call handling
├── cash_management.py        # Cash flow management
├── consolidation.py          # Sub-portfolio aggregation
├── portfolio.py              # Fund/portfolio aggregation
├── reporting.py              # Annual aggregation tables, formatting utilities
├── ownership_tree.py         # Investor ownership structures
├── utils.py                  # Helper utilities
├── waterfall.db              # SQLite database (not in git, >100MB)
│
├── flask_app/                # Flask REST API backend
│   ├── __init__.py           # App factory (create_app)
│   ├── run.py                # Dev server entry point
│   ├── config.py             # Flask configuration (dynamic defaults, ACTUALS_THROUGH)
│   ├── extensions.py         # Flask extensions
│   ├── serializers.py        # JSON serialization helpers (NumpyEncoder, safe_json)
│   ├── auth/                 # JWT authentication (login, SSO config)
│   ├── api/                  # API blueprints
│   │   ├── dashboard.py      # Dashboard endpoints (KPIs, charts, SSE init-stream)
│   │   ├── data.py           # Data endpoints (deals, import/export, config, list-csvs)
│   │   ├── deals.py          # Deal analysis endpoints + Excel downloads
│   │   ├── financials.py     # Property Financials + One Pager endpoints
│   │   ├── reports.py        # Report generation endpoints
│   │   ├── reviews.py        # Review workflow endpoints (status, submit, approve, return, tracking, roles)
│   │   └── ...               # Additional route blueprints
│   └── services/             # Business logic (reuses compute.py, database.py, etc.)
│       ├── dashboard_service.py  # KPI calculations, NOI pipeline, chart data
│       ├── data_service.py       # Data loading and caching
│       ├── compute_service.py    # Deal computation cache, ROE/MOIC audit builders, Excel generators
│       ├── review_service.py     # Review workflow business logic (approval pipeline)
│       ├── financials_service.py # Property Financials + One Pager data aggregation
│       └── ...
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

## Documentation

- **DOCUMENTATION.md** - Complete project documentation (setup, data files, concepts, troubleshooting)
- **waterfall_setup_rules.txt** - Waterfall step configuration guide for deal modeling team
- **typename_rules.txt** - Capital pool routing rules based on Typename field

## Running the Application

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run Flask API backend
python -m flask_app.run          # API on http://localhost:5000

# Run Vue frontend (separate terminal)
cd vue_app && npm run dev        # Frontend on http://localhost:5173
# Default login: admin / admin
```

## Key Concepts

### Waterfall Types
- **CF Waterfall**: Operating cash distributions (does NOT reduce capital outstanding)
- **Capital Waterfall**: Refi/sale proceeds (DOES reduce capital outstanding)

### Preferred Returns
- Daily accrual using Act/365 Fixed day count
- Compounds annually on 12/31 (with 45-day grace period)
- Tracked per investor via `InvestorState`

### Capital Calls
- **Data pipeline**: `MRI_Capital_Calls.csv` → `capital_calls` SQLite table → `load_capital_calls()` → `build_capital_call_schedule()` → `apply_capital_calls_to_states()`
- **Date handling**: Uses `pd.to_datetime(format='mixed', dayfirst=False)` to handle both CSV US dates ("6/30/2026") and HTML ISO dates ("2026-06-01")
- **Null filtering**: `dropna(subset=['deal_name', 'call_date', 'amount'])` removes empty rows from CSV imports
- **Column mapping**: Supports `entityid` → `deal_name`, `propcode` → `investor_id`, `calldate` → `call_date`
- **Table auto-creation**: `database.py` creates `capital_calls` table via `CREATE TABLE IF NOT EXISTS` in `create_additional_tables()`
- **CRUD endpoints**: `POST/PUT/DELETE /api/deals/<vcode>/capital-calls` in `deals.py`. All use `data_service.reload()` for full cache invalidation
- **Refi shortfall auto-clear**: After capital calls are applied, if total capital calls cover the refi shortfall (within $1 tolerance), `refi_capital_call_required` is set to False

### Balloon Loan Payoff at Sale
- **Detection**: Loan schedule's last row has `ending_balance < 1.0` (float tolerance), `principal > 0`, and prior row's `ending_balance > 0`
- **Forecast exclusion**: Balloon principal payments are excluded from forecast debt service rows (they are NOT operating expenses)
- **Sale proceeds**: Net sale proceeds deduct `total_loan_balance_at(sale_date) + balloon_total` — uses pre-balloon balance since balloon is paid at sale
- **Groupby**: Loan schedule grouped by `["vcode", "LoanID", "event_date"]` to distinguish loans with same dates

### Prospective Loans (Refinancing)
- `size_prospective_loan()` in `planned_loans.py` returns sizing dict including `refi_date` for Vue form pre-fill
- Prospective loan extends sale date to new maturity when `Sale_ME` < new maturity
- Net sale proceeds formula: `sale_price - loan_balances - balloon_total + cash_reserves`

### Sub-Portfolio Aggregation
- Deals can have child properties linked via `Portfolio_Name`
- Loans aggregate UP from properties to parent deal level
- See `consolidation.py` for implementation

### Actuals Through Cutoff
- Global setting (`actuals_through`): date or None (default None = full forecast)
- **Partner cash flows**: Actual distributions from `accounting_feed` through cutoff (via `seed_states_from_accounting`); waterfall-computed distributions only for periods AFTER cutoff
- **Operating forecast**: Forecast Rev+Exp rows for months <= cutoff are removed from `fc_deal_full`
- **Waterfall**: `cf_period_cash` and `cap_period_cash` filtered to post-cutoff periods only
- **Cache key**: includes `actuals_through` so toggling triggers recomputation
- **Dynamic defaults**: `DEFAULT_START_YEAR = date.today().year`, `PRO_YR_BASE_DEFAULT = date.today().year - 1`
- **UI**: Streamlit sidebar checkbox + month-end selector; Vue sidebar in Report Settings
- **Flask**: `ACTUALS_THROUGH` in config, passed via query params / request body, included in `/api/data/config`

## Application Tabs

### 1. Dashboard
Executive portfolio-level view with instant-load KPIs and charts. Vue: `DashboardView.vue`. Flask: `dashboard.py` + `dashboard_service.py`.
- **KPI Cards** (6): Portfolio Value, Debt Outstanding, Wtd Avg Cap Rate, Portfolio Occupancy, Deal Count, Total Preferred Equity
- **Portfolio NOI Trend** — Dual-axis Altair chart (occupancy bars + Actual/U/W NOI lines) aggregated across all deals. Values in $ million. Frequency and period-end selectors (defaults to most recently ended quarter), trailing 12 periods. Capped at last closed quarter.
- **Portfolio Capital Structure** — Consolidated vertical stacked bar (Debt blue / Pref Equity green / OP Equity grey) with Avg LTV and Pref Exposure annotations at dividing lines. Values in $ million.
- **Occupancy by Type** — Horizontal bars showing weighted-average occupancy per Asset_Type, colored above/below portfolio average with dashed reference line.
- **Asset Allocation** — Donut chart by Asset_Type sized by preferred equity, with % of total in legend table and hover tooltips.
- **Loan Maturities** — Stacked bar chart by maturity year (Fixed blue / Floating orange) with weighted avg rate labels on fixed sections, total dollar labels, and "Show Data" expander with loan-level detail table (includes child property loans).
- **Computed Returns** (button-gated) — Progress bar → IRR by Deal bar chart + formatted summary table (Contributions, Distributions, IRR, ROE, MOIC)

### 2. Deal Analysis
Main waterfall computation, partner returns, capital accounts, XIRR/MOIC metrics. Vue: `DealAnalysisView.vue`. Flask: `deals.py` + `compute_service.py`.

**Layout**: Deal Information + Capitalization → Deal-Level Summary (KPI cards) → Partner Returns (non-OP partners highlighted bold with blue-grey background) → Annual Forecast (whole-dollar formatting, DSCR as 2-decimal, blank spacer/header cells) → expandable sections (Diagnostics, Debt Service, Cash Management, Capital Calls, XIRR Cash Flows, ROE Audit, MOIC Audit).

**XIRR Cash Flows**: Merged side-by-side table with columns Date, Description (typename from `cashflow_details`), one amount column per partner, and Deal total column.

**Annual Forecast Formatting**: Black border lines under Expenses and Capital Expenditures rows (`underline-row` CSS class), black border above Total Distributions (`topline-row` CSS class).

**Excel Downloads** (Vue): Per-section download buttons ("Excel") on each section header + "Download Full Deal Analysis (Excel)" button at top of results. Uses `fetch` + `Blob` with `Authorization: Bearer` header. Sections: Partner Returns, Annual Forecast, Debt Service, Cash Management, Capital Calls, XIRR Cash Flows, ROE Audit, MOIC Audit. Full workbook combines all 7 sheets.

**Excel API Endpoints** (`/api/deals/<vcode>/excel/`): `partner-returns`, `forecast`, `debt-service`, `cash-schedule`, `capital-calls`, `xirr-cashflows`, `full` (7-sheet workbook). All GET, login_required.

**Excel Generators** (`compute_service.py`): Shared helpers `_excel_styles()`, `_write_header_row()`, `_autosize_columns()`. Per-section: `generate_partner_returns_excel()`, `generate_forecast_excel()`, `generate_debt_service_excel()`, `generate_cash_schedule_excel()`, `generate_capital_calls_excel()`, `generate_xirr_cashflows_excel()`. Full: `generate_full_deal_excel()` — 7 sheets including ROE/MOIC audit via `load_workbook` copy.

**Audit Expanders** (after XIRR Cash Flows):
- **ROE Audit — Return on Equity Breakdown**: Capital Balance Timeline table (each capital event with balance, days held, weighted capital), CF Distributions table (numerator detail), 5 metric cards per partner (Inception→End, Days/Years, CF Distributions, Wtd Avg Capital, ROE). Deal-level section with same breakdown. Excel download.
- **MOIC Audit — Multiple on Invested Capital**: Cashflow Breakdown table (Date, Description, Type, Amount), 6 metric cards per partner (Contributions, CF/Cap/Total Distributions, Unrealized NAV, MOIC). Deal-level section with note that deal MOIC uses realized distributions only. Excel download.

### 3. Property Financials
Vue: `PropertyFinancialsView.vue`. Flask: `financials.py` + `financials_service.py`. Sections in order:
- **Performance Chart** — Actual vs U/W NOI lines + occupancy bars (ECharts). Supports Monthly/Quarterly/Annually with configurable period window. Defaults to most recently completed actual period.
- **Income Statement** — Two-column comparison (TTM, YTD, Full Year, Estimate, Custom). Sources: Actual, Budget, Underwriting, Valuation. Independent left/right "As of Date" selectors for cross-period comparison. Valuation source negates MRI sign convention (`-mAmount_norm`).
- **Balance Sheet** — Two-period comparison with variance.
- **Tenant Roster** — Commercial lease data with rollover report, maturity chart, and printable HTML. $/SF displayed as currency with 2 decimal places.

### 4. One Pager
Standalone route at `/one-pager`. Vue: `OnePagerView.vue`. Flask: `financials.py` + `financials_service.py`. Professional investor report matching printed PDF layout.
- **Data Logic** (`one_pager.py`): `get_general_information()`, `get_capitalization_stack()`, `get_property_performance()`, `get_pe_performance()`, `get_one_pager_comments()`/`save_one_pager_comments()`.
- **General Information** — Partner, Asset Type, Location, Investment Strategy, Units/SF, Date Closed, Year Built, Underwritten Exit.
- **Capitalization / Exposure / Deal Terms** — Purchase Price (from deals `Acquisition_Price` or valuations), P.E. Coupon/Participation (from waterfall Pref/Share steps), Loan Terms string (maturity + rate + type), 2nd Loan Terms, Rate Cap, P.E. Yield on Exposure (NOI / (Debt + PE), computed in service layer). Capitalization table: Debt/Pref. Equity/Partner Equity/Total Cap with %. Valuation with year label. P.E. Exposure on Total Cap and on Value. Pref Equity capitalization (investor breakdown from accounting contributions).
- **Property Performance** — Table with YTD (Actual), YTD (Budget), Variance (% of budget), At Close, Actual YE, U/W YE. Rows: Economic Occ., Revenue, Expenses, NOI, DSCR. Amounts in $M, DSCR as X.XXX. Editable performance comments.
- **Preferred Equity Performance** — Committed PE, Remaining to Fund, Funded to Date, Return of Capital, Current PE Balance, Accrued Balance, Coupon, Participation, ROE to Date, U/W ROE to Date. Editable accrued pref comment.
- **Business Plan & Updates** — Editable free-text comments.
- **Occupancy vs. NOI Chart** — ECharts dual-axis. Occupancy bars + NOI U/W and NOI ACT lines. Trailing 10-12 quarters. Values in $ millions.
- **Comments** — Three editable fields (performance, accrued pref, business plan) persisted to `one_pager_comments` table per vcode + quarter. Comments are locked (read-only) when the document is in review or approved.
- **Review Workflow** — Sequential approval pipeline: Asset Manager → Head of AM → President → CCO → CEO → Approved. `ReviewPanel.vue` component shows status indicator, approve/return buttons (role-gated), and threaded review notes. Return sends document back to Draft with a required note. Comments locked when status is not draft/returned.
- **Print** — `@media print` CSS produces clean single-page output matching the PDF template. Textareas render as plain text in print. ReviewPanel hidden in print.
- **API Endpoints**: `GET /api/financials/<vcode>/one-pager` (all data), `GET /api/financials/<vcode>/one-pager/chart` (quarterly chart data), `PUT /api/financials/<vcode>/one-pager/comments` (save comments, blocked when in review).
- **Review API Endpoints** (`/api/reviews`): `GET /<vcode>/<quarter>` (status + notes + permissions), `POST /<vcode>/<quarter>/submit` (submit for review), `POST /<vcode>/<quarter>/approve` (advance step), `POST /<vcode>/<quarter>/return` (return to draft), `POST /<vcode>/<quarter>/note` (add discussion note), `GET /tracking` (production pipeline data), `GET /roles` (list assignments), `POST /roles` (assign role), `DELETE /roles/<id>` (remove role).
- **Database Tables**: `review_roles` (user↔review_role, UNIQUE), `review_submissions` (vcode+quarter, status, current_step), `review_notes` (audit trail with action/note_text). All three in `PROTECTED_TABLES`.

### 4a. Review Tracking
Standalone view at `/review-tracking` (`ReviewTrackingView.vue`). Production pipeline dashboard for One Pager approval status across all active deals.
- **Summary Cards**: Draft / In Review / Returned / Approved counts (clickable to filter).
- **Filters**: Quarter (text input), Status (dropdown). Refresh button.
- **Table**: Deal name, Quarter, Status badge, Step label, Updated date. Click row → navigates to `/one-pager?vcode=X&quarter=Y`.
- **Data**: LEFT JOINs deals with `review_submissions` so unsubmitted deals show as "Draft". Excludes sold deals and child properties.

### Review Role Management (Settings)
Admin-only section in `SettingsView.vue`. Table of current review role assignments (username + role) with remove button. Add form: select user + select review role → "Assign Review Role". Available roles: `asset_manager`, `head_am`, `president`, `cco`, `ceo`. A user can hold multiple review roles.

### 5. Ownership & Partnerships
Ownership tree visualization and relationship data.

### 6. Waterfall Setup
View, edit, and create waterfall structures for any entity. Vue: `WaterfallSetupView.vue`. Flask: `waterfall_service.py`.
- **Entity Navigation** — Selectbox of all entities with waterfalls + entities from relationships.
- **Waterfall Editor** — Editable table for CF_WF and Cap_WF steps. Columns: iOrder, PropCode, vState, FXRate, nPercent, mAmount, vtranstype, vAmtType, vNotes.
- **Validation** — Inline warnings/errors: FXRate sums, Operating Capital Add vs Tag, Pref FX=1.0, lead/tag pairing, AMFee/Promote vNotes requirements.
- **New Waterfall** — Pre-fills template from relationships/accounting: Pref steps per investor, Initial steps for Cap_WF, residual Share+Tag.
- **Actions** — Save to Database (with audit trail), Reset to Saved, Copy CF_WF->Cap_WF, Export CSV, Preview Waterfall ($100k test).
- **Guidance Panel** — Collapsible reference from `waterfall_setup_rules.txt`.

### Sidebar: Database Tools
Vue: `AppSidebar.vue` database tools section. Flask: `data.py` API endpoints.
- **Import CSVs** — Folder path input with Scan button to discover available CSVs. Supports importing individual CSVs or all at once. Protected tables (`waterfalls`, `one_pager_comments`, `waterfall_audit`, `review_roles`, `review_submissions`, `review_notes`) are never overwritten. Clears data and computation caches.
- **Export Database** — Export all tables as `waterfall_db_export_{timestamp}.zip` containing `{table_name}_db_export.csv` for every table.

### 7. Reports
Projected Returns Summary with Excel export. Vue: `ReportsView.vue`. Flask: `reports.py` + `reports_service.py`.
- **Report Type**: Extensible selector (currently: Projected Returns Summary)
- **Population Selectors**: Current Deal, Select Deals, By Partner, By Upstream Investor, All Deals
- **Output**: Partner-level rows (Contributions, CF Distributions, Capital Distributions, IRR, ROE, MOIC) plus bold deal-level total row with solid top border
- **Excel Export**: Formatted workbook via openpyxl (currency/pct/multiple formats, auto-width, deal-total rows bold with top border)

### 8. Sold Portfolio
Historical returns for sold deals computed from accounting_feed (no forecast waterfalls). Vue: `SoldPortfolioView.vue`. Flask: `sold_service.py`.
- **Data Source**: Accounting history only — contributions (`is_contribution`), distributions (`is_distribution`), capital events (`is_capital`). Raw `acct` is normalised via `normalize_accounting_feed()` on first use.
- **Pref Equity Only**: Filters out OP partners (`InvestorID` starting with "OP"). Case-insensitive InvestorID grouping handles mixed-case entity IDs.
- **Summary Table**: One row per deal + bold Portfolio Total row. Columns: Investment Name, Acquisition Date, Sale Date, Total Contributions, Total Distributions, IRR, ROE, MOIC. Portfolio Total computes IRR/ROE/MOIC from the combined cashflow pool across all deals (not simple averages).
- **Deal Detail Drill-Down**: Selectbox to pick a deal → expander with every pref equity accounting row sorted by date. Columns: Date, InvestorID, MajorType, Typename, Capital, Amount, Cashflow (XIRR), Capital Balance (running). IRR/ROE/MOIC metric cards below. Download Activity Detail exports the table + summary metrics to Excel for independent return verification.
- **Excel Exports**: Summary workbook (`sold_portfolio_returns.xlsx`) and per-deal activity detail (`sold_activity_{name}.xlsx`) via openpyxl.

### 9. PSCKOC
Upstream waterfall analysis for the PSCKOC holding entity, showing how deal-level distributions flow through PPI entities to PSCKOC members. Vue: `PsckocView.vue`. Flask: `psckoc_service.py`.
- **Members**: PSC1 (GP co-invest, Capital Units), KCREIT (LP, Capital Units), PCBLE (GP promote + AM fee recipient, Carry Units)
- **Deal Discovery**: Traces chain: relationships (PSCKOC as investor) → PPI entities → deals referencing those PPIs in waterfalls. Currently discovers 8 underlying deals.
- **Computation**: Button-gated. Runs `get_cached_deal_result()` per deal + `run_recursive_upstream_waterfalls()` for CF and Cap. Results cached in `st.session_state['_psckoc_results']`.
- **Partner Returns**: KPI cards (IRR, ROE, MOIC) per member + styled summary table with deal-level totals.
- **Income Schedule**: PSCKOC's projected income by period and source deal (CF vs Cap).
- **Waterfall Allocations**: Allocation tables showing how income is distributed among PSC1/KCREIT/PCBLE.
- **AM Fee Schedule**: Quarterly AM fee amounts (date, KCREIT balance, fee amount) per Section 6.02.
- **XIRR Cash Flows**: Combined cashflow table per member (contributions + distributions).
- **Excel Export**: 4-sheet workbook (Partner Returns, Income Schedule, AM Fee Schedule, XIRR Cash Flows).
- **New Waterfall vStates** (in `waterfall.py`):
  - `AMFee`: Post-distribution fee deducted from source investor (vNotes), paid to recipient (PropCode). Pool-neutral. `nPercent` = annual rate, `mAmount` = periods/yr.
  - `Promote`: Cumulative catch-up. `FXRate` = carry share, `nPercent` = target carry %. `vNotes` = comma-separated capital investors. Math: `E >= target/(1-target) * P`.
- **New InvestorState Fields** (`models.py`): `promote_base` (cumulative pref for catch-up denominator), `promote_carry` (cumulative carry from catch-up).

## Key Functions

### Core Engine
- `compute_deal_analysis()` - Main deal computation orchestration (compute.py)
- `build_partner_results()` - Single source of truth for all partner & deal metrics (compute.py)
- `run_interleaved_waterfalls()` - Merges CF/Cap timelines chronologically with shared InvestorState (compute.py)
- `prepare_cap_lookups()` - Pre-compute normalized DataFrames and lookup dicts for batch capitalization (compute.py)
- `xirr(cfs)` - Calculate IRR with irregular dates (metrics.py)
- `accrue_pref_to_date()` - Daily pref accrual (waterfall.py)
- `InvestorState` - Tracks capital, pref, cashflows per investor (models.py)
- `Loan` - Debt structure with fixed/variable rates (models.py)
- `get_property_vcodes_for_deal()` - Get child properties for aggregation (consolidation.py)
- `cashflows_monthly_fad()` - Monthly FAD from modeled forecast (reporting.py)
- `annual_aggregation_table()` - Annual pivot table for forecast display (reporting.py)

### One Pager Data
- `get_general_information()` - Deal general info from investment_map (one_pager.py)
- `get_capitalization_stack()` - Cap stack, loan terms, PE exposure, investor breakdown (one_pager.py)
- `get_property_performance()` - YTD/Budget/Variance/AtClose/YE metrics per quarter (one_pager.py)
- `get_pe_performance()` - PE funding, ROE, balances from accounting (one_pager.py)
- `get_one_pager_comments()` / `save_one_pager_comments()` - Comments CRUD per vcode+quarter (one_pager.py)

### Capital Calls
- `load_capital_calls()` - Load and normalize capital calls with mixed date format handling (capital_calls.py)
- `build_capital_call_schedule()` - Build list of capital call events, optionally filtered by deal (capital_calls.py)
- `apply_capital_calls_to_states()` - Apply capital calls to investor states with pool routing (capital_calls.py)

### Database
- `save_waterfall_steps()` - Replace all waterfall steps for a vcode with audit trail (database.py)
- `create_additional_tables()` - Creates capital_calls and other tables if they don't exist (database.py)
- `import_csvs_to_database()` - Refresh all tables from CSVs, protecting DB-managed tables (database.py)
- `import_single_csv()` - Import a single CSV into its database table (database.py)
- `export_all_tables_to_zip()` - Export all tables as labeled CSVs in a zip archive (database.py)

### Flask Services
- `get_cached_deal_result()` - Shared multi-deal cache wrapper (compute_service.py)
- `build_roe_audit()` / `build_moic_audit()` - Audit data builders (compute_service.py)
- `generate_roe_audit_excel()` / `generate_moic_audit_excel()` - Audit Excel workbooks (compute_service.py)
- `generate_partner_returns_excel()` - Partner Returns Excel with deal total row (compute_service.py)
- `generate_forecast_excel()` - Annual Forecast Excel pivoted by year (compute_service.py)
- `generate_debt_service_excel()` - Loan Summary + Amortization Schedule (2-sheet) (compute_service.py)
- `generate_cash_schedule_excel()` - Cash flow schedule Excel (compute_service.py)
- `generate_capital_calls_excel()` - Capital calls Excel (compute_service.py)
- `generate_xirr_cashflows_excel()` - Merged XIRR cashflows by partner (compute_service.py)
- `generate_full_deal_excel()` - 7-sheet comprehensive Deal Analysis workbook (compute_service.py)
- `get_one_pager_data()` - Aggregates all one-pager sections + computes pe_yield_on_exposure (financials_service.py)
- `get_one_pager_chart()` - Quarterly NOI chart data for one-pager (financials_service.py)
- `get_submission()` - Get or create review submission with notes and permissions (review_service.py)
- `submit_for_review()` - Submit draft/returned document for review (review_service.py)
- `approve()` - Approve at current step and advance to next (review_service.py)
- `return_to_draft()` - Return document to draft with required note (review_service.py)
- `is_editable()` - Check if comments can be edited based on review status (review_service.py)
- `get_tracking_data()` - Production tracking data with filters, LEFT JOINs deals with submissions (review_service.py)

## Account Classifications

- Revenue accounts: 4xxx series
- Expense accounts: 5xxx series
- Interest/Principal/CapEx tracked separately

## Conventions

- Cashflow signs: negative = contribution, positive = distribution
- Rates as decimals (0.08 = 8%)
- Use Python date objects for dates
