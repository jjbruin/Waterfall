# Waterfall XIRR - Multi-Layer Waterfall Model

## Project Overview

A Streamlit-based financial modeling application for calculating investment waterfalls, XIRR, and related performance metrics for real estate investments. The application supports multi-layer distribution waterfalls with preferred returns, capital accounts, and investor-level tracking.

## Tech Stack

- **Python 3.x** with virtual environment (`.venv/`)
- **Streamlit** - Web UI framework
- **pandas/numpy** - Data manipulation
- **scipy** - XIRR/NPV calculations (Brent's method)
- **altair** - Interactive charts (performance chart, lease maturity)
- **SQLite** - Local database (`waterfall.db`)

## Project Structure

```
waterfall-xirr/
├── app.py                    # Main Streamlit entry point & sidebar
├── config.py                 # Constants, account classifications, rates
├── compute.py                # Deal computation logic (extracted from app.py)
├── dashboard_ui.py           # Dashboard tab UI (KPI cards, portfolio charts, computed returns)
├── debt_service_ui.py        # Debt Service display (Loan Summary, Amortization Schedules, Sale Proceeds)
├── property_financials_ui.py # Property Financials tab UI (Performance Chart, IS, BS, Tenants, One Pager)
├── reports_ui.py             # Reports tab UI (Projected Returns Summary, Excel export)
├── sold_portfolio_ui.py      # Sold Portfolio tab UI (historical returns from accounting)
├── psckoc_ui.py              # PSCKOC tab UI (upstream entity analysis, member returns)
├── waterfall_setup_ui.py     # Waterfall Setup tab UI (view, edit, create waterfall structures)
├── one_pager_ui.py           # One Pager Investor Report UI (Streamlit components)
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
├── consolidation.py          # Sub-portfolio aggregation
├── portfolio.py              # Fund/portfolio aggregation
├── reporting.py              # Report generation
├── ownership_tree.py         # Investor ownership structures
├── utils.py                  # Helper utilities
└── waterfall.db              # SQLite database (not in git, >100MB)
```

## Documentation

- **DOCUMENTATION.md** - Complete project documentation (setup, data files, concepts, troubleshooting)
- **waterfall_setup_rules.txt** - Waterfall step configuration guide for deal modeling team
- **typename_rules.txt** - Capital pool routing rules based on Typename field

## Running the Application

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run Streamlit app
streamlit run app.py
```

## Key Concepts

### Waterfall Types
- **CF Waterfall**: Operating cash distributions (does NOT reduce capital outstanding)
- **Capital Waterfall**: Refi/sale proceeds (DOES reduce capital outstanding)

### Preferred Returns
- Daily accrual using Act/365 Fixed day count
- Compounds annually on 12/31 (with 45-day grace period)
- Tracked per investor via `InvestorState`

### Sub-Portfolio Aggregation
- Deals can have child properties linked via `Portfolio_Name`
- Loans aggregate UP from properties to parent deal level
- See `consolidation.py` for implementation

## Application Tabs

### 1. Dashboard
Rendered by `dashboard_ui.py`. Executive portfolio-level view with instant-load KPIs and charts.
- **KPI Cards** (6): Portfolio Value, Debt Outstanding, Wtd Avg Cap Rate, Portfolio Occupancy, Deal Count, Total Equity
- **Portfolio NOI Trend** — Dual-axis Altair chart (occupancy bars + Actual/U/W NOI lines) aggregated across all deals. Values in $ million. Frequency and period-end selectors (defaults to most recently ended quarter), trailing 12 periods. Capped at last closed quarter.
- **Portfolio Capital Structure** — Consolidated vertical stacked bar (Debt blue / Pref Equity green / OP Equity grey) with Avg LTV and Pref Exposure annotations at dividing lines. Values in $ million.
- **Occupancy by Type** — Horizontal bars showing weighted-average occupancy per Asset_Type, colored above/below portfolio average with dashed reference line.
- **Asset Allocation** — Donut chart by Asset_Type sized by preferred equity, with % of total in legend table and hover tooltips.
- **Loan Maturities** — Stacked bar chart by maturity year (Fixed blue / Floating orange) with weighted avg rate labels on fixed sections, total dollar labels, and "Show Data" expander with loan-level detail table (includes child property loans).
- **Computed Returns** (button-gated) — Progress bar → IRR by Deal bar chart + formatted summary table (Contributions, Distributions, IRR, ROE, MOIC)

### 2. Deal Analysis
Wrapped in `_deal_analysis_fragment()` (`@st.fragment`, defined at module level in `app.py`). Main waterfall computation, partner returns, capital accounts, XIRR/MOIC metrics. Debt service display rendered by `debt_service_ui.py` (Loan Summary, Detailed Amortization Schedules, Sale Proceeds Calculation). Deal switching only reruns this fragment — not all six tabs. Cross-tab state (`_current_deal_vcode`, `_current_fc_deal_modeled`) stored in session_state for Property Financials and Ownership tabs.

**Audit Expanders** (after XIRR Cash Flows):
- **ROE Audit — Return on Equity Breakdown**: Capital Balance Timeline table (each capital event with balance, days held, weighted capital), CF Distributions table (numerator detail), 5 metric cards per partner (Inception→End, Days/Years, CF Distributions, Wtd Avg Capital, ROE). Deal-level section with same breakdown. Excel download.
- **MOIC Audit — Multiple on Invested Capital**: Cashflow Breakdown table (Date, Description, Type, Amount), 6 metric cards per partner (Contributions, CF/Cap/Total Distributions, Unrealized NAV, MOIC). Deal-level section with note that deal MOIC uses realized distributions only. Excel download.

### 3. Property Financials
Rendered by `property_financials_ui.py`. Sections in order:
- **Performance Chart** — Actual vs U/W NOI lines + occupancy bars (Altair). Supports Monthly/Quarterly/Annually with configurable period window.
- **Income Statement** — Two-column comparison (TTM, YTD, Full Year, Estimate, Custom). Sources: Actual, Budget, Underwriting, Valuation.
- **Balance Sheet** — Two-period comparison with variance.
- **Tenant Roster** — Commercial lease data with rollover report, maturity chart, and printable HTML.
- **One Pager** — Investor report with cap stack, property performance, PE metrics, NOI/Occupancy chart (trailing 12 quarters), editable comments, print/export.

### 4. Ownership & Partnerships
Ownership tree visualization and relationship data.

### 5. Waterfall Setup
Rendered by `waterfall_setup_ui.py`. View, edit, and create waterfall structures for any entity.
- **Entity Navigation** — Selectbox of all entities with waterfalls + entities from relationships. Mini ownership tree and investor list. Default set once from Deal Analysis deal; user's explicit selection persists across reruns (not overridden by `_current_deal_vcode`).
- **Waterfall Editor** — `st.data_editor` with `num_rows="dynamic"` for CF_WF and Cap_WF steps. Columns: iOrder, PropCode, vState, FXRate, nPercent, mAmount, vtranstype, vAmtType, vNotes. Draft (`_wf_draft|{entity_id}|{wf_type}`) is a stable base — only updated on Save/Reset/Copy, never on render (avoids editor reinitialization that discards edits).
- **Validation** — Inline warnings/errors: FXRate sums, Operating Capital Add vs Tag, Pref FX=1.0, lead/tag pairing, AMFee/Promote vNotes requirements.
- **New Waterfall** — Pre-fills template from relationships/accounting: Pref steps per investor, Initial steps for Cap_WF, residual Share+Tag.
- **Actions** — Save to Database (with audit trail, returns bool), Reset to Saved (clears editor widget state), Copy CF_WF->Cap_WF (clears Cap_WF editor state), Export CSV, Preview Waterfall ($100k test).
- **Guidance Panel** — Collapsible reference from `waterfall_setup_rules.txt`: vState reference, Add vs Tag rule, pool routing table, common patterns, modeling checklist.

### Sidebar: Database Tools (SQLite mode)
Expander in sidebar when using SQLite Database mode.
- **Import CSVs** — Folder path input + "Import All CSVs" button. Refreshes all MRI-sourced tables from CSVs. Protected tables (`waterfalls`, `one_pager_comments`, `waterfall_audit`) are never overwritten. Shows summary (updated/protected/skipped/errors). Clears data and computation caches.
- **Export Database** — "Prepare Export" button builds zip in session_state. "Download Database Export" button for `waterfall_db_export_{timestamp}.zip` containing `{table_name}_db_export.csv` for every table.

### 6. Reports
Rendered by `reports_ui.py`. Projected Returns Summary with Excel export.
- **Report Type**: Extensible selector (currently: Projected Returns Summary)
- **Population Selectors**: Current Deal, Select Deals, By Partner, By Upstream Investor, All Deals
- **Output**: Partner-level rows (Contributions, CF Distributions, Capital Distributions, IRR, ROE, MOIC) plus bold deal-level total row with solid top border
- **Excel Export**: Formatted workbook via openpyxl (currency/pct/multiple formats, auto-width, deal-total rows bold with top border)

### 7. Sold Portfolio
Rendered by `sold_portfolio_ui.py`. Historical returns for sold deals computed from accounting_feed (no forecast waterfalls).
- **Data Source**: Accounting history only — contributions (`is_contribution`), distributions (`is_distribution`), capital events (`is_capital`). Raw `acct` is normalised via `normalize_accounting_feed()` on first use.
- **Pref Equity Only**: Filters out OP partners (`InvestorID` starting with "OP"). Case-insensitive InvestorID grouping handles mixed-case entity IDs.
- **Summary Table**: One row per deal + bold Portfolio Total row. Columns: Investment Name, Acquisition Date, Sale Date, Total Contributions, Total Distributions, IRR, ROE, MOIC. Portfolio Total computes IRR/ROE/MOIC from the combined cashflow pool across all deals (not simple averages).
- **Deal Detail Drill-Down**: Selectbox to pick a deal → expander with every pref equity accounting row sorted by date. Columns: Date, InvestorID, MajorType, Typename, Capital, Amount, Cashflow (XIRR), Capital Balance (running). IRR/ROE/MOIC metric cards below. Download Activity Detail exports the table + summary metrics to Excel for independent return verification.
- **Excel Exports**: Summary workbook (`sold_portfolio_returns.xlsx`) and per-deal activity detail (`sold_activity_{name}.xlsx`) via openpyxl.

### 8. PSCKOC
Rendered by `psckoc_ui.py`. Upstream waterfall analysis for the PSCKOC holding entity, showing how deal-level distributions flow through PPI entities to PSCKOC members.
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

- `get_cached_deal_result()` - Shared multi-deal cache wrapper; all consumers call this instead of `compute_deal_analysis()` directly (compute.py)
- `build_partner_results()` - Single source of truth for all partner & deal metrics (compute.py)
- `xirr(cfs)` - Calculate IRR with irregular dates (metrics.py)
- `accrue_pref_to_date()` - Daily pref accrual (waterfall.py)
- `InvestorState` - Tracks capital, pref, cashflows per investor (models.py)
- `Loan` - Debt structure with fixed/variable rates (models.py)
- `get_property_vcodes_for_deal()` - Get child properties for aggregation (consolidation.py)
- `_render_performance_chart()` - NOI + occupancy chart (property_financials_ui.py)
- `_build_quarterly_noi_chart()` - One Pager chart, trailing 12 quarters (one_pager_ui.py)
- `render_debt_service()` - Debt service display: loan summary, amortization, sale proceeds (debt_service_ui.py)
- `render_dashboard()` - Dashboard tab entry point (dashboard_ui.py)
- `_get_portfolio_caps()` - Lightweight cap data per deal, cached in session_state (dashboard_ui.py)
- `_render_computed_returns()` - Button-gated IRR/MOIC computation (dashboard_ui.py)
- `render_waterfall_setup()` - Waterfall Setup tab entry point (waterfall_setup_ui.py)
- `save_waterfall_steps()` - Replace all waterfall steps for a vcode with audit trail (database.py)
- `import_csvs_to_database()` - Refresh tables from CSVs, protecting DB-managed tables (database.py)
- `export_all_tables_to_zip()` - Export all tables as labeled CSVs in a zip archive (database.py)
- `render_reports()` - Reports tab entry point (reports_ui.py)
- `_build_partner_returns()` - Partner + deal-level metrics from compute result (reports_ui.py)
- `_generate_excel()` - Formatted Excel workbook via openpyxl (reports_ui.py)
- `_build_roe_timeline()` - Replay ROE calculation with audit trail: timeline df, cf distributions df, summary (app.py)
- `_build_moic_breakdown()` - MOIC cashflow breakdown with Type classification (app.py)
- `_generate_roe_audit_excel()` - Formatted ROE audit workbook with per-partner + deal-level sections (app.py)
- `_generate_moic_audit_excel()` - Formatted MOIC audit workbook with per-partner + deal-level sections (app.py)
- `render_sold_portfolio()` - Sold Portfolio tab entry point (sold_portfolio_ui.py)
- `_compute_all_sold_returns()` - Pref-equity returns from accounting for sold deals, with portfolio total (sold_portfolio_ui.py)
- `_build_deal_detail()` - Cashflow detail table for a single sold deal with running capital balance (sold_portfolio_ui.py)
- `_generate_detail_excel()` - Activity detail Excel with summary metrics for return verification (sold_portfolio_ui.py)
- `render_psckoc_tab()` - PSCKOC tab entry point (psckoc_ui.py)
- `_find_psckoc_deals()` - Trace deal chain: relationships → PPI entities → underlying deals (psckoc_ui.py)
- `_run_psckoc_computation()` - Run deal computations + upstream waterfalls for PSCKOC (psckoc_ui.py)
- `_generate_psckoc_excel()` - 4-sheet PSCKOC workbook via openpyxl (psckoc_ui.py)

## Account Classifications

- Revenue accounts: 4xxx series
- Expense accounts: 5xxx series
- Interest/Principal/CapEx tracked separately

## Conventions

- Cashflow signs: negative = contribution, positive = distribution
- Rates as decimals (0.08 = 8%)
- Use Python date objects for dates
