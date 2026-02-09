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
├── waterfall_setup_ui.py     # Waterfall Setup tab UI (view, edit, create waterfall structures)
├── one_pager_ui.py           # One Pager Investor Report UI (Streamlit components)
├── one_pager.py              # One Pager data logic (performance calcs, cap stack, PE metrics)
├── models.py                 # Data classes (InvestorState, Loan)
├── waterfall.py              # Waterfall calculation engine
├── metrics.py                # XIRR, XNPV, ROE, MOIC calculations
├── loaders.py                # Data loading from database/CSV
├── database.py               # SQLite management, migrations, table definitions
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
Main waterfall computation, partner returns, capital accounts, XIRR/MOIC metrics. Debt service display rendered by `debt_service_ui.py` (Loan Summary, Detailed Amortization Schedules, Sale Proceeds Calculation).

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
- **Entity Navigation** — Selectbox of all entities with waterfalls + entities from relationships. Mini ownership tree and investor list.
- **Waterfall Editor** — `st.data_editor` with `num_rows="dynamic"` for CF_WF and Cap_WF steps. Columns: iOrder, PropCode, vState, FXRate, nPercent, mAmount, vtranstype, vAmtType, vNotes.
- **Validation** — Inline warnings/errors: FXRate sums, Operating Capital Add vs Tag, Pref FX=1.0, lead/tag pairing.
- **New Waterfall** — Pre-fills template from relationships/accounting: Pref steps per investor, Initial steps for Cap_WF, residual Share+Tag.
- **Actions** — Save to Database (with audit trail), Reset to Saved, Copy CF_WF->Cap_WF, Export CSV, Preview Waterfall ($100k test).
- **Guidance Panel** — Collapsible reference from `waterfall_setup_rules.txt`: vState reference, Add vs Tag rule, pool routing table, common patterns, modeling checklist.

### 6. Reports
Rendered by `reports_ui.py`. Projected Returns Summary with Excel export.
- **Report Type**: Extensible selector (currently: Projected Returns Summary)
- **Population Selectors**: Current Deal, Select Deals, By Partner, By Upstream Investor, All Deals
- **Output**: Partner-level rows (Contributions, CF Distributions, Capital Distributions, IRR, ROE, MOIC) plus bold deal-level total row with solid top border
- **Excel Export**: Formatted workbook via openpyxl (currency/pct/multiple formats, auto-width, deal-total rows bold with top border)

## Key Functions

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
- `render_reports()` - Reports tab entry point (reports_ui.py)
- `_build_partner_returns()` - Partner + deal-level metrics from compute result (reports_ui.py)
- `_generate_excel()` - Formatted Excel workbook via openpyxl (reports_ui.py)

## Account Classifications

- Revenue accounts: 4xxx series
- Expense accounts: 5xxx series
- Interest/Principal/CapEx tracked separately

## Conventions

- Cashflow signs: negative = contribution, positive = distribution
- Rates as decimals (0.08 = 8%)
- Use Python date objects for dates
