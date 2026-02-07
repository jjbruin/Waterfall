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
├── property_financials_ui.py # Property Financials tab UI (Performance Chart, IS, BS, Tenants, One Pager)
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

### 1. Deal Analysis
Main waterfall computation, partner returns, capital accounts, XIRR/MOIC metrics.

### 2. Property Financials
Rendered by `property_financials_ui.py`. Sections in order:
- **Performance Chart** — Actual vs U/W NOI lines + occupancy bars (Altair). Supports Monthly/Quarterly/Annually with configurable period window.
- **Income Statement** — Two-column comparison (TTM, YTD, Full Year, Estimate, Custom). Sources: Actual, Budget, Underwriting, Valuation.
- **Balance Sheet** — Two-period comparison with variance.
- **Tenant Roster** — Commercial lease data with rollover report, maturity chart, and printable HTML.
- **One Pager** — Investor report with cap stack, property performance, PE metrics, NOI/Occupancy chart (trailing 12 quarters), editable comments, print/export.

### 3. Ownership & Partnerships
Ownership tree visualization and relationship data.

## Key Functions

- `xirr(cfs)` - Calculate IRR with irregular dates (metrics.py)
- `accrue_pref_to_date()` - Daily pref accrual (waterfall.py)
- `InvestorState` - Tracks capital, pref, cashflows per investor (models.py)
- `Loan` - Debt structure with fixed/variable rates (models.py)
- `get_property_vcodes_for_deal()` - Get child properties for aggregation (consolidation.py)
- `_render_performance_chart()` - NOI + occupancy chart (property_financials_ui.py)
- `_build_quarterly_noi_chart()` - One Pager chart, trailing 12 quarters (one_pager_ui.py)

## Account Classifications

- Revenue accounts: 4xxx series
- Expense accounts: 5xxx series
- Interest/Principal/CapEx tracked separately

## Conventions

- Cashflow signs: negative = contribution, positive = distribution
- Rates as decimals (0.08 = 8%)
- Use Python date objects for dates
