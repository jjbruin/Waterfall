# Waterfall XIRR - Multi-Layer Waterfall Model

## Project Overview

A Streamlit-based financial modeling application for calculating investment waterfalls, XIRR, and related performance metrics for real estate investments. The application supports multi-layer distribution waterfalls with preferred returns, capital accounts, and investor-level tracking.

## Tech Stack

- **Python 3.x** with virtual environment (`.venv/`)
- **Streamlit** - Web UI framework
- **pandas/numpy** - Data manipulation
- **scipy** - XIRR/NPV calculations (Brent's method)
- **SQLite** - Local database (`waterfall.db`)

## Project Structure

```
waterfall-xirr/
├── app.py              # Main Streamlit entry point
├── config.py           # Constants, account classifications, rates
├── models.py           # Data classes (InvestorState, Loan)
├── waterfall.py        # Waterfall calculation engine
├── metrics.py          # XIRR, XNPV, ROE, MOIC calculations
├── loaders.py          # Data loading from database/CSV
├── database.py         # SQLite management, migrations
├── loans.py            # Debt service modeling
├── planned_loans.py    # Future loan projections
├── capital_calls.py    # Capital call handling
├── cash_management.py  # Cash flow management
├── consolidation.py    # Sub-portfolio aggregation
├── portfolio.py        # Fund/portfolio aggregation
├── reporting.py        # Report generation
├── ownership_tree.py   # Investor ownership structures
├── utils.py            # Helper utilities
└── waterfall.db        # SQLite database
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

## Key Functions

- `xirr(cfs)` - Calculate IRR with irregular dates (metrics.py)
- `accrue_pref_to_date()` - Daily pref accrual (waterfall.py)
- `InvestorState` - Tracks capital, pref, cashflows per investor (models.py)
- `Loan` - Debt structure with fixed/variable rates (models.py)
- `get_property_vcodes_for_deal()` - Get child properties for aggregation (consolidation.py)

## Account Classifications

- Revenue accounts: 4xxx series
- Expense accounts: 5xxx series
- Interest/Principal/CapEx tracked separately

## Conventions

- Cashflow signs: negative = contribution, positive = distribution
- Rates as decimals (0.08 = 8%)
- Use Python date objects for dates
