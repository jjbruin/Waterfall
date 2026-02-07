# Waterfall Model - Complete Documentation

**Last Updated:** February 2026

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
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

A Streamlit-based financial modeling application for calculating investment waterfalls, XIRR, and related performance metrics for real estate investments. Supports multi-layer distribution waterfalls with preferred returns, capital accounts, and investor-level tracking.

### Tech Stack

- **Python 3.x** with virtual environment (`.venv/`)
- **Streamlit** - Web UI framework
- **pandas/numpy** - Data manipulation
- **scipy** - XIRR/NPV calculations (Brent's method)
- **altair** - Interactive charts (performance chart, lease maturity)
- **SQLite** - Local database (`waterfall.db`)

### Project Structure

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
├── consolidation.py          # Sub-portfolio deal/property aggregation
├── portfolio.py              # Fund/portfolio aggregation
├── reporting.py              # Report generation
├── ownership_tree.py         # Investor ownership structures
├── utils.py                  # Helper utilities
└── waterfall.db              # SQLite database (not in git, >100MB)
```

### Module Reference

| Module | Purpose |
|--------|---------|
| `app.py` | Streamlit entry point, sidebar controls, data loading, tab routing |
| `compute.py` | Deal computation orchestration (extracted from app.py for caching) |
| `property_financials_ui.py` | Property Financials tab: Performance Chart, IS, BS, Tenants, One Pager |
| `one_pager_ui.py` | One Pager UI: quarter selector, section renderers, NOI/Occupancy chart, print/export |
| `one_pager.py` | One Pager data: property performance, cap stack, PE metrics, comments CRUD |
| `waterfall.py` | Pref accrual, waterfall step processing, investor state management |
| `loans.py` | Loan amortization schedules |
| `consolidation.py` | Sub-portfolio aggregation (deals with properties) |
| `ownership_tree.py` | Multi-tier ownership tracing |
| `metrics.py` | XIRR, ROE, MOIC calculations |
| `capital_calls.py` | Capital calls processing |
| `cash_management.py` | Cash reserves and CapEx management |
| `database.py` | Database operations, table definitions, migrations, indexes |
| `reporting.py` | Annual tables and formatting |
| `config.py` | BS_ACCOUNTS, IS_ACCOUNTS, capital pool routing, default settings |

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
streamlit run app.py
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
| `MRI_Investor_ROE_Feed.csv` | Investor ROE data |
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
| `MRI_Investor_ROE_Feed.csv` | All investor financial activity |
| `MRI_Capital_Calls.csv` | Planned future calls |

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
- Wrapped in `st.form()` with "Apply Settings" button

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

### Validation Checks

1. **Commitment percentages sum to 100%** for each entity
2. **Funded <= Committed** for each investor
3. **Capital calls <= Unfunded balance**
4. **Outstanding equity >= 0**
5. **FXRates at each iOrder sum to 1.0** (typically)

### Database Issues

- **"File not found"**: Check CSV path and filename spelling
- **"Database is locked"**: Close Streamlit app and Excel
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
| 2.3 | Feb 7 2026 | Performance chart (NOI + occupancy), occupancy table, One Pager chart upgrade, section reorder (IS before BS) |
| 2.2 | Feb 2026 | Property Financials tab extraction (IS, BS, Tenants, One Pager into property_financials_ui.py), compute.py extraction |
| 2.1 | Feb 2026 | Loan aggregation for sub-portfolios, 45-day grace period compounding, NaT handling |
| 2.0 | Jan 2026 | Capital calls, cash management, database migration |
| 1.0 | Initial | Core waterfall engine, ownership tree, metrics |
