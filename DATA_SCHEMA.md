# Waterfall XIRR — Data Schema Reference

## Table Overview

| # | Table | CSV Source | Rows | Protected | Description |
|---|-------|-----------|------|-----------|-------------|
| 1 | deals | investment_map.csv | ~103 | No | Investment properties and deal information |
| 2 | forecasts | forecast_feed.csv | ~140K | No | Monthly operating forecasts (P&L) |
| 3 | waterfalls | waterfalls.csv | ~1,485 | Yes | Cash distribution waterfall logic |
| 4 | accounting | accounting_feed.csv | ~11K | No | Historical accounting transactions |
| 5 | coa | coa.csv | ~189 | No | Chart of accounts |
| 6 | loans | MRI_Loans.csv | ~78 | No | Existing loan structures |
| 7 | valuations | MRI_Val.csv | ~213 | No | Property valuations and cap rates |
| 8 | planned_loans | MRI_Supp.csv | ~1 | Yes | Planned second mortgage parameters |
| 9 | capital_calls | MRI_Capital_Calls.csv | ~5K | No | Planned capital calls at entity level |
| 10 | relationships | MRI_IA_Relationship.csv | ~721 | No | Ownership relationships and structures |
| 11 | commitments | MRI_Commitments.csv | ~534 | No | Investor commitments to entities |
| 12 | occupancy | MRI_Occupancy_Download.csv | ~2.8K | No | Quarterly occupancy data by property |
| 13 | isbs | ISBS_Download.csv | ~798K | No | Income statement and balance sheet data |
| 14 | tenants | Tenant_Report.csv | ~1.6K | No | Commercial tenant roster by property |
| 15 | one_pager_comments | OnePager_Comments.csv | ~1 | Yes | One Pager report comments |
| 16 | fund_deals | fund_deals.csv | — | No | Fund to deal mappings |
| 17 | investor_waterfalls | investor_waterfalls.csv | — | No | LP/GP waterfall definitions |
| 18 | investor_accounting | investor_accounting.csv | — | No | LP/GP historical transactions |
| 19 | investor_roe_feed | investor_roe_feed.csv | ~10.7K | No | Investor ROE feed data |

**App-managed tables** (no CSV source — created by the application):

| # | Table | Protected | Description |
|---|-------|-----------|-------------|
| 20 | users | — | Application users (admin/admin default) |
| 21 | waterfall_audit | Yes | Audit trail for waterfall step changes |
| 22 | prospective_loans | Yes | Refinance / new mortgage proposals |
| 23 | prospective_loans_audit | Yes | Audit trail for prospective loan changes |
| 24 | review_roles | Yes | Review role assignments for approval workflow |
| 25 | review_submissions | Yes | One Pager review workflow status |
| 26 | review_notes | Yes | Discussion notes in review workflow |
| 27 | import_log | — | Data import history |
| 28 | narratives | — | Report text sections |
| 29 | report_templates | — | Report template definitions |
| 30 | calculation_cache | — | Cache for expensive calculations |

---

## Table Definitions

### 1. deals (investment_map.csv)

Investment properties and deal information. Central reference table — most other tables join through `vcode`.

| Column | Type | Description |
|--------|------|-------------|
| vcode | TEXT | Property virtual code |
| InvestmentID | TEXT | MRI investment identifier (e.g., "30BEAR") |
| Investment_Name | TEXT | Deal display name |
| Portfolio_Name | TEXT | Parent portfolio name (for sub-portfolio aggregation) |
| Asset_Type | TEXT | Property type (Office, Industrial, Residential, etc.) |
| Acquisition_Price | REAL | Purchase price |
| Sale_Date | DATE | Projected sale date |
| Sale_ME | DATE | Sale date month-end |
| Year_Built | INTEGER | Construction year |
| Units | REAL | Number of units |
| Sqft | REAL | Gross rentable square footage |
| Location | TEXT | Geographic location |
| City | TEXT | City |
| Investment_Strategy | TEXT | Strategy description |
| Cap_Rate | REAL | Underwritten cap rate (decimal, 0.06 = 6%) |
| Anticipated_Exit | DATE | Underwritten exit date |
| Cash_Reserves | REAL | Beginning cash reserves |

**Keys:**
- **Primary/Natural key:** `vcode`, `InvestmentID` (key_columns)
- No explicit PRIMARY KEY constraint (CSV-loaded table)

**Relationships:**
- `vcode` → forecasts, waterfalls, loans, occupancy, isbs, capital_calls, valuations
- `InvestmentID` → accounting.InvestmentID, relationships.InvestmentID
- `Portfolio_Name` matches another deal's `Investment_Name` for sub-portfolio grouping

---

### 2. forecasts (forecast_feed.csv)

Monthly operating forecasts — revenue, expenses, interest, principal, capex.

| Column | Type | Description |
|--------|------|-------------|
| vcode | TEXT | Property code |
| event_date | DATE | Month-end date of forecast period |
| vAccount | INTEGER | Chart of accounts code (4xxx=revenue, 5xxx=expense) |
| mAmount | REAL | Raw amount from MRI |
| mAmount_norm | REAL | Normalized amount (standardized sign convention) |
| Pro_Yr | INTEGER | Pro forma year offset |
| Year | INTEGER | Calendar year |
| vAccountType | TEXT | Account classification |

**Keys:**
- **Natural key:** `vcode`, `event_date`, `vAccount` (key_columns)
- **Indexes:** `idx_forecasts_vcode(vcode)`, `idx_forecasts_date(event_date)`, `idx_forecasts_account(vAccount)`

**Normalization rules:**
- Revenue (4xxx except contra): positive as-is
- Contra-Revenue (4040, 4043, 4030, 4042): negated
- Expenses (5xxx): negated (stored as negative)
- Interest/Principal/CapEx: negated

---

### 3. waterfalls (waterfalls.csv) — PROTECTED

Cash distribution waterfall step definitions. Two types: CF_WF (cash flow) and Cap_WF (capital events).

| Column | Type | Description |
|--------|------|-------------|
| vcode | TEXT | Property code |
| vmisc | TEXT | Waterfall type: `CF_WF` or `Cap_WF` |
| iOrder | INTEGER | Step execution order (ascending) |
| PropCode | TEXT | Investor/partner identifier |
| vState | TEXT | Step type (see below) |
| vtranstype | TEXT | Step description (e.g., "Preferred Return 8.5%") |
| vAmtType | TEXT | `Lead` or `Tag` (for paired distribution steps) |
| mAmount | REAL | Fixed amount or default value |
| nPercent | REAL | Percentage split |
| nPercent_dec | REAL | Normalized percentage (0–1 decimal) |
| FXRate | REAL | Allocation rate |
| dteffective | DATE | Effective date |
| vNotes | TEXT | Pool routing, entity routing, variable rate info |

**Keys:**
- **Natural key:** `vcode`, `vmisc`, `iOrder` (key_columns)
- **Indexes:** `idx_waterfalls_vcode(vcode)`, `idx_waterfalls_vmisc(vmisc)`

**Step types (vState):**

| vState | Description |
|--------|-------------|
| Pref | Preferred return accrual and distribution |
| Initial | Unpreferred (common equity) distributions |
| Add | Additional capital pools (Operating Capital, Cost Overrun, Special) |
| Share | Operating company equity splits |
| Tag | Paired partner distributions (linked via Lead/Tag) |
| IRR | IRR hurdle achievements |
| Def&Int | Default interest |
| AMFee | Asset management fee (post-distribution) |
| Promote | Cumulative carry-up distribution |

---

### 4. accounting (accounting_feed.csv)

Historical accounting transactions — actual contributions and distributions by investor.

| Column | Type | Description |
|--------|------|-------------|
| InvestmentID | TEXT | MRI investment identifier (maps to deals via InvestmentID) |
| InvestorID | TEXT | Investor/partner identifier (case-insensitive) |
| EffectiveDate | DATE | Transaction date |
| MajorType | TEXT | "Contribution", "Distribution", "Capital Event" |
| Typename | TEXT | Detail (e.g., "Return of Capital", "Operating Distribution") |
| Amt | REAL | Transaction amount (signed per MajorType) |
| Capital | TEXT | "Y" = capital event |

**Keys:**
- **Natural key:** `InvestmentID`, `InvestorID`, `EffectiveDate` (key_columns)
- **Indexes:** `idx_accounting_investment(InvestmentID)`, `idx_accounting_investor(InvestorID)`, `idx_accounting_date(EffectiveDate)`

**Derived flags:**
- `is_capital` = Capital == "Y"
- `is_contribution` = MajorType contains "Contribution"
- `is_distribution` = MajorType contains "Distribution"

---

### 5. coa (coa.csv)

Chart of accounts mapping account codes to classifications.

| Column | Type | Description |
|--------|------|-------------|
| vAccount | INTEGER | Account code |
| vAccountType | TEXT | Classification: Revenue, Expense, Interest, Principal, CapEx, Other |

**Keys:**
- **Natural key:** `vAccount` (key_columns — single column, effectively the primary key)

**Account ranges:** 4xxx = Revenue, 5xxx = Expense, 7xxx = Capital & Other

---

### 6. loans (MRI_Loans.csv)

Existing loan structures for debt service modeling.

| Column | Type | Description |
|--------|------|-------------|
| vCode | TEXT | Property code |
| LoanID | TEXT | Loan identifier |
| dtEvent / dtMaturity | DATE | Maturity date |
| mOrigLoanAmt | REAL | Original loan amount |
| iLoanTerm | INTEGER | Loan term (months) |
| iAmortTerm | INTEGER | Amortization term (months) |
| mNominalPenalty | INTEGER | Interest-only period (months) |
| vIntType | TEXT | "Fixed" or "Variable" |
| nRate | REAL | Fixed rate (decimal, 0.05 = 5%) |
| vIndex | TEXT | Rate index for variable loans (SOFR, LIBOR, WSJ) |
| vSpread | REAL | Spread above index (decimal) |
| nFloor | REAL | Rate floor (decimal) |
| vIntRatereset | REAL | Rate cap for variable loans |

**Keys:**
- **Natural key:** `vCode`, `LoanID` (key_columns)
- **Indexes:** `idx_loans_vcode(vCode)`

---

### 7. valuations (MRI_Val.csv)

Property valuations and cap rates over time.

| Column | Type | Description |
|--------|------|-------------|
| vcode | TEXT | Property code |
| dtVal | DATE | Valuation date |
| mAmount | REAL | Valuation amount |
| nCapRate | REAL | Cap rate (decimal) |

**Keys:**
- **Natural key:** `vcode`, `dtVal` (key_columns)

**Note:** `dtValuation` may be an Excel serial date (origin 1899-12-30).

---

### 8. planned_loans (MRI_Supp.csv) — PROTECTED

Planned second mortgage / refinance parameters.

| Column | Type | Description |
|--------|------|-------------|
| vCode | TEXT | Property code |
| Orig_Date | DATE | Planned origination date |
| mAmount | REAL | Proposed loan amount |
| iLoanTerm | INTEGER | Term (months) |
| iAmortTerm | INTEGER | Amortization (months) |
| mNominalPenalty | INTEGER | IO period (months) |
| nRate | REAL | Rate (decimal) |
| vIntType | TEXT | Fixed or Variable |

**Keys:**
- **Natural key:** `vCode`, `Orig_Date` (key_columns)

---

### 9. capital_calls (MRI_Capital_Calls.csv)

Planned capital calls at entity level.

| Column | Type | Description |
|--------|------|-------------|
| deal_name / EntityID / Vcode | TEXT | Deal identifier |
| investor_id / PropCode | TEXT | Investor identifier |
| call_date / CallDate | DATE | Date of capital call |
| amount / Amount | REAL | Call amount |
| Typename | TEXT | Capital pool routing key |
| CallType | TEXT | Call type description |
| Notes | TEXT | Additional notes |

**Date handling:** Accepts both US dates ("6/30/2026") and ISO dates ("2026-06-01") via `pd.to_datetime(format='mixed', dayfirst=False)`.

**Pool routing via Typename:**

| Typename contains | Pool |
|-------------------|------|
| "operating capital" | operating |
| "cost overrun" | cost_overrun |
| "special capital" | special |
| "additional capital" | additional |
| (default) | initial |

**Keys:**
- **Natural key:** `EntityID`, `CallDate` (key_columns — mapped from deal_name/Vcode + call_date)
- **Indexes:** `idx_capital_calls_vcode(vcode)`, `idx_capital_calls_date(CallDate)`

---

### 10. relationships (MRI_IA_Relationship.csv)

Ownership relationships between entities.

| Column | Type | Description |
|--------|------|-------------|
| InvestmentID | TEXT | Child entity / investment |
| InvestorID | TEXT | Parent investor / fund |
| RelType | TEXT | Relationship type |
| Ownership_Pct | REAL | Ownership percentage |
| CommitmentAmount | REAL | Capital commitment |

**Keys:**
- **Natural key:** `InvestmentID`, `InvestorID` (key_columns)
- **Indexes:** `idx_relationships_investment(InvestmentID)`, `idx_relationships_investor(InvestorID)`

---

### 11. commitments (MRI_Commitments.csv)

Investor commitments to entities.

| Column | Type | Description |
|--------|------|-------------|
| CommitmentUID | TEXT | Unique commitment ID |
| EntityID | TEXT | Fund/entity identifier |
| InvestorID | TEXT | Investor identifier |
| CommittedCapital | REAL | Total committed |
| FundedCapital | REAL | Amount funded |
| RemainingCommitment | REAL | Unfunded |
| CouponRate | REAL | Preferred return rate (decimal) |
| ParticipationRate | REAL | Participation rate (decimal) |

**Keys:**
- **Natural key:** `CommitmentUID`, `EntityID`, `InvestorID` (key_columns)

---

### 12. occupancy (MRI_Occupancy_Download.csv)

Quarterly occupancy data by property.

| Column | Type | Description |
|--------|------|-------------|
| vCode | TEXT | Property code |
| Qtr | TEXT | Quarter identifier |
| Occ% | REAL | Occupancy percentage |
| dtReported | REAL | Reported date (Excel serial format) |

**Keys:**
- **Natural key:** `vCode`, `Qtr` (key_columns)
- **Indexes:** `idx_occupancy_vcode(vCode)`, `idx_occupancy_qtr(Qtr)`

**Note:** Code prefers `Occ%` over `OccupancyPercent` if both exist.

---

### 13. isbs (ISBS_Download.csv)

Income Statement and Balance Sheet data — the largest table (~800K rows).

| Column | Type | Description |
|--------|------|-------------|
| vcode | TEXT | Property code |
| dtEntry | REAL/DATE | Period date (Excel serial or ISO) |
| vSource | TEXT | Data source (see below) |
| vAccount | INTEGER | Account code |
| mAmount | REAL | Period amount |
| mAmount_norm | REAL | Normalized amount |
| mYTDAmount | REAL | Year-to-date amount |
| mBudget | REAL | Budgeted amount |
| Quarter | TEXT | Derived quarter (e.g., "2025-Q1") |

**Sources:** `Interim IS` (actuals), `Budget`, `Underwriting`, `Valuation` (sign-inverted per MRI convention).

**Keys:**
- **Natural key:** `vcode`, `dtEntry`, `vSource`, `vAccount` (key_columns)
- **Indexes:** `idx_isbs_vcode(vcode)`, `idx_isbs_source(vSource)`, `idx_isbs_date(dtEntry)`

**Date parsing:** Attempts Excel serial date first (`origin='1899-12-30'`), falls back to standard datetime.

---

### 14. tenants (Tenant_Report.csv)

Commercial tenant roster by property.

| Column | Type | Description |
|--------|------|-------------|
| Code | TEXT | Property code |
| Tenant Code | TEXT | Tenant identifier |
| Tenant Name | TEXT | Tenant name |
| Suite | TEXT | Suite / unit number |
| Lease Start | DATE | Lease commencement |
| Lease End | DATE | Lease expiration |
| Rentable SF | REAL | Rentable square footage |
| Rent per SF | REAL | Annual rent per SF (displayed as currency with 2 decimals) |
| Annual Rent | REAL | Total annual rent |

**Keys:**
- **Natural key:** `Code`, `Tenant Code` (key_columns)

---

### 15–18. Fund-Level Tables

**fund_deals** (fund_deals.csv): Fund → deal mappings (`FundID`, `vcode`, `PPI_PropCode`, `Ownership_Pct`)
- **Natural key:** `FundID`, `vcode`

**investor_waterfalls** (investor_waterfalls.csv): LP/GP waterfall definitions at fund level (same schema as `waterfalls` but keyed by `FundID`)
- **Natural key:** `FundID`, `iOrder`

**investor_accounting** (investor_accounting.csv): LP/GP historical transactions (`FundID`, `InvestorID`, `EffectiveDate`, `TransType`, `Amount`)
- **Natural key:** `FundID`, `InvestorID`, `EffectiveDate`

**investor_roe_feed** (investor_roe_feed.csv): Investor ROE feed data

---

## Key Types

| Key Type | Description |
|----------|-------------|
| **PRIMARY KEY** | Database-enforced unique row identifier (auto-increment `id` or `SERIAL` for PostgreSQL) |
| **UNIQUE** | Database-enforced uniqueness constraint on one or more columns |
| **Natural key** | Logical key defined in `TABLE_DEFINITIONS.key_columns` — used for deduplication and lookups but not enforced as a DB constraint (CSV-loaded tables) |
| **Index** | Database index for query performance (created by `create_indexes()`) |
| **Foreign key** | Referential integrity constraint |

---

## App-Managed Tables

### users

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER/SERIAL | **PRIMARY KEY** (AUTOINCREMENT / SERIAL) |
| username | TEXT | User identifier |
| password_hash | TEXT | Hashed password |
| salt | TEXT | Password salt |
| role | TEXT | User role (default: 'viewer') |
| created_at | TIMESTAMP | Created time |

**Keys:**
- **Primary key:** `id` (auto-increment)
- **UNIQUE:** `username`

### review_roles

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| user_id | INTEGER | References users.id |
| review_role | TEXT | Role: asset_manager, head_am, president, cco, ceo |

**Keys:**
- **Primary key:** `id` (auto-increment)
- **UNIQUE:** (`user_id`, `review_role`)
- **Foreign key:** `user_id` → `users(id)`

### review_submissions

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| vcode | TEXT | Property code |
| quarter | TEXT | Quarter (e.g., "2025-Q1") |
| status | TEXT | draft, in_review, returned, approved |
| current_step | INTEGER | Current approval step number |
| submitted_by | INTEGER | Submitting user ID |
| returned_to_step | INTEGER | Step returned to (if returned) |
| created_at | TIMESTAMP | Created time |
| updated_at | TIMESTAMP | Last updated |

**Keys:**
- **Primary key:** `id` (auto-increment)
- **UNIQUE:** (`vcode`, `quarter`)

### review_notes

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| vcode | TEXT | Property code |
| quarter | TEXT | Quarter |
| user_id | INTEGER | Author user ID |
| username | TEXT | Author username |
| review_role | TEXT | Author's review role |
| action | TEXT | submitted, approved, returned, note |
| note_text | TEXT | Note content |
| created_at | TIMESTAMP | Created time |

**Keys:**
- **Primary key:** `id` (auto-increment)

### one_pager_comments

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| vcode | TEXT | Property code |
| reporting_period | TEXT | Quarter |
| econ_comments | TEXT | Performance comments |
| business_plan_comments | TEXT | Business plan text |
| accrued_pref_comment | TEXT | Accrued pref commentary |
| last_updated | TIMESTAMP | Last updated |

**Keys:**
- **Primary key:** `id` (auto-increment)
- **UNIQUE:** (`vcode`, `reporting_period`)
- **Indexes:** `idx_one_pager_vcode(vcode)`, `idx_one_pager_period(reporting_period)`

### prospective_loans

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| vcode | TEXT | Property code |
| loan_name | TEXT | Description |
| status | TEXT | draft, accepted, rejected |
| refi_date | TEXT | Planned refinance date |
| existing_loan_id | TEXT | Loan to replace |
| loan_amount | REAL | Proposed amount |
| lender_uw_noi | REAL | Lender underwritten NOI |
| max_ltv | REAL | Maximum LTV |
| min_dscr | REAL | Minimum DSCR |
| min_debt_yield | REAL | Minimum debt yield |
| interest_rate | REAL | Rate (decimal) |
| rate_spread_bps | INTEGER | Rate spread in basis points |
| rate_index | TEXT | Rate index (SOFR, etc.) |
| term_years | INTEGER | Term |
| amort_years | INTEGER | Amortization |
| io_years | REAL | IO period |
| int_type | TEXT | Fixed / Variable |
| closing_costs | REAL | Closing costs |
| reserve_holdback | REAL | Reserve holdback amount |
| notes | TEXT | Notes |
| created_by | TEXT | Creator |
| created_at | TIMESTAMP | Created |

**Keys:**
- **Primary key:** `id` (auto-increment)

### prospective_loans_audit

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| loan_id | INTEGER | References prospective_loans.id |
| action | TEXT | Audit action |
| vcode | TEXT | Property code |
| loan_name | TEXT | Loan description |
| status | TEXT | Status at time of action |
| loan_amount | REAL | Amount at time of action |
| all_fields | TEXT | JSON snapshot of all fields |
| changed_by | TEXT | User who made change |
| changed_at | TIMESTAMP | Change time |

**Keys:**
- **Primary key:** `id` (auto-increment)

### waterfall_audit

| Column | Type | Description |
|--------|------|-------------|
| audit_id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| vcode | TEXT | Property code |
| action | TEXT | Audit action |
| changed_by | TEXT | User who made change |
| changed_at | TIMESTAMP | Change time |
| details | TEXT | Change details |

**Keys:**
- **Primary key:** `audit_id` (auto-increment)

### narratives

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| vcode | TEXT | Property code |
| section_name | TEXT | Report section |
| content | TEXT | Narrative text |
| last_updated | TIMESTAMP | Last updated |
| updated_by | TEXT | Last editor |

**Keys:**
- **Primary key:** `id` (auto-increment)
- **UNIQUE:** (`vcode`, `section_name`)

### report_templates

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| template_name | TEXT | Template identifier |
| sections | TEXT | JSON array of section names |
| format | TEXT | Output format (default: 'PDF') |
| created_at | TIMESTAMP | Created time |

**Keys:**
- **Primary key:** `id` (auto-increment)
- **UNIQUE:** `template_name`

### import_log

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| table_name | TEXT | Table that was imported |
| rows_imported | INTEGER | Row count |
| import_mode | TEXT | Import method |
| imported_at | TIMESTAMP | Import time |
| imported_by | TEXT | User/system |
| source_file | TEXT | Source file name |

**Keys:**
- **Primary key:** `id` (auto-increment)

### calculation_cache

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | **PRIMARY KEY** (AUTOINCREMENT) |
| vcode | TEXT | Property code |
| calculation_type | TEXT | Type of calculation |
| input_hash | TEXT | Hash of input parameters |
| result_json | TEXT | Cached result |
| calculated_at | TIMESTAMP | Calculation time |

**Keys:**
- **Primary key:** `id` (auto-increment)
- **UNIQUE:** (`vcode`, `calculation_type`, `input_hash`)

---

## Key Identifier Mappings

```
deals.vcode ─────────────┬── forecasts.vcode
  (P0000001)             ├── waterfalls.vcode
                         ├── loans.vCode
                         ├── occupancy.vCode
                         ├── isbs.vcode
                         ├── capital_calls.deal_name
                         ├── valuations.vcode
                         └── tenants.Code

deals.InvestmentID ──────┬── accounting.InvestmentID
  (30BEAR)               └── relationships.InvestmentID

waterfalls.PropCode ─────── accounting.InvestorID
  (investor identifier)     (same namespace)
```

## Sign Conventions

| Context | Negative | Positive |
|---------|----------|----------|
| Forecast (normalized) | Expenses | Revenue |
| Accounting | Contributions (investor outflow) | Distributions (investor inflow) |
| XIRR cashflows | Contributions | Distributions |
| Rates | — | Always decimal (0.08 = 8%) |

## Percentages

All percentages are stored as decimals (0–1) after normalization. Input may be 0–100 range; the code normalizes via `x / 100 if x > 1 else x`.

## Date Handling

| Source | Format | Parsing |
|--------|--------|---------|
| Forecast dates | ISO (YYYY-MM-DD) | Standard |
| Accounting dates | Mixed | `pd.to_datetime` |
| Capital call dates | US or ISO | `format='mixed', dayfirst=False` |
| ISBS dtEntry | Excel serial | `origin='1899-12-30'` with fallback |
| Occupancy dtReported | Excel serial | `origin='1899-12-30'` |
| Valuation dtVal | Excel serial or ISO | Try serial first |

## Protected Tables

Never overwritten by CSV import:

```
waterfalls, one_pager_comments, waterfall_audit,
review_roles, review_submissions, review_notes,
prospective_loans, prospective_loans_audit, planned_loans
```

These are managed exclusively through the application UI.
