# One Pager Audit Variance Analysis — Q1 2026

Audit comparison of Azure One Pager vs Excel model across 39 deals (959 total discrepancies).
PDF report generated: `OnePager_Audit_Variance_Analysis.pdf` (project root).
Source audit file: `audit_comparison_final.xlsx` (SharePoint / Downloads).

## Status: Pending Team Review (Jul 22, 2026)

Team is reviewing recommendations. No code changes made yet.

## Code/Logic Bugs to Fix (~293 discrepancies, 31%)

### Bug 1: DSCR Principal Accounts (P0) — ~87 rows
- `one_pager.py:566-568` uses BS accounts {2145, 2150, 2152, 2154, 2156} for debt service principal
- These accounts don't exist in vSource='Interim IS', so principal = $0, DSCR = NOI/Interest only
- Should use IS account {7060} per `config.py:53` PRINCIPAL_ACCTS
- **Fix**: Change line 568 to `'Principal': ['7060'],`
- Example: 30 Bearfoot DSCR: Azure 2.96X vs Excel 1.72X

### Bug 2: PE Capitalization Shows PPI Entities (P0) — 39 rows
- `one_pager.py:495-518` reads InvestorID from accounting directly
- Accounting records contributions at PPI entity level (e.g., PPI27), not underlying investors
- Excel maps through PPI to actual investors (PSC 69%, Declaration 31%)
- **Fix**: Add logic to resolve PPI entities to upstream investors via ownership/relationships table

### Bug 3: PE Exposure on Value Formula (P1) — ~22 rows
- Azure shows ~1% while Excel shows ~107% — wrong denominator or valuation amount
- Example: 45th & Main: Azure 1.4% vs Excel 107.2%
- **Fix**: Verify pe_exposure_on_value uses pref_equity / most_recent_valuation

### Bug 4: Capitalization Debt Source (P1) — ~53 rows
- Construction loans: Azure falls back to MRI mOrigLoanAmt (commitment) instead of ISBS drawn balance
- Example: 45th & Main Debt: Azure $47.0M (commitment) vs Excel $31.4M (drawn)
- **Fix**: Ensure ISBS Interim BS data loaded for Q1 2026; verify quarter-date selection logic

### Bug 5: ROE to Date (P1) — ~30 rows
- Example: 30 Bearfoot: Azure 21.3% vs Excel 14.3%
- **Fix**: Step-through audit of ROE calculation for 30 Bearfoot vs Excel to isolate formula vs data

### Bug 6: Economic Occupancy Variances (P2) — ~56 rows
- Small but consistent differences in Variance, Projected YE, Budget occupancy
- Likely different budget_econ_occ data source or bad debt formula
- **Fix**: Audit budget_econ_occ source and bad debt formula (4040+4043 / abs(4010))

### Bug 7: Chart Issues (P2) — 17 missing quarters, 24 value mismatches
- Town Fair Tire: values "shifted one quarter late"
- **Fix**: Audit period-end alignment in cumulative_to_periodic and aggregate_periodic()

## Data Gaps for Team (~540 discrepancies, 56%)

| Pri | Action | Owner | Count |
|-----|--------|-------|-------|
| P0 | Import at_close_noi (Prop_Info_AtClose.sql) + deal_terms (Prop_Info_DealTerms.sql) | Data Team | ~200 |
| P0 | Import/verify ISBS Projected IS (all deals, Dec 31 rows) | Data Team | ~62 |
| P1 | Enter Q1 2026 Business Plan comments for all 39 deals | Asset Managers | 39 |
| P1 | Enter Q1 2026 Accrued Pref comments for all 39 deals | Asset Managers | 39 |
| P1 | Verify/update Anticipated_Exit dates in investment map | Data Team | 39 |
| P2 | Import MRI valuations, PE commitments, ISBS gaps | Data Team | ~35 |
| P3 | Misc: Location, Year Built, Partner names, Rate Cap | Data Team | ~8 |

## Architecture Differences — By Design (~126 discrepancies, 13%)

- **Date Closed** (39): Azure derives from earliest accounting activity (intentional, per CLAUDE.md)
- **Loan Terms format** (14): Different display format, same data
- **Units/SF rounding** (21): Cosmetic 1K rounding differences
- **U/W ROE availability** (~29 partial): Azure computes from Projected IS acct 7071; Excel doesn't for some deals
- **PE Yield on Exposure** (~22 partial): Depends on upstream cap stack/NOI fixes

## 19 Deals Not Found in Azure
Child properties (Burton subs, Apple subs, PMAT subs), sold deals (Crowne Plaza, Plaza Del Mar, etc.), or entity-level records. Expected — consolidated under parent deals.
