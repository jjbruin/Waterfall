# One Pager Audit Variance Analysis — Q1 2026

Audit comparison of Azure One Pager vs Excel model across 39 deals (959 total discrepancies).
PDF report generated: `OnePager_Audit_Variance_Analysis.pdf` (project root).
Source audit file: `audit_comparison_final.xlsx` (SharePoint / Downloads).

## Status: v168 Deployed (Jul 30, 2026)
**Azure Container App**: `app-waterfall-dev-v2` (renamed from `app-waterfall-dev`).
**URL**: `https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io`

- **Fix 1+7 (DSCR)**: Deployed v144. Principal from IS acct 7060 (YTD Actual), BS balance change fallback. U/W uses acct 7010.
- **Fix 5 (Budget Econ Occ)**: Deployed v144. Bad debt % deducted from Budget IS (4040+4041+4043 / abs(4010)).
- **Fix 2 (PE Exposure)**: Already correct — verified with Ascent on Steamboat (107.5% match). 45th & Main gap is Bug 4 (construction draws).
- **Fix 3 (At Close)**: Investigated — see below. Data gap, not code bug.
- **Fix 4 (ROE to Date)**: Investigated — see below. Methodology difference, not data freshness.
- **Fix 6 (Chart Quarters)**: Investigated — see below. Code correct, data freshness issue.

### One Pager Snapshot System (v164, Jul 30, 2026)
- **Approved snapshot**: On CEO final approval, all computed One Pager data + chart frozen into `one_pager_snapshots` table as JSON.
- **Table**: `one_pager_snapshots` (vcode, quarter, snapshot_data JSON, approved_by, approved_at). UNIQUE(vcode, quarter). In `PROTECTED_TABLES`.
- **Hook**: `_save_snapshot()` in `review_service.py`, triggered when `approve()` advances to status "approved" (step 5).
- **API**: `GET /api/financials/<vcode>/one-pager/snapshot?quarter=X` returns frozen data. Review status includes `has_snapshot` flag.
- **Vue**: "View Approved Version" / "View Live Data" toggle button. Blue banner shows approver + date. Comments read-only from snapshot. All computed sections (gen, cap, perf, pe, chart) render from frozen data.

### Print Layout (v157-v163, Jul 30, 2026)
- Business plan textarea replaced with `print-hide`/`print-only` div pattern for clean print rendering.
- All font sizes +2px across categories. `.op-sheet` set to exact page height with `overflow: hidden` for single-page constraint.
- Section headers all uppercase (GENERAL INFORMATION, PROPERTY PERFORMANCE, etc.).
- Uniform section spacing, flex layout for business plan to fill available space.

### Desktop Shortcut Installer (v166-v168, Jul 30, 2026)
- **One-click install**: `GET /auth/shortcut/install` returns a `.bat` file that creates "Waterfall XIRR" desktop shortcut with custom icon.
- **Icon**: `GET /auth/shortcut/icon` serves `waterfall_xirr.ico` from project root. Downloaded via `curl.exe` (Windows 10+ built-in, corporate proxy compatible).
- **Tech**: Base64-encoded PowerShell (`-EncodedCommand`) wrapped in `.bat` for non-technical users. Icon cached in `%LOCALAPPDATA%\WaterfallXIRR\`.
- **Welcome email**: Green "Download Desktop Shortcut Setup" button with step-by-step instructions added to `send_welcome_email()` in `email_utils.py`.
- **Dockerfile**: `COPY waterfall_xirr.ico ./` added for production container.

### Action Plan v9 Fixes (Jul 30, 2026)
- **AP Fix 1 (Loan Extensions)**: DEPLOYED v147. `ExtensionOptions` column exists on PG `loans` table (not local SQLite). No code change needed — data-driven.
- **AP Fix 2b (Comment Fallback)**: DEPLOYED v154. Comments now fall back to most recent quarter when exact match is empty. Vue default quarter changed from hardcoded to dynamic.
- **AP Fix 4 (Construction Debt)**: BLOCKED. No `Inspection` table in PG (44 tables). MRI query written (`queries/MRI_Inspection.sql`) and sent to Charlene. Awaiting data.
- **AP Fix 5 (Chart Classification)**: Complete. 95 rows: 38 Azure data gap, 57 need account-level comparison, 0 code bugs. Excel sent to Charlene.
- **AP Fix 9 (U/W ROE)**: DEPLOYED v155. Fixed `_get_uw_pe_distributions()` mid-year cumulative-to-periodic bug. Was treating full YTD cumulative as single period when data starts mid-year (e.g. Donald Lynch July 2021 = $70K YTD treated as one month). Now pro-rates by dividing cumulative by month number.
- **AP Fix 10 (Budget Econ Occ)**: Already on main (commit 2285431). No additional changes needed.

### Prior Action Plan Fixes (v146-v147, Jul 27, 2026)
- **AP Fix 1 (Loan Extensions)**: DEPLOYED v147. `_get_extension_options()` + `(+2x12)` appended to maturity.
- **AP Fix 2 (PPI Entities)**: DEPLOYED v146+v147. `underlying_investors` column in `one_pager_comments` overrides with human-readable names.
- **AP Fix 2b (Comments Import)**: DEPLOYED v147. 139 rows imported from Charlene's spreadsheet (71 deals x 2 quarters).
- **AP Fix 3 (U/W Exit)**: BLOCKED. No "UW Exit Changes" table exists. Decision needed on approach.
- **AP Fix 5 (Chart Quarters)**: Investigation complete. 18 deals no data, 25 one-month lag, 24 stale.

## Code/Logic Bugs to Fix (~293 discrepancies, 31%)

### Bug 1: DSCR Principal Accounts (P0) — ~87 rows — FIXED (v144)
- Was using BS accounts {2145, 2150, 2152, 2154, 2156} — don't exist in Interim IS
- Fixed: YTD Actual uses IS acct 7060, fallback to BS balance change estimation
- U/W YE uses acct 7010 (total debt service from Projected IS)

### Bug 2: PE Capitalization Shows PPI Entities (P0) — 39 rows — FIXED (v147)
- Was showing PPI entity IDs (e.g., PPI27 100%) instead of underlying investors
- Fixed: `underlying_investors` from `one_pager_comments` table (human-readable), fallback to relationships-based PPI resolution

### Bug 3: PE Exposure on Value Formula (P1) — ~22 rows — VERIFIED CORRECT
- Formula `(debt + pref_equity) / current_valuation` is correct
- Verified: Ascent on Steamboat matches Excel exactly (107.5%)
- 45th & Main gap (140.6%) caused by Bug 4 (construction draw debt source), not formula

### Bug 4: Capitalization Debt Source (P1) — ~53 rows
- Construction loans: Azure falls back to MRI mOrigLoanAmt (commitment) instead of ISBS drawn balance
- Example: 45th & Main Debt: Azure $47.0M (commitment) vs Excel $31.4M (drawn)
- **Fix**: Ensure ISBS Interim BS data loaded for Q1 2026; verify quarter-date selection logic

### Bug 5: ROE to Date (P1) — ~30 rows — INVESTIGATED (methodology difference)
- Example: 30 Bearfoot: Azure 21.3% vs Excel 14.3%
- **Data freshness ruled out**: 30 Bearfoot has distributions through May 2026 (well past Q1 end)
- **Root cause**: Methodology/formula difference vs Excel, NOT data timing
- Possible causes: different annualization (simple vs compound), OP investor inclusion, weighted avg capital method
- **Next step**: Needs side-by-side Excel workbook comparison to identify exact formula difference

### Bug 6: Economic Occupancy Variances (P2) — ~56 rows — PARTIALLY FIXED (v144)
- Budget econ occ now deducts bad debt % from Budget IS (4040+4041+4043 / abs(4010))
- Remaining small variances likely budget_econ_occ data source differences

### Bug 7: Chart Issues (P2) — 17 missing quarters, 24 value mismatches — INVESTIGATED (data freshness)
- Chart code is correct: `cumulative_to_periodic()` and `aggregate_periodic()` logic verified
- "One quarter lag" for Town Fair Tire: Q1 2026 has only 1/3 months loaded in Interim IS → correctly excluded
- `aggregate_periodic` requires 3 months per quarter — incomplete quarters are rightfully omitted
- Missing quarters across deals = incomplete ISBS Interim IS data, not a code bug
- **No code fix needed** — charts auto-correct as MRI data is refreshed

## Data Gaps for Team (~540 discrepancies, 56%)

### Fix 3 Investigation: At Close — Burton Retail & Pontchartrain
- **Pontchartrain (P0000037)**: Data EXISTS in `at_close_noi` (NOI=$933K). Should display correctly. No issue.
- **Burton Retail (P0000109)**: Acquired 08/28/2025. NOT in `at_close_noi` table, no Projected IS fallback. Data gap — AM team needs to load at-close data via `Prop_Info_AtClose.sql`.

| Pri | Action | Owner | Count |
|-----|--------|-------|-------|
| P0 | Import at_close_noi (Prop_Info_AtClose.sql) + deal_terms (Prop_Info_DealTerms.sql) — Burton Retail confirmed missing | Data Team | ~200 |
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
