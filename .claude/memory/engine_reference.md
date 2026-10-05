# Engine reference — the computation invariants

Moved verbatim out of CLAUDE.md on Oct 5 2026, where it had grown past what a
file loaded into every session should carry. CLAUDE.md keeps the one-line rule
for each of these; the working detail — column names, account sets, fallbacks,
sign conventions, the reasons a rule is shaped the way it is — is here.

## Dates, waterfalls, capital calls, abatements, loans, sale overrides, parcel sales

### Acquisition Date
- **Derived from accounting feed**: `min(EffectiveDate)` per `InvestmentID` from the accounting table, mapped to vcode via investment map
- **Enriched at load time**: `data_service.py:load_all()` overwrites `Acquisition_Date` in `inv` DataFrame after loading both `inv` and `acct`. Also re-applied in `refresh_table()` when the deals table is refreshed.
- **Date parsing**: Raw `EffectiveDate` in the accounting adapter is strings (e.g., `"11/21/2016 0:00"`). Must parse via `pd.to_datetime()` before `groupby().min()` — string comparison is alphabetical, not chronological.
- **Rationale**: The MRI `Acquisition_Date` field may not reflect the true closing date. The earliest accounting activity (e.g., acquisition fee collected at closing) is the authoritative date. For development deals, first funding may occur months after acquisition (e.g., 3rd Ave & Indian School: acquisition fee 11/21/2016, first funding 5/15/2017).
- **Fallback**: If a deal has no accounting activity, the original MRI `Acquisition_Date` is preserved
- **Consumers**: Deal Analysis metadata, Sold Portfolio summary, One Pager general info — all use `inv["Acquisition_Date"]` automatically

### Sale Date & Anticipated Exit
- **Sale date priority**: (1) Sale date override from UI, (2) `event_dates` projected disposition closing (`vEventType='Disposition', vEvent='Closing', vDateType='Projected'`), (3) horizon end / max loan maturity. The `Sale_Date` column in the deals table is no longer consulted.
- **Underwritten Exit**: From MRI `event_dates` table (`vEventType='Asset Management', vEvent='U/W Exit', vDateType='Actual'`). Latest `dtEvent` wins. Function: `get_underwritten_exit()` in `one_pager.py`. Previously sourced from `inv.Sale_Date` — now sourced directly from `event_dates`.
- **Current Anticipated Exit**: From `event_dates` table (`vEventType='Disposition', vEvent='Closing', vDateType='Projected'`). Latest `dtEvent` wins when multiple rows exist. Used on One Pager and as default sale date for Deal Analysis projections. Function: `get_current_anticipated_exit()` in `one_pager.py`.
- **Shared helper**: Both functions use `_lookup_event_date()` in `one_pager.py` — generic event_dates query by vCode + vEventType + vEvent + vDateType.
- **MRI refresh safety**: `MRI_COLUMNS` in `mri_service.py` lists only columns that MRI refresh may overwrite. `Sale_Date`, `Sale_Status`, `InvestmentID`, and `Portfolio_Name` are explicitly excluded — they are preserved during upsert.
- **Sale date override**: Separate `sale_overrides` table (see Sale Overrides section) — completely independent of MRI refresh.
- **Event dates refresh**: `MRI_Event_Dates` and `MRI_Inspection` queries are included in the "Refresh All Data from MRI" process (`QUERY_REGISTRY` in `mri_service.py`).

### Waterfall Types
- **CF Waterfall**: Operating cash distributions (does NOT reduce capital outstanding)
- **Capital Waterfall**: Refi/sale proceeds (DOES reduce capital outstanding)

### Preferred Returns
- Daily accrual using Act/365 Fixed day count
- Compounds annually on 12/31 (with 45-day grace period)
- Tracked per investor via `InvestorState`

### Capital Calls
- **Data pipeline**: app entry → `capital_calls` table → `load_capital_calls()` → `build_capital_call_schedule()` → `apply_capital_calls_to_states()`
- **`capital_calls` IS IN `PROTECTED_TABLES`** (Sep 10 2026, Jim's instruction). The CSV import runs `to_sql(if_exists="replace")`, which DROPS the table, so a single `MRI_Capital_Calls.csv` upload silently destroyed every call typed into Deal Analysis. **Capital calls are now app-entered only** — added, edited and deleted per row via `/api/deals/<vcode>/raw-capital-calls`. Cost is only the bulk upload: the table is NOT in `mri_service.QUERY_REGISTRY`, so "Refresh All Data from MRI" never touched it and no automated feed is interrupted. The upload UI badges a protected file "locked" and excludes it from the importable count, so a skipped import announces itself. **Known consequence**: rows with a blank `Vcode` (5,127 of 5,130 locally — empty trailing rows from a past CSV) are invisible to the vcode-filtered GET and the replace that used to clear them is now blocked, so they cannot be cleaned from the UI. Harmless to every computation (`load_capital_calls` drops them via `dropna`), but permanent; purging them is a separate deliberate act. Guardrail: `scripts/capital_call_crud_check.py` asserts all four import entry points refuse the table AND that the rows survive, plus that app add/edit/delete still work.
- **Date handling**: Uses `pd.to_datetime(format='mixed', dayfirst=False)` to handle both CSV US dates ("6/30/2026") and HTML ISO dates ("2026-06-01")
- **Null filtering**: `dropna(subset=['deal_name', 'call_date', 'amount'])` removes empty rows from CSV imports
- **Column mapping**: Supports `entityid` → `deal_name`, `propcode` → `investor_id`, `calldate` → `call_date`
- **Table auto-creation**: `database.py` creates `capital_calls` table via `CREATE TABLE IF NOT EXISTS` in `create_additional_tables()`
- **CRUD endpoints**: `POST/PUT/DELETE /api/deals/<vcode>/capital-calls` in `deals.py`. All use `data_service.reload()` for full cache invalidation
- **Refi shortfall auto-clear**: After capital calls are applied, if total capital calls cover the refi shortfall (within $1 tolerance), `refi_capital_call_required` is set to False

### Tax Abatements
- **Account**: `TAX_ABATEMENT_ACCTS = {7070}` in `config.py` — stored in `forecast_feed` / `forecasts` table
- **Sign convention**: MRI stores abatements as negative (credit); `normalize_forecast_signs()` in `loaders.py` forces `base.abs()` (positive) since abatements increase cash flow
- **Annual Forecast**: Displayed as "Tax Abatement" row below NOI, included in FAD calculation. FAD adds back CapEx funded from cash reserves (`reporting.py`)
- **NPV at Sale**: Remaining abatement payments after sale date are discounted to PV at `TAX_ABATEMENT_DISCOUNT_RATE` (5%) and added to net sale proceeds (`compute.py`). Shown as "NPV (@5%) Tax Abatements" in Sale Proceeds Calculation (Debt Service section)
- **Below-the-line items**: Former "Excluded Accounts" renamed to "Other Below-the-Line" (`OTHER_EXCLUDED_ACCTS`): Interest Income (4050), Other Income/Expenses (5220, 5210, 5195, 7065), Partnership Expenses (5120, 5130), Depreciation & Amortization (5160, 5165), Extraordinary Expenses (5400). Other Revenue (4075) and Maintenance Flex (5092) are in NOI.
- **Conditional display**: Tax Abatement NPV line only appears in Sale Proceeds Calculation when deal has 7070 data
- **Comparability fix**: In U/W Projected IS, 7070 appears below NOI as a separate line item, but in actuals the abatement is netted into account 5090 (Real Estate Taxes). To ensure apples-to-apples comparison across all columns:
  - **One Pager** (`one_pager.py`): `calc_amounts()` folds 7070 into expenses (negative credit reduces expenses). `TAX_ABATEMENT: ['7070']` added to `IS_ACCOUNTS`. Separate adjustment for the pre-computed `at_close_noi_df` path.
  - **Property Financials** (`config.py`): `'7070'` added to `Real Estate Taxes` account list in `IS_ACCOUNTS['EXPENSES']`, so the income statement includes it alongside 5090 in all source comparisons.

### Paid-Off Loan Exclusion
- **Filter**: Loans with `vDateType = "Paid Off"` are excluded from all analysis at the data layer
- **Implementation**: `data_service.py:load_all()` filters `mri_loans_raw` after loading; same filter applied in `refresh_table()` for the `loans` table
- **Column detection**: Case-insensitive lookup (`c.lower() == "vdatetype"`) for robustness across CSV/MRI sources
- **Scope**: Affects all consumers — waterfall, capitalization, dashboard, debt service, assistant tools

### Balloon Loan Payoff at Sale
- **Detection**: Loan schedule's last row has `ending_balance < 1.0` (float tolerance), `principal > 0`, and prior row's `ending_balance > 0`
- **Forecast exclusion**: Balloon principal payments are excluded from forecast debt service rows (they are NOT operating expenses)
- **Sale proceeds**: Net sale proceeds deduct `total_loan_balance_at(sale_date) + balloon_total` — uses pre-balloon balance since balloon is paid at sale
- **Sale proceeds formula**: `max(0, value_net_selling_cost - loan_balances + tax_abatement_npv)`
- **Groupby**: Loan schedule grouped by `["vcode", "LoanID", "event_date"]` to distinguish loans with same dates
- **ISBS debt fallback**: When modeled loans aren't active at sale date (e.g. all loans paid off, or loan originates after sale), `compute.py` falls back to ISBS balance sheet debt (`get_isbs_debt_balance()`) for loan payoff amount. Also handles case where `loan_sched` is completely empty but ISBS shows outstanding debt.

### Sale Overrides
- **UI**: Contract Sale Price, Selling Costs (% or $), Sale Date — input row on Deal Analysis below capitalization table
- **Contract Sale Price**: Overrides the NOI/cap rate implied value when populated
- **Selling Costs**: Overrides the default 2% selling cost; supports percentage (`pct`) or fixed dollar (`fixed`) modes
- **Sale Date**: Overrides the default sale date (from `event_dates` projected disposition); threads through `compute_service.py` as `sale_date_override`
- **Database**: `sale_overrides` table (vcode PK, contract_sale_price, selling_cost_value, selling_cost_type, sale_date_override, updated_at, updated_by). In `PROTECTED_TABLES`.
- **Cache behavior**: Sale overrides do NOT create separate cache keys — `force=True` is set automatically when any override is present, overwriting the existing cache entry
- **Workflow**: Enter values → Recompute → optionally Save (persists to DB) or Clear. Saved overrides auto-load on deal select.
- **API**: `GET/PUT/DELETE /api/deals/<vcode>/sale-override`. POST `/compute` accepts overrides in body; falls back to saved overrides if not provided.
- **ISBS-anchored loan payoff**: Modeled amortization balance at sale date is scaled by `(ISBS_actual_debt / modeled_debt_at_anchor_date)` ratio. The ISBS Interim BS gives the real current outstanding; the scale factor corrects for differences between MRI origination amount and actual paydown. Diagnostic message shows anchor date, actual/modeled balances, and scale factor.

### Parcel Sales (Interim Sales)
Sales of part of a property before the final disposition. A deal can carry
several. Expandable box on Deal Analysis, directly above the Sale row.

- **Inputs per sale**: label, projected sale date, sale price, cost of sale (% or $), debt paydown allocated across mortgages, amount held in the CapEx reserve, distribution mode, and the revenue/expenses that leave with the parcel.
- **Database**: `parcel_sales` table (id, vcode, label, sale_date, sale_price, cost_of_sale_value/type, debt_application, capex_reserve_hold, distribution_mode, distribution_fixed, lost_revenue, lost_expense, notes, sort_order). The four structured columns are JSON blobs, following `prospect_assumptions`. In `PROTECTED_TABLES`.
- **Money flow** (`compute_economics()` in `parcel_sale_service.py`, the single definition the UI and engine share): price → less cost of sale → less debt paydown → less reserve hold → remainder distributed.
- **Reserve hold**: passed to `build_cash_flow_schedule_from_fad()` as `reserve_deposits`; joins the cash balance *before* CapEx is funded, then is spent by the ordinary reserve rules. Anything unspent returns at the final sale through the existing `get_sale_period_total_cash()` path — no separate logic. Note the reserve can also fund operating deficits that were previously unfunded, so a parcel sale may improve returns by more than its proceeds.
- **Debt paydown**: `Loan.curtailments` + the amortization builder. Convention is **re-amortize** — the payment is recalculated over the remaining term and the maturity date does not move, so debt service and DSCR improve from the paydown date. An interest-only loan simply drops its interest.
- **Distribution modes**: `waterfall` feeds the remainder into `cap_period_cash` as a Cap event at the parcel date (exact — `run_interleaved_waterfalls` processes it in date order); `pro_rata` splits on contributed capital; `fixed` uses per-partner amounts. All three are a **return of capital** — they reduce capital outstanding, so weighted average capital falls and ROE rises gradually rather than spiking.
- **Lost income**: `apply_parcel_income_loss()` spreads annual amounts across the months from the sale date, adjusting `mAmount_norm` only (the column every NOI/FAD/DSCR/terminal-value path reads). Revenue subtracts; expense adds back. Because the exit value is forward NOI / cap rate, **removing revenue automatically lowers the final sale price** — correct, and stated in diagnostics so it is not read as an error.
- **Tenant picker**: `get_deal_tenants()` reads the MRI roster and seeds Rental Income (4010); the figure stays editable because a point-in-time rent roll will not tie to a forecast carrying growth and rollover. Identical roster rows are collapsed — the roster returns every tenant twice and summing raw rows doubles the rent removed.
- **Cache**: parcel sales are loaded *inside* `get_cached_deal_result()` and fingerprinted into the cache key, so Deal Analysis, the reports and the assistant share one set of assumptions. The fingerprint covers only fields that change the projection, so a rename does not force a recompute.
- **Reporting**: four lines on the Annual Forecast below DSCR — Net Proceeds, Debt Paydown, To CapEx Reserve, Distributed — added only when the deal has a sale, and picked up by the Excel export automatically. `result['parcel_sales_applied']` is a JSON-safe audit record of what was applied.
- **Guards** (each reported rather than silent): a parcel date after the deal's sale date is ignored; a paydown at or before the actuals boundary is not modelled, since it is already in the reported balance and would otherwise corrupt the ISBS anchoring; a paydown allocated to a loan that is not modelled is not applied; a loan retired early by a paydown is not misread as a balloon and deducted again at sale; fixed amounts that miss the remainder are distributed as entered with the difference reported; an amount naming a partner not on the deal is not distributed.
- **All three modes process in date order**: `pro_rata`/`fixed` remainders are runner events (`build_manual_parcel_events()` + `manual_events` on `run_interleaved_waterfalls`) — pref accrues to the parcel date on the pre-cut balance, then the return of capital reduces the pools, so pref for every later period accrues on the reduced capital, same as the `waterfall` mode.
- **API**: `GET/POST /api/deals/<vcode>/parcel-sales`, `PUT/DELETE /api/deals/<vcode>/parcel-sales/<id>`, `GET /api/deals/<vcode>/parcel-sales/tenants`. Validation runs server-side on create and update; errors block the save, warnings do not.

## Cap rate, prospective loans, ISBS, At Close, occupancy, forecasts, the cutoff

### Cap Rate at Sale / Refinance
- **Source column**: `fCapRate` from `valuations` table (MRI_Val)
- **Date column**: `dtValuation` — the date each cap rate was assessed
- **Function**: `projected_cap_rate_at_date()` in `planned_loans.py` — sorts by `dtValuation`, takes the most recent `fCapRate` as the base rate
- **Escalation**: +0.05% (0.0005) per year from the `dtValuation` of the selected row to the target date (sale or refi)
- **Example**: fCapRate=0.0575 at 12/31/2025, sale at 12/31/2030 → 0.0575 + 5×0.0005 = 0.06 (6.00%)
- **Sale value**: `NOI_12_months / cap_rate` — uses `twelve_month_noi_after_date()` for forward 12-month NOI from forecast
- **Callers**: `compute_deal_analysis()` for sale proceeds, `size_planned_second_mortgage()` and `size_prospective_loan()` for loan sizing

### Prospective Loans (Refinancing)
- `size_prospective_loan()` in `planned_loans.py` returns sizing dict including `refi_date` for Vue form pre-fill
- Prospective loan extends sale date to new maturity when `Sale_ME` < new maturity
- Net sale proceeds formula: `sale_price - loan_balances - balloon_total + cash_reserves`
- **Loan type**: Radio selector — Supplemental (keep all existing loans) or Refinance (replace selected loans via checkbox list)
- **Multi-loan replacement**: `existing_loan_id` stores comma-separated LoanIDs; `compute.py` filters with `not in replacing_ids`; `planned_loans.py` uses `.isin(replacing_ids)` for balance lookup
- **Sizing constraints**: LTV (value x max LTV), DSCR (NOI / min DSCR → solve principal), Debt Yield (NOI / min yield), Quoted amount — binding = minimum
- **Workflow**: Draft → Save & Analyze (sizing only) → Accept (full waterfall recompute) → Revert. Editing an accepted loan auto-recomputes waterfall on save.
- **CRUD endpoints**: `GET/POST/PUT/DELETE /api/deals/<vcode>/prospective-loans`, plus `/accept`, `/revert`, `/sizing` per loan
- **PostgreSQL**: `ensure_pg_tables()` in `database.py` creates tables (prospective_loans, prospective_loans_audit, waterfall_audit) with `SERIAL` id + proper column types at startup; `_pg_fix_column_types()` migrates TEXT→INTEGER/DOUBLE PRECISION. Column names with mixed case (iOrder, PropCode, FXRate) must be double-quoted in SQL statements.

### ISBS Data Formats
- **Source column**: `vSource` in ISBS table (`ISBS_Download.csv`)
- **Interim IS** (Actuals): YTD cumulative trial balance snapshots — use `_get_cumulative_balances()` at a single date
- **Interim BS** (Balance Sheet): Current outstanding balances — used for debt via `get_isbs_debt_balance()`
- **Budget IS**: Periodic monthly amounts — use `_get_budget_sum()` over date range
- **Projected IS** (Underwriting): YTD cumulative trial balance snapshots — use `_get_cumulative_balances()` (same as Actuals)
- **Valuation**: Periodic monthly from `forecast_feed` — use `_get_valuation_sum()` with negated `mAmount_norm`
- **TTM from cumulative**: Current YTD + prior year Dec YTD - prior year same-month YTD
- **Performance chart / Dashboard**: Both correctly convert cumulative→periodic via `_cumulative_to_periodic()`

### ISBS Split Tables
MRI's query record limits make exporting the monolithic `ISBS_Download.csv` (800K+ rows) impractical. The ISBS data is split into 6 tables by `vSource`:

| Table | CSV Filename | vSource | Description |
|-------|-------------|---------|-------------|
| `isbs_interim_is` | `ISBS_Interim_IS.csv` | Interim IS | Actuals — YTD cumulative. **ALL of it**, not just 2025+ — 2018-03-31 to 2026-08-31 as of Sep 14 2026 |
| `isbs_interim_is_historical` | — (see below) | Interim IS | **Empty, and normally stays that way.** Extension point, not a second source |
| `isbs_interim_bs` | `ISBS_Interim_BS.csv` | Interim BS | Balance Sheet |
| `isbs_budget_is` | `ISBS_Budget_IS.csv` | Budget IS | Budget — periodic monthly |
| `isbs_projected_is` | `ISBS_Projected_IS.csv` | Projected IS | Underwriting — YTD cumulative |
| `isbs_valuation_is` | `ISBS_Valuation_IS.csv` | Valuation IS | Valuation — periodic monthly |
| `isbs_uw_supplements` | `ISBS_UW_Supplements.csv` | Projected IS | Supplemental UW records (e.g. 7073 capital contributions) — importable via CSV upload, persists across MRI refreshes (not in QUERY_REGISTRY) |

| `isbs_budget_is_supplements` | — (app-written) | Budget IS | Partner budgets imported and vetted in the valuation section, before MRI has them |

- **`isbs_interim_is_historical` does NOT hold pre-2025 actuals** (corrected Sep 14
  2026 — the earlier description here said it did, and sent a session hunting for data
  that was never missing). The refresh's split branch sends EVERY `Interim IS` row to
  `isbs_interim_is`; there is no date filter in that path. The 2025 cutoff exists only in
  `split_isbs_table()`, the one-time migration off the legacy monolithic `isbs` table.
  The `ISBS_Interim_IS_Historical.csv` name in `TABLE_DEFINITIONS` is a mapping with no
  file and no import behind it.
  **Keep the table anyway**: `_ISBS_SPLIT` in `data_service.py` concatenates it into
  `isbs_raw`, so rows put there DO reach every ISBS consumer, and the legacy migration
  still writes to it. It is a working extension point that happens to be empty — an
  MRI refresh reports it as `preserved` rather than importing it (`mri_service.py`,
  guardrail `scripts/isbs_historical_preserve_check.py`).

- **Supplement ownership decides protection** (Sep 11 2026). `isbs_budget_is_supplements` is in `PROTECTED_TABLES` because the APP writes it and holds the only copy of a budget between import and approval. The other four supplements are **deliberately not protected**: a CSV is their source of record and `replace` is their designed refresh. Protecting `isbs_uw_supplements`, which has no app write path, briefly froze its 56 rows (they feed One Pager PE ROE via 7073) — protection without a write path is a lockout, not a safeguard. Guardrail: `scripts/isbs_supplement_precedence_check.py`.
- **Where MRI and an app supplement share a key, the app wins** — `(vcode, dtEntry, vSource, vAccount)`. Written as "remove MRI rows whose key a supplement covers", NOT `drop_duplicates`: ISBS is a **journal**, one key legitimately carries many rows that consumers SUM, and the `drop_duplicates` formulation was measured taking `isbs_raw` from 797,660 to 439,268 rows.
- **Assembly**: `_assemble_isbs()` in `data_service.py` loads split tables, restores `vSource` column, concatenates into `isbs_raw`. Supplements from legacy monolithic `isbs` table for any missing vSource categories. Falls back entirely to legacy table if no split tables exist. After assembly, `_append_uw_supplements()` appends rows from `isbs_uw_supplements` (defaults `vSource='Projected IS'` if not present in CSV).
- **CSV upload**: Auto-detects split table filenames via `TABLE_DEFINITIONS` in `database.py`
- **Cache**: `refresh_table()` reassembles `isbs_raw` when any split table or `isbs_uw_supplements` is updated
- **Migration**: `split_isbs_table()` in `database.py` can migrate legacy monolithic table into splits (idempotent)
- **Consumers unchanged**: All code continues to use `isbs_raw` with `vSource` filtering
- **Pending**: Direct MRI database access via VPN (requested Apr 2026) to bypass record limits entirely

### ISBS Debt Balance
- **Config**: `DEBT_BS_ACCTS = {'2150', '2152', '2210'}` in `config.py` — Balance Sheet debt accounts
- **Function**: `get_isbs_debt_balance(isbs_raw, vcode, as_of_date=None)` in `compute.py`
- **Logic**: Filters ISBS to `vSource='Interim BS'`, deal vcode (case-insensitive), debt accounts; picks most recent period (or specific `as_of_date`); returns `abs(sum(mAmount))`
- **Hierarchy**: ISBS current outstanding preferred over MRI_Loans `mOrigLoanAmt` (static origination). Falls back to MRI_Loans if ISBS unavailable.
- **Usage**: Dashboard (`get_portfolio_caps()`), Deal Analysis (`get_deal_capitalization()`), One Pager (`get_capitalization_stack()` with quarter-specific date)
- **Date parsing**: Uses `pd.to_datetime(format='mixed')` first; Excel serial fallback only when >50% NaT

### At Close Data (One Pager)
- **Primary source**: `at_close_noi` table (from `Prop_Info_AtClose.sql` MRI query) — pre-computed per deal with dynamic date selection
- **Fallback**: ISBS `vSource='Projected IS'` at the earliest December 31 date per deal
- **Meaning**: Due diligence audit performed at original closing — represents underwritten expectations
- **Fields**: Revenue, Expenses, NOI, DSCR (debt service from Interest + Principal accounts)
- **Sign convention**: MRI stores revenue as negative (credit); negated to positive for display
- **Deal terms**: `deal_terms` table (from `Prop_Info_DealTerms.sql`) provides `econ_occ_at_close` and PE coupon/participation overrides
- **Implementation**: `get_property_performance()` in `one_pager.py` checks `at_close_noi_df` first, falls back to ISBS Projected IS scan

### Economic Occupancy (One Pager)
- **Formula**: `avg(physical occupancy YTD months) - bad_debt_concessions_pct`
- **Physical occupancy**: Average of `Occ%` from MRI_Occupancy_Download for YTD months of current year through quarter end
- **Bad debt/concessions %**: `(sum of vAccounts 4040 + 4043) / abs(sum of vAccount 4010) × 100` from ISBS Interim IS YTD
- **Accounts**: 4040 = Residential Concessions, 4043 = Bad Debt & Collection Loss, 4010 = Rental Income

### Sub-Portfolio Aggregation
- Deals can have child properties linked via `Portfolio_Name`
- Loans aggregate UP from properties to parent deal level
- See `consolidation.py` for implementation

### Forecast Assembly (Multi-Source)
- **Priority 1**: `forecast_feed.csv` (admin-uploaded via CSV import → `forecasts` table). Overrides everything for deals it covers. Used for draft/updated valuations before MRI approval.
- **Priority 2**: ISBS `vSource = 'Valuation IS'` — annual valuation cash flow projections (periodic monthly). These are the MRI-approved 10-year projections from the December 31 valuation exercise.
- **Priority 3**: ISBS `vSource = 'Projected IS'` — underwriting projections (YTD cumulative, auto-converted to periodic monthly).
- **Assembly**: `_assemble_forecasts()` in `data_service.py` — identifies deals covered by forecast_feed, then fills gaps from ISBS Valuation IS, then ISBS Projected IS. Combined result passed to `load_forecast()`.
- **Cumulative→Periodic**: Projected IS conversion subtracts prior same-year cumulative value; January = start of new year (no subtraction).
- **Pro_Yr derivation**: For ISBS-sourced rows, `Pro_Yr = date.year - pro_yr_base`.
- **vcode case**: ISBS normalizes vcodes to lowercase; `_restore_case()` converts back to original case (e.g., `p0000008` → `P0000008`) for case-sensitive matching in `compute_deal_analysis()`.
- **Cache refresh**: `refresh_table()` reassembles forecasts when `forecasts`, `isbs`, or any ISBS split table changes.

### Forecast Date Filtering
- **Anomalous dates**: MRI valuation exports can include "Year 0" base entries with dates far in the past (e.g. 2015-12-31 for a 2025 deal). `compute_deal_analysis()` filters out forecast rows with `event_date` before `start_year - 2` to prevent `model_start` from being set to an unreasonable date.
- **Beginning cash date selection**: `load_beginning_cash_balance()` uses the **most recent** available Interim BS date (not restricted to pre-model_start). This ensures account reclassifications (e.g. security deposits moved from 1010→1080) are reflected in the beginning cash. Previously restricted to dates before `model_start`, which could include misclassified balances from older periods.

### Actuals Through Cutoff
- Global setting (`actuals_through`): date or None (default None = full forecast)
- **Actuals/Forecast boundary**: Always enforced. If `actuals_through` is set, that date is the cutoff. Otherwise, defaults to Dec 31 of `start_year - 1`. XIRR cash flows come from `accounting_feed` only before the boundary, and from `forecast_feed` (waterfall) only after.
- **Partner cash flows**: `seed_states_from_accounting()` accepts `cutoff_date` parameter — only accounting entries on or before cutoff are used to seed InvestorState. Waterfall-computed distributions cover periods AFTER cutoff only.
- **Operating forecast**: Forecast Rev+Exp rows for months <= cutoff are removed from `fc_deal_full`
- **Waterfall**: `cf_period_cash` and `cap_period_cash` filtered to post-cutoff periods only (always, not just when `actuals_through` is set)
- **Date comparison safety**: Period filtering uses `pd.to_datetime()` on `event_date`/`EffectiveDate` columns before comparing with `pd.Timestamp` cutoff to avoid `Timestamp vs datetime.date` errors
- **Cache key**: includes `actuals_through` so toggling triggers recomputation
- **Defaults**: `DEFAULT_START_YEAR = 2026`, `DEFAULT_HORIZON_YEARS = 10`, `PRO_YR_BASE_DEFAULT = 2025`, `DEFAULT_ACTUALS_THROUGH = "2026-07-31"` (YTD Actuals enabled through July 2026)
- **UI**: Vue sidebar in Report Settings (checkbox + month-end selector)
- **Flask**: `ACTUALS_THROUGH` in config, passed via query params / request body, included in `/api/data/config`

## Account classifications (full)

## Account Classifications

- Revenue accounts: 4xxx series (positive in `mAmount_norm`)
- Expense accounts: 5xxx series (negative in `mAmount_norm`)
- Interest: `INTEREST_ACCTS` {5190, 7030}
- Principal: `PRINCIPAL_ACCTS` {7060}
- CapEx: `CAPEX_ACCTS` {7050}
- Tax Abatement: `TAX_ABATEMENT_ACCTS` {7070} — forced positive (income)
- Other Below-the-Line: `OTHER_EXCLUDED_ACCTS` {4050, 5220, 5210, 5195, 7065, 5120, 5130, 5400}
- `ALL_EXCLUDED` = Interest | Principal | CapEx | Other Below-the-Line (does NOT include Tax Abatement — separate sign handling)
- Debt (Balance Sheet): `DEBT_BS_ACCTS` {2150, 2152, 2210} — ISBS Interim BS accounts for current debt outstanding
- Cash & Reserves (Balance Sheet): `CASH_BALANCE_ACCTS` {1010, 1012, 1014, 1070, 1090, 1091, 1092, 1100, 1120, 1130, 1140, 1141, 1142, 1144, 1145} — ISBS Interim BS accounts for beginning cash balance in Cash Management section
- Bad Debt/Concessions: 4040 (Residential Concessions), 4043 (Bad Debt & Collection Loss), 4010 (Rental Income) — used in Economic Occupancy calculation
- Underwritten PE (Projected IS): `UW_PE_DIST_ACCT` 7071 (PE distributions — ROE numerator), `UW_PE_ROC_ACCT` 7073 (PE capital events — sign convention: positive = contribution, negative = return of capital). 7071 is YTD cumulative converted to periodic by `_get_uw_pe_periodic()`. 7073 uses `_get_uw_7073_signed()` for sign-preserving YTD→periodic conversion (positive MRI → negative cashflow/contribution, negative MRI → positive cashflow/ROC). Supplemental 7073 records loaded from `isbs_uw_supplements` table (importable via CSV upload, persists across MRI refreshes).
