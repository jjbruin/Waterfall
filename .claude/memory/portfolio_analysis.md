# Portfolio Analysis Tab (May 2026)

## Overview
Generalized upstream entity analysis — traces deal cash flows through PPI entities to a holding company (e.g., TGA23). Two modes: Actual (saved waterfalls) and Proposed (simplified assumptions at entity level only).

## Architecture
- **3-layer upstream waterfall chain**: Deal waterfalls → PPI waterfalls → Entity waterfall
- **Entity-level accounting seeding**: `_build_entity_seeded_states()` creates InvestorStates with `capital_outstanding` from entity accounting so pref accrues. Passed as `pre_seeded_states` to `run_recursive_upstream_waterfalls()`.
- **Pref accrual in upstream waterfalls**: `accrue_all_pools()` called before each period's step processing in `run_upstream_waterfall_period()` (waterfall.py ~line 1548). Without this, pref is always $0.
- **Actual mode**: Uses saved waterfalls at all tiers via `run_recursive_upstream_waterfalls()`
- **Proposed mode**: Replaces only the selected entity's waterfall with simplified assumptions (AM Fee, Hurdle, Promote, Expenses). All lower-tier waterfalls remain actual.

## Files Created/Modified
- `flask_app/services/portfolio_analysis_service.py` — Main service (entity discovery, deal tracing, computation, Excel)
- `flask_app/api/portfolio_analysis.py` — 4 endpoints: entities, deals, compute, excel
- `vue_app/src/views/PortfolioAnalysisView.vue` — Full Vue frontend (~450 lines)
- `flask_app/__init__.py` — Blueprint registered (line 121-122)
- `vue_app/src/router/index.ts` — Route at `/portfolio-analysis` (line 73-76)
- `vue_app/src/components/layout/AppSidebar.vue` — Nav entry (line 27)

## API Endpoints
- `GET /api/portfolio-analysis/entities` — List portfolio entities (PSC1, PSC3, TGA23, TGA24, TGA25)
- `GET /api/portfolio-analysis/entities/<id>/deals` — Deals + investors for entity
- `POST /api/portfolio-analysis/entities/<id>/compute` — Body: `{mode, assumptions?}`
- `POST /api/portfolio-analysis/entities/<id>/excel` — Excel download (same body as compute)

## Allocation Table Features (May 7 session 2)
- **Combined CF/Cap table**: Single `<table>` with section headers ("CF Waterfall (Section 6.02)" / "Capital Waterfall (Section 6.03)"), unified year columns aligned vertically
- **Step labels**: `iOrder | FXRate% | vState | PropCode | rate%` (e.g., "2 | 90% | Pref | TGAM | 8.0%")
- **Step detail from wf_raw**: `step_detail` dict keyed by `(wf_type, iOrder, PropCode)` with FXRate summed from multiple rows (e.g., Capital Share 10% + GP Promote 20% = 30%)
- **Lead before Tag**: `_step_sort_key()` sorts by (iOrder, is_tag, label)
- **Capital Event Sources**: Shows which deals produced capital events by year
- **Entity Investor Balances**: End-of-projection diagnostic showing capital_outstanding and accrued pref per investor
- **Source deal tracking**: Extracted from upstream allocation `Path` column (first segment before "->")

## Bugs Fixed
1. **IRR=N/A, MOIC=0 in actual mode**: Entity states only had distributions, no contributions. Fix: seed contributions from `acct` DataFrame in `_build_entity_results()`
2. **TypeError in proposed mode ROE**: `cf_dists` was `List[float]` but `calculate_roe()` expects `List[Tuple[date, float]]`. Fix: changed to `cf_dists.append((ev_d, inv_share))`
3. **sum(cf_dists) TypeError**: `cf_dists` contains `(date, float)` tuples. Fix: `sum(a for _, a in cf_dists)` on line 733
4. **TGA23 waterfall had AMB23 instead of INV23**: AMB23 appeared in residual steps (CF step 4, Cap step 5) but AMB23 is not a direct investor — it's one level up (AMB23 owns INV23 which owns 10% of TGA23). Fixed in both local SQLite and Azure PostgreSQL.
5. **Pref steps always $0**: Entity InvestorStates created empty (capital_outstanding=0), pref never accrues. Fix: `_build_entity_seeded_states()` seeds capital from accounting; `accrue_all_pools()` added to upstream waterfall period processing.
6. **FXRate doubled (180% instead of 90%)**: Same (iOrder, PropCode) appears in both CF_WF and Cap_WF. Fix: keyed step_detail by `(wf_type, iOrder, PropCode)`.
7. **TGA23 CF step 3 (Default) modeling error**: iOrder=3, TGAM Default FXRate=0.9, INV23 Tag FXRate=0.1. Deleted from SQLite `waterfalls` table (vAmtType=6.02(c) only). Cap step 3 retained.

## Current Status (end of May 7 session 2)
- **Backend API**: Both modes working. TGA23 actual: Def_Int 18% now allocates cash (TGAM ~$1.7M/yr, INV23 ~$190K/yr). Pref 8% is $0 because 18% default pref consumes all available CF — correct behavior.
- **Vue frontend**: Combined CF/Cap table with aligned year columns, section headers, enriched step labels
- **TGA23 seeded capital**: TGAM=$102M, INV23=$11.6M (from entity accounting net contributions)
- **NOT YET COMMITTED**: 3 files modified (waterfall.py, portfolio_analysis_service.py, PortfolioAnalysisView.vue)
- **SQLite waterfall.db modified**: TGA23 CF step 3 deleted (not version-controlled)
- **NOT browser-tested**: Vue changes compiled but user hasn't visually verified the combined table yet

## Transfer-Aware Returns
- See [transfer_aware_returns.md](transfer_aware_returns.md) for full design doc
- PSC1 owned 10% of TGA23 from inception (2023-02-21) to 2024-03-31, then INV23 replaced it
- Scoped but NOT approved for implementation

## Next Steps
1. Browser test the combined allocation table in Vue
2. Commit all changes
3. Deploy to Azure (`az acr build` + `az containerapp update`)
4. Delete TGA23 CF step 3 from Azure PostgreSQL too (only deleted from local SQLite so far)

## Response Structure Differences
- **Actual mode** returns: `partner_results`, `deal_summary`, `deal_info`, `income_schedule`, `member_allocations`, `allocation_table`, `investors`, `errors`
- **Proposed mode** returns: `partner_results`, `deal_summary`, `deal_info`, `waterfall_detail`, `assumptions`, `investors`, `errors`
- Vue handles both via `computed()` with empty defaults

## Key Functions
- `_build_entity_seeded_states(entity_id, acct)` — Seeds InvestorStates with capital from accounting (portfolio_analysis_service.py)
- `_pivot_allocations_by_year(member_alloc_rows, ...)` — Builds combined CF/Cap allocation table with step labels (portfolio_analysis_service.py)
- `_step_label(order, state, member, fx_from_row)` — Builds enriched step label string (nested in _pivot_allocations_by_year)
- `run_recursive_upstream_waterfalls(..., pre_seeded_states)` — New param for entity capital seeding (waterfall.py)
