# Portfolio Snapshot Performance Optimizations

Status: PENDING — waiting for Charlene to confirm current deploy works before implementing.

## Context
The `/api/portfolio-snapshot/bundle` endpoint computes all four subtabs live for ~32 deals (TIAA).
Load time is slow. Three bottlenecks identified (Aug 2026):

## Optimization 1: Trim the One Pager provider (biggest win)
**File**: `flask_app/services/portfolio_snapshot_freeze.py` — `_one_pager_provider()`

Currently calls `get_one_pager_data()` per deal, which runs 5 sub-functions:
- `get_capitalization_stack()` — USED by subtabs
- `get_property_performance()` — USED by subtabs
- `get_pe_performance()` — NOT USED, wasted work
- `get_general_information()` — NOT USED, wasted work
- `get_one_pager_comments()` — NOT USED, wasted work (DB round-trip)

**Fix**: Replace `get_one_pager_data()` with direct calls to only `get_capitalization_stack()` and `get_property_performance()`. Eliminates ~60% of per-deal work.

## Optimization 2: Direct ISBS lookup for quarterly NOI
**File**: `flask_app/services/portfolio_snapshot_freeze.py` — `_quarterly_noi_provider()`

The Loan subtab needs one number per deal: that quarter's periodic NOI for Debt Yield.
Currently runs the full Property Financials chart pipeline (`get_performance_chart_data()`)
which processes 12 quarters of ISBS cumulative-to-periodic-to-aggregate data.

**Fix**: Replace with direct ISBS query — single-quarter periodic NOI is just
`YTD_at_quarter_end - YTD_at_prior_quarter_end` for revenue/expense accounts.
Could be a single bulk query for all 32 deals instead of 32 separate chart pipelines.

## Optimization 3: Batch pe_cap_comments query
**File**: `flask_app/api/portfolio_snapshot.py` — `_pe_cap_comments()`

Guardrails loop calls `get_one_pager_comments()` individually per deal (32 DB round-trips).

**Fix**: One SQL query for all vcodes at once instead of looping.
