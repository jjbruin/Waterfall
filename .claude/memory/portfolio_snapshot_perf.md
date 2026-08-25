# Portfolio Snapshot Performance Optimizations

## Context
The `/api/portfolio-snapshot/bundle` endpoint computes all four subtabs live for ~32 deals (TIAA).
Load time was slow. Three bottlenecks identified (Aug 2026):

## Optimization 1: Trim the One Pager provider (biggest win) — DONE
**Commit**: `1f679c9` (v342, Aug 24 2026)
**File**: `flask_app/services/portfolio_snapshot_freeze.py` — `_one_pager_provider()`

Replaced `get_one_pager_data()` with direct calls to only `get_capitalization_stack()` and
`get_property_performance()`. Skips `pe_performance`, `general`, and `comments` — the 3
sections no subtab reads. Eliminates ~60% of per-deal work. Does NOT modify the shared
`get_one_pager_data`; lean path calls the two functions directly with the same args.

## Optimization 2: Direct ISBS lookup for quarterly NOI — PENDING
**File**: `flask_app/services/portfolio_snapshot_freeze.py` — `_quarterly_noi_provider()`

The Loan subtab needs one number per deal: that quarter's periodic NOI for Debt Yield.
Currently runs the full Property Financials chart pipeline (`get_performance_chart_data()`)
which processes 12 quarters of ISBS cumulative-to-periodic-to-aggregate data.

**Fix**: Replace with direct ISBS query — single-quarter periodic NOI is just
`YTD_at_quarter_end - YTD_at_prior_quarter_end` for revenue/expense accounts.
Could be a single bulk query for all 32 deals instead of 32 separate chart pipelines.

**Charlene's requirement**: Must reproduce the None rule (11 of 32 deals return None NOI —
dev deals + Giant 7 rollup), the account mapping, YTD-to-periodic conversion, and
parent/child rollup. Verify byte-for-byte against her baseline before shipping.

## Optimization 3: Batch pe_cap_comments query — DONE
**Commit**: `1f679c9` (v342, Aug 24 2026)
**File**: `flask_app/api/portfolio_snapshot.py` — `_pe_cap_comments()`

Replaced per-deal loop (32 DB round-trips) with a single batch SQL query.
Preserves behaviors: only non-empty comments returned, degrades to `{}` on failure.
